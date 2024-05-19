# from flax import linen as nn
# import jax.numpy as jnp
# import jax
#
# # Inspired by https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
# # Imported from https://github.com/CG80499/KAN-GPT-2/blob/master/transformer.py
#
# class ChebyKAN(nn.Module):
#     in_features: int
#     out_features: int
#     degree: int # degree of the basis polynomials
#
#     def setup(self):
#         assert self.degree > 0, "Degree of the Chebyshev polynomials must be greater than 0"
#         mean, std = 0.0, 1/ (self.in_features * (self.degree + 1))
#         self.coefficients = self.param("coefficients", lambda key, shape: mean + std * jax.random.normal(key, shape), (self.in_features, self.out_features, self.degree+1))
#
#     def __call__(self, x):
#         # x: (batch_size, in_features)
#         # normalize x between -1 and 1
#         x = jnp.tanh(x)
#         cheby_values = jnp.ones((x.shape[0], self.in_features, self.degree+1))
#         cheby_values = cheby_values.at[:, :, 1].set(x)
#         for i in range(2, self.degree+1):
#             next_value = 2 * x * cheby_values[:, :, i-1] - cheby_values[:, :, i-2]
#             cheby_values = cheby_values.at[:, :, i].set(next_value)
#         # cheby_values: (batch_size, in_features, degree+1)
#         # multiply by coefficients (in_features, out_features, degree+1)
#         return jnp.einsum('bid,ijd->bj', cheby_values, self.coefficients)
#
# class KANLayer(nn.Module):
#     polynomial_degree: int
#
#     @nn.compact
#     def __call__(self, x, det):
#         # y has shape (batch_size, seq_len, d_model) -> (batch_size * seq_len, d_model)
#         y = x.reshape((-1, x.shape[-1]))
#         y = ChebyKAN(in_features=x.shape[-1], out_features=x.shape[-1], degree=self.polynomial_degree)(y)
#         y = y.reshape(x.shape)
#         return y
from flax import linen as nn
import jax.numpy as jnp
import jax


def kan_ops2(x, coefficients):
    # x: (batch_size, in_features)
    # normalize x between -1 and 1
    # x = nn.tanh(x)
    # x = nn.gelu(x)

    in_features = x.shape[1]
    degree = coefficients.shape[-1] - 1

    cheby_values = jnp.ones((x.shape[0], in_features, 2), dtype=x.dtype)

    """ """
    cheby_values = cheby_values.at[:, :, 1].set(x)
    prev_values = cheby_values[:, :, 0]
    values = cheby_values[:, :, 1]

    # print(cheby_values)

    out = jnp.einsum('bi,ij->bj', prev_values,
                     coefficients[:, :, 0], ) + jnp.einsum('bi,ij->bj', values,
                                                           coefficients[:, :, 1],
                                                           )

    for i in range(2, degree + 1):
        temp = 2 * x * values - prev_values

        temp=nn.LayerNorm()(temp)

        prev_values = values
        values = temp
        out += jnp.einsum('bi,ij->bj', temp, coefficients[:, :, i], )

    # def loop_body(i, carry):
    #     prev_values, values, out = carry
    #     temp = 2 * x * values - prev_values
    #     prev_values = values
    #     values = temp
    #     out += jnp.einsum('bi,ij->bj', temp, coefficients[:, :, i])
    #     return (prev_values, values, out)
    #
    # prev_values, values, out = jax.lax.fori_loop(2, degree + 1, body_fun=loop_body, init_val=(prev_values, values, out))

    return out


# Inspired by https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
# Imported from https://github.com/CG80499/KAN-GPT-2/blob/master/transformer.py

class ChebyKAN(nn.Module):
    in_features: int
    out_features: int
    degree: int  # degree of the basis polynomials

    def setup(self):
        assert self.degree > 0, "Degree of the Chebyshev polynomials must be greater than 0"
        mean, std = 0.0, 1 / (self.in_features * (self.degree + 1))
        # self.coefficients = self.param("coefficients", lambda key, shape: mean + std * jax.random.normal(key, shape), (self.in_features, self.out_features, self.degree+1))

        # self.coefficients = self.param('coefficients', nn.initializers.kaiming_uniform(in_axis=(0,2),out_axis=2),
        #                                (self.in_features, self.out_features, self.degree + 1))

        kan_features = self.out_features  // (self.degree + 1)

        self.coefficients = self.param('coefficients', nn.initializers.kaiming_uniform(in_axis=(0, 2), out_axis=2),
                                       (kan_features, self.out_features, self.degree + 1))

        self.projection = nn.Dense(kan_features)
        self.projection_out = nn.Dense(self.out_features)

        self.linear1 = nn.Dense(self.out_features)
        self.linear2 = nn.Dense(self.in_features*4)

    def __call__(self, x):
        kan = self.kan_ops2_self(self.projection(x), self.coefficients)

        return kan + self.linear1(nn.silu(self.linear2(x)))
    @nn.compact
    def kan_ops2_self(self,x, coefficients):
        # x: (batch_size, in_features)
        # normalize x between -1 and 1
        x = nn.tanh(x)
        # x = nn.gelu(x)

        in_features = x.shape[1]
        degree = coefficients.shape[-1] - 1

        cheby_values = jnp.ones((x.shape[0], in_features, 2), dtype=x.dtype)

        """ """
        cheby_values = cheby_values.at[:, :, 1].set(x)
        prev_values = cheby_values[:, :, 0]
        values = cheby_values[:, :, 1]

        # print(cheby_values)

        out = jnp.einsum('bi,ij->bj', prev_values,
                         coefficients[:, :, 0], ) + jnp.einsum('bi,ij->bj', values,
                                                               coefficients[:, :, 1],
                                                               )

        for i in range(2, degree + 1):
            temp = 2 * x * values - prev_values
            prev_values = values
            values = temp
            out += jnp.einsum('bi,ij->bj', temp, coefficients[:, :, i], )

        # def loop_body(i, carry):
        #     prev_values, values, out = carry
        #     temp = 2 * x * values - prev_values
        #     prev_values = values
        #     values = temp
        #     out += jnp.einsum('bi,ij->bj', temp, coefficients[:, :, i])
        #     return (prev_values, values, out)
        #
        # prev_values, values, out = jax.lax.fori_loop(2, degree + 1, body_fun=loop_body, init_val=(prev_values, values, out))

        return out



class KANLayer(nn.Module):
    polynomial_degree: int

    @nn.compact
    def __call__(self, x, det):
        # y has shape (batch_size, seq_len, d_model) -> (batch_size * seq_len, d_model)
        y = x.reshape((-1, x.shape[-1]))
        y = ChebyKAN(in_features=x.shape[-1], out_features=x.shape[-1], degree=self.polynomial_degree)(y)
        y = y.reshape(x.shape)
        return y


if __name__ == "__main__":
    ChebyKAN(10, 10, 10).init(jax.random.PRNGKey(1), jnp.ones((1, 1)))
