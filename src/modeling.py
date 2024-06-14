# Copyright 2024 Jungwoo Park (affjljoo3581)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Literal

import einops
import flax.linen as nn
import flax.linen.initializers as init
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
from flax.training.train_state import TrainState

from utils import fixed_sincos2d_embeddings, get_layer_index_fn
from kan import KANLayer

import optax
import jax
import flax

DenseGeneral = partial(nn.DenseGeneral, kernel_init=init.truncated_normal(0.02))
Dense = partial(nn.Dense, kernel_init=init.truncated_normal(0.02))
Conv = partial(nn.Conv, kernel_init=init.truncated_normal(0.02))


@dataclass
class ViTBase:
    layers: int = 12
    dim: int = 768
    heads: int = 12
    labels: int | None = 1000
    layerscale: bool = False

    patch_size: int = 16
    image_size: int = 224
    posemb: Literal["learnable", "sincos2d"] = "learnable"
    pooling: Literal["cls", "gap"] = "cls"

    dropout: float = 0.0
    droppath: float = 0.0
    grad_ckpt: bool = False
    use_kan: bool = False
    polynomial_degree: int = 8
    dtype:Any=jnp.bfloat16

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(ViTBase)}

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2


class PatchEmbed(ViTBase, nn.Module):
    def setup(self):
        self.wte = Conv(
            self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
        )
        if self.pooling == "cls":
            self.cls_token = self.param(
                "cls_token", init.truncated_normal(0.02), (1, 1, self.dim)
            )

        if self.posemb == "learnable":
            self.wpe = self.param(
                "wpe", init.truncated_normal(0.02), (*self.num_patches, self.dim)
            )
        elif self.posemb == "sincos2d":
            self.wpe = fixed_sincos2d_embeddings(*self.num_patches, self.dim)

    def __call__(self, x: Array) -> Array:
        x = (self.wte(x) + self.wpe).reshape(x.shape[0], -1, self.dim)
        if self.pooling == "cls":
            cls_token = jnp.repeat(self.cls_token, x.shape[0], axis=0)
            x = jnp.concatenate((cls_token, x), axis=1)
        return x


class Attention(ViTBase, nn.Module):
    def setup(self):
        self.wq = DenseGeneral((self.heads, self.head_dim),dtype=self.dtype)
        self.wk = DenseGeneral((self.heads, self.head_dim),dtype=self.dtype)
        self.wv = DenseGeneral((self.heads, self.head_dim),dtype=self.dtype)
        self.wo = DenseGeneral(self.dim, axis=(-2, -1))
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        z = jnp.einsum("bqhd,bkhd->bhqk", self.wq(x) / self.head_dim ** 0.5, self.wk(x))
        z = jnp.einsum("bhqk,bkhd->bqhd", self.drop(nn.softmax(z), det), self.wv(x))
        return self.drop(self.wo(z), det)


class FeedForward(ViTBase, nn.Module):
    def setup(self):
        self.w1 = Dense(self.hidden_dim,dtype=self.dtype)
        self.w2 = Dense(self.dim,dtype=self.dtype)
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        return self.drop(self.w2(self.drop(nn.gelu(self.w1(x)), det)), det)


class ViTLayer(ViTBase, nn.Module):
    def setup(self):
        self.attn = Attention(**self.kwargs)
        if self.use_kan:
            self.ff = KANLayer(self.polynomial_degree)
        else:
            self.ff = FeedForward(**self.kwargs)

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.drop = nn.Dropout(self.droppath, broadcast_dims=(1, 2))

        self.scale1 = self.scale2 = 1.0
        if self.layerscale:
            self.scale1 = self.param("scale1", init.constant(1e-4), (self.dim,))
            self.scale2 = self.param("scale2", init.constant(1e-4), (self.dim,))

    def __call__(self, x: Array, det: bool = True) -> Array:
        x = x + self.drop(self.scale1 * self.attn(self.norm1(x), det), det)
        x = x + self.drop(self.scale2 * self.ff(self.norm2(x), det), det)
        return x


class ViT(ViTBase, nn.Module):
    def setup(self):
        self.embed = PatchEmbed(**self.kwargs)
        self.drop = nn.Dropout(self.dropout)

        # The layer class should be wrapped with `nn.remat` if `grad_ckpt` is enabled.
        layer_fn = nn.remat(ViTLayer) if self.grad_ckpt else ViTLayer
        self.layer = [layer_fn(**self.kwargs) for _ in range(self.layers)]

        self.norm = nn.LayerNorm()
        self.head = Dense(self.labels) if self.labels is not None else None

    def __call__(self, x: Array, det: bool = True) -> Array:
        x = self.drop(self.embed(x), det)
        for layer in self.layer:
            x = layer(x, det)
        x = self.norm(x)

        # If the classification head is not defined, then return the output of all
        # tokens instead of pooling to a single vector and then calculate class logits.
        if self.head is None:
            return x

        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "gap":
            x = x.mean(1)
        return self.head(x)


@dataclass
class MAEBase:
    mask_ratio: int = 0.75
    decoder_dim: int = 512
    decoder_layers: int = 1
    decoder_heads: int = 16
    decoder_posemb: Literal["learnable", "sincos2d"] = "learnable"


class MAE(ViTBase, MAEBase, nn.Module):
    def setup(self):
        self.embed = PatchEmbed(**self.kwargs)
        self.drop = nn.Dropout(self.dropout)

        # The layer class should be wrapped with `nn.remat` if `grad_ckpt` is enabled.
        layer_fn = nn.remat(ViTLayer) if self.grad_ckpt else ViTLayer
        self.layer = [layer_fn(**self.kwargs) for _ in range(self.layers)]

        self.norm = nn.LayerNorm()

        self.decoder_embed = Dense(self.decoder_dim)

        self.mask_token = self.param(
            "mask_token", init.truncated_normal(0.02), (1, 1, self.decoder_dim)
        )

        self.decoder_pos_embed = self.param(
            "decoder_pos_embed", init.truncated_normal(0.02), (1, self.num_patches[0] ** 2, self.decoder_dim)
        )

        kwargs = self.kwargs
        kwargs.update({'dim': self.decoder_dim, 'heads': self.decoder_heads, 'layers': self.decoder_layers})
        print(kwargs)
        self.decoder_layer = [layer_fn(**kwargs) for _ in range(self.decoder_layers)]

        self.decoder_norm = nn.LayerNorm()
        self.decoder_pred = Dense(self.patch_size ** 2 * 3)

    def random_masking(self, x):
        rng = self.make_rng("random_masking")
        # x = einops.rearrange(x, 'b (h k1) (w k2) c->b (h w) (c k1 k2)', k1=self.patch_size, k2=self.patch_size)

        b, n, d = x.shape
        len_keep = int(n * (1 - self.mask_ratio))

        noise = jax.random.uniform(rng, shape=(b, n))
        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        mask = jnp.ones((b, n))
        mask = mask.at[:, :len_keep].set(0)
        mask = jnp.take(mask, ids_restore)

        x = jnp.take_along_axis(x, ids_shuffle[:, :len_keep, None], axis=1)
        print(x.shape)

        return x, mask, ids_restore

    def forward_encoder(self, x, det: bool = True):
        x = self.drop(self.embed(x), det)



        x, mask, ids_restore = self.random_masking(x)

        for layer in self.layer:
            x = layer(x, det)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # print('\n' * 5)
        # print(x.shape)
        x = self.decoder_embed(x)
        # print(x.shape)

        mask_tokens = jnp.tile(self.mask_token, (x.shape[0], ids_restore.shape[1] - x.shape[1], 1))

        # print(mask_tokens.shape,x.shape[0])

        x = jnp.concatenate([x, mask_tokens], axis=1)
        x = jnp.take_along_axis(x, ids_restore[..., None], axis=1)
        x = x + self.decoder_pos_embed

        for layer in self.decoder_layer:
            x = layer(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        # print(mask_tokens.shape, x.shape)
        return x

    def forward_loss(self, x, pred, mask):
        target = einops.rearrange(x, 'b (h k1) (w k2) c->b (h w) (c k1 k2)', k1=self.patch_size, k2=self.patch_size)

        mean = target.mean(axis=-1, keepdims=True)
        var = target.var(axis=-1, keepdims=True)
        target = (target - mean) / (var + 1.e-6) ** .5


        loss = (pred - target) ** 2
        loss = loss.mean(axis=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def __call__(self, images: Array, det: bool = True, rng=None):
        print(images.shape)
        latent, mask, ids_restore = self.forward_encoder(images, det)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(images, pred, mask)
        return loss, pred, mask


class TrainState(train_state.TrainState):
    mixup_rng: PRNGKey
    dropout_rng: PRNGKey
    random_masking_rng: PRNGKey

    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None

    def split_rngs(self) -> tuple[ArrayTree, ArrayTree]:
        mixup_rng, new_mixup_rng = jax.random.split(self.mixup_rng)
        dropout_rng, new_dropout_rng = jax.random.split(self.dropout_rng)
        random_masking_rng, new_random_masking_rng = jax.random.split(self.random_masking_rng)

        rngs = {"mixup": mixup_rng, "dropout": dropout_rng, 'random_masking': random_masking_rng}
        updates = {"mixup_rng": new_mixup_rng, "dropout_rng": new_dropout_rng,
                   'random_masking_rng': new_random_masking_rng}
        return rngs, updates

    def replicate(self) -> TrainState:
        return flax.jax_utils.replicate(self).replace(
            mixup_rng=shard_prng_key(self.mixup_rng),
            dropout_rng=shard_prng_key(self.dropout_rng),
            random_masking_rng=shard_prng_key(self.random_masking_rng),
        )

    def replace_tx(self, tx):
        return flax.jax_utils.unreplicate(self).replace(tx=tx)


def create_optimizer(learning_rate, weight_decay, warmup_steps, training_steps, decay=True):
    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
            learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = optax.lion(
            learning_rate=learning_rate,
            # b1=0.95,b2=0.98,
            # eps=args.adam_eps,
            weight_decay=weight_decay,
            mask=partial(jax.tree_util.tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        from jax.tree_util import tree_map_with_path
        if decay:
            lr_decay = 1.0
            layers = 12
            # if args.lr_decay < 1.0:
            layerwise_scales = {
                i: optax.scale(lr_decay ** (layers - i))
                for i in range(layers + 1)
            }
            label_fn = partial(get_layer_index_fn, num_layers=layers)
            label_fn = partial(tree_map_with_path, label_fn)
            tx = optax.chain(tx, optax.multi_transform(layerwise_scales, label_fn))
            tx = optax.chain(optax.clip_by_global_norm(1), tx)
        return tx

    print(learning_rate, weight_decay, warmup_steps, training_steps)
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=training_steps,
        end_value=1e-5,
    )

    tx = create_optimizer_fn(learning_rate)
    return tx


def create_train_state(rng,
                       layers=12,
                       dim=192,
                       heads=3,
                       labels=10,
                       layerscale=True,
                       patch_size=2,
                       image_size=32,
                       posemb="learnable",
                       pooling='gap',
                       dropout=0.0,
                       droppath=0.0,
                       clip_grad=1.0,
                       warmup_steps=None,
                       training_steps=None,
                       learning_rate=None,
                       weight_decay=None,
                       ema_decay=0.9999,
                       trade_beta=5.0,
                       label_smoothing=0.1,
                       aux_rng_keys: list = ["random_masking"],
                       decay=True

                       ):
    """Creates initial `TrainState`."""

    cnn = MAE(
        layers=layers,
        dim=dim,
        heads=heads,
        labels=labels,
        layerscale=layerscale,
        patch_size=patch_size,
        image_size=image_size,
        posemb=posemb,
        pooling=pooling,
        dropout=dropout,
        droppath=droppath,
    )

    # cnn=RNGModule()

    num_keys = len(aux_rng_keys)
    key, *subkeys = jax.random.split(rng, num_keys + 1)
    rng_keys = {aux_rng_keys[ix]: subkeys[ix] for ix in range(len(aux_rng_keys))}

    # image_shape = [1, 28, 28, 1]
    image_shape = [1, 32, 32, 3]

    # print(rng_keys)
    # cnn.init({'params': rng, **rng_keys}, jnp.ones(image_shape))

    params = cnn.init({'params': rng, }, jnp.ones(image_shape))['params']
    """
    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
            learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = optax.lion(
            learning_rate=learning_rate,
            # b1=0.95,b2=0.98,
            # eps=args.adam_eps,
            weight_decay=weight_decay,
            mask=partial(jax.tree_util.tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        # if args.lr_decay < 1.0:
        #     layerwise_scales = {
        #         i: optax.scale(args.lr_decay ** (args.layers - i))
        #         for i in range(args.layers + 1)
        #     }
        #     label_fn = partial(get_layer_index_fn, num_layers=args.layers)
        #     label_fn = partial(tree_map_with_path, label_fn)
        #     tx = optax.chain(tx, optax.multi_transform(layerwise_scales, label_fn))
        if clip_grad > 0:
            tx = optax.chain(optax.clip_by_global_norm(clip_grad), tx)
        return tx

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=training_steps,
        end_value=1e-5,
    )
    """

    # learning_rate = optax.warmup_cosine_decay_schedule(
    #     init_value=1e-7,
    #     peak_value=LEARNING_RATE,
    #     warmup_steps=50000 * 5 // TRAIN_BATCH_SIZE,
    #     decay_steps=50000 * EPOCHS // TRAIN_BATCH_SIZE,
    #     end_value=1e-6,
    # )

    # tx = create_optimizer_fn(learning_rate)

    tx = create_optimizer(learning_rate, weight_decay, warmup_steps, training_steps, decay=decay)

    return TrainState.create(
        apply_fn=cnn.apply,
        params=params,
        tx=tx,
        mixup_rng=jax.random.PRNGKey(1 + jax.process_index()),
        dropout_rng=jax.random.PRNGKey(2 + jax.process_index()),
        random_masking_rng=jax.random.PRNGKey(3 + jax.process_index()),
        micro_step=0,
        micro_in_mini=1,
        grad_accum=1 if 1 > 1 else None,
    )


if __name__ == "__main__":
    rng = jax.random.PRNGKey(1)
    state = create_train_state(rng, layers=1, warmup_steps=1000, training_steps=10000000, weight_decay=0.05,
                               learning_rate=1e-3).replicate()
    batch = 2
    image_shape = [batch, 32, 32, 3]

    k1 = 1
    k2 = 1
    x = jnp.ones(image_shape)
    # x = jax.random.normal(rng, image_shape)
    """
    x = einops.rearrange(x, 'b (h k1) (w k2) c ->b (h w) (c k1 k2) ', k1=k1, k2=k2, )

    b, n, d = x.shape
    # print(rng)
    noise = jax.random.uniform(rng, shape=(b, n))
    # ids_shuffle = jnp.argsort(noise, axis=1)
    ids_shuffle = jnp.arange(0, n)[None, :].repeat(b, 0)
    ids_restore = jnp.argsort(noise, axis=1)

    mask_ratio = 0.998
    len_keep = int(n * (1 - mask_ratio))
    ids_shuffle_expand = ids_shuffle[:, :len_keep, None]  #.repeat(d, -1)

    # print(x[ids_shuffle[:, :len_keep], ids_shuffle[:, :len_keep]])

    # print(x[ids_shuffle[:, :len_keep], ids_shuffle[:, :len_keep]])
    # print('\n'*5)
    print(x[:, :len_keep])
    print('\n' * 5)
    print(jnp.take_along_axis(x, ids_shuffle_expand, axis=1))
"""


    @partial(jax.pmap, axis_name="batch", )
    def test(state):
        def loss(params):
            # y = state.apply_fn({'params': params, }, x, rngs={'random_masking': rng})
            # return optax.losses.l2_loss(y, jnp.zeros_like(y)).mean()

            loss, pred, mask = state.apply_fn({'params': params, }, x, rngs={'random_masking': rng})
            return loss

        grad = jax.grad(loss)(state.params)
        state = state.apply_gradients(grads=grad)

        return state


    # print(state.opt_state)

    new_tx = create_optimizer(1e-5, 1, 1, 100, decay=True)

    new_state = create_train_state(rng, layers=1, warmup_steps=1000, training_steps=10000000, weight_decay=0.05,
                                   learning_rate=1e-3, decay=False).replicate()

    # print('\n'*5)
    # print(new_state.opt_state)

    # loss(state.params)
    # state.opt_state

    # state.replace()

    old_opt_state = state.opt_state
    state = test(state)
    # grad = jax.grad(loss)(state.params)
    # state = state.apply_gradients(grads=grad)

    # state.replace(opt_state=old_opt_state)

    print(state.opt_state.hyperparams)

    state = new_state.replace(opt_state=state.opt_state)
    # print(state.opt_state)
    state = test(state)
    # state.replace(opt_state=old_opt_state)

    print(state.opt_state)

    # state=state.replace(step=100)
    # print(state.opt_state)

    """
    x, mask, ids_restore = loss(state.params)
   
    mask_tokens = jnp.zeros((x.shape[0], ids_restore.shape[1] - x.shape[1], x.shape[2]))

    x = jnp.concatenate([x, mask_tokens], axis=1)
    print(ids_restore[0])

    # ids_restore=jnp.arange(0,256)[None,:].repeat(x.shape[0],0)
    # print(ids_restore[0])
    # ids_restore = ids_restore[:, :, None].repeat(x.shape[-1], -1)
    print(x[0, :, 0])
    print(ids_restore)
    x = jnp.take_along_axis(x, ids_restore[..., None], axis=1)

    # x = jnp.take(x, ids_restore[:, :, None].repeat(x.shape[-1], -1))

    print(x.shape, ids_restore.shape)
    # print(x[0]-y[0])
    print(x[0, :, 0])
    print(jnp.take_along_axis(mask, ids_restore, axis=1)[0, :])

    while True:
        pass

    x = einops.rearrange(x, 'b (h w) (c k1 k2) ->b (h k1) (w k2) c', k1=2, k2=2, h=16)

    
    import matplotlib.pyplot as plt

    print(x.shape, mask_tokens.shape)
    plt.imshow(x[0])
    plt.show()

    plt.imshow(x[1])
    plt.show()

    """
