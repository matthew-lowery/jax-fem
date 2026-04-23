from __future__ import annotations

import equinox as eqx
import jax.random as jr


def create_lifted_module(base_layer, *, lift_dim, key):
    keys = jr.split(key, lift_dim)
    return eqx.filter_vmap(lambda current_key: base_layer(key=current_key))(keys)
