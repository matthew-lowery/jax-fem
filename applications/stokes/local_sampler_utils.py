from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class GaussianSampler(eqx.Module):
    mu: jax.Array
    raw_sig: jax.Array

    def __init__(self, *, dim, key):
        key_mu, key_sig = jr.split(key)
        self.mu = 0.01 * jr.normal(key_mu, (dim,))
        self.raw_sig = -2.0 + 0.1 * jr.normal(key_sig, (dim,))

    @property
    def sig(self):
        return jax.nn.softplus(self.raw_sig)

    @property
    def num_components(self):
        return 1

    def sample_base_noise(self, key, batch_shape=()):
        return jr.normal(key, batch_shape + self.mu.shape)

    def sample_eps(self, eps):
        return self.mu + eps * self.sig

    def scale_mean(self):
        return self.sig.mean()


class RelaxedMixtureGaussianSampler(eqx.Module):
    logits: jax.Array
    mu: jax.Array
    raw_sig: jax.Array
    temperature: float = eqx.field(static=True)

    def __init__(self, *, dim, num_components, temperature, key):
        key_logits, key_mu, key_sig = jr.split(key, 3)
        self.logits = 0.01 * jr.normal(key_logits, (num_components,))
        self.mu = 0.01 * jr.normal(key_mu, (num_components, dim))
        self.raw_sig = -2.0 + 0.1 * jr.normal(key_sig, (num_components, dim))
        self.temperature = temperature

    @property
    def sig(self):
        return jax.nn.softplus(self.raw_sig)

    @property
    def mixture_weights(self):
        return jax.nn.softmax(self.logits)

    @property
    def num_components(self):
        return self.logits.shape[0]

    def sample_base_noise(self, key, batch_shape=()):
        key_eps, key_gumbel = jr.split(key)
        eps = jr.normal(key_eps, batch_shape + self.mu.shape)
        gumbel = jr.gumbel(key_gumbel, batch_shape + self.logits.shape)
        return eps, gumbel

    def sample_eps(self, eps):
        gaussian_eps, gumbel_noise = eps
        component_samples = self.mu + gaussian_eps * self.sig
        weights = jax.nn.softmax((self.logits + gumbel_noise) / self.temperature)
        return weights @ component_samples

    def scale_mean(self):
        return jnp.sum(self.mixture_weights[:, None] * self.sig) / self.sig.shape[1]


def build_sampler(*, sampler_type, dim, key, num_components, temperature):
    if sampler_type == "gaussian":
        return GaussianSampler(dim=dim, key=key)
    if sampler_type == "mog":
        return RelaxedMixtureGaussianSampler(
            dim=dim,
            num_components=num_components,
            temperature=temperature,
            key=key,
        )
    raise ValueError(f"Unsupported sampler_type={sampler_type}")


def sample_noise_batch(sampler, key, batch_size):
    return sampler.sample_base_noise(key, (batch_size,))


def sampler_scale_mean(sampler):
    return sampler.scale_mean()
