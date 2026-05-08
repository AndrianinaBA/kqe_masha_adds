import jax.numpy as jnp
from kqe.kernels import GaussianKernel
from kqe.kqd import ekqd

# Sample data
X = jnp.array([[1.0], [2.0], [3.0]])
Y = jnp.array([[1.5], [2.5], [3.5]])

# Kernel
k = GaussianKernel(l=1.0)

# Compute e-KQD²
ekqd_val = ekqd(X, Y, kernel_fn=k, num_projections=10)
print("eKQD²:", ekqd_val)