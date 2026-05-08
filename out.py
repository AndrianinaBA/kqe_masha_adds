import jax.numpy as jnp
from kqe.kernels import GaussianKernel
from kqe.kqd import ekqd, ekqd_centered

X1 = jnp.array([[4.0], [5.0]])
Y1 = jnp.array([[6.0], [7.0]])
print(ekqd(X1, Y1, num_projections=3, kernel_fn=GaussianKernel(l=2.0)))

X2 = jnp.array([[7.0], [9.0]])
Y2 = jnp.array([[8.0], [10.0]])
print(ekqd_centered(X2, Y2, num_projections=2, kernel_fn=GaussianKernel(l=1.0)))

X3 = jnp.array([[1.0], [2.0], [3.0]])
Y3 = jnp.array([[4.0], [5.0], [6.0]])
print(ekqd(X3, Y3, num_projections=3, kernel_fn=GaussianKernel(l=2.0), normalise=False, p=1, nu_shape="triangle", nu_ratio=0.0))
