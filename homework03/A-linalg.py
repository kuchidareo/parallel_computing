import numpy as np


def solver(a, b):
    return np.linalg.solve(a, b)


n = 10
rng = np.random.default_rng(2026)
a = rng.random((n, n))
b = rng.random(n)
x = solver(a, b)

np.testing.assert_allclose(a @ x, b)
print("max error:", np.max(np.abs(a @ x - b)))
