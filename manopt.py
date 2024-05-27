import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

anp.random.seed(42)

dim = 2
circle = pymanopt.manifolds.Sphere(dim)

matrix = anp.random.normal(size=(dim, dim))
matrix = 0.5 * (matrix + matrix.T)


@pymanopt.function.autograd(circle)
def cost(point):
    return -point @ matrix @ point


problem = pymanopt.Problem(circle, cost)

optimizer = pymanopt.optimizers.SteepestDescent()
result = optimizer.run(problem)

eigenvalues, eigenvectors = anp.linalg.eig(matrix)
dominant_eigenvector = eigenvectors[:, eigenvalues.argmax()]

print("Dominant eigenvector:", dominant_eigenvector)
print("Pymanopt solution:", result.point)
