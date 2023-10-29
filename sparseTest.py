from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

numQubits = 15

DenseMat = lil_matrix((2**numQubits, 2**numQubits))


DenseMat[0, 0] = 1

DenseMat[1000, 27] = 15



print(DenseMat)