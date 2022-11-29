import numpy as np
from context import ipfn

N = 12000

def print_error(m, xip, xpj):
    colsum_err = np.sum((np.sum(m, 0) - xip.squeeze())**2)
    rowsum_err = np.sum((np.sum(m, 1) - xpj.squeeze())**2)
    print(f"Error:  {colsum_err:8.3f}  {rowsum_err:8.3f}")

# Construct NxN matrix with ones on the main and upper diagonal
m = np.eye(N, dtype='float32')
m.flat[1:N*N:N+1] = 1

aggregates = [np.ones(N), np.ones(N)]
dimensions = [[0], [1]]

print_error(m, *aggregates)
m, converged, history = ipfn.ipfn(m, aggregates, dimensions, max_iteration=10, verbose=2).iteration()
print_error(m, *aggregates)
print(history)
