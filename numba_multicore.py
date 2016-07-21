import numpy as np
from numba import float32, jit, vectorize


@jit('float32[:,:], float32[:,:]', cache=True)
def pairwise_numba(X, D):
    M = X.shape[0]
    N = X.shape[1]
    for i in xrange(M):
        for j in xrange(M):
            d = 0.0
            for k in xrange(N):
                tmp = X[i,k] - X[j,k]
                d += tmp*tmp
            D[i,j] = np.sqrt(d)


@vectorize('float32(float32, float32)', target='cpu')
def subpow2(x,y):
    return (x-y)*(x-y)


class multicore(object):
    """docstring for multicore"""
    def __init__(self):
        from ipyparallel import Client
        rc = Client()
        rc.block=True
        self.cpu = rc[:]
        
    def kde(self, BT, X):
        self.cpu.push(dict(pairwise_numba=pairwise_numba))
        self.cpu.scatter('x_query', X)
        self.cpu.execute('import numpy as np')
        self.cpu.execute('dist = BT.query(x_query, k=15)[0]')
        self.cpu.execute('dens = np.mean(dist,axis=1)')
        return self.cpu.gather('dens')


@jit('float32[:,:], int64[:], float64[:], int64[:], float32[:]', cache=False)
def _get_tao_numba(X, ind, rho, reduced, tao):
    N = X.shape[0]
    M = X.shape[1]
    k = 0
    for i in range(N-1):
        if ind[i] in reduced:
            _min = 10
            for j in range(N-ind[i]):
                a = ind[i] + j
                for l in range(M):
                    tmp = 0.0
                    tmp += (X[i,l] - X[a,l])*(X[i,l] - X[a,l])
                if tmp<_min:
                    _min=tmp
            tao[k] = _min
            k += 1
