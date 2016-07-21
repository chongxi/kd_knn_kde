import numpy as np
from sklearn.neighbors import KDTree, BallTree
from sklearn.neighbors import KernelDensity
from sklearn.datasets import make_blobs
from hdbscan._hdbscan_linkage import mst_linkage_core, mst_linkage_core_cdist
import matplotlib.pyplot as plt
X = np.random.randn(17000,32)
BT = BallTree(X)



class multicore_KNN(object):
    """docstring for multicore"""
    def __init__(self):
        from ipyparallel import Client
        rc = Client()
        rc.block=True
        self.cpu = rc[:]
        print '{} cores ready'.format(len(self.cpu))
        self.cpu.execute('import numpy as np')
        self.cpu.execute('from sklearn.neighbors import KDTree, BallTree')
        
    def _kde(self, BT, X):
        self.cpu.push(dict(BT=BT))
#         self.cpu.push(dict(k=k))
        self.cpu.scatter('x_query', X)
        self.cpu.execute('dist = BT.query(x_query, k=5)[0]')
        self.cpu.execute('dens = np.mean(dist,axis=1)')
        return self.cpu.gather('dens')
    
    def kde(self, X, k=15, method='kd'):
        self.cpu.push(dict(X=X, k=k))
        self.cpu.scatter('ind', range(X.shape[0]))
        if method == 'kd':
            self.cpu.execute('KT = KDTree(X)')
            self.cpu.execute('dist = KT.query(X[ind], k)[0]')
            self.cpu.execute('dens = 1/np.mean(dist,axis=1)')
        if method == 'ball':
            self.cpu.execute('BT = BallTree(X)')
            self.cpu.execute('dist = BT.query(X[ind], k)[0]')
            self.cpu.execute('dens = 1/np.mean(dist,axis=1)')            
        return self.cpu.gather('dens')
    
    def knn_fullscan(self, X, k=15, method='kd'):
        self.cpu.push(dict(X=X, k=k))
        self.cpu.scatter('ind', range(X.shape[0]))
        if method == 'kd':
            self.cpu.execute('KD = KDTree(X)')
            self.cpu.execute('knn_dist, knn_indices = KD.query(X[ind], k=k, dualtree=True)')
            self.cpu.execute('knn_dens = 1/np.mean(knn_dist, axis=1)')
        if method == 'ball':
            self.cpu.execute('BT = BallTree(X)')
            self.cpu.execute('knn_dist, knn_indices = BT.query(X[ind], k=k, dualtree=True)')
            self.cpu.execute('knn_dens = 1/np.mean(knn_dist, axis=1)')
        return self.cpu.gather('knn_dist'), self.cpu.gather('knn_indices'), self.cpu.gather('knn_dens')
    
    def compress(self, X, k=15):
        self.knndist, self.knnind, self.rho = self.knn_fullscan(X, k=k, method='kd')
        self.reduced = np.asarray([i for i in np.arange(X.shape[0]) if self.rho[self.knnind[i]].argmax() == 0])
        return self.rho, self.reduced
    
    def view_compressed(self, points, reduced):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(points[:, 0], points[:, 1], s=20, alpha=0.3)
        ax[0].scatter(points[reduced, 0], points[reduced, 1], s=40, c="red", alpha=0.7)
        ax[0].set_aspect('equal')

        # KD = KDTree(points)
        # new_rho = KD.kernel_density(points[reduced], h=0.2)
        # ax[1].scatter(points[reduced, 0], points[reduced, 1], s=20, c=new_rho, alpha=0.7)
        # ax[1].set_aspect('equal')

        random_sample = np.random.choice(points.shape[0], len(reduced), replace=False)
        ax[1].scatter(points[:, 0], points[:, 1], s=20, alpha=0.3)
        ax[1].scatter(points[random_sample, 0], points[random_sample, 1], s=40, c="red", alpha=0.7)
        ax[1].set_aspect('equal')
        fig.tight_layout()

if __name__ == '__main__':
    mc = multicore()
    dens = mc.kde(BT, X)
    print dens
