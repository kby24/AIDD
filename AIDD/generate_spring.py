import numpy as np
import time
import pickle
import networkx as nx
import scipy.sparse
import torch

#node number
n_ball = 100
# data volume = sample_T/sample_frequence
sample_T = 1000000
sample_frequence = 100
#network
net = 'ER'

#use_cuda = torch.cuda.is_available()

class SpringSim(object):
    def __init__(self, n_balls=n_ball, box_size=5., loc_std=.5, vel_norm=.5,
                 interaction_strength=.1, noise_var=0.):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

        if net == 'ER':
            print('ER')
            self.G = nx.random_graphs.erdos_renyi_graph(self.n_balls, 0.04)

        if net == 'WS':
            print('WS')
            self.G = nx.random_graphs.watts_strogatz_graph(self.n_balls, 2, 0.3)

        if net == 'BA':
            print('BA')
            self.G = nx.random_graphs.barabasi_albert_graph(self.n_balls, 1)
        A = nx.to_scipy_sparse_matrix(self.G, format='csr')

        self.obj_matrix = torch.FloatTensor(A.toarray())



    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size

        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(self, T=10000, sample_freq=10):
        n = self.n_balls
        assert (T % sample_freq == 0)
        #T_save = int(T / sample_freq - 1)
        T_save = int(T / sample_freq )
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        edges = self.obj_matrix.numpy()
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)

            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T+1):
                if i% 10000 == 0:
                    print('sample we have:',i/100)
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)

                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges

if __name__ == '__main__':
    #Generate data for multiple experiments at once
    for exp_id in range(1,11):

        sim = SpringSim()
        t = time.time()
        print('Simulating time series...')
        loc, vel, edges = sim.sample_trajectory(T=sample_T, sample_freq=sample_frequence)
        print('Simulation finished!')
        # loc:sample_time,loc,nodes
        # vel:sample_time,vel,nodes
        data_num = str(sample_T//sample_frequence)

        vel_path = './data/vel_' + net+ '_' + str(n_ball) + '_id' + str(exp_id) + '.pickle'
        loc_path = './data/sim_' + net + '_' + str(n_ball) + '_id' + str(exp_id) + '.pickle'
        edges_path = './data/edges_' + net + '_' + str(n_ball) + '_id' + str(exp_id) + '.pickle'

        with open(vel_path,'wb') as f:
            pickle.dump(vel,f)
        with open(loc_path,'wb') as f:
            pickle.dump(loc,f)
        with open(edges_path,'wb') as f:
            pickle.dump(edges,f)
        print(edges)
        print(edges.shape)
        print(np.sum(edges))

        print("Simulation time: {}".format(time.time() - t))
