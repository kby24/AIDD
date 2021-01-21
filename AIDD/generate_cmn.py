import networkx as nx
import scipy.sparse
import argparse
import torch
import numpy as np
import pickle
import random

torch.cuda.set_device(2)
use_cuda = torch.cuda.is_available()


parser = argparse.ArgumentParser()
# data volume = samples * eolving-steps
parser.add_argument('--nodes', type=int, default=100, help='Number of nodes, default=10')
parser.add_argument('--network', type=str, default='ER', help='type of network')
parser.add_argument('--samples', type=int, default=240, help='Number of samples in simulation, default=7000')
parser.add_argument('--prediction-steps', type=int, default=2, help='prediction steps, default=10')
parser.add_argument('--evolving-steps', type=int, default=100, help='evolving steps, default=100')
parser.add_argument('--lambd', type=float, default=3.5, help='lambda in logistic map, default=3.6')
parser.add_argument('--coupling', type=float, default=0.2, help='coupling coefficent, default=0.2')
args = parser.parse_args()



def logistic_map(x, lambd=args.lambd):
    # return 1 - lambd * x ** 2
    return lambd * x * (1 - x)


class CMLDynamicSimulator():
    def __init__(self, batch_size, sz, s):
        self.s = s
        self.thetas = torch.rand(batch_size, sz)

        if args.network=='ER':
            print('ER')
            #Avoid all elements in a column of the generated matrix are all 0
            flag = True
            while flag:
                self.G = nx.random_graphs.erdos_renyi_graph(sz, 0.04)
                A = nx.to_scipy_sparse_matrix(self.G, format='csr')
                n, m = A.shape
                diags = A.sum(axis=1)
                if 0 not in diags:
                    flag =False


        if args.network == 'WS':
            print('WS')

            #Avoid all elements in a column of the generated matrix are all 0
            flag = True
            while flag:
                self.G = nx.random_graphs.watts_strogatz_graph(sz, 2, 0.3)
                A = nx.to_scipy_sparse_matrix(self.G, format='csr')
                n, m = A.shape
                diags = A.sum(axis=1)
                if 0 not in diags:
                    flag =False

        if args.network == 'BA':
            print('BA')
            #Avoid all elements in a column of the generated matrix are all 0
            flag = True
            while flag:
                self.G = nx.random_graphs.barabasi_albert_graph(sz, 1)
                A = nx.to_scipy_sparse_matrix(self.G, format='csr')
                n, m = A.shape
                diags = A.sum(axis=1)
                if 0 not in diags:
                    flag =False

        A = nx.to_scipy_sparse_matrix(self.G, format='csr')
        n, m = A.shape
        diags = A.sum(axis=1)
        D = scipy.sparse.spdiags(diags.flatten(), [0], m, n, format='csr')

        self.obj_matrix = torch.FloatTensor(A.toarray())
        self.inv_degree_matrix = torch.FloatTensor(np.linalg.inv(D.toarray()))

        if use_cuda:
            self.thetas = self.thetas.cuda()
            self.obj_matrix = self.obj_matrix.cuda()
            self.inv_degree_matrix = self.inv_degree_matrix.cuda()


    def SetMatrix(self, matrix):
        self.obj_matrix = matrix
        if use_cuda:
            self.obj_matrix = self.obj_matrix.cuda()

    def SetThetas(self, thetas):
        self.thetas = thetas
        if use_cuda:
            self.thetas = self.thetas.cuda()

    def OneStepDiffusionDynamics(self):
        self.thetas = (1 - self.s) * logistic_map(self.thetas) + self.s * \
                      torch.matmul(torch.matmul(logistic_map(self.thetas), self.obj_matrix), self.inv_degree_matrix)
        return self.thetas


if __name__ == '__main__':

    # num_nodes = args.nodes
    # num_samples = args.samples
    prediction_steps = args.prediction_steps
    evolve_steps = args.evolving_steps
    lambd = args.lambd
    coupling = args.coupling
    num_nodes=args.nodes
    num_samples=args.samples
    # Generate data for multiple experiments at once
    for exp_id in range(1,11):
        print('Simulating time series...')

        # generate data
        simulator = CMLDynamicSimulator(batch_size=num_samples, sz=num_nodes, s=coupling)
        simulates = np.zeros([num_samples, num_nodes, evolve_steps, 1])
        sample_freq = 1
        for t in range((evolve_steps + 1) * sample_freq):
            locs = simulator.OneStepDiffusionDynamics()
            print(locs.size())
            if t % sample_freq == 0:
                locs = locs.cpu().data.numpy() if use_cuda else locs.data.numpy()
                simulates[:, :, t // sample_freq - 1, 0] = locs
        print('Simulation finished!')
        results = [simulator.obj_matrix, simulates]
        print(str(torch.sum(simulator.obj_matrix)))
        data_path = './data/cmn_'+args.network+'_'+str(args.nodes)+ '_id' + str(exp_id) + '.pickle'
        with open(data_path, 'wb') as f:
            pickle.dump(results, f)


