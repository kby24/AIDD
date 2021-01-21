import argparse
import numpy as np
import networkx as nx
import pickle
import time
import pandas as pd
from scipy.integrate import odeint


#data volume = samples * times
parser = argparse.ArgumentParser()
parser.add_argument('--node_num', type=int, default=100, help='Number of nodes, default=100')
parser.add_argument('--dims', type=int, default=1, help='dims of nodes, default=1')
parser.add_argument('--samples', type=int, default=20000, help='sample times, default=20000')
parser.add_argument('--times', type=int, default=10, help='time steps, default=5000')
parser.add_argument('--network', type=str, default='Gene', help='network, default=ER')
parser.add_argument('--e_times', type=int, default=100, help='执行多少次,default=4')
parser.add_argument('--dt', type=float, default=0.01, help='ER P,default=0.04')
args = parser.parse_args()

def generate_network():
    dg = nx.DiGraph()
    # add nodes
    for i in range(10):
        dg.add_node(i)
    path = './gene_data/insilico_size100_3_goldstandard.tsv'
    df = pd.read_csv(path, header=None, sep='\t')

    for i in range(len(df)):
        if(int(df.at[i,2])!=0):
            first = int(df.at[i,0].lstrip('G')) - 1
            second = int(df.at[i,1].lstrip('G')) - 1
            dg.add_edge(first, second)
    print(len(dg.edges()))
    return dg


#calculate degree
def cal_degree(matrix):
    return np.sum(matrix,axis=0)



# return:{0:[1,2,3],1:[0,4]...}
def get_innode(adj):
	innodes = {}
	for i in range(adj.shape[0]):
		innode = []
		for j in range(adj.shape[0]):
			if adj[j][i] == 1:
				innode.append(j)
		innodes[i] = innode
	return innodes


def michaelis_menten(y,t,kesi):
    dydt = np.zeros((y.shape[0],))
    #J = np.loadtxt('Data/connectivity.dat')
    k = edges.shape[0]

    for i in range(k):
        if Ni[i]==0:
            #print('节点度为0'+str(i))
            dydt[i] = -y[i] + kesi[i]
        else:
            sum=0.0
            for j in neibour[i]:
                sum += (edges[j][i] * (y[j]/(1+y[j])))
            dydt[i] = -y[i] + sum*(1/Ni[i]) + kesi[i]

    return(dydt)


def generate_data():
    Y = np.array([])
    resolution=1
    for s in range(args.samples):
        kesi_array = np.zeros((args.node_num))
        init = 1+np.random.uniform(0.,1.,size=(args.node_num,))
        tspan=np.arange(0,args.times,resolution)
        y = odeint(michaelis_menten, init, tspan,args=(kesi_array,))
        Y = np.vstack((Y,y)) if Y.size else y

    new_data = Y
    data = new_data[:,:,np.newaxis]
    print(data.shape)
    results = [edges, data]
    return results




graph = generate_network()
edges = nx.adjacency_matrix(graph).toarray()
Ni = cal_degree(edges)
neibour = get_innode(edges)

for exp_id in range(1,11):
    print('exp_id:'+str(exp_id))
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('start_time:', start_time)
    print('Simulating time series...')
    result=generate_data()
    data_path =  './data/gene_id'+str(exp_id)+'.pickle'

    with open(data_path, 'wb') as f:
        pickle.dump(result, f)

    print('Simulation finished!')
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('end_time:', end_time)
