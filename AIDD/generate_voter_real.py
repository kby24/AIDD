

import random
import time
import numpy as np
import argparse
import torch
import networkx as nx
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='voter',
                    help='What simulation to generate.')
parser.add_argument('--num_samples', type=int, default=2000,
                    help='Number of training simulations to generate.')

parser.add_argument('--length', type=int, default=50,
                    help='Length of trajectory.')

parser.add_argument('--n_nodes', type=int, default=1133,
                    help='Number of balls in the simulation.')
parser.add_argument('--network', type=str, default='EMAIL',
                    help='network of the simulation')


parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')

args = parser.parse_args()

# torch.cuda.set_device(3)
# use_cuda = torch.cuda.is_available()


simulates = np.zeros((args.num_samples*args.length,args.n_nodes))

# get the innode of each node
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


# init node data randomly
def init_node(dg):
    for i in range(dg.number_of_nodes()):
        dg.nodes[i]['value'] = random.randint(0, 1)


# let the net spread by probility
def spread_prob(dg, innodes, step=100):
    node_num = dg.number_of_nodes()
    # data to be returned
    data = []
    # add initial value to data
    origin_val = []
    for i in range(node_num):
        origin_val.append(dg.nodes[i]['value'])
    data.append(origin_val)


    # control the circulates
    run = 0
    # step is the only limitation because there is no conception like attractor and so on...
    while run < step:
        run += 1
        # each step
        next_val = []

        for i in range(node_num):
            # num for neighbors who vote for agree
            k = 0.
            # num for all neighbors
            m = len(innodes[i])
            for iter, val in enumerate(innodes[i]):
                if dg.nodes[val]['value'] == 1:
                    k += 1.
            try:
                if random.random() < k / m:
                    next_val.append(1)
                else:
                    next_val.append(0)
            except ZeroDivisionError:
                if random.random() < 0.5:
                    next_val.append(1)
                else:
                    next_val.append(0)

        # print(next_val)
        # set value to the net
        for i in range(node_num):
            dg.nodes[i]['value'] = next_val[i]

        # just add to data to record
        data.append(next_val)
    return np.array(data)

def generate_network():
    if args.network == 'ROAD':
        print('ROAD')
        G = nx.Graph()
        node_num = 1174
        for i in range(node_num):
            G.add_node(i, value=random.randint(0, 1))
        path = './real_network_data/out.subelj_euroroad_euroroad'
        # 读取文件
        f = open(path)
        flag = 0
        for line in f:
            if flag < 2:
                flag = flag + 1
                continue
            first = int(line.split(' ')[0]) - 1
            second = int(line.split(' ')[1]) - 1
            G.add_edge(first, second)
        print(len(G.edges()))

    if args.network== 'BLOG':
        print('BLOG')
        G = nx.DiGraph()
        node_num = 1224
        for i in range(node_num):
            G.add_node(i, value=random.randint(0, 1))
        path = './real_network_data/out.moreno_blogs_blogs'
        # 读取文件
        f = open(path)
        flag = 0
        for line in f:
            if flag < 2:
                flag = flag + 1
                continue
            first = int(line.split(' ')[0]) - 1
            second = int(line.split(' ')[1]) - 1
            G.add_edge(first, second)
        print(len(G.edges()))

    if args.network == 'EMAIL':
        print('EMAIL')
        G=nx.Graph()
        node_num = 1133
        for i in range(node_num):
            G.add_node(i, value=random.randint(0, 1))
        path = './real_network_data/out.arenas-email'
        # 读取文件
        f = open(path)
        flag = 0
        for line in f:
            if flag < 2:
                flag = flag + 1
                continue
            first = int(line.split(' ')[0]) - 1
            second = int(line.split(' ')[1]) - 1
            G.add_edge(first, second)
        print(len(G.edges()))

    if args.network == 'DORM':
        print('DORM')
        G=nx.DiGraph()
        node_num = 217
        for i in range(node_num):
            G.add_node(i, value=random.randint(0, 1))
        path = './real_network_data/out.moreno_oz_oz'
        # 读取文件
        f = open(path)
        flag = 0
        for line in f:
            if flag < 2:
                flag = flag + 1
                continue
            first = int(line.split(' ')[0]) - 1
            second = int(line.split(' ')[1]) - 1
            G.add_edge(first, second)
        print(len(G.edges()))

    return G

for exp_id in range(1,4):
    dg = generate_network()
    edges = nx.adjacency_matrix(dg).toarray()
    print(edges)
    innodes = get_innode(edges)


    for i in range(args.num_samples):
        init_node(dg)
        data = spread_prob(dg,  innodes,step=args.length-1)
        simulates[i*args.length:(i+1)*args.length,:] = data

    print(simulates.shape)
    print(str(np.sum(edges)))
    all_data = simulates[:, :, np.newaxis]
    print(all_data.shape)

    # save the data
    # save time series data
    # data path
    series_path = './data/voter_' + args.network + '_' + str(args.n_nodes) + '_id' + str(exp_id) + '_data.pickle'
    adj_path = './data/voter_' + args.network + '_' + str(args.n_nodes) + '_id' + str(exp_id) + '_adj.pickle'

    with open(series_path, 'wb') as f:
        pickle.dump(all_data, f)

    with open(adj_path, 'wb') as f:
        pickle.dump(edges, f)


