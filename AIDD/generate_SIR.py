import numpy as np
from scipy import *
import pandas as pd
import datetime
import pickle
import time
from scipy.integrate import odeint
import argparse
import sys
import random

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int, default=5000, help='sample times,default=5000')
parser.add_argument('--sample_freq', type=int, default=50, help='sample frequency')
args = parser.parse_args()

# load city data
df1 = pd.read_csv('./city_data/county_city_province.csv')
df2 = pd.read_csv('./city_data/citypopulation.csv')
cities1 = set((df1['CITY']))
cities2 = set((df2['city']))
cities = set(list(cities1) + list(cities2))
nodes = {}
city_properties = {}
id_city = {}

for ct in cities:
    nodes[ct] = len(nodes)
    city_properties[ct] = {'pop': 1, 'prov': '', 'id': -1}

for i in df2.iterrows():
    city_properties[i[1][0]] = {'pop': float(i[1][1])}

for i in df1.iterrows():
    dc = city_properties.get(i[1]['CITY'], {})
    dc['prov'] = i[1]['PROV']
    dc['id'] = i[1]['CITY_ID']
    city_properties[i[1]['CITY']] = dc
    id_city[dc['id']] = i[1]['CITY']

def flushPrint(variable):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % variable)
    sys.stdout.flush()


def generate_network():
    df = pd.read_csv('./city_data/city_flow_v1.csv')
    flows = {}
    for n, i in enumerate(df.iterrows()):
        if n % 1000 == 0:
            flushPrint(n / len(df))
        cityi = (i[1]['cityi.id'])
        cityj = (i[1]['cityj.id'])
        value = flows.get((cityi, cityj), 0)
        flows[(cityi, cityj)] = value + i[1]['flowij']
        if cityi == 341301:
            print(flows[(cityi, cityj)])

    # save to flux matrix
    matrix = np.zeros([len(nodes), len(nodes)])
    self_flux = np.zeros(len(nodes))
    pij1 = np.zeros([len(nodes), len(nodes)])
    for key, value in flows.items():
        id1 = nodes.get(id_city[key[0]], -1)
        id2 = nodes.get(id_city[key[1]], -1)
        matrix[id1, id2] = value
    for i in range(matrix.shape[0]):
        self_flux[i] = matrix[i, i]
        matrix[i, i] = 0
        if np.sum(matrix[i, :]) > 0:
            pij1[i, :] = matrix[i, :] / np.sum(matrix[i, :])

    df = pd.read_csv('./city_data/Pij_BAIDU.csv', encoding='gbk')
    cities = {d: i for i, d in enumerate(df['Cities'])}
    pij2 = np.zeros([len(nodes), len(nodes)])
    for k, ind in cities.items():
        row = df[k]
        for city, column in cities.items():
            i_indx = nodes.get(city, -1)
            if i_indx < 0:
                print(city)
            j_indx = nodes.get(k, -1)
            if j_indx < 0:
                print(k)
            if i_indx >= 0 and j_indx >= 0:
                pij2[j_indx, i_indx] = row[column] / 100
                if i_indx == j_indx:
                    pij2[i_indx, j_indx] = 0
    bools = pij2 <= 0
    pij = np.zeros([pij1.shape[0], pij1.shape[0]])
    for i in range(pij1.shape[0]):
        row = pij1[i]
        bool1 = bools[i]
        values = row * bool1
        if np.sum(values) > 0:
            ratios = values / np.sum(values)
            sum2 = np.sum(pij2[i, :])
            pij[i, :] = (1 - sum2) * ratios + pij2[i, :]
    zeros = np.argwhere(np.sum(pij, axis=1) == 0).reshape(-1)
    for idx in zeros:
        pij[idx][idx] = 1

    print(np.sum(pij, 1))  # Testing normalization

    # > 0.02210023336 2%
    pij_c = (pij > 0.02210023336) + 0
    pij_c = pij_c.astype(int)
    pij_c2 = np.zeros((pij_c.shape[0],pij_c.shape[0]))
    for i in range(pij_c.shape[0]):
        pij_c2[i, :] = pij_c[i, :] / np.sum(pij_c[i, :])
    print(np.sum(pij_c2,1))
    # return pij
    return pij_c2.T

def diff(sicol, t, r_0, t_l, gamma, pijt):

    sz = sicol.shape[0] // 3
    Is = sicol[:sz]
    Rs = sicol[sz:2 * sz]
    ss = sicol[2 * sz:]

    I_term = Is.dot(pijt) - Is * np.sum(pijt, axis=0)
    S_term = ss.dot(pijt) - ss * np.sum(pijt, axis=0)
    R_term = Rs.dot(pijt) - Rs * np.sum(pijt, axis=0)
    cross_term = r_0 * Is * ss / t_l

    delta_I = cross_term - Is / t_l + gamma * I_term
    delta_S = - cross_term + gamma * S_term
    deta_R = Is / t_l + gamma * R_term
    output = np.r_[delta_I, deta_R, delta_S]
    return output


def generate_data(matrix):

    experiments = pd.read_pickle('./city_data/parameters/experiments_ti_tr_120_new.pkl')
    experiments = experiments + pd.read_pickle('./city_data/parameters/experiments_ti_tr_120_new_2.pkl')
    experiments = experiments + pd.read_pickle('./city_data/parameters/experiments_ti_tr_120_new_3.pkl')
    best_para = experiments[0]
    fit_param = sorted([(vvv[1], i) for i, vvv in enumerate(best_para)])
    itm = best_para[fit_param[0][1]]

    t_days = 300
    steps = 1000  # 1000
    r0 = itm[0][0][0].item()
    t_l = itm[2][1]
    gamma = itm[2][2]
    timespan = np.linspace(0, t_days, steps)

    for i in range(args.times):
        if i % 100 ==0:
            print(10*i)

        Is0 = np.zeros(len(nodes))
        Ss0 = np.ones(len(nodes))
        Rs0 = np.zeros(len(nodes))

        #Randomly select a city
        city_chose = random.randint(0, 370)
        Is0[city_chose] = random.uniform(1, 10) * (1e-4)
        Rs0[city_chose] = random.uniform(1, 10) * (1e-4)
        Ss0[city_chose] = 1 - Is0[city_chose] - Rs0[city_chose]
        #odeint ,Solve differential equations
        result = odeint(diff, np.r_[Is0, Rs0, Ss0], timespan, args=(r0,  t_l, gamma, matrix))

        sz = result.shape[1] // 3
        Is = result[:, :sz]  # infected
        Rs = result[:, sz:2 * sz]  # recover
        Ss = result[:, 2 * sz:3 * sz]  # susceptible

        len_node = Is.shape[1]
        Is_list = np.zeros((1000 // args.sample_freq, len_node))
        Rs_list = np.zeros((1000 // args.sample_freq, len_node))
        Ss_list = np.zeros((1000 // args.sample_freq, len_node))
        for j in range(Is_list.shape[0]):
            Is_list[j] = Is[args.sample_freq * j, :]
            Rs_list[j] = Rs[args.sample_freq * j, :]
            Ss_list[j] = Ss[args.sample_freq * j, :]

        Is_data2 = Is_list[:, :, np.newaxis]
        Rs_data2 = Rs_list[:, :, np.newaxis]
        Ss_data2 = Ss_list[:, :, np.newaxis]
        simu = np.concatenate((Is_data2, Rs_data2, Ss_data2), axis=2)

        if(i==0):
            simu_all = simu
        else :
            simu_all =  np.concatenate((simu_all,simu),axis=0)

    print(simu_all.shape)

    results = [matrix, simu_all]

    return results


for exp_id in range(1,11):
    edges= generate_network()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('start_time:', start_time)
    print('Simulating time series...')
    result=generate_data(edges)
    print('Simulation finished!')

    data_path = './data/SIR_id'+str(exp_id)+'.pickle'

    with open(data_path, 'wb') as f:
        pickle.dump(result, f)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('end_time:', end_time)
