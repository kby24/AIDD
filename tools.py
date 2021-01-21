import torch
import numpy as np
import os
import math
import pickle
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sgm import sgraphmatch
import torch.nn.functional as F
import networkx as nx
from networkx.convert import from_dict_of_dicts
from networkx.classes.graph import Graph
# if use cuda
use_cuda = torch.cuda.is_available()

def calc_tptnfpfn(out,adj):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if adj[i][j] == 1:
                # positive
                if out[i][j] == 1:
                    # true positive
                    tp += 1
                else:
                    # false positive
                    fn += 1
            else:
                # negative
                if out[i][j] == 1:
                    # true negative
                    fp += 1
                else:
                    # false neg
                    tn += 1
    return tp,tn,fp,fn

def tpr_fpr(out,adj):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if adj[i][j] == 1:
                # positive
                if out[i][j] == 1:
                    # true positive
                    tp += 1
                else:
                    # false positive
                    fn += 1
            else:
                # negative
                if out[i][j] == 1:
                    # true negative
                    fp += 1
                else:
                    # false negative
                    tn += 1
    # tpr = tp /  (tp + fp)
    # return tp,tn,fp,fn
    try:
        tpr = float(tp) / (tp + fn)
    except ZeroDivisionError:
        tpr=0
    try:
        fpr = float(fp) / (fp + tn)
    except ZeroDivisionError:
        fpr = 0
    return tpr,fpr

def calc_tpr_fpr(matrix, matrix_pred):
    matrix = matrix.to('cpu').data.numpy()
    matrix_pred = matrix_pred.to('cpu').data.numpy()


    tpr,fpr = tpr_fpr(matrix_pred,matrix)

    return tpr, fpr

def evaluation_indicator(tp,tn,fp,fn):
    try:
        tpr = float(tp) / (tp + fn)
    except ZeroDivisionError:
        tpr=0
    try:
        fpr = float(fp) / (fp + tn)
    except ZeroDivisionError:
        fpr = 0
    try:
        tnr = float(tn) / (tn + fp)
    except ZeroDivisionError:
        tnr = 0
    try:
        fnr = float(fn) / (tp + fn)
    except ZeroDivisionError:
        fnr = 0
    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        p = 0
    try:
        r = tp / (tp + fn)
    except ZeroDivisionError:
        r = 0
    try:
        f1_score = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1_score = 0
    return tpr, fpr, tnr, fnr, f1_score

def calc_tptnfpfn_dyn(out,target):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if target[i][j] == 1:
                # positive
                if out[i][j] == 1:
                    # true positive
                    tp += 1
                else:
                    # false positive
                    fn += 1
            else:
                # negative
                if out[i][j] == 1:
                    # true negative
                    fp += 1
                else:
                    # false neg
                    tn += 1
    return tp,tn,fp,fn

def evaluator(output, target):
    out = torch.zeros(output.size(0), output.size(1))
    output = output.cpu()
    target = target.cpu()
    for i in range(output.size(0)):
        for j in range(output.size(1)):
            if output[i, j, 0] > output[i, j, 1]:
                out[i, j] = 0
            else:
                out[i, j] = 1

    err = torch.sum(torch.abs(out - target))

    err = err.item() / out.size(0)

    fpr, tpr, threshold = roc_curve(target.view(-1).numpy(),out.view(-1).numpy())
    roc_auc = auc(fpr, tpr)
    tp, tn, fp, fn = calc_tptnfpfn_dyn(out, target)
    tp = tp / output.size(0)
    tn = tn / output.size(0)
    fp = fp / output.size(0)
    fn = fn / output.size(0)
    tpr, fpr, tnr, fnr, f1_score = evaluation_indicator(tp, tn, fp, fn)
    return err, tp, tn, fp, fn, tpr, fpr, tnr, fnr, f1_score, roc_auc



def constructor_evaluator(gumbel_generator, tests, obj_matrix,e):

    err_list = []
    tp_list=[]
    tn_list = []
    fp_list = []
    fn_list = []
    tpr_list = []
    fpr_list = []
    tnr_list = []
    fnr_list = []
    f1_list=[]
    auc_list=[]
    soft_auc_list=[]

    obj_matrix = torch.from_numpy(obj_matrix)
    for t in range(tests):

        mat = gumbel_generator.gen_matrix.cpu().view(-1, 2)
        y_score1 = torch.nn.functional.softmax(mat, dim=1)[:, 0].detach().numpy()
        fpr, tpr, threshold = roc_curve(obj_matrix.numpy().reshape(-1), y_score1)
        soft_auc = auc(fpr, tpr)

        out_matrix = gumbel_generator.sample_all(hard=True, epoch=e)
        out_matrix = out_matrix.cpu()
        fpr, tpr, threshold = roc_curve(obj_matrix.numpy().reshape(-1), out_matrix.numpy().reshape(-1))
        roc_auc = auc(fpr, tpr)
        err = torch.sum(torch.abs(out_matrix - obj_matrix))

        err = err.cpu() if use_cuda else err
        # if we got nan in err
        if math.isnan(err):
            print('problem cocured')
            # torch.save(gumbel_generator,'problem_generator_genchange.model')
            d()
            t=t-1
            continue
        err_list.append(err.data.numpy().tolist())
        tp, tn, fp, fn = calc_tptnfpfn(out_matrix,obj_matrix)
        tpr, fpr, tnr, fnr, f1_score = evaluation_indicator(tp,tn,fp,fn)
        tp_list.append(tp)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        tnr_list.append(tnr)
        fnr_list.append(fnr)
        f1_list.append(f1_score)
        auc_list.append(auc)
        soft_auc_list.append(soft_auc)
    print('err:', np.mean(err_list))
    print('tp:', np.mean(tp_list))
    print('tn:', np.mean(tn_list))
    print('fp:', np.mean(fp_list))
    print('fn:', np.mean(fn_list))
    print('tpr:', np.mean(tpr_list))
    print('fpr:', np.mean(fpr_list))
    print('tnr:', np.mean(tnr_list))
    print('fnr:', np.mean(fnr_list))
    print('f1:', np.mean(f1_list))
    print('auc:', np.mean(auc_list))
    print('soft_auc:', np.mean(soft_auc_list))






def constructor_evaluator_SIR(gumbel_generator, tests, obj_matrix, e):
    # obj_matrix = obj_matrix.cuda()
    errs = []
    # tprs = []
    # fprs = []
    obj_matrix = obj_matrix > 0 + 0
    obj_matrix= obj_matrix.astype(int)
    obj_matrix = torch.from_numpy(obj_matrix)
    #print(obj_matrix)
    for t in range(tests):

        mat = gumbel_generator.gen_matrix.cpu().view(-1, 2)
        y_score1 = torch.nn.functional.softmax(mat, dim=1)[:, 0].detach().numpy()
        y_score2 = torch.nn.functional.softmax(mat, dim=1)[:, 1].detach().numpy()
        fpr, tpr, threshold = roc_curve(obj_matrix.view(-1).numpy(), y_score1)
        roc_auc1 = auc(fpr, tpr)
        fpr, tpr, threshold = roc_curve(obj_matrix.view(-1).numpy(), y_score2)
        roc_auc2 = auc(fpr, tpr)

        out_matrix = gumbel_generator.sample_all(hard=True, epoch=e)
        out_matrix = out_matrix.cpu()
        fpr, tpr, threshold = roc_curve(obj_matrix.view(-1).numpy(), out_matrix.view(-1).numpy())
        roc_auc = auc(fpr, tpr)
        out = torch.abs(out_matrix - obj_matrix)
        err = torch.sum(torch.abs(out_matrix - obj_matrix))

        err = err.cpu() if use_cuda else err
        # if we got nan in err
        if math.isnan(err):
            print('problem cocured')
            # torch.save(gumbel_generator,'problem_generator_genchange.model')
            d()
            # d()代表什么
            t = t - 1
            continue
        errs.append(err.data.numpy().tolist())
        tp, tn, fp, fn = calc_tptnfpfn(out_matrix, obj_matrix)
        tpr, fpr, tnr, fnr, f1_score = evaluation_indicator(tp, tn, fp, fn)
        print('err:', errs[0])
        print('tp:', tp)
        print('tn:', tn)
        print('fp:', fp)
        print('fn:', fn)
        print('tpr:', tpr)
        print('fpr:', fpr)
        print('tnr:', tnr)
        print('fnr:', fnr)
        print('f1:', f1_score)
        print('auc:', roc_auc)
        print('auc1:', roc_auc1)
        print('auc2:', roc_auc2)




def load_voter(batch_size=128,node_num=100,network='ER',exp_id=1):

    #data path
    series_path = './data/voter_' + network + '_' + str(node_num) + '_id' + str(exp_id) + '_data.pickle'
    adj_path = './data/voter_' + network + '_' + str(node_num) + '_id' + str(exp_id) + '_adj.pickle'


    # adj matrix
    with open(series_path, 'rb') as f:
        info_train = pickle.load(f, encoding='latin1')
    # time series data
    with open(adj_path, 'rb') as f:
        edges = pickle.load(f, encoding='latin1')
    print('voter')
    print(network)
    print(node_num)
    print(exp_id)
    print(info_train.shape)

    # 即将用到的数据，先填充为全0
    data_x = np.zeros((int(info_train.shape[0] / 2), info_train.shape[1], 2))
    data_y = np.zeros((int(info_train.shape[0] / 2), info_train.shape[1]))

    # 预处理成分类任务常用的数据格式
    for i in range(int(info_train.shape[0] / 2)):
        for j in range(info_train.shape[1]):
            if info_train[2 * i][j][0] == 0.:
                data_x[i][j] = [1, 0]
            else:
                data_x[i][j] = [0, 1]
            if info_train[2 * i + 1][j][0] == 0.:
                data_y[i][j] = 0
            else:
                data_y[i][j] = 1

    # random permutation
    indices = np.random.permutation(data_x.shape[0])
    data_x_temp = [data_x[i] for i in indices]
    data_y_temp = [data_y[i] for i in indices]
    data_x = np.array(data_x_temp)
    data_y = np.array(data_y_temp)

    # seperate train set,val set and test set
    # train / val / test == 5 / 1 / 1
    train_len = int(data_x.shape[0] * 5 / 7)
    val_len = int(data_x.shape[0] * 6 / 7)
    # seperate
    feat_train = data_x[:train_len]
    target_train = data_y[:train_len]
    feat_val = data_x[train_len:val_len]
    target_val = data_y[train_len:val_len]
    feat_test = data_x[val_len:]
    target_test = data_y[val_len:]

    # change to torch.tensor
    feat_train = torch.DoubleTensor(feat_train)
    feat_val = torch.DoubleTensor(feat_val)
    feat_test = torch.DoubleTensor(feat_test)
    target_train = torch.LongTensor(target_train)
    target_val = torch.LongTensor(target_val)
    target_test = torch.LongTensor(target_test)

    # put into tensor dataset
    train_data = TensorDataset(feat_train, target_train)
    val_data = TensorDataset(feat_val, target_val)
    test_data = TensorDataset(feat_test, target_test)

    # put into dataloader
    train_data_loader = DataLoader(train_data, batch_size=batch_size, drop_last=False)
    valid_data_loader = DataLoader(val_data, batch_size=batch_size, drop_last=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, drop_last=False)

    return train_data_loader, valid_data_loader, test_data_loader, torch.from_numpy(edges)


def load_voter_completetion(batch_size=128,node_num=100,network='ER',exp_id=1):

    #data path
    series_path = './data/voter_' + network + '_' + str(node_num) + '_id' + str(exp_id) + '_data.pickle'
    adj_path = './data/voter_' + network + '_' + str(node_num) + '_id' + str(exp_id) + '_adj.pickle'


    # adj matrix
    with open(series_path, 'rb') as f:
        info_train = pickle.load(f, encoding='latin1')
    # time series data
    with open(adj_path, 'rb') as f:
        edges = pickle.load(f, encoding='latin1')
    print('voter')
    print(network)
    print(node_num)
    print(exp_id)
    print(info_train.shape)

    # random delete graph
    node_order,node_order_r,new_object_matrix  = random_del_graph(edges,'voter')

    data_x = np.zeros((int(info_train.shape[0] / 2), info_train.shape[1], 2))
    data_y = np.zeros((int(info_train.shape[0] / 2), info_train.shape[1]))

    for i in range(int(info_train.shape[0] / 2)):
        for j in range(info_train.shape[1]):
            if info_train[2 * i][j][0] == 0.:
                data_x[i][j] = [1, 0]
            else:
                data_x[i][j] = [0, 1]
            if info_train[2 * i + 1][j][0] == 0.:
                data_y[i][j] = 0
            else:
                data_y[i][j] = 1

    # random permutation
    indices = np.random.permutation(data_x.shape[0])
    data_x_temp = torch.DoubleTensor([data_x[i] for i in indices])
    data_y_temp = torch.LongTensor([data_y[i] for i in indices])

    # states 顺序
    data_x = states_r(data_x_temp, node_order, node_order_r, 'voter')
    data_y = states_r(data_y_temp, node_order, node_order_r, 'voter')

    order_data_path = './data/voter_random_' + network + '_' + str(node_num) + '_id' + str(exp_id) + '.pickle'
    results = [new_object_matrix, data_x, data_y]
    with open(order_data_path, 'wb') as f:
        pickle.dump(results, f)

    # seperate train set,val set and test set
    # train / val / test == 5 / 1 / 1
    train_len = int(data_x.shape[0] * 5 / 7)
    val_len = int(data_x.shape[0] * 6 / 7)
    # seperate
    feat_train = data_x[:train_len]
    target_train = data_y[:train_len]
    feat_val = data_x[train_len:val_len]
    target_val = data_y[train_len:val_len]
    feat_test = data_x[val_len:]
    target_test = data_y[val_len:]

    # change to torch.tensor
    feat_train = torch.DoubleTensor(feat_train)
    feat_val = torch.DoubleTensor(feat_val)
    feat_test = torch.DoubleTensor(feat_test)
    target_train = torch.LongTensor(target_train)
    target_val = torch.LongTensor(target_val)
    target_test = torch.LongTensor(target_test)

    # put into tensor dataset
    train_data = TensorDataset(feat_train, target_train)
    val_data = TensorDataset(feat_val, target_val)
    test_data = TensorDataset(feat_test, target_test)

    # put into dataloader
    train_data_loader = DataLoader(train_data, batch_size=batch_size, drop_last=False)
    valid_data_loader = DataLoader(val_data, batch_size=batch_size, drop_last=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, drop_last=False)

    return train_data_loader, valid_data_loader, test_data_loader, torch.from_numpy(edges)

def load_cmn(batch_size=128,node_num=100,network='ER',exp_id=1):
    data_path = './data/cmn_'+network+'_'+str(node_num)+ '_id' + str(exp_id) + '.pickle'

    with open(data_path, 'rb') as f:
        object_matrix, simulates = pickle.load(f)

    print(object_matrix.shape)
    print(simulates.shape)
    print('cmn')
    print(network)
    print(node_num)
    print(exp_id)

    data = torch.Tensor(simulates)
    prediction_num = data.size()[2] // 2
    for i in range(prediction_num):
        last = min((i + 1) * 2, data.size()[2])
        feat = data[:, :, i * 2: last, :]
        if i == 0:
            features = feat
        else:
            features = torch.cat((features, feat), dim=0)

    print(features.shape)


    # split train, val, test
    train_data = features[: features.shape[0] // 6 * 5, :, :, :]
    val_data = features[features.shape[0] // 6 * 5: features.shape[0] // 12 * 11, :, :, :]
    test_data = features[features.shape[0] // 12 * 11:, :, :, :]
    print(train_data.shape, val_data.shape, test_data.shape)



    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader,test_loader,object_matrix


def load_cmn_completetion( batch_size=128, node_num=10, network='ER',exp_id=1):
    data_path = './data/cmn_' + network + '_' + str(node_num) + '_id' + str(exp_id) + '.pickle'

    with open(data_path, 'rb') as f:
        object_matrix, simulates = pickle.load(f)

    print(object_matrix.shape)
    print(simulates.shape)
    print('cmn')
    print(network)
    print(node_num)
    print(exp_id)

    data = torch.Tensor(simulates)
    prediction_num = data.size()[2] // 2
    for i in range(prediction_num):
        last = min((i + 1) * 2, data.size()[2])
        feat = data[:, :, i * 2: last, :]
        if i == 0:
            features = feat
        else:
            features = torch.cat((features, feat), dim=0)

    print(features.shape)

    # split train, val, test
    train_data = features[: features.shape[0] // 6 * 5, :, :, :]
    val_data = features[features.shape[0] // 6 * 5: features.shape[0] // 12 * 11, :, :, :]
    test_data = features[features.shape[0] // 12 * 11:, :, :, :]
    print(train_data.shape, val_data.shape, test_data.shape)
    order, new_order, object_matrix = random_del_graph(object_matrix,  'cmn')
    train_data = states_r(train_data, order, new_order, 'cmn')
    val_data = states_r(val_data, order, new_order, 'cmn')
    test_data = states_r(test_data, order, new_order, 'cmn')

    order_data_path='./data/cml_random_' + network + '_' + str(node_num) + '_id' + str(exp_id) + '.pickle'
    results = [object_matrix, train_data, val_data, test_data]
    with open(order_data_path, 'wb') as f:
        pickle.dump(results, f)


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, object_matrix



def load_cmn_control(batch_size=128,node_num=100,network='ER',control_steps=20,exp_id=1):
    data_path = './data/cmn_' + network + '_' + str(node_num) + '_id' + str(exp_id) + '.pickle'

    with open(data_path, 'rb') as f:
        object_matrix, simulates = pickle.load(f)

    print(object_matrix.shape)
    print(simulates.shape)
    print('cmn')
    print(network)
    print(node_num)
    print(exp_id)

    data = torch.Tensor(simulates)
    test_data=data[data.size(0) // 12 * 11:,:,:,:]
    prediction_num = test_data.size(2) -control_steps

    for i in range(prediction_num):

        feat = test_data[:, :, i : i+control_steps, :]
        if i == 0:
            features = feat
        else:
            features = torch.cat((features, feat), dim=0)

    print(features.shape)

    test_loader = DataLoader(features, batch_size=batch_size, shuffle=False)

    return test_loader, object_matrix

def load_spring(batch_size=128,node_num=100,network='ER',exp_id=1):
    vel_path = './data/vel_'+network+'_'+str(node_num)+ '_id' + str(exp_id) + '.pickle'
    loc_path = './data/sim_'+network+'_'+str(node_num)+ '_id' + str(exp_id) + '.pickle'
    edges_path = './data/edges_'+network+'_'+str(node_num)+ '_id' + str(exp_id) + '.pickle'

    with open(edges_path, 'rb') as f:
        edges = pickle.load(f)
    with open(loc_path, 'rb') as f:
        loc_np = pickle.load(f)
    with open(vel_path, 'rb') as f:
        vel_np = pickle.load(f)

    print('spring')
    print(network)
    print(node_num)
    print(exp_id)

    loc = torch.from_numpy(loc_np)
    vel = torch.from_numpy(vel_np)
    loc = torch.cat((loc,vel),1)
    P = 2
    sample = int(loc.size(0)/P)
    node = loc.size(2)
    dim = loc.size(1)

    loc = loc.transpose(1,2)
    data = torch.zeros(sample,node,P,dim)
    for i in range(data.size(0)):
        data[i] = loc[i*P:(i+1)*P].transpose(0,1)

    # cut to train val and test
    train_data = data[:int(sample*5/7)]
    val_data = data[int(sample*5/7):int(sample*6/7)]
    test_data = data[int(sample*6/7):]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader,test_loader,torch.from_numpy(edges)

def load_spring_multi(batch_size=128,node_num=100,network='ER',prediction_steps=10,exp_id=1):
    vel_path = './data/vel_'+network+'_'+str(node_num)+ '_id' + str(exp_id) + '.pickle'
    loc_path = './data/sim_'+network+'_'+str(node_num)+ '_id' + str(exp_id) + '.pickle'
    edges_path = './data/edges_'+network+'_'+str(node_num)+ '_id' + str(exp_id) + '.pickle'

    with open(edges_path, 'rb') as f:
        edges = pickle.load(f)
    with open(loc_path, 'rb') as f:
        loc_np = pickle.load(f)
    with open(vel_path, 'rb') as f:
        vel_np = pickle.load(f)

    print('spring')
    print(network)
    print(node_num)
    print(exp_id)

    loc = torch.from_numpy(loc_np)
    vel = torch.from_numpy(vel_np)
    loc = torch.cat((loc,vel),1)
    P = prediction_steps+1

    sample = int(loc.size(0)/P)
    node = loc.size(2)
    dim = loc.size(1)

    loc = loc.transpose(1,2)
    data = torch.zeros(sample,node,P,dim)
    for i in range(data.size(0)):
        data[i] = loc[i*P:(i+1)*P].transpose(0,1)

    # cut to train val and test
    train_data = data[:int(sample*5/7)]
    val_data = data[int(sample*5/7):int(sample*6/7)]
    test_data = data[int(sample*6/7):]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader,test_loader,torch.from_numpy(edges)



def load_spring_control(batch_size=128,node_num=100,network='ER',control_steps=20,exp_id=1):
    # vel_path = './data/vel_'+network+'_'+str(node_num)+ '_id' + str(exp_id) + '.pickle'
    # loc_path = './data/sim_'+network+'_'+str(node_num)+ '_id' + str(exp_id) + '.pickle'
    # edges_path = './data/edges_'+network+'_'+str(node_num)+ '_id' + str(exp_id) + '.pickle'
    #for example
    vel_path = './data/vel_BA_10_id1.pickle'
    loc_path = './data/sim_BA_10_id1.pickle'
    edges_path = './data/edges_BA_10_id1.pickle'


    print('spring')
    print(network)
    print(node_num)
    print(exp_id)

    with open(edges_path, 'rb') as f:
        edges = pickle.load(f)
    with open(loc_path, 'rb') as f:
        loc_np = pickle.load(f)
    with open(vel_path, 'rb') as f:
        vel_np = pickle.load(f)

    loc = torch.from_numpy(loc_np)
    vel = torch.from_numpy(vel_np)
    loc = torch.cat((loc,vel),1)


    node = loc.size(2)

    loc = loc.transpose(1,2)# 499,5,4
    data = torch.zeros(int(loc.shape[0] -control_steps), node, loc.shape[2])
    for i in range(data.size(0)):
        data[i] = loc[i]
    #data = loc
    sample = int(loc.size(0))
    # cut to train val and test
    train_data = data[:int(sample*5/7)]
    val_data = data[int(sample*5/7):int(sample*6/7)]
    test_data = data[int(sample*6/7):]

    data_path='./data/real_test_data.pickle'
    with open(data_path,'wb') as f:
        pickle.dump(test_data.numpy(),f)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, torch.from_numpy(edges)

def load_SIR(batch_size = 512,exp_id=1):

    data_path = './data/SIR_id'+str(exp_id)+'.pickle'

    with open(data_path, 'rb') as f:
        edges, loc = pickle.load(f)

    data = torch.zeros(int(loc.shape[0] / 2), 371, 2, loc.shape[2])
    for i in range(data.size(0)):
        data[i] = torch.from_numpy(loc[i * 2:(i + 1) * 2]).transpose(0,1)
    gates = data.shape[0]

    train_loader = DataLoader(data[:int(gates * (5 / 7))], batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(data[int(gates * (5 / 7)):int(gates * (6 / 7))], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data[int(gates * (6 / 7)):], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, torch.from_numpy(edges)




def load_Gene(batch_size = 512,exp_id=1):
    data_path = './data/gene_id' + str(exp_id) + '.pickle'
    with open(data_path, 'rb') as f:
        edges, loc= pickle.load(f)
    data = torch.zeros(int(loc.shape[0]/2),100,2,loc.shape[2])

    for i in range(data.size(0)):
        data[i] = torch.from_numpy(loc[i * 2:(i + 1) * 2]).transpose(0,1)
    gates = data.shape[0]

    train_loader = DataLoader(data[:int(gates*(5/7))], batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(data[int(gates*(5/7)):int(gates*(6/7))], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data[int(gates*(6/7)):], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, torch.from_numpy(edges)



# output:[batch_num,node_num,2]
# target[batch_num,node_num]
def cacu_accu(output,target):
    if output.size(1) == 2:
        output=  output.permute(0,2,1)
    output = output.cpu()
    target = target.cpu()
    right = 0.
    accu_all_list = []
    for i in range(output.size(0)):
        accu_batch = []
        for j in range(output.size(1)):
            if output[i][j][0] >= output[i][j][1]:
                if target[i][j] == 0:
                    right += 1
                elif target[i][j] == 1:
                    continue
                else:
                    print('error pos 1')
                    debug()
            elif output[i][j][0] < output[i][j][1]:
                if target[i][j] == 1:
                    right += 1
                elif target[i][j] == 0:
                    continue
                else:
                    print('error pos 2')
                    debug()
            else:
                print('error pos 0')
                debug()
    return right / target.size(0) /target.size(1)

def get_test_accu(gumbel_generator,dyn_learner,data_train,data_target):
    out_matrix = gumbel_generator.sample()
    out_matrix = out_matrix.unsqueeze(0)
    out_matrix = out_matrix.repeat(data_train.size()[0], 1, 1)
    gumbel_generator.drop_temperature()

    output = dyn_learner(data_train,out_matrix)

    # caculate the difference
    data_target = data_target.long()
    accus = cacu_accu(output,data_target)
    return accus


def partial_mask(sz, del_num):
    '''mask the known part and un know part'''
    kn_mask = torch.zeros(sz, sz)
    kn_mask[:-del_num, :-del_num] = 1

    un_un_mask = torch.zeros(sz, sz)
    un_un_mask[-del_num:, -del_num:] = 1
    un_un_mask = un_un_mask - torch.diag(torch.diag(un_un_mask))

    left_mask = torch.ones(sz, sz)
    left_mask[:-del_num, :-del_num] = 0
    left_mask = left_mask - torch.diag(torch.diag(left_mask))
    kn_un_mask = left_mask - un_un_mask

    return kn_mask, left_mask, un_un_mask, kn_un_mask


def part_constructor_evaluator_sgm(generator, tests, obj_matrix, sz, del_num):
    kn_nodes = sz - del_num
    precision = []

    kn_mask, un_mask, un_un_mask, kn_un_mask = partial_mask(sz, del_num)
    pre_adj = gumbel_p(generator, sz, un_mask).detach()
    index_order, P = sgraphmatch(obj_matrix, pre_adj, kn_nodes, iteration=20)
    pre_adj = torch.mm(torch.mm(P, pre_adj), P.T)
    print("index_order", index_order)
    auc_net = aucs(pre_adj, obj_matrix, un_mask, un_un_mask, kn_un_mask)

    # print("auc:%f,un_un_auc:%f,kn_un_auc:%f"%(auc_net[0],auc_net[1],auc_net[2]))

    print("auc:", auc_net[0])
    print("un_un_auc:", auc_net[1])
    print("kn_un_auc:", auc_net[2])

    for t in range(tests):
        obj_matrix = obj_matrix.cuda()
        out_matrix = generator.sample_all(hard=True)  # 使用hard采样 才能计算tpr 与fpr

        index_order, P = sgraphmatch(obj_matrix, out_matrix, kn_nodes, iteration=20)
        out_matrix = torch.mm(torch.mm(P, out_matrix), P.T)

        metrics_all = tpr_fpr_part(out_matrix, obj_matrix, un_mask, kn_un_mask, un_un_mask, del_num)
        precision.append(metrics_all)

    (f1, err_net, tp, fn, fp, tn, tpr, fpr) = np.mean([precision[i][0] for i in range(tests)], 0)
    un_precision = (f1, err_net, tp, fn, fp, tn, tpr, fpr)
    # print("f1:%f,err_net:%d,tp:%d,fn:%d,fp:%d,tn:%d,tpr:%f,fpr:%f"%(f1,err_net,tp,fn,fp,tn,tpr,fpr))
    print('!!!un_precision!!!')
    print('f1:', f1)
    print('err_net:', err_net)
    print('tp:', tp)
    print('tn:', tn)
    print('fp:', fp)
    print('fn:', fn)
    print('tpr:', tpr)
    print('fpr:', fpr)

    (f1, err_net, tp, fn, fp, tn, tpr, fpr) = np.mean([precision[i][1] for i in range(tests)], 0)
    kn_un_precision = (f1, err_net, tp, fn, fp, tn, tpr, fpr)
    # print("kn_un_precision: f1:%f,err_net:%d,tp:%d,fn:%d,fp:%d,tn:%d,tpr:%f,fpr:%f"%(f1,err_net,tp,fn,fp,tn,tpr,fpr))
    print('!!!kn_un_precision!!!')
    print('f1:', f1)
    print('err_net:', err_net)
    print('tp:', tp)
    print('tn:', tn)
    print('fp:', fp)
    print('fn:', fn)
    print('tpr:', tpr)
    print('fpr:', fpr)
    (f1, err_net, tp, fn, fp, tn, tpr, fpr) = np.mean([precision[i][2] for i in range(tests)], 0)
    un_un_precision = (f1, err_net, tp, fn, fp, tn, tpr, fpr)
    # print("un_un_precision: f1:%f,err_net:%d,tp:%d,fn:%d,fp:%d,tn:%d,tpr:%f,fpr:%f"%(f1,err_net,tp,fn,fp,tn,tpr,fpr))
    print('!!!un_un_precision!!!')
    print('f1:', f1)
    print('err_net:', err_net)
    print('tp:', tp)
    print('tn:', tn)
    print('fp:', fp)
    print('fn:', fn)
    print('tpr:', tpr)
    print('fpr:', fpr)

    return index_order, auc_net, un_precision, kn_un_precision, un_un_precision

def all_constructor_evaluator_softmax(generator,generator_nc, tests, obj_matrix, sz, del_num):
    kn_nodes = sz - del_num
    errs = []

    kn_mask, un_mask, un_un_mask, kn_un_mask = partial_mask(sz, del_num)
    pre_adj = gumbel_p(generator_nc, sz, un_mask).detach()
    index_order, P = sgraphmatch(obj_matrix, pre_adj, kn_nodes, iteration=20)
    pre_adj = torch.mm(torch.mm(P, pre_adj), P.T)

    print("index_order", index_order)
    mat = generator.gen_matrix
    known_matrix_p = torch.nn.functional.softmax(mat, dim=2)[:, :,0]
    pre_adj[:-del_num, :-del_num] = known_matrix_p
    pre_adj = pre_adj.cpu().detach()
    obj_matrix=obj_matrix.cpu()
    fpr, tpr, threshold = roc_curve(obj_matrix.numpy().reshape(-1), pre_adj.numpy().reshape(-1))

    roc_auc1 = auc(fpr, tpr)
    print('soft_auc:', roc_auc1)

def gumbel_p(generator,node_num,left_mask):
    p = F.softmax(generator.gen_matrix,dim =1)[:,0]
    matrix = torch.zeros(node_num,node_num).cuda()
    un_index = torch.triu(left_mask).nonzero()
    matrix[(un_index[:,0],un_index[:,1])] = p
    out_matrix = matrix + matrix.T
    return out_matrix

def aucs(pre_adj,object_matrix,un_mask,un_un_mask,kn_un_mask):
    roc_auc = cal_auc(pre_adj,object_matrix,un_mask)
    un_un_roc_auc = cal_auc(pre_adj,object_matrix,un_un_mask)
    kn_un_roc_auc = cal_auc(pre_adj,object_matrix,kn_un_mask)
    return roc_auc,un_un_roc_auc,kn_un_roc_auc

def cal_auc(pre,true_adj,un_mask):
    # print(1,un_mask)
    pre_un = pre[un_mask.bool()].cpu().detach().numpy()
    true_un = true_adj[un_mask.bool()].cpu().detach().numpy()
    fpr,tpr,threshold = roc_curve(true_un,pre_un)
    roc_auc = auc(fpr,tpr)
    return roc_auc

def tpr_fpr_part(pre,true_adj,un_mask,kn_un_mask,un_un_mask,del_num):
    '''Calculate the accuracy of the unknown part'''
    un_metics = cal_tpfp(pre,true_adj,un_mask)
    kn_un_metrics = cal_tpfp(pre,true_adj,kn_un_mask)
    un_un_metrics = cal_tpfp(pre,true_adj,un_un_mask)
    return (un_metics,kn_un_metrics,un_un_metrics)


def cal_tpfp(pre, true_adj, mask):
    true_mask = true_adj[mask.bool()]
    pre_un = pre[mask.bool()]
    err = torch.sum(torch.abs(pre_un - true_mask))
    tp = torch.sum(pre_un[true_mask.bool()])
    fn = torch.sum(1 - pre_un[true_mask.bool()])
    tn = torch.sum(1 - pre_un[(1 - true_mask).bool()])
    fp = torch.sum(pre_un[(1 - true_mask).bool()])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    pf = tn / (tn + fn)
    rf = tn / (tn + fp)

    f1 = precision * recall / (precision + recall) + pf * rf / (pf + rf)
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return (f1, err, tp, fn, fp, tn, tpr, fpr)


def states_r(states, order, new_order, data_type='cml'):
    new_states = torch.zeros_like(states)
    if data_type == 'cml':
        for i, j in zip(order, new_order):
            new_states[:, i, :, :] = states[:, j, :, :]
    if data_type == 'voter':
        if len(new_states.shape) == 2:
            for i, j in zip(order, new_order):
                new_states[:, i] = states[:, j]

        else:
            for i, j in zip(order, new_order):
                new_states[:, i, :] = states[:, j, :]

    return new_states


def random_del_graph(adj, data_type='voter'):
    if data_type == 'cmn':
        G = nx.from_numpy_matrix(adj.cpu().data.numpy())
    if data_type == 'voter':
        G = nx.from_numpy_matrix(adj)

    # np.random.seed(seed)
    node_order = list(G.nodes)
    node_order_r = np.random.permutation(node_order)
    new_order_graph = dict()
    for node in node_order_r:
        new_order_graph.update({node: G[node]})

    new_order_graph = from_dict_of_dicts(new_order_graph, create_using=Graph)
    new_object_matrix = torch.FloatTensor(nx.adjacency_matrix(new_order_graph).todense()).cuda()
    return node_order, node_order_r, new_object_matrix
