import torch
import time

import torch.nn.utils as U
import torch.optim as optim
import argparse
import sys

from model import *
from tools import *

parser = argparse.ArgumentParser(description="ER network")
parser.add_argument('--nodes', type=int, default=100,
                    help='number of epochs to train')  # '--epoch_num'
parser.add_argument('--node_size', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--exp_id', type=int, default=1, help='实验id, default=1')
parser.add_argument('--device_id', type=int, default=1, help='Gpu_id, default=5')
parser.add_argument('--network', type=str, default='ER', help='type of network')
parser.add_argument("--seed", type=int, default=2050, help="random seed (default: 2050)")
parser.add_argument("--dim", type=int, default=2, help="information diminsion of each node voter")
parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of GGN model (default:128)")
parser.add_argument("--epoch_num", type=int, default=300, help="train epoch of model (default:500)")
parser.add_argument("--batch_size", type=int, default=1024, help="input batch size for training (default: 128)")
parser.add_argument("--lr_net", type=float, default=0.004, help="gumbel generator learning rate (default:0.004) ")
parser.add_argument("--lr_net_comp", type=float, default=0.004, help="gumbel generator learning rate (default:0.004) ")
parser.add_argument("--lr_dyn", type=float, default=0.001, help="dynamic learning rate (default:0.001)")
parser.add_argument("--lr_dyn_comp", type=float, default=0.01, help="dynamic learning rate (default:0.001)")

parser.add_argument("--lr_stru", type=float, default=0.0001,
                    help="sparse network adjustable rate (default:0.000001)")
parser.add_argument("--lr_stru_comp", type=float, default=0.00001,
                    help="sparse network adjustable rate (default:0.000001)")

parser.add_argument("--lr_state", type=float, default=0.05, help="state learning rate (default:0.1)")
parser.add_argument("--miss_percent", type=float, default=0.1, help="missing percent node (default:0.1)")

args = parser.parse_args()

# print("no lr stru")
# from model_old_generator import *
# configuration
HYP = {
    'note': 'try init',
    'node_num': args.nodes,  # node num
    'node_size': args.node_size,  # node size
    'conn_p': '25',  # connection probility : 25 means 1/25
    'seed': args.seed,  # the seed
    'dim': args.dim,  # information diminsion of each node cml:1 spring:4
    'hid': args.hidden_size,  # hidden size
    'epoch_num': args.epoch_num,  # epoch
    'batch_size': args.batch_size,  # batch size
    'lr_net': args.lr_net,  # lr for net generator
    'lr_net_comp': args.lr_net_comp,  # lr for net generator
    'lr_dyn': args.lr_dyn,  # lr for dyn learner
    'lr_dyn_comp': args.lr_dyn_comp,  # lr for dyn learner
    'lr_stru': args.lr_stru,  # lr for structural loss 0.0001
    'lr_stru_comp': args.lr_stru_comp,  # lr for structural loss 0.0001
    'lr_state': args.lr_state,
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'sys': 'voter',  # simulated system to model
    'isom': True,  # isomorphic, no code for sync
    'temp': 1,  # 温度
    'drop_frac': 1,  # temperature drop frac
    'save': True,  # weather to save the result
}

print("has structure parameter:", HYP)
# partial known adj
del_num = int(args.nodes * args.miss_percent)
print("del_num", del_num)
known_num = args.nodes-del_num

#cuda_number = 0
#torch.set_num_threads(1)
torch.cuda.set_device(args.device_id)
print('cuda:', args.device_id)
#torch.manual_seed(HYP['seed'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_loader, val_loader, test_loader, object_matrix = load_voter_completetion(batch_size=HYP['batch_size'],node_num=args.nodes, network=args.network,exp_id=args.exp_id)
unedges  = torch.sum(object_matrix[-del_num:,-del_num:])
while unedges.item() == 0:
    print(object_matrix.sum(0))
    print("sample 0 edges")
    sys.exit()

start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#for AIDD
# dyn learner
dyn_isom = IO_B_Voter(HYP['dim'], HYP['hid']).to(device)
# optimizer
op_dyn = optim.Adam(dyn_isom.parameters(), lr=HYP['lr_dyn'])


#for AIDD
#generator
generator = Gumbel_Generator_Old(sz=known_num, temp=HYP['temp'], temp_drop_frac=HYP['drop_frac']).to(device)
generator.init(0, 0.1)
# optimizer
op_net = optim.Adam(generator.parameters(), lr=HYP['lr_net'])



#for network completetion
# dyn learner isomorphism
dyn_isom_nc = IO_B_Voter(HYP['dim'], HYP['hid']).to(device)
op_dyn_nc = optim.Adam(dyn_isom_nc.parameters(), lr=HYP['lr_dyn_comp'])

#for network completetion
# net learner 
generator_nc = Gumbel_Generator_nc(sz=HYP['node_num'],del_num = del_num,temp=HYP['temp'], temp_drop_frac=HYP['drop_frac']).to(device)
generator_nc.init(0, 0.1)
op_net_nc = optim.Adam(generator_nc.parameters(), lr=HYP['lr_net_comp'])


loss_fn = torch.nn.NLLLoss()
#已知部分的邻接矩阵
known_matrix=object_matrix[:-del_num, :-del_num]
known_matrix = known_matrix.cpu().numpy()
def train_dyn_gen():
    loss_batch = []
    accus_batch = []
    err_batch = []
    tp_batch =[]
    tn_batch=[]
    fp_batch = []
    fn_batch =[]
    tpr_batch=[]
    fpr_batch =[]
    tnr_batch=[]
    fnr_batch=[]
    f1_score_batch=[]
    roc_auc_batch=[]


    print('current temp:', generator.temperature)
    for idx, data in enumerate(train_loader):
        print('batch idx:', idx)
        # data
        x = data[0].float().to(device)
        y = data[1].float().to(device)
        x = x[:, :-del_num, :]
        y = y[:, :-del_num]
        # drop temperature
        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1), 2)
        loss_node = []
        for j in range(known_num):
            # zero grad
            op_net.zero_grad()
            op_dyn.zero_grad()
            # predict and caculate the loss
            adj_col = generator.sample_adj_i(j, hard=HYP['hard_sample'], sample_time=HYP['sample_time']).to(device)
            num = int(known_num / HYP['node_size'])
            remainder = int(known_num % HYP['node_size'])
            if remainder == 0:
                num = num - 1
            y_hat = dyn_isom(x, adj_col, j, num, HYP['node_size'])

            loss = loss_fn(y_hat, y[:, j].long())

            # backward and optimize
            loss.backward()
            # cut gradient in case nan shows up
            U.clip_grad_norm_(generator.gen_matrix, 0.000075)

            op_net.step()
            op_dyn.step()

            # use outputs to caculate mse
            outputs[:, j, :] = y_hat

            # record
            loss_node.append(loss.item())

        loss_batch.append(np.mean(loss_node))
        accus_batch.append(cacu_accu(outputs, y.long()))

        err,tp, tn, fp, fn, tpr, fpr, tnr, fnr, f1_score, roc_auc = evaluator(outputs, y.long())
        err_batch.append(err)
        tp_batch.append(tp)
        tn_batch.append(tn)
        fp_batch.append(fp)
        fn_batch.append(fn)
        tpr_batch.append(tpr)
        fpr_batch.append(fpr)
        tnr_batch.append(tnr)
        fnr_batch.append(fnr)
        f1_score_batch.append(f1_score)
        roc_auc_batch.append(roc_auc)


    print('err_dyn:', np.mean(err_batch))
    print('tp_dyn:', np.mean(tp_batch))
    print('tn_dyn:', np.mean(tn_batch))
    print('fp_dyn:', np.mean(fp_batch))
    print('fn_dyn:', np.mean(fn_batch))
    print('tpr_dyn:', np.mean(tpr_batch))
    print('fpr_dyn:', np.mean(fpr_batch))
    print('tnr_dyn:', np.mean(tnr_batch))
    print('fnr_dyn:', np.mean(fnr_batch))
    print('f1_dyn:', np.mean(f1_score_batch))
    print('auc_dyn:', np.mean(roc_auc_batch))

    # used for more than 10 nodes
    op_net.zero_grad()
    loss = (torch.sum(generator.sample_all())) * HYP['lr_stru']
    loss.backward()
    op_net.step()

    # each item is the mean of all batches, means this indice for one epoch
    return np.mean(loss_batch), np.mean(accus_batch)


'''voter states,dyn,NET  '''


# indices to draw the curve
def train_dyn_gen_state_voter(observed_adj):
    loss_batch = []
    accus_batch = []
    err_batch = []
    tp_batch = []
    tn_batch = []
    fp_batch = []
    fn_batch = []
    tpr_batch = []
    fpr_batch = []
    tnr_batch = []
    fnr_batch = []
    f1_score_batch = []
    roc_auc_batch = []


    for idx, (data, states_id) in enumerate(zip(train_loader, train_index)):
        # print('batch idx:', idx)
        # data
        x = data[0].float().to(device)
        y = data[1].to(device).long()

        x_kn = x[:, :-del_num, :]
        generator_nc.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1), 2)

        loss_node = []

        for j in range(HYP['node_num'] - del_num):
            # zero grad
            op_net_nc.zero_grad()
            op_dyn_nc.zero_grad()
            opt_states.zero_grad()

            x_un_pre = states_learner(states_id.cuda())  # .detach()
            x_hypo = torch.cat((x_kn, x_un_pre.float()), 1)

            hypo_adj = generator_nc.sample_all()
            hypo_adj[:-del_num, :-del_num] = observed_adj

            adj_col = hypo_adj[:, j].cuda()  # hard = true

            num = int(HYP['node_num'] / HYP['node_size'])
            remainder = int(HYP['node_num'] % HYP['node_size'])
            if remainder == 0:
                num = num - 1

            y_hat = dyn_isom_nc(x_hypo, adj_col, j, num, HYP['node_size'])
            loss = loss_fn(y_hat, y[:, j])


            # backward and optimize
            loss.backward()
            # cut gradient in case nan shows up
            U.clip_grad_norm_(generator_nc.gen_matrix, 0.000075)

            # op_net.step()
            op_dyn_nc.step()
            op_net_nc.step()
            opt_states.step()

            # use outputs to caculate mse
            outputs[:, j, :] = y_hat

            # record
            loss_node.append(loss.item())

        # dif_batch.append(np.mean(dif_node))
        loss_batch.append(np.mean(loss_node))
        accus_batch.append(cacu_accu(outputs, y.long()))

        err, tp, tn, fp, fn, tpr, fpr, tnr, fnr, f1_score, roc_auc = evaluator(outputs, y.long())
        err_batch.append(err)
        tp_batch.append(tp)
        tn_batch.append(tn)
        fp_batch.append(fp)
        fn_batch.append(fn)
        tpr_batch.append(tpr)
        fpr_batch.append(fpr)
        tnr_batch.append(tnr)
        fnr_batch.append(fnr)
        f1_score_batch.append(f1_score)
        roc_auc_batch.append(roc_auc)

    print('err_dyn:', np.mean(err_batch))
    print('tp_dyn:', np.mean(tp_batch))
    print('tn_dyn:', np.mean(tn_batch))
    print('fp_dyn:', np.mean(fp_batch))
    print('fn_dyn:', np.mean(fn_batch))
    print('tpr_dyn:', np.mean(tpr_batch))
    print('fpr_dyn:', np.mean(fpr_batch))
    print('tnr_dyn:', np.mean(tnr_batch))
    print('fnr_dyn:', np.mean(fnr_batch))
    print('f1_dyn:', np.mean(f1_score_batch))
    print('auc_dyn:', np.mean(roc_auc_batch))


    # op_net.zero_grad()
    # loss = torch.abs(torch.sum(generator.sample_all())- infer_edges) * HYP['lr_stru']
    # loss.backward()
    # op_net.step()

    return np.mean(loss_batch), np.mean(accus_batch)


def onehot_state(x_un_pre):
    if x_un_pre.shape[1] == 1:
        state = torch.argmax(x_un_pre, 2)
        pre_state = torch.cat((state, 1 - state), 1).unsqueeze(1)
    else:
        state = torch.argmax(x_un_pre, 2).unsqueeze(2)
        pre_state = torch.cat((state, 1 - state), 2)
    return pre_state


def cal_states_loss():
    mae = []
    for data, states_id in zip(train_loader, train_index):
        x = data[0].float().to(device)
        x_un = x[:, -del_num:, :]
        x_un_pre = states_learner(states_id.cuda()).detach()
        pre_state = onehot_state(x_un_pre)
        state_erro_rate = abs(torch.argmax(x_un, 2) - torch.argmax(pre_state, 2)).float().mean()
        state_erro_rate = state_erro_rate.cpu().numpy()
        mae.append(state_erro_rate)

    return np.mean(mae)


# start training
best_val_mse = 1000000
best = 0
best_loss = 10000000


real_time_list=[]
# each training epoch
for e in range(HYP['epoch_num']):
    print('\nepoch', e)

    t_s = time.time()
    # train both dyn learner and generator together
    t_s1 = time.time()
    # loss, mse = train_dyn_gen()

    try:
        loss, accus = train_dyn_gen()
    except RuntimeError as sss:
        if 'out of memory' in str(sss):
            print('|WARNING: ran out of memory')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise sss
    t_e1 = time.time()
    print('loss:' + str(loss) + ' accus:' + str(accus))
    print('time for this dyn_adj epoch:' + str(round(t_e1 - t_s1, 2)))


    dyn_path='./model/dyn_voter_'+args.network+'_'+str(args.nodes)+'_miss'+str(args.miss_percent)+'_id'+str(args.exp_id)+'.pkl'
    gen_path = './model/gen_voter_' + args.network + '_' + str(args.nodes) + '_miss' + str(args.miss_percent) + '_id' + str(
        args.exp_id) + '.pkl'
    adj_path = './model/adj_voter_' + args.network + '_' + str(args.nodes) + '_miss' + str(args.miss_percent) + '_id' + str(
        args.exp_id) + '.pkl'
    if loss < best_loss:
        print('best epoch:', e)
        best_loss = loss
        best = e
        torch.save(dyn_isom, dyn_path)
        torch.save(generator, gen_path)
        out_matrix = generator.sample_all(hard=HYP['hard_sample'], ).to(device)
        torch.save(out_matrix, adj_path)
    print('best epoch:', best)

    t_s2 = time.time()
    constructor_evaluator(generator, 1, np.float32(known_matrix), e)
    t_e2 = time.time()
    print('time for this adj_eva epoch:' + str(round(t_e2 - t_s2, 2)))

    t_e = time.time()
    real_time_list.append(round(t_e - t_s, 2))
    real_time = sum(real_time_list)

    print('time for this whole epoch:' + str(round(t_e - t_s, 2)))
    print('realtime until this epoch:' + str(round(real_time, 2)))


end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('end_time:', end_time)



# states learner
sample_num = len(train_loader.dataset)
if HYP["sys"] == "voter":
    states_learner = Generator_states_discrete(sample_num,del_num).double()
if use_cuda:
    states_learner = states_learner.cuda()
    
opt_states = optim.Adam(states_learner.parameters(),lr = HYP['lr_state'])

train_index = DataLoader([i for i in range(sample_num)],HYP['batch_size'])

#observed_adj = object_matrix[:-del_num,:-del_num]
generator = torch.load(gen_path).to(device)
observed_adj = generator.sample_all().detach()

kn_mask,left_mask,un_un_mask,kn_un_mask = partial_mask(HYP['node_num'],del_num)
infer_edges = torch.sum(object_matrix).item() - torch.sum(observed_adj).item()


min_loss=10000
com_dyn_path='./model/com_dyn_voter_'+args.network+'_'+str(args.nodes)+'_miss'+str(args.miss_percent)+'_id'+str(args.exp_id)+'.pkl'
com_gen_path = './model/com_gennc_voter_' + args.network + '_' + str(args.nodes) + '_miss' + str(args.miss_percent) + '_id' + str(
    args.exp_id) + '.pkl'
com_states_path = './model/com_states_voter_' + args.network + '_' + str(args.nodes) + '_miss' + str(args.miss_percent) + '_id' + str(
    args.exp_id) + '.pkl'

print('start learn kn_un ！！！')
for epoch in range(HYP["epoch_num"]):
    print('\nepoch', epoch)
    start_time = time.time()
    loss,accus = train_dyn_gen_state_voter(observed_adj)
    print('epoch:',epoch)
    print('loss:' + str(loss) + ' accus:' + str(accus))
    print('-------------part_constructor_evaluator-------------')
    (index_order,auc_net,precision,kn_un_precision,un_un_precision) = part_constructor_evaluator_sgm(generator_nc,1,object_matrix,HYP["node_num"],del_num)


    print('Net Error:')
    print('un:',round(float(precision[1].item() / left_mask.sum()), 2))
    print('kn_un:', round(float(kn_un_precision[1].item() / kn_un_mask.sum()), 2))
    print('un_un:',round(float(un_un_precision[1].item() / un_un_mask.sum()), 2))
    print('-------------all_constructor_evaluator-------------')

    all_constructor_evaluator_softmax(generator,generator_nc,1,object_matrix,HYP["node_num"],del_num)

    x_state_erro = cal_states_loss()
    print('x_states_erro:',x_state_erro)
    end_time = time.time()
    print("cost_time", str(round(end_time - start_time, 2)))

    if loss < min_loss:
        print('best epoch:', epoch)
        min_loss = loss
        best = epoch
        torch.save(dyn_isom_nc.state_dict(), com_dyn_path)
        torch.save(generator_nc.state_dict(), com_gen_path)
        torch.save(states_learner.state_dict(), com_states_path)
    print('best epoch:', best)