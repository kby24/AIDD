import torch
import time
from model import *
from tools import *
import argparse

# configuration
HYP = {
    'node_size': 100,
    'hid': 128,  # hidden size
    'epoch_num': 200,  # epoch
    'batch_size': 2048,  # batch size
    'lr_net': 0.004,  # lr for net generator 0.004
    'lr_dyn': 0.001,  # lr for dyn learner
    'lr_stru': 0.0001,  # lr for structural loss 0.0001 2000 0.01  0.00001
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'temp': 1,  # temperature
    'drop_frac': 1,  # temperature drop frac
}

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=100, help='Number of nodes, default=10')
parser.add_argument('--network', type=str, default='ER', help='type of network')
parser.add_argument('--sys', type=str, default='voter', help='simulated system to model,spring or cmn')
parser.add_argument('--dim', type=int, default=2, help='# information dimension of each node spring:4 cmn:1 ')
parser.add_argument('--exp_id', type=int, default=1, help='experiment_id, default=1')
parser.add_argument('--device_id', type=int, default=5, help='Gpu_id, default=5')
args = parser.parse_args()

torch.cuda.set_device(args.device_id)



start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model load path
dyn_path = './model/dyn_voter_' + args.network + '_' + str(args.nodes) + '_id' + str(args.exp_id) + '.pkl'
gen_path = './model/gen_voter_' + args.network + '_' + str(args.nodes) + '_id' + str(args.exp_id) + '.pkl'

generator = torch.load(gen_path).to(device)
dyn_isom = torch.load(dyn_path).to(device)

# load_data
if args.sys== 'voter':
    train_loader, val_loader, test_loader, object_matrix = load_voter(batch_size=HYP['batch_size'],node_num=args.nodes,network=args.network,exp_id=args.exp_id)



object_matrix = object_matrix.cpu().numpy()

# loss function
loss_fn = torch.nn.NLLLoss()


def test_dyn_gen():
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


    for idx, data in enumerate(test_loader):
        print('batch idx:', idx)
        # data
        x = data[0].float().to(device)
        y = data[1].float().to(device)
        # drop temperature
        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1), 2)
        loss_node = []
        for j in range(args.nodes):
            # predict and caculate the loss
            adj_col = generator.sample_adj_i(j, hard=HYP['hard_sample'], sample_time=HYP['sample_time']).to(device)
            num = int(args.nodes / HYP['node_size'])
            remainder = int(args.nodes % HYP['node_size'])
            if remainder == 0:
                num = num - 1
            y_hat = dyn_isom(x, adj_col, j, num, HYP['node_size'])
            loss = loss_fn(y_hat, y[:, j].long())

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


    # each item is the mean of all batches, means this indice for one epoch
    return np.mean(loss_batch), np.mean(accus_batch)

with torch.no_grad():
    loss, accus= test_dyn_gen()
    print('loss:' + str(loss) + ' accus:' + str(accus))


