import torch
import time
from model import *
from tools import *
import argparse

# configuration
HYP = {
    'node_size': 371,
    'hid': 128,  # hidden size
    'epoch_num': 1000,  # epoch
    'batch_size': 2048,  # batch size
    'lr_net': 0.004,  # lr for net generator 0.004
    'lr_dyn': 0.0033,  # lr for dyn learner
    'lr_stru': 0.00001,  # lr for structural loss 0.0001 2000 0.01  0.00001
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'temp': 1,  # temperature
    'drop_frac': 1,  # temperature drop frac
}

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=371, help='Number of nodes, default=10')
parser.add_argument('--sys', type=str, default='SIR', help='simulated system to model,spring or cmn')
parser.add_argument('--dim', type=int, default=3, help='# information dimension of each node SIR 3 ')
parser.add_argument('--exp_id', type=int, default=1, help='experiment_id, default=1')
parser.add_argument('--device_id', type=int, default=5, help='Gpu_id, default=5')
args = parser.parse_args()

start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model load path
dyn_path = './model/dyn_SIR_' + args.network + '_' + str(args.nodes) + '_id' + str(args.exp_id) + '.pkl'
gen_path = './model/gen_SIR_' + args.network + '_' + str(args.nodes) + '_id' + str(args.exp_id) + '.pkl'

generator = torch.load(gen_path).to(device)
dyn_isom = torch.load(dyn_path).to(device)

# load_data
if args.sys== 'SIR':
    train_loader, val_loader, test_loader, object_matrix = load_SIR(batch_size=HYP['batch_size'],exp_id=args.exp_id)

object_matrix = object_matrix.cpu().numpy()


def test_dyn_gen():
    loss_batch = []
    mse_batch = []
    print('current temp:', generator.temperature)
    for idx, data in enumerate(test_loader):
        print('batch idx:', idx)
        # data
        data = data.to(device)
        x = data[:, :, 0, :]
        y = data[:, :, 1, :]
        # drop temperature
        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1), y.size(2))
        loss_node = []
        for j in range(args.nodes):

            # predict and caculate the loss
            adj_col = generator.sample_adj_i(j, hard=HYP['hard_sample'], sample_time=HYP['sample_time']).to(device)

            num = int(args.nodes / HYP['node_size'])
            remainder = int(args.nodes % HYP['node_size'])
            if remainder == 0:
                num = num - 1
            y_hat = dyn_isom(x, adj_col, j, num, HYP['node_size'])
            loss = torch.mean(torch.abs(y_hat - y[:, j, :]))
            # use outputs to caculate mse
            outputs[:, j, :] = y_hat

            # record
            loss_node.append(loss.item())

        loss_batch.append(np.mean(loss_node))
        mse_batch.append(F.mse_loss(y.cpu(), outputs).item())

    # each item is the mean of all batches, means this indice for one epoch
    return np.mean(loss_batch), np.mean(mse_batch)



with torch.no_grad():
    loss, mse = test_dyn_gen()
    print('loss:' + str(loss) + ' mse:' + str(mse))


