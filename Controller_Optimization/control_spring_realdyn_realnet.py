import torch
import time
import torch.optim as optim
from model import *
from tools import *
import argparse


# configuration
HYP = {
    'node_size': 10, #node size
    'hid': 128,  # hidden size
    'epoch_num': 1000,  # epoch
    'batch_size': 512,  # batch size
    'lr_con':0.01,
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'temp': 1,  # temperature
    'drop_frac': 1,  # temperature drop frac
}

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=100, help='Number of nodes, default=10')
parser.add_argument('--network', type=str, default='BA', help='type of network')
parser.add_argument('--sys', type=str, default='spring', help='simulated system to model,spring or cmn')
parser.add_argument('--dim', type=int, default=4, help='# information dimension of each node spring:4 cmn:1 ')
parser.add_argument('--control_steps', type=int, default=20, help='control steps')
parser.add_argument('--exp_id', type=int, default=1, help='experiment_id, default=1')
parser.add_argument('--device_id', type=int, default=3, help='Gpu_id, default=5')
args = parser.parse_args()

torch.cuda.set_device(args.device_id)



start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# controller
control_isom = Controller(HYP['dim'],HYP['hid']).to(device)
# optimizer
op_con = optim.Adam(control_isom .parameters(), lr=HYP['lr_con'])

# load_data
if args.sys== 'spring':
    train_loader, val_loader, test_loader, object_matrix = load_spring_control(batch_size=HYP['batch_size'],node_num=args.nodes,network=args.network,control_steps=args.control_steps,exp_id=args.exp_id)


object_matrix = object_matrix.cpu().numpy()



_delta_T = 0.001
n_balls = args.nodes
loc_std = 0.5
vel_norm = 0.5
interaction_strength = 0.1
_max_F = 0.1 /_delta_T



def train():
    loss_batch = []

    node_list = list(np.arange(10))
    # driver node
    control_node_list = [0,1,3]
    # target node
    controlled_node_list = [i for i in node_list if i not in control_node_list]

    forces_size = - interaction_strength * object_matrix

    for i in range(forces_size.size(0)):
        forces_size[i][i] = 0
    Y = np.array([])
    LOSS=np.array([])
    for idx, data in enumerate(test_loader):
        print('batch idx:', idx)
        # data
        data = data.float().to(device)
        x = data.float()
        temp_x = x

        outputs = torch.zeros(x.size(0), 20,len(node_list), 4).to(device)
        outputs[:, 0, :, :] = temp_x[:, node_list, :]

        loss_time = []
        for time in range(19):
            controlled_x = temp_x[:,controlled_node_list,:].float()

            for k in control_node_list:
                con_adj_col = object_matrix[:,k]
                control_x = control_isom(temp_x[:,k,:],controlled_x,con_adj_col,controlled_node_list)
                temp_x[:, k, 2:] = control_x


            for i in range(100):
                temp_x[:, :, :2] += _delta_T * temp_x[:, :, 2:]
                F = torch.sum((forces_size
                               .unsqueeze(0).expand(2 * temp_x.size(0), args.nodes, args.nodes) *
                               torch.cat((
                                   (temp_x[:, :, 0].unsqueeze(2).expand(temp_x.size(0), args.nodes,
                                                                        args.nodes) - temp_x[:, :, 0].unsqueeze(
                                       1).expand(temp_x.size(0), args.nodes, args.nodes)),

                                   (temp_x[:, :, 1].unsqueeze(2).expand(temp_x.size(0), args.nodes,
                                                                        args.nodes) - temp_x[:, :,
                                                                                      1].unsqueeze(
                                       1).expand(temp_x.size(0), args.nodes, args.nodes))
                               ))), -1)

                F = F.unsqueeze(2)
                F = torch.cat((F[:temp_x.size(0)], F[temp_x.size(0):]), -1)
                F[F > _max_F] = _max_F
                F[F < -_max_F] = -_max_F
                temp_x[:, :, 2:] += _delta_T * F


            controlled_node_list_copy = controlled_node_list.copy()
            loss=0
            count=0
            while len(controlled_node_list_copy)>1:
                cur_con_node = controlled_node_list_copy.pop(0)
                for no in controlled_node_list_copy:
                    count = count +1
                    loss = loss + torch.mean(torch.abs(temp_x[:, cur_con_node, 2:] / torch.norm(
                        temp_x[:, cur_con_node, 2:], p=2, dim=-1).unsqueeze(1).expand(
                        temp_x[:, cur_con_node, 2:].size(0), 2) - temp_x[:, no, 2:] / torch.norm(
                        temp_x[:, no, 2:], p=2, dim=-1).unsqueeze(1).expand(
                        temp_x[:, no, 2:].size(0), 2)))

            loss = loss/count
            temp_x = temp_x.detach()
            outputs[:, time+1, :, :]=temp_x[:,node_list,:]
            loss_time.append(loss.item())

        LOSS = np.vstack((LOSS, np.array(loss_time))) if LOSS.size else np.array(loss_time)
        Y = np.vstack((Y, outputs.cpu().detach().numpy())) if Y.size else outputs.cpu().detach().numpy()

        loss_batch.append(np.mean(loss_time))
    # each item is the mean of all batches, means this indice for one epoch
    return np.mean(loss_batch), Y,LOSS


main_path = './model/'
control_path = main_path + 'control_spring_'+str(args.network)+'_realdyn_realnet_model_id'+str(args.exp_id)+'.pkl'
data_path = './data/control_spring_'+str(args.network)+'_realdyn_realnet_data_id'+str(args.exp_id)+'.pickle'
loss_path = './data/control_spring_'+str(args.network)+'_realdyn_realnet_loss_id'+str(args.exp_id)+'.pickle'

best_loss=1000
real_time_list = []
# each training epoch
for e in range(HYP['epoch_num']):
    print('\nepoch', e)
    if e % 50 == 0:
        torch.cuda.empty_cache()

    t_s = time.time()
    # train both dyn learner and generator together
    t_s1 = time.time()

    try:
        loss, Y ,LOSS= train()
    except RuntimeError as sss:
        if 'out of memory' in str(sss):
            print('|WARNING: ran out of memory')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise sss
    t_e1 = time.time()
    print('loss:' + str(loss) )
    print('time for this dyn_adj epoch:' + str(round(t_e1 - t_s1, 2)))

    if loss < best_loss:
        print('best epoch:', e)
        best = e
        best_loss = loss
        torch.save(control_isom, control_path)
        with open(data_path, 'wb') as f:
            pickle.dump(Y, f)

        with open(loss_path, 'wb') as f:
            pickle.dump(LOSS, f)
    print('best epoch:', best)

    t_e = time.time()
    real_time_list.append(round(t_e - t_s, 2))
    real_time = sum(real_time_list)

    print('time for this whole epoch:' + str(round(t_e - t_s, 2)))
    print('realtime until this epoch:' + str(round(real_time, 2)))

end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


print('end_time:', end_time)