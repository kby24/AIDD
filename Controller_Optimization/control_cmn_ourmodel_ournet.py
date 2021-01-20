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
parser.add_argument('--sys', type=str, default='cmn', help='simulated system to model,spring or cmn')
parser.add_argument('--dim', type=int, default=1, help='# information dimension of each node spring:4 cmn:1 ')
parser.add_argument('--control_steps', type=int, default=20, help='control steps')
parser.add_argument('--exp_id', type=int, default=1, help='experiment_id, default=1')
parser.add_argument('--device_id', type=int, default=3, help='Gpu_id, default=5')
args = parser.parse_args()

torch.cuda.set_device(args.device_id)



start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model load path
# dyn_path = './model/dyn_cmn_' + args.network + '_' + str(args.nodes) + '_id' + str(args.exp_id) + '.pkl'
# gen_path = './model/gen_cmn_' + args.network + '_' + str(args.nodes) + '_id' + str(args.exp_id) + '.pkl'
#for example
dyn_path = './model/dyn_cmn_BA_10_id1.pkl'
gen_path = './model/gen_cmn_BA_10_id1.pkl'
#load model
dyn_isom = torch.load(dyn_path).to(device)
generator = torch.load(gen_path).to(device)
out_matrix=generator.sample_all()
# controller
control_isom = Controller(HYP['dim'],HYP['hid']).to(device)
# optimizer
op_con = optim.Adam(control_isom .parameters(), lr=HYP['lr_con'])

# load_data
if args.sys== 'cmn':
    test_loader, object_matrix = load_cmn_control(batch_size=HYP['batch_size'],node_num=args.nodes,network=args.network,control_steps=args.control_steps,exp_id=args.exp_id)


object_matrix = object_matrix.cpu().numpy()
def train():
    loss_batch = []

    node_list = list(np.arange(10))
    #dirver node
    control_node_list = [2,8]
    #target node
    controlled_node_list = [i for i in node_list if i not in control_node_list]

    Y=np.array([])


    for idx, data in enumerate(test_loader):
        print('batch idx:', idx)
        # data
        data = data.float().to(device)
        x = data.float()
        temp_x = x

        #target trace
        new_y = (torch.ones(x.size(0),args.control_steps,len(controlled_node_list),1)*0.6).float().to(device)
        outputs = torch.zeros(x.size(0), args.control_steps+1,len(node_list), 1).to(device)
        outputs[:, 0, :, :] = temp_x[:, node_list, :]


        loss_time=[]
        for time in range(args.control_steps):
            controlled_x = temp_x[:,controlled_node_list,:].float()
            # zero grad
            op_con.zero_grad()

            for k in control_node_list:
                con_adj_col = out_matrix[:,k]
                control_x = control_isom(temp_x[:,k,:],controlled_x,con_adj_col,controlled_node_list)
                temp_x[:,k,:]=control_x
            current_tempx = temp_x

            for j in node_list:
                adj_col  = out_matrix[:,j]
                num = int(args.nodes / HYP['node_size'])
                remainder = int(args.nodes % HYP['node_size'])
                if remainder == 0:
                    num = num - 1
                y_hat = dyn_isom(current_tempx, adj_col, j, num, HYP['node_size'])
                temp_x[:,j,:]=y_hat

            loss = torch.mean(torch.abs(temp_x[:,controlled_node_list,:]-new_y[:,time,:,:]))

            loss.backward()
            op_con.step()
            temp_x = temp_x.detach()
            outputs[:, time + 1, :, :] = temp_x
            loss_time.append(loss.item())

        Y = np.vstack((Y, outputs.cpu().detach().numpy())) if Y.size else outputs.cpu().detach().numpy()
        loss_batch.append(np.mean(loss_time))


    return np.mean(loss_batch), Y




main_path = './model/'
control_path = main_path + 'control_cmn_'+str(args.network)+'_ourmodel_ournet_model_id'+str(args.exp_id)+'.pkl'
data_path = './data/control_cmn_'+str(args.network)+'_ourmodel_ournet_data_id'+str(args.exp_id)+'.pickle'

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
    # loss, mse = train_dyn_gen()

    try:
        loss, Y = train()
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
    print('best epoch:', best)

    t_e = time.time()
    real_time_list.append(round(t_e - t_s, 2))
    real_time = sum(real_time_list)

    print('time for this whole epoch:' + str(round(t_e - t_s, 2)))
    print('realtime until this epoch:' + str(round(real_time, 2)))

end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

print('end_time:', end_time)