import argparse
import pickle
from feeders.feeder import Feeder_UCLA
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
#### nut120
# parser.add_argument('--datasets', default='ntu120/xview',
#                     choices={'kinetics', 'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xsetup'},
#                     help='the work folder for storing results')
# parser.add_argument('--alpha', default=1, help='weighted summation')
# parser.add_argument('--beta', default=0.5, help='weighted summation')
# parser.add_argument('--gama', default=1.6, help='weighted summation')
# parser.add_argument('--delta', default=0.8, help='weighted summation')
parser.add_argument('--datasets', default='nw_ucla',
                    choices={'kinetics', 'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xsetup', 'nw_ucla'},
                    help='the work folder for storing results')
arg = parser.parse_args()
class sum_weight():
    def __init__(self,alpha,beta,gama,delta):
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.delta = delta

if arg.datasets == 'ntu/xview':
    weight = sum_weight(1, 0.6, 1, 0.6)
if arg.datasets == 'ntu/xsub':
    # weight = sum_weight(0.6, 0.6, 1, 0.5)
    weight = sum_weight(1, 0.6, 1, 0.6)
    # weight = sum_weight(0.8, 0.2, 0.9, 0.2)
if arg.datasets == 'ntu120/xsub':
    # weight = sum_weight(1, 0.5, 1.6, 0.8)
    weight = sum_weight(0.8, 0.3, 1.6, 0.8)
if arg.datasets == 'ntu120/xsetup':
    weight = sum_weight(1, 0.4, 1.5, 0.7)
    # weight = sum_weight(1, 1, 1, 1)
if arg.datasets == 'kinetics':
    weight = sum_weight(1.8, 1.0,1.7, 0.2)
    # weight = sum_weight(1., 0.7,1.2,0.1)
if arg.datasets == 'nw_ucla':
    weight = sum_weight(1, 0.3, 1, 0.3)




dataset = arg.datasets
if dataset == 'nw_ucla':
    data_feeder = Feeder_UCLA(None, 'val')
    label = np.array(data_feeder.label)
    ind = np.array(range(len(label)))
    label = np.stack([ind,label])
else:
    label = open('./data/' + dataset + '/val_label.pkl', 'rb')
    label = np.array(pickle.load(label))
    ind = np.argsort(label[0])
    label = label[:,ind]
r1 = open('./work_dir/' + dataset + '/rsgcn_test_joint/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r1 =sorted(r1,key=lambda x: x[0])
r2 = open('./work_dir/' + dataset + '/rsgcn_test_joint_motion/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r2 =sorted(r2,key=lambda x: x[0])
r3 = open('./work_dir/' + dataset + '/rsgcn_test_bone/epoch1_test_score.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r3 =sorted(r3,key=lambda x: x[0])
r4 = open('./work_dir/' + dataset + '/rsgcn_test_bone_motion/epoch1_test_score.pkl', 'rb')
r4 = list(pickle.load(r4).items())
r4 =sorted(r4,key=lambda x: x[0])

right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]
    _, r44 = r4[i]
    r = r11* weight.alpha + r22* weight.beta+  r33* weight.gama + r44* weight.delta
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)
