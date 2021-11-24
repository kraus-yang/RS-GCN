import argparse
import pickle
from feeders.feeder import Feeder_UCLA
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='nw_ucla', choices={'kinetics', 'ntu/xsub', 'ntu/xview','nw_ucla'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

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
r2 = open('./work_dir/' + dataset + '/rsgcn_test_bone/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r2 =sorted(r2,key=lambda x: x[0])
right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    r = r11 + r22 * arg.alpha
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)
