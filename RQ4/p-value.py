import collections

import numpy
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import os
import pandas as pd
import pingouin as pg


def get_loss(path):
    f = open(path)
    lines = f.readlines()
    loss = 0
    key = "step = 4900"
    cur = ''
    for line in lines:
        pre = cur
        cur = line
        if key in line:

            #index = line.find(key)
            #res = line[index + len(key):index + len(key)+ 10]
            res = pre.split(' ')[-3]
            print(res)
            loss = float(res)
            break
    return loss
def cal():
    res = collections.defaultdict(list)
    dep = "log/resnet-1.13/originallog"

    # log/resnet-1.13/conv2d", "log/resnet-1.13/dense", "log/resnet-1.13/max_pooling2d", "log/resnet-1.13/batch_normalization"
    new = "log/resnet-1.13/experiment"
    deprecated, new_data = [], []
    for i in os.listdir(dep):
        data = get_loss(os.path.join(dep, i))
        if data != 0:
            deprecated.append(data)
        else:
            print(i)
    res['deprecated'] = deprecated
    for i in os.listdir(new):
        data = get_loss(os.path.join(new, i))
        if data != 0:
            new_data.append(data)
        else:
            print('new:zero')
    res['new'] = new_data[0:17]
    rev = pg.ttest(res['deprecated'], res['new'])
    print(len(res['deprecated']), len(res['new']))
    print(numpy.average(res['deprecated']), numpy.average(res['new']))
    print(rev['p-val'], rev['cohen-d'])
    res['final'] = [numpy.average(res['deprecated']), numpy.average(res['new']), rev['p-val'], rev['cohen-d'], 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pd.DataFrame.from_dict(res).to_csv(new + 'acc.csv')
def get_cohen():
    path = ["log/mnist-1.13/to_int32_log.csv","log/transform_2.2.0/logv2.nn.soft.csv", "log/transform-1.13/logto_int32.csv",
        "log/transform-1.13/logto_float.csv","log/cifar-10_2.1.0/log_sparse_new.csv","log/resnet-1.13/conv2d.csv","log/resnet-1.13/max_pooling2d.csv",
           "log/resnet-1.13/dense.csv", "log/resnet-1.13/batch_normalization.csv","log/xlnet-2.1.0/map_and_batch_new.csv",
           "log/xlnet-2.1.0/interleave.csv", "log/od_2.1.0/log_new.csv"]
    for i in path:
        csv = pd.read_csv(i)
        dat = csv['deprecated']
        dat_new = csv['new']

        res = pg.ttest(dat, dat_new)

        print(i,'\n', numpy.average(dat), numpy.average(dat_new))
        #print(res['p-val'], res['cohen-d'])

if __name__ == '__main__':
    #get_scipy()
    #get_cohen()
    cal()