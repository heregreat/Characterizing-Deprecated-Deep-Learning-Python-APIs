import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import os


def get_loss(path):
    f = open(path)
    lines = f.readlines()
    loss = 0
    for line in lines:
        if "steps" in line:
            print(line)
            index = line.find("loss:")
            loss = float(line[index + 6:index + 12])
            break
    return loss


if __name__ == '__main__':
    dat = []
    dat_new = []
    path = "/Users/nianliu/Documents/GitHub/ModelGarden/models-r2.2.0/official/nlp/transformer/log"
    for i in os.listdir(path):
        loss = get_loss(os.path.join(path, i))
        if "new" in i:
            dat_new.append(loss)
        else:
            dat.append(loss)
    print(dat, dat_new)
    stat_val, p_val = stats.ttest_ind(dat, dat_new, equal_var=False)
    print ('Two-sample t-statistic D = %6.3f, p-value = %6.4f' % (stat_val, p_val))