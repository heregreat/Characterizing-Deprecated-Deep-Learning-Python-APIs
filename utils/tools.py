import pandas as pd
import sklearn.utils as utils
import os


def move(x, i):
    path = "/Users/nianliu/Documents/GitHub/ModelGarden/models-r2.2.0/official/nlp/transformer"
    new_dir = path + "/log" + str(i)
    try:
        os.mkdir(new_dir)
    except:
        pass
    for lable in x:
        files = os.listdir(path+'/log')
        for file in files:
            if str(lable) in file:
                path1 = path + '/log/' + file
                path2 = new_dir
                if os.path.exists(path2 +'/' + file):
                    command = 'cp ' + path1 +' ' + path2 + file
                else:
                    command = 'cp ' + path1 +' ' + path2

                os.system(command)




if __name__ == '__main__':
    x = [i for i in range(1, 11)]
    for i in range(10):
        x = utils.resample(x, replace=1, n_samples=10)
        move(x, i)