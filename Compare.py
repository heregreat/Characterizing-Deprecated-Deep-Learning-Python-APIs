import pandas as pd
import os
import collections
import utils

def check(root, depre_api):
    files = utils.PyExtract().run(root) # return a list of the .py paths given a root path
    res = collections.defaultdict(list)
    for i in files:
        f = open(i)
        code = f.read()
        for api in depre_api:
            index = code.find(api)
            if index != -1 and code[index-1] == '.' and code[index + len(api)] == '(':
                res[i].append(api)
    rev = []
    for i in res.values():
        rev += i
    print(collections.Counter(rev))
    f = collections.Counter(rev)
    print(root, len(rev))
    #pd.DataFrame.from_dict(f, orient='index').to_csv(root + "(official)_counter.csv")
    pd.DataFrame.from_dict(res, orient='index').to_csv(root + "(official).csv")

if __name__ == '__main__':
    # diff is all the deprecated api name from tensorflow 1.10 to 2.20
    #root = "/Users/nianliu/Documents/GitHub/Tensorflow_APIs_NianLiu/output"
    version = ['r1.10', 'r1.11', 'r1.12', 'r1.13', 'r2.0', 'r2.1', 'r2.2', 'r2.3']
    for i in version:
        api_path = "/Users/nianliu/Documents/GitHub/Tensorflow_APIs_NianLiu/output/tensorflow-"+ i + ".csv"
        depre = pd.read_csv(api_path)
        depre_api = depre['Unnamed: 0'].tolist()
        root = "./models-"+i+'.0'
        check(root, depre_api)

