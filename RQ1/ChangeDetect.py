import ast
import collections
import os
import utils
import pandas as pd
from FindDeprecated import FuncLister

def count_loc(file_list):
    """count the LOC of files"""
    total = 0
    for file in file_list:
        lines = open(file, encoding='utf-8').readlines()
        count = 0
        for line in lines:
            if line == '\n':
                continue
            elif line.startswith('#'):
                continue
            else:
                count += 1
        total += count
    return total

def parse_def(file_list):
    """given a python file list, do the ast parse for every file,"""
    def_list = dict()
    deprecated_api = dict()
    for file in file_list:
        #print("=============\n", file)
        f = open(file)
        code = f.read()
        root_node = ast.parse(code)
        a = FuncLister(file)
        def_info, api = a.get_Function(root_node)
        # def_info.insert(0, file)
        # print(def_info)
        def_list.update(def_info)
        deprecated_api.update(api)
    loc = count_loc(file_list)
    return def_list, deprecated_api, loc


class tensorflow_diff:
    def __init__(self):
        self.version = ['tensorflow-r1.0', 'tensorflow-r1.1', 'tensorflow-r1.2', 'tensorflow-r1.3', 'tensorflow-r1.4',
                        'tensorflow-r1.5', 'tensorflow-r1.6', 'tensorflow-r1.7', 'tensorflow-r1.8', 'tensorflow-r1.9',
                        'tensorflow-r1.10', 'tensorflow-r1.11', 'tensorflow-r1.12', 'tensorflow-r1.13',
                        'tensorflow-r1.14',
                        'tensorflow-r1.15', 'tensorflow-r2.0', 'tensorflow-r2.1', 'tensorflow-r2.2', 'tensorflow-r2.3']

        root = "/Users/nianliu/tensorflow/tensorflow-r2.3/"
        folders = os.listdir(root)
        folders = sorted(folders)
        #del folders[0]
        self.res = collections.defaultdict(set)
        self.method = collections.defaultdict(set)
        method, depre, loc = parse_def(utils.PyExtract().run(root))
        print(len(depre))
        pd.DataFrame.from_dict(depre, orient='index').to_csv('arg' + ".csv")
        """
        for i in folders:
            path = os.path.join(root, i)
            if os.path.isdir(path):
                method, depre, loc = parse_def(utils.PyExtract().run(path))
                self.res[i] = set(depre.keys())
                self.method[i] = set(method.keys())
                print(i, len(method), loc, len(depre))
                print(depre)
                pd.DataFrame.from_dict(depre, orient='index').to_csv('arg' + ".csv")
        """
        #self.newly_depre()
        #self.be_removed()
        #self.get_age()

    def newly_depre(self):
        """get the apis newly deprecated in each TensorFlow version"""
        res = []
        for i in self.version:
            res.extend(self.res[i])
            print(i, len(set(res)))

        self.diff = collections.defaultdict(set)

        for i in range(len(self.res)):
            if (i == 0):
                self.diff[self.version[i]] = set(self.res[self.version[i]])
            else:
                for api in self.res[self.version[i]]:
                    if (api not in self.res[self.version[i - 1]]):
                        self.diff[self.version[i]].add(api)
        print(self.diff)
        pd.DataFrame.from_dict(self.diff, orient='index').to_csv("diff.csv")

    def be_removed(self):
        """get the removed APIs"""
        removed = collections.defaultdict(set)
        for i in range(len(self.res)-1):
            for api in self.res[self.version[i]]:
                    if api not in self.res[self.version[i+1]]:
                        removed[self.version[i+1]].add(api)
        print(removed)
        pd.DataFrame.from_dict(removed, orient='index').to_csv("removed_apis.csv")

    def get_age(self):
        """get the number of versions API gets deprecated"""
        self.age_info = collections.defaultdict(list)
        self.age = collections.defaultdict(list)
        for i in range(1, len(self.version)):
            for deprecated in self.diff[self.version[i]]:
                for j in range(i):
                    if deprecated in self.method[self.version[j]]:
                        self.age[i-j].append(deprecated)
                        self.age_info[deprecated].append((self.version[i], self.version[j]))
                        break
        pd.DataFrame.from_dict(self.age, orient='index').to_csv("age.csv")
        pd.DataFrame.from_dict(self.age_info, orient='index').to_csv("age_info.csv")

if __name__ == '__main__':
    tensorflow_diff()