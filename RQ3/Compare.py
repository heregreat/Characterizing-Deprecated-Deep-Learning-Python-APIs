import pandas as pd
import os
import collections
from RQ1 import utils
def count(a):
    # diff is all the deprecated api name from tensorflow 1.10 to 2.20
    diff = pd.read_csv("/Users/nianliu/Documents/GitHub/Tensorflow_APIs_NianLiu/output_old/diff_withoutendpoint.csv")
    diff = diff.iloc[:, 1:]
    depre_api = []
    for i in diff.iloc[:, :14]:
        # print(i)
        for i in diff[i]:
            if str(i) != 'nan':
                depre_api.append(i)
    apis = ['to_float', 'to_int32', 'sparse_to_dense', 'parallel_interleave', "make_one_shot_iterator",
            'tf_record_iterator', 'softmax_cross_entropy_with_logits', 'max_pooling2d', 'dense', 'batch_normalization'
        , 'conv2d', 'map_and_batch', 'py_func', 'is_gpu_available', 'test_session', 'set_learning_phase',
            'experimental_run_functions_eagerly']

    name = "models-master"
    root = "/Users/nianliu/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Characterizing_Dep_DP_APIs:/models-r" + a
    files = utils.PyExtract().run(root)  # return a list of the .py paths given a root path
    version = {
        '1.13.0': ['to_float', 'to_int32', 'flatten', 'tf_record_iterator' 'sparse_to_dense', 'max_pooling2d', 'dense',
                   'batch_normalization', 'test_session', 'conv2d']
        , '2.0.0': ['to_float', 'to_int32', 'tf_record_iterator' 'sparse_to_dense', 'max_pooling2d', 'dense',
                    'batch_normalization', 'conv2d', 'parallel_interleave', 'softmax_cross_entropy_with_logits',
                    'parallel_interleave']
        , '2.1.0': ['to_float', 'to_int32', 'tf_record_iterator' 'sparse_to_dense', 'max_pooling2d', 'dense',
                    'batch_normalization', 'conv2d', 'parallel_interleave', 'softmax_cross_entropy_with_logits',
                    'parallel_interleave',
                    'py_func', 'is_gpu_available']
        , '2.2.0': ['to_float', 'to_int32', 'tf_record_iterator' 'sparse_to_dense', 'max_pooling2d', 'dense',
                    'batch_normalization', 'conv2d', 'parallel_interleave', 'softmax_cross_entropy_with_logits',
                    'parallel_interleave',
                    'py_func', 'is_gpu_available']
        , '2.3.0': ['to_float', 'to_int32', 'tf_record_iterator' 'sparse_to_dense', 'parallel_interleave',
                    'softmax_cross_entropy_with_logits', 'parallel_interleave',
                    'py_func', 'set_learning_phase', 'experimental_run_functions_eagerly']
        }

    res = collections.defaultdict(list)
    for i in files:
        f = open(i)
        # code = f.read()
        # f = open(path)
        lines = f.readlines()
        for api in version[a]:
            for line in lines:
                index = line.find(api)
                if index != -1 and line[index - 1] == '.' and line[index + len(api)] == '(':
                    print(api, i)
                    res[i].append(api)
    rev = []
    for i in res.values():
        rev += i
    print(collections.Counter(rev))
    f = collections.Counter(rev)
    print(len(rev))
    return rev
    # pd.DataFrame.from_dict(f, orient='index').to_csv(name + "(official)_counter.csv")
    # pd.DataFrame.from_dict(res, orient='index').to_csv(name + "(research).csv")
if __name__ == '__main__':
    total = []
    a = ['1.13.0']
    for i in a:
        total.extend(count(i))

    print(collections.Counter(total))