import pandas as pd
import collections, os
def tool():
    count = pd.read_csv("/Users/nianliu/Desktop/models_counter.csv", names=["models", "num"])
    res = collections.defaultdict(int)
    models = list(count["models"])[1:]
    num = list(count["num"])[1:]
    for i, j in zip(models, num):
        res[i] += j
    print(res)
    pd.DataFrame.from_dict(res, orient='index').to_csv("models_all(official).csv")

if __name__ == '__main__':

    path = "Tools/arg.csv"
    data = pd.read_csv(path)
    res = collections.defaultdict(list)
    for a, row in data.iterrows():
        api = row[0]
        file = row[1]

        f = open(file)
        code = f.read()
        index = code.find("def " + api)
        #print(api, '\n')
        #print(code[index-300:index+300])
        message = ''.join(code[index-300:index+300])
        res[api].append(message)
    print(res)
    pd.DataFrame.from_dict(res).to_csv("arg_deprecation_message.csv")



