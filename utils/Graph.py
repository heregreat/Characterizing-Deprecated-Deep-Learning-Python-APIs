import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def retained():
    total = [13, 13, 13, 15, 15, 23, 24, 24, 24, 24, 25, 31, 47, 119, 137, 155, 156, 164, 175, 199]
    retained = [13, 13, 13, 15, 15, 21, 22, 22, 22, 22, 23, 29, 45, 116, 132, 150, 150, 155, 166, 190]
    x = range(20)
    l1 = plt.plot(x, total, 'k', label='total', marker='o', markersize=3)
    l2 = plt.plot(x, retained, 'k--', label='retained', marker='o', markersize=3)
    my_x_ticks = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 2.0, 2.1, 2.2, 2.3]
    plt.xticks(range(20), my_x_ticks, rotation=270)
    plt.legend()
    plt.show()
    plt.savefig('retained.png')

def distri():
    method = [1525,1659,1888,2002, 2612,2798,2948, 3033,3172,3251,3357,3470,3786,4419,5338,5707,5714,5934, 6251, 6586]
    retained = [13, 13, 13, 15, 15, 21, 22, 22, 22, 22, 23, 29, 45, 116, 132, 150, 150, 155, 166, 190]
    ratio = []
    for i in range(20):
        ratio.append(round(retained[i]/method[i] * 100, 2))
    print(ratio)
    x = range(20)
    plt.bar(x, ratio, color = 'k', width=0.5)
    my_x_ticks = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, '1.10', 1.11, 1.12, 1.13, 1.14, 1.15, 2.0, 2.1, 2.2, 2.3]
    my_y_ticks = ['0.00%', '0.50%', '1.00%', '1.50%', '2.00%', '2.50%', '3.00%', '3.50%']
    plt.xticks(range(20), my_x_ticks, rotation=270)
    plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], my_y_ticks)
    for a, b in zip(x, ratio):
        plt.text(a, b+0.1, str(b)+'%', ha='center', rotation = 'vertical')
    plt.legend()
    plt.show()
    #plt.savefig('ratio')


def ratio():
    data2 = [81,49,18,2]
    data2 = [107, 58, 19, 15]
    data = [3, 33]
    label = ['Unnecessary Parameter (3)', 'Parameter Name Change (33)']
    label2 = ['API Optimization (107)', 'API Name Change (58)', 'Compatibility Issue (19)', 'Feature Deleting (15)']
    plt.figure(figsize=(12, 8))
    plt.pie(data2, explode=(0.02, 0.02, 0.02, 0.02), autopct="%1.2f%%", labels=label2, textprops={'fontsize': 16})
    plt.show()

def violin():
    survival= [2,2,2,1,1,2,2,2,2]
    a = [3, 5, 7, 4, 15, 3, 9, 2, 2, 9, 9, 18, 51, 7, 10, 2, 3, 4]
    b = [3, 5, 4, 4, 15, 3, 9, 2, 1, 9, 9, 18, 50, 6, 7, 2, 3, 4]
    removed = [3, 3, 3, 9, 13, 14, 15, 15, 15]
    retained = []
    index = 1
    for i in b:
        for j in range(i):
            retained.append(index)
        index += 1
    print(retained)
    removed.extend([ None for i in range(len(retained)-len(removed))])
    res = {}
    cat = ['retained' for _ in range(len(retained))]
    #cat.extend(['removed' for _ in range(len(removed))])
    #retained.extend(removed)
    res['cat'] = cat
    res['retained'] = retained
    res['removed'] = removed
    print(res)
    data2 = pd.DataFrame.from_dict(res)
    print(data2)
    #sns.set_theme(style="whitegrid")
    #tips = sns.load_dataset("tips")
    ax = sns.displot(survival, kind="kde")
    ax.set_axis_labels( 'survival time', 'Density')
    #ax.fig.set_figwidth(8)
    ax.fig.set_figheight(6)
    #ax = sns.violinplot(data, palette=['grey', 'grey', 'grey', 'grey'])
    #ax.set_title("Life Expectancy By Country")
    #ax.set_ylabel("version")
    #ax.set_xlabel("datasets")
    ax.set(xticks=[1,2])
    #ax.xticks(range(2), rotation=270)
    #plt.bar([1,2], [2,7], color='k', width=0.5)
    #plt.xticks([0, 1, 2, 3], ['0', '1', '2', '3'])
    plt.show()
    #plt.savefig('dfs')

if __name__ == '__main__':
    violin()
    #ratio()














