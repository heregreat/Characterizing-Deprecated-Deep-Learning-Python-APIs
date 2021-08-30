import os


class PyExtract():
    """Extract all .py file path from project path """

    def __init__(self):
        self.all_file = []
    def run(self, root_path):
        files = os.listdir(root_path)
        for file in files:
            file = os.path.join(root_path, file)
            if os.path.isdir(file):
                self.run(file)
            elif os.path.isfile(file) and file.endswith(".py"):
                if not file.endswith("test.py"):
                    if not "__" in file and not "contrib" in file:
                        self.all_file.append(file)
        return self.all_file


def check_diff(a, b):
    diff_para = {}
    diff_rm = {}
    diff_add = {}
    for key in a:
        if not key in b:
            diff_rm[key] = ''
            #print(key)
        elif b[key] != a[key]:
            diff_para[key] = list([a[key], b[key]])
            #print(diff_para[key])
    for key in b:
        if not key in a:
            diff_add[key] = ''
    return diff_rm, diff_add, diff_para

def check(a,b):
    if(a in b):
        return True
    else:
        return False
