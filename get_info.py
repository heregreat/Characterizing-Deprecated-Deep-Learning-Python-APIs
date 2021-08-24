import ast
import os
import pandas as pd
import ast

class FuncLister(ast.NodeVisitor):
    """get all Function name given a python code"""

    def __init__(self, file_path):
        self.file = file_path.split("/")[-1]
        self.path = file_path
        self.Function_info = {}
        self.Deprecated = {}

    def get_deprecated(self,node):
        """获取deprecation annotation"""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if "depreca" in decorator.id and "endpoints" not in decorator.id:
                    #print("1", decorator.id,self.path)
                    self.Deprecated[node.name] = self.path
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    if "deprecat" in decorator.func.id and "endpoints" not in decorator.func.id:
                        #print('2', decorator.func.id,self.path)
                        self.Deprecated[node.name] = self.path
                elif isinstance(decorator.func, ast.Attribute):
                    #if "deprecat" in decorator.func.value.id and "endpoints" not in decorator.func.attr and "args" not in decorator.func.attr:
                        #print('3', decorator.func.attr, self.path)
                        #self.Deprecated[node.name] = self.path
                    pass

    def visit_FunctionDef(self, node):
        #print(self.path)
        if not (node.name.startswith("_") or node.name == "main"): # 获取API名
            self.get_deprecated(node)
            #returns = self.get_return(node)
            args_list = ""
            arg_len = len(node.args.args)
            count = 0
            for arg in node.args.args:
                if not arg.arg == "self":
                    if count < arg_len-1:
                        args_list += arg.arg + ', '  # save all agrs
                    elif count == arg_len - 1:
                        args_list += arg.arg
                count +=1
            def_name = node.name
            if def_name not in self.Function_info:
                self.Function_info[def_name] = args_list  # save function name
            else:
                #print(def_name)
                self.Function_info[def_name] += args_list
            # self.Function_info.append(a)  # save function name and args as one list

        self.generic_visit(node)

    def get_Function(self, node):
        """get all Function info, function name, args, return"""

        self.generic_visit(node)
        return self.Function_info, self.Deprecated

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


def count_loc(file_list):

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

if __name__ == '__main__':
    root = "/Users/nianliu/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/ModelGarden"
    version = ["models-r1.10.0", "models-r1.11.0", "models-r1.12.0", "models-r1.13.0", "models-r2.0.0", "models-r2.1.0", "models-r2.2.0", "models-r2.3.0"]
    for i in version:
        path = os.path.join(root, i)
        paths = PyExtract().run(path)
        a, b, c = parse_def(paths)
        print(i,'\n', len(a), c)