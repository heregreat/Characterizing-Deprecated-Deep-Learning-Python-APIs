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
                if "depreca" in decorator.id and "endpoints" not in decorator.id and "args" in decorator.id:
                    print("1", decorator.id,self.path)
                    #self.Deprecated[node.name] = self.path
                    pass
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    if "deprecat" in decorator.func.id and "endpoints" not in decorator.func.id and "args" in decorator.func.id:
                        print('2', decorator.func.id,self.path)
                        #self.Deprecated[node.name] = self.path
                        pass
                elif isinstance(decorator.func, ast.Attribute):
                    if "deprecat" in decorator.func.value.id and "endpoints" not in decorator.func.attr and "args" in decorator.func.attr:
                        print('3', decorator.func.attr, self.path)
                        self.Deprecated[node.name] = self.path

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