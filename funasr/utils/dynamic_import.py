import importlib.util

import importlib.util
import inspect


def load_module_from_path(file_path):
    """
    从给定的文件路径动态加载模块。

    :param file_path: 模块文件的绝对路径。
    :return: 加载的模块
    """
    module_name = file_path.split("/")[-1].replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#
# def load_module_from_path(module_name, file_path):
#     """
#     从给定的文件路径动态加载模块。
#
#     :param module_name: 动态加载的模块的名称。
#     :param file_path: 模块文件的绝对路径。
#     :return: 加载的模块
#     """
#     # 创建加载模块的spec（规格）
#     spec = importlib.util.spec_from_file_location(module_name, file_path)
#
#     # 根据spec创建模块
#     module = importlib.util.module_from_spec(spec)
#
#     # 执行模块的代码来实际加载它
#     spec.loader.exec_module(module)
#
#     return module
