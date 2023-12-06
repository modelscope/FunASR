import importlib


def dynamic_import(import_path):
    """dynamic import module and class

    :param str import_path: syntax 'module_name:class_name'
    :return: imported class
    """

    module_name, objname = import_path.split(":")
    m = importlib.import_module(module_name)
    return getattr(m, objname)
