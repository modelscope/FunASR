import importlib.util
import sys

def load_class_from_path(model_path):
    path, class_name = model_path
    # import pdb;
    # pdb.set_trace()
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

