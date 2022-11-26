import collections
import sys

from torch import multiprocessing


def get_size(obj, seen=None):
    """Recursively finds size of objects

    Taken from https://github.com/bosswissam/pysize

    """

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif isinstance(obj, (list, set, tuple)):
        size += sum([get_size(i, seen) for i in obj])

    return size


class SizedDict(collections.abc.MutableMapping):
    def __init__(self, shared: bool = False, data: dict = None):
        if data is None:
            data = {}

        if shared:
            # NOTE(kamo): Don't set manager as a field because Manager, which includes
            # weakref object, causes following error with method="spawn",
            # "TypeError: can't pickle weakref objects"
            self.cache = multiprocessing.Manager().dict(**data)
        else:
            self.manager = None
            self.cache = dict(**data)
        self.size = 0

    def __setitem__(self, key, value):
        if key in self.cache:
            self.size -= get_size(self.cache[key])
        else:
            self.size += sys.getsizeof(key)
        self.size += get_size(value)
        self.cache[key] = value

    def __getitem__(self, key):
        return self.cache[key]

    def __delitem__(self, key):
        self.size -= get_size(self.cache[key])
        self.size -= sys.getsizeof(key)
        del self.cache[key]

    def __iter__(self):
        return iter(self.cache)

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)
