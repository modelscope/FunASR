import logging
import inspect
from dataclasses import dataclass
import re


@dataclass
class RegisterTables:
    model_classes = {}
    frontend_classes = {}
    specaug_classes = {}
    normalize_classes = {}
    encoder_classes = {}
    decoder_classes = {}
    joint_network_classes = {}
    predictor_classes = {}
    stride_conv_classes = {}
    tokenizer_classes = {}
    dataloader_classes = {}
    batch_sampler_classes = {}
    dataset_classes = {}
    index_ds_classes = {}

    def print(self, key=None):
        print("\ntables: \n")
        fields = vars(self)
        for classes_key, classes_dict in fields.items():

            flag = True
            if key is not None:
                flag = key in classes_key
            if classes_key.endswith("_meta") and flag:
                print(f"-----------    ** {classes_key.replace('_meta', '')} **    --------------")
                headers = ["register name", "class name", "class location"]
                metas = []
                for register_key, meta in classes_dict.items():
                    metas.append(meta)
                metas.sort(key=lambda x: x[0])
                data = [headers] + metas
                col_widths = [max(len(str(item)) for item in col) for col in zip(*data)]

                for row in data:
                    print(
                        "| "
                        + " | ".join(str(item).ljust(width) for item, width in zip(row, col_widths))
                        + " |"
                    )
        print("\n")

    def register(self, register_tables_key: str, key=None):
        def decorator(target_class):

            if not hasattr(self, register_tables_key):
                setattr(self, register_tables_key, {})
                logging.info("new registry table has been added: {}".format(register_tables_key))

            registry = getattr(self, register_tables_key)
            registry_key = key if key is not None else target_class.__name__

            # assert not registry_key in registry, "(key: {} / class: {}) has been registered already，in {}".format(
            #     registry_key, target_class, register_tables_key)

            registry[registry_key] = target_class

            # meta， headers = ["class name", "register name", "class location"]
            register_tables_key_meta = register_tables_key + "_meta"
            if not hasattr(self, register_tables_key_meta):
                setattr(self, register_tables_key_meta, {})
            registry_meta = getattr(self, register_tables_key_meta)
            # doc = target_class.__doc__
            class_file = inspect.getfile(target_class)
            class_line = inspect.getsourcelines(target_class)[1]
            pattern = r"^.+/funasr/"
            class_file = re.sub(pattern, "funasr/", class_file)
            # meata_data = [f"{target_class.__name__}", f"{class_file}:{class_line}"]
            meata_data = [
                f"{registry_key}",
                f"{target_class.__name__}",
                f"{class_file}:{class_line}",
            ]
            registry_meta[registry_key] = meata_data
            # print(f"Registering class: {class_file}:{class_line} - {target_class.__name__} as {registry_key}")
            return target_class

        return decorator


tables = RegisterTables()


import funasr
