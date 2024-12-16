import logging
import inspect
from dataclasses import dataclass
import re


@dataclass
class RegisterTables:
    """Registry system for classes."""

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

    def print(self, key: str = None) -> None:
        """Print registered classes."""
        print("\ntables: \n")
        fields = vars(self)
        headers = ["register name", "class name", "class location"]
        for classes_key, classes_dict in fields.items():
            if classes_key.endswith("_meta") and (key is None or key in classes_key):
                print(f"-----------    ** {classes_key.replace('_meta', '')} **    --------------")
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

    def register(self, register_tables_key: str, key: str = None) -> callable:
        """Decorator to register a class."""

        def decorator(target_class):
            if not hasattr(self, register_tables_key):
                setattr(self, register_tables_key, {})
                logging.debug(f"New registry table added: {register_tables_key}")

            registry = getattr(self, register_tables_key)
            registry_key = key if key is not None else target_class.__name__

            if registry_key in registry:
                logging.debug(
                    f"Key {registry_key} already exists in {register_tables_key}, re-register"
                )

            registry[registry_key] = target_class

            register_tables_key_meta = register_tables_key + "_meta"
            if not hasattr(self, register_tables_key_meta):
                setattr(self, register_tables_key_meta, {})
            registry_meta = getattr(self, register_tables_key_meta)

            class_file = inspect.getfile(target_class)
            class_line = inspect.getsourcelines(target_class)[1]
            pattern = r"^.+/funasr/"
            class_file = re.sub(pattern, "funasr/", class_file)
            meta_data = [
                registry_key,
                target_class.__name__,
                f"{class_file}:{class_line}",
            ]
            registry_meta[registry_key] = meta_data
            return target_class

        return decorator


tables = RegisterTables()
