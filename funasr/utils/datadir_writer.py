from pathlib import Path
from typing import Union
import warnings


class DatadirWriter:
    """Writer class to create kaldi like data directory.

    Examples:
        >>> with DatadirWriter("output") as writer:
        ...     # output/sub.txt is created here
        ...     subwriter = writer["sub.txt"]
        ...     # Write "uttidA some/where/a.wav"
        ...     subwriter["uttidA"] = "some/where/a.wav"
        ...     subwriter["uttidB"] = "some/where/b.wav"

    """

    def __init__(self, p: Union[Path, str]):
        self.path = Path(p)
        self.chilidren = {}
        self.fd = None
        self.has_children = False
        self.keys = set()

    def __enter__(self):
        return self

    def __getitem__(self, key: str) -> "DatadirWriter":
        if self.fd is not None:
            raise RuntimeError("This writer points out a file")

        if key not in self.chilidren:
            w = DatadirWriter((self.path / key))
            self.chilidren[key] = w
            self.has_children = True

        retval = self.chilidren[key]
        return retval

    def __setitem__(self, key: str, value: str):
        if self.has_children:
            raise RuntimeError("This writer points out a directory")
        if key in self.keys:
            warnings.warn(f"Duplicated: {key}")

        if self.fd is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = self.path.open("w", encoding="utf-8")

        self.keys.add(key)
        self.fd.write(f"{key} {value}\n")
        self.fd.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.has_children:
            prev_child = None
            for child in self.chilidren.values():
                child.close()
                if prev_child is not None and prev_child.keys != child.keys:
                    warnings.warn(
                        f"Ids are mismatching between " f"{prev_child.path} and {child.path}"
                    )
                prev_child = child

        elif self.fd is not None:
            self.fd.close()
