"""Reporter module."""
import dataclasses
import datetime
import logging
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import ContextManager
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

Num = Union[float, int, complex, torch.Tensor, np.ndarray]

_reserved = {"time", "total_count"}


def to_reported_value(v: Num, weight: Num = None) -> "ReportedValue":
    assert check_argument_types()
    if isinstance(v, (torch.Tensor, np.ndarray)):
        if np.prod(v.shape) != 1:
            raise ValueError(f"v must be 0 or 1 dimension: {len(v.shape)}")
        v = v.item()

    if isinstance(weight, (torch.Tensor, np.ndarray)):
        if np.prod(weight.shape) != 1:
            raise ValueError(f"weight must be 0 or 1 dimension: {len(weight.shape)}")
        weight = weight.item()

    if weight is not None:
        retval = WeightedAverage(v, weight)
    else:
        retval = Average(v)
    assert check_return_type(retval)
    return retval


def aggregate(values: Sequence["ReportedValue"]) -> Num:
    assert check_argument_types()

    for v in values:
        if not isinstance(v, type(values[0])):
            raise ValueError(
                f"Can't use different Reported type together: "
                f"{type(v)} != {type(values[0])}"
            )

    if len(values) == 0:
        warnings.warn("No stats found")
        retval = np.nan

    elif isinstance(values[0], Average):
        retval = np.nanmean([v.value for v in values])

    elif isinstance(values[0], WeightedAverage):
        # Excludes non finite values
        invalid_indices = set()
        for i, v in enumerate(values):
            if not np.isfinite(v.value) or not np.isfinite(v.weight):
                invalid_indices.add(i)
        values = [v for i, v in enumerate(values) if i not in invalid_indices]

        if len(values) != 0:
            # Calc weighed average. Weights are changed to sum-to-1.
            sum_weights = sum(v.weight for i, v in enumerate(values))
            sum_value = sum(v.value * v.weight for i, v in enumerate(values))
            if sum_weights == 0:
                warnings.warn("weight is zero")
                retval = np.nan
            else:
                retval = sum_value / sum_weights
        else:
            warnings.warn("No valid stats found")
            retval = np.nan

    else:
        raise NotImplementedError(f"type={type(values[0])}")
    assert check_return_type(retval)
    return retval


def wandb_get_prefix(key: str):
    if key.startswith("valid"):
        return "valid/"
    if key.startswith("train"):
        return "train/"
    if key.startswith("attn"):
        return "attn/"
    return "metrics/"


class ReportedValue:
    pass


@dataclasses.dataclass(frozen=True)
class Average(ReportedValue):
    value: Num


@dataclasses.dataclass(frozen=True)
class WeightedAverage(ReportedValue):
    value: Tuple[Num, Num]
    weight: Num


class SubReporter:
    """This class is used in Reporter.

    See the docstring of Reporter for the usage.
    """

    def __init__(self, key: str, epoch: int, total_count: int):
        assert check_argument_types()
        self.key = key
        self.epoch = epoch
        self.start_time = time.perf_counter()
        self.stats = defaultdict(list)
        self._finished = False
        self.total_count = total_count
        self.count = 0
        self._seen_keys_in_the_step = set()

    def get_total_count(self) -> int:
        """Returns the number of iterations over all epochs."""
        return self.total_count

    def get_epoch(self) -> int:
        return self.epoch

    def next(self):
        """Close up this step and reset state for the next step"""
        for key, stats_list in self.stats.items():
            if key not in self._seen_keys_in_the_step:
                # Fill nan value if the key is not registered in this step
                if isinstance(stats_list[0], WeightedAverage):
                    stats_list.append(to_reported_value(np.nan, 0))
                elif isinstance(stats_list[0], Average):
                    stats_list.append(to_reported_value(np.nan))
                else:
                    raise NotImplementedError(f"type={type(stats_list[0])}")

            assert len(stats_list) == self.count, (len(stats_list), self.count)

        self._seen_keys_in_the_step = set()

    def register(
            self,
            stats: Dict[str, Optional[Union[Num, Dict[str, Num]]]],
            weight: Num = None,
    ) -> None:
        assert check_argument_types()
        if self._finished:
            raise RuntimeError("Already finished")
        if len(self._seen_keys_in_the_step) == 0:
            # Increment count as the first register in this step
            self.total_count += 1
            self.count += 1

        for key2, v in stats.items():
            if key2 in _reserved:
                raise RuntimeError(f"{key2} is reserved.")
            if key2 in self._seen_keys_in_the_step:
                raise RuntimeError(f"{key2} is registered twice.")
            if v is None:
                v = np.nan
            r = to_reported_value(v, weight)

            if key2 not in self.stats:
                # If it's the first time to register the key,
                # append nan values in front of the the value
                # to make it same length to the other stats
                # e.g.
                # stat A: [0.4, 0.3, 0.5]
                # stat B: [nan, nan, 0.2]
                nan = to_reported_value(np.nan, None if weight is None else 0)
                self.stats[key2].extend(
                    r if i == self.count - 1 else nan for i in range(self.count)
                )
            else:
                self.stats[key2].append(r)
            self._seen_keys_in_the_step.add(key2)

    def log_message(self, start: int = None, end: int = None, num_updates: int = None) -> str:
        if self._finished:
            raise RuntimeError("Already finished")
        if start is None:
            start = 0
        if start < 0:
            start = self.count + start
        if end is None:
            end = self.count

        if self.count == 0 or start == end:
            return ""

        message = f"{self.epoch}epoch:{self.key}:" f"{start + 1}-{end}batch:"
        if num_updates is not None:
            message += f"{num_updates}num_updates: "

        for idx, (key2, stats_list) in enumerate(self.stats.items()):
            assert len(stats_list) == self.count, (len(stats_list), self.count)
            # values: List[ReportValue]
            values = stats_list[start:end]
            if idx != 0 and idx != len(stats_list):
                message += ", "

            v = aggregate(values)
            if abs(v) > 1.0e3:
                message += f"{key2}={v:.3e}"
            elif abs(v) > 1.0e-3:
                message += f"{key2}={v:.3f}"
            else:
                message += f"{key2}={v:.3e}"
        return message

    def tensorboard_add_scalar(self, summary_writer, start: int = None):
        if start is None:
            start = 0
        if start < 0:
            start = self.count + start

        for key2, stats_list in self.stats.items():
            assert len(stats_list) == self.count, (len(stats_list), self.count)
            # values: List[ReportValue]
            values = stats_list[start:]
            v = aggregate(values)
            summary_writer.add_scalar(f"{key2}", v, self.total_count)

    def wandb_log(self, start: int = None):
        import wandb

        if start is None:
            start = 0
        if start < 0:
            start = self.count + start

        d = {}
        for key2, stats_list in self.stats.items():
            assert len(stats_list) == self.count, (len(stats_list), self.count)
            # values: List[ReportValue]
            values = stats_list[start:]
            v = aggregate(values)
            d[wandb_get_prefix(key2) + key2] = v
        d["iteration"] = self.total_count
        wandb.log(d)

    def finished(self) -> None:
        self._finished = True

    @contextmanager
    def measure_time(self, name: str):
        start = time.perf_counter()
        yield start
        t = time.perf_counter() - start
        self.register({name: t})

    def measure_iter_time(self, iterable, name: str):
        iterator = iter(iterable)
        while True:
            try:
                start = time.perf_counter()
                retval = next(iterator)
                t = time.perf_counter() - start
                self.register({name: t})
                yield retval
            except StopIteration:
                break


class Reporter:
    """Reporter class.

    Examples:

        >>> reporter = Reporter()
        >>> with reporter.observe('train') as sub_reporter:
        ...     for batch in iterator:
        ...         stats = dict(loss=0.2)
        ...         sub_reporter.register(stats)

    """

    def __init__(self, epoch: int = 0):
        assert check_argument_types()
        if epoch < 0:
            raise ValueError(f"epoch must be 0 or more: {epoch}")
        self.epoch = epoch
        # stats: Dict[int, Dict[str, Dict[str, float]]]
        # e.g. self.stats[epoch]['train']['loss']
        self.stats = {}

    def get_epoch(self) -> int:
        return self.epoch

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be 0 or more: {epoch}")
        self.epoch = epoch

    @contextmanager
    def observe(self, key: str, epoch: int = None) -> ContextManager[SubReporter]:
        sub_reporter = self.start_epoch(key, epoch)
        yield sub_reporter
        # Receive the stats from sub_reporter
        self.finish_epoch(sub_reporter)

    def start_epoch(self, key: str, epoch: int = None) -> SubReporter:
        if epoch is not None:
            if epoch < 0:
                raise ValueError(f"epoch must be 0 or more: {epoch}")
            self.epoch = epoch

        if self.epoch - 1 not in self.stats or key not in self.stats[self.epoch - 1]:
            # If the previous epoch doesn't exist for some reason,
            # maybe due to bug, this case also indicates 0-count.
            if self.epoch - 1 != 0:
                warnings.warn(
                    f"The stats of the previous epoch={self.epoch - 1}"
                    f"doesn't exist."
                )
            total_count = 0
        else:
            total_count = self.stats[self.epoch - 1][key]["total_count"]

        sub_reporter = SubReporter(key, self.epoch, total_count)
        # Clear the stats for the next epoch if it exists
        self.stats.pop(epoch, None)
        return sub_reporter

    def finish_epoch(self, sub_reporter: SubReporter) -> None:
        if self.epoch != sub_reporter.epoch:
            raise RuntimeError(
                f"Don't change epoch during observation: "
                f"{self.epoch} != {sub_reporter.epoch}"
            )

        # Calc mean of current stats and set it as previous epochs stats
        stats = {}
        for key2, values in sub_reporter.stats.items():
            v = aggregate(values)
            stats[key2] = v

        stats["time"] = datetime.timedelta(
            seconds=time.perf_counter() - sub_reporter.start_time
        )
        stats["total_count"] = sub_reporter.total_count
        if LooseVersion(torch.__version__) >= LooseVersion("1.4.0"):
            if torch.cuda.is_initialized():
                stats["gpu_max_cached_mem_GB"] = (
                        torch.cuda.max_memory_reserved() / 2 ** 30
                )
        else:
            if torch.cuda.is_available() and torch.cuda.max_memory_cached() > 0:
                stats["gpu_cached_mem_GB"] = torch.cuda.max_memory_cached() / 2 ** 30

        self.stats.setdefault(self.epoch, {})[sub_reporter.key] = stats
        sub_reporter.finished()

    def sort_epochs_and_values(
            self, key: str, key2: str, mode: str
    ) -> List[Tuple[int, float]]:
        """Return the epoch which resulted the best value.

        Example:
            >>> val = reporter.sort_epochs_and_values('eval', 'loss', 'min')
            >>> e_1best, v_1best = val[0]
            >>> e_2best, v_2best = val[1]
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must min or max: {mode}")
        if not self.has(key, key2):
            raise KeyError(f"{key}.{key2} is not found: {self.get_all_keys()}")

        # iterate from the last epoch
        values = [(e, self.stats[e][key][key2]) for e in self.stats]

        if mode == "min":
            values = sorted(values, key=lambda x: x[1])
        else:
            values = sorted(values, key=lambda x: -x[1])
        return values

    def sort_epochs(self, key: str, key2: str, mode: str) -> List[int]:
        return [e for e, v in self.sort_epochs_and_values(key, key2, mode)]

    def sort_values(self, key: str, key2: str, mode: str) -> List[float]:
        return [v for e, v in self.sort_epochs_and_values(key, key2, mode)]

    def get_best_epoch(self, key: str, key2: str, mode: str, nbest: int = 0) -> int:
        return self.sort_epochs(key, key2, mode)[nbest]

    def check_early_stopping(
            self,
            patience: int,
            key1: str,
            key2: str,
            mode: str,
            epoch: int = None,
            logger=None,
    ) -> bool:
        if logger is None:
            logger = logging
        if epoch is None:
            epoch = self.get_epoch()

        best_epoch = self.get_best_epoch(key1, key2, mode)
        if epoch - best_epoch > patience:
            logger.info(
                f"[Early stopping] {key1}.{key2} has not been "
                f"improved {epoch - best_epoch} epochs continuously. "
                f"The training was stopped at {epoch}epoch"
            )
            return True
        else:
            return False

    def has(self, key: str, key2: str, epoch: int = None) -> bool:
        if epoch is None:
            epoch = self.get_epoch()
        return (
                epoch in self.stats
                and key in self.stats[epoch]
                and key2 in self.stats[epoch][key]
        )

    def log_message(self, epoch: int = None) -> str:
        if epoch is None:
            epoch = self.get_epoch()

        message = ""
        for key, d in self.stats[epoch].items():
            _message = ""
            for key2, v in d.items():
                if v is not None:
                    if len(_message) != 0:
                        _message += ", "
                    if isinstance(v, float):
                        if abs(v) > 1.0e3:
                            _message += f"{key2}={v:.3e}"
                        elif abs(v) > 1.0e-3:
                            _message += f"{key2}={v:.3f}"
                        else:
                            _message += f"{key2}={v:.3e}"
                    elif isinstance(v, datetime.timedelta):
                        _v = humanfriendly.format_timespan(v)
                        _message += f"{key2}={_v}"
                    else:
                        _message += f"{key2}={v}"
            if len(_message) != 0:
                if len(message) == 0:
                    message += f"{epoch}epoch results: "
                else:
                    message += ", "
                message += f"[{key}] {_message}"
        return message

    def get_value(self, key: str, key2: str, epoch: int = None):
        if not self.has(key, key2):
            raise KeyError(f"{key}.{key2} is not found in stats: {self.get_all_keys()}")
        if epoch is None:
            epoch = self.get_epoch()
        return self.stats[epoch][key][key2]

    def get_keys(self, epoch: int = None) -> Tuple[str, ...]:
        """Returns keys1 e.g. train,eval."""
        if epoch is None:
            epoch = self.get_epoch()
        return tuple(self.stats[epoch])

    def get_keys2(self, key: str, epoch: int = None) -> Tuple[str, ...]:
        """Returns keys2 e.g. loss,acc."""
        if epoch is None:
            epoch = self.get_epoch()
        d = self.stats[epoch][key]
        keys2 = tuple(k for k in d if k not in ("time", "total_count"))
        return keys2

    def get_all_keys(self, epoch: int = None) -> Tuple[Tuple[str, str], ...]:
        if epoch is None:
            epoch = self.get_epoch()
        all_keys = []
        for key in self.stats[epoch]:
            for key2 in self.stats[epoch][key]:
                all_keys.append((key, key2))
        return tuple(all_keys)

    def tensorboard_add_scalar(
            self, summary_writer, epoch: int = None, key1: str = None
    ):
        if epoch is None:
            epoch = self.get_epoch()
            total_count = self.stats[epoch]["train"]["total_count"]
            if key1 == "train":
                summary_writer.add_scalar("iter_epoch", epoch, total_count)

        if key1 is not None:
            key1_iterator = tuple([key1])
        else:
            key1_iterator = self.get_keys(epoch)

        for key1 in key1_iterator:
            for key2 in self.get_keys2(key1):
                summary_writer.add_scalar(
                    f"{key2}", self.stats[epoch][key1][key2], total_count
                )

    def wandb_log(self, epoch: int = None):
        import wandb

        if epoch is None:
            epoch = self.get_epoch()

        d = {}
        for key1 in self.get_keys(epoch):
            for key2 in self.stats[epoch][key1]:
                if key2 in ("time", "total_count"):
                    continue
                key = f"{key1}_{key2}_epoch"
                d[wandb_get_prefix(key) + key] = self.stats[epoch][key1][key2]
        d["epoch"] = epoch
        wandb.log(d)

    def state_dict(self):
        return {"stats": self.stats, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        self.epoch = state_dict["epoch"]
        self.stats = state_dict["stats"]
