# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import dataclasses
import logging
import os
import socket
from typing import Optional

import torch
import torch.distributed


@dataclasses.dataclass
class DistributedOption:
    # Enable distributed Training
    distributed: bool = False
    # torch.distributed.Backend: "nccl", "mpi", "gloo", or "tcp"
    dist_backend: str = "nccl"
    # if init_method="env://",
    # env values of "MASTER_PORT", "MASTER_ADDR", "WORLD_SIZE", and "RANK" are referred.
    dist_init_method: str = "env://"
    dist_world_size: Optional[int] = None
    dist_rank: Optional[int] = None
    local_rank: Optional[int] = None
    ngpu: int = 0
    dist_master_addr: Optional[str] = None
    dist_master_port: Optional[int] = None
    dist_launcher: Optional[str] = None
    multiprocessing_distributed: bool = True

    def init_options(self):
        if self.distributed:
            if self.dist_init_method == "env://":
                if get_master_addr(self.dist_master_addr, self.dist_launcher) is None:
                    raise RuntimeError(
                        "--dist_master_addr or MASTER_ADDR must be set "
                        "if --dist_init_method == 'env://'"
                    )
                if get_master_port(self.dist_master_port) is None:
                    raise RuntimeError(
                        "--dist_master_port or MASTER_PORT must be set "
                        "if --dist_init_port == 'env://'"
                    )

    def init_torch_distributed(self, args):
        if self.distributed:
            # See:
            # https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html
            os.environ.setdefault("NCCL_DEBUG", "INFO")

            # See:
            # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
            os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

            torch.distributed.init_process_group(backend=self.dist_backend,
                                                 init_method=self.dist_init_method,
                                                 world_size=args.dist_world_size,
                                                 rank=args.dist_rank)
            self.dist_rank = torch.distributed.get_rank()
            self.dist_world_size = torch.distributed.get_world_size()
            self.local_rank = args.local_rank

    def init_options_pai(self):
        if self.distributed:
            if self.dist_init_method == "env://":
                if get_master_addr(self.dist_master_addr, self.dist_launcher) is None:
                    raise RuntimeError(
                        "--dist_master_addr or MASTER_ADDR must be set "
                        "if --dist_init_method == 'env://'"
                    )
                if get_master_port(self.dist_master_port) is None:
                    raise RuntimeError(
                        "--dist_master_port or MASTER_PORT must be set "
                        "if --dist_init_port == 'env://'"
                    )

            self.dist_rank = get_rank(self.dist_rank, self.dist_launcher)
            self.dist_world_size = get_world_size(
                self.dist_world_size, self.dist_launcher
            )
            self.local_rank = get_local_rank(self.local_rank, self.dist_launcher)

            if (
                    self.dist_rank is not None
                    and self.dist_world_size is not None
                    and self.dist_rank >= self.dist_world_size
            ):
                raise RuntimeError(
                    f"RANK >= WORLD_SIZE: {self.dist_rank} >= {self.dist_world_size}"
                )

            if self.dist_init_method == "env://":
                self.dist_master_addr = get_master_addr(
                    self.dist_master_addr, self.dist_launcher
                )
                self.dist_master_port = get_master_port(self.dist_master_port)
                if (
                        self.dist_master_addr is not None
                        and self.dist_master_port is not None
                ):
                    self.dist_init_method = (
                        f"tcp://{self.dist_master_addr}:{self.dist_master_port}"
                    )

    def init_torch_distributed_pai(self, args):
        if self.distributed:
            # See:
            # https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html
            os.environ.setdefault("NCCL_DEBUG", "INFO")

            # See:
            # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
            os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

            torch.distributed.init_process_group(backend=self.dist_backend, init_method='env://')
            self.dist_rank = torch.distributed.get_rank()
            self.dist_world_size = torch.distributed.get_world_size()
            self.local_rank = args.local_rank


def resolve_distributed_mode(args):
    # Note that args.distributed is set by only this function.
    # and ArgumentParser doesn't have such option

    if args.multiprocessing_distributed:
        num_nodes = get_num_nodes(args.dist_world_size, args.dist_launcher)
        # a. multi-node
        if num_nodes > 1:
            args.distributed = True
        # b. single-node and multi-gpu with multiprocessing_distributed mode
        elif args.ngpu > 1:
            args.distributed = True
        # c. single-node and single-gpu
        else:
            args.distributed = False

        if args.ngpu <= 1:
            # Disable multiprocessing_distributed mode if 1process per node or cpu mode
            args.multiprocessing_distributed = False
        if args.ngpu == 1:
            # If the number of GPUs equals to 1 with multiprocessing_distributed mode,
            # LOCAL_RANK is always 0
            args.local_rank = 0

        if num_nodes > 1 and get_node_rank(args.dist_rank, args.dist_launcher) is None:
            raise RuntimeError(
                "--dist_rank or RANK must be set "
                "if --multiprocessing_distributed == true"
            )

        # Note that RANK, LOCAL_RANK, and WORLD_SIZE is automatically set,
        # so we don't need to check here
    else:
        # d. multiprocess and multi-gpu with external launcher
        #    e.g. torch.distributed.launch
        if get_world_size(args.dist_world_size, args.dist_launcher) > 1:
            args.distributed = True
        # e. single-process
        else:
            args.distributed = False

        if args.distributed and args.ngpu > 0:
            if get_local_rank(args.local_rank, args.dist_launcher) is None:
                raise RuntimeError(
                    "--local_rank or LOCAL_RANK must be set "
                    "if --multiprocessing_distributed == false"
                )
        if args.distributed:
            if get_node_rank(args.dist_rank, args.dist_launcher) is None:
                raise RuntimeError(
                    "--dist_rank or RANK must be set "
                    "if --multiprocessing_distributed == false"
                )
    if args.distributed and args.dist_launcher == "slurm" and not is_in_slurm_step():
        raise RuntimeError("Launch by 'srun' command if --dist_launcher='slurm'")


def is_in_slurm_job() -> bool:
    return "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ


def is_in_slurm_step() -> bool:
    return (
            is_in_slurm_job()
            and "SLURM_STEP_NUM_NODES" in os.environ
            and "SLURM_STEP_NODELIST" in os.environ
    )


def _int_or_none(x: Optional[str]) -> Optional[int]:
    if x is None:
        return x
    return int(x)


def free_port():
    """Find free port using bind().

    There are some interval between finding this port and using it
    and the other process might catch the port by that time.
    Thus it is not guaranteed that the port is really empty.

    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def get_rank(prior=None, launcher: str = None) -> Optional[int]:
    if prior is None:
        if launcher == "slurm":
            if not is_in_slurm_step():
                raise RuntimeError("This process seems not to be launched by 'srun'")
            prior = os.environ["SLURM_PROCID"]
        elif launcher == "mpi":
            raise RuntimeError(
                "launcher=mpi is used for 'multiprocessing-distributed' mode"
            )
        elif launcher is not None:
            raise RuntimeError(f"launcher='{launcher}' is not supported")

    if prior is not None:
        return int(prior)
    else:
        # prior is None and RANK is None -> RANK = None
        return _int_or_none(os.environ.get("RANK"))


def get_world_size(prior=None, launcher: str = None) -> int:
    if prior is None:
        if launcher == "slurm":
            if not is_in_slurm_step():
                raise RuntimeError("This process seems not to be launched by 'srun'")
            prior = int(os.environ["SLURM_NTASKS"])
        elif launcher == "mpi":
            raise RuntimeError(
                "launcher=mpi is used for 'multiprocessing-distributed' mode"
            )
        elif launcher is not None:
            raise RuntimeError(f"launcher='{launcher}' is not supported")

    if prior is not None:
        return int(prior)
    else:
        # prior is None and WORLD_SIZE is None -> WORLD_SIZE = 1
        return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank(prior=None, launcher: str = None) -> Optional[int]:
    # LOCAL_RANK is same as GPU device id

    if prior is None:
        if launcher == "slurm":
            if not is_in_slurm_step():
                raise RuntimeError("This process seems not to be launched by 'srun'")

            prior = int(os.environ["SLURM_LOCALID"])
        elif launcher == "mpi":
            raise RuntimeError(
                "launcher=mpi is used for 'multiprocessing-distributed' mode"
            )
        elif launcher is not None:
            raise RuntimeError(f"launcher='{launcher}' is not supported")

    if prior is not None:
        return int(prior)

    elif "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        # There are two possibility:
        # - "CUDA_VISIBLE_DEVICES" is set to multiple GPU ids. e.g. "0.1,2"
        #   => This intends to specify multiple devices to to be used exactly
        #      and local_rank information is possibly insufficient.
        # - "CUDA_VISIBLE_DEVICES" is set to an id. e.g. "1"
        #   => This could be used for LOCAL_RANK
        cvd = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(cvd) == 1 and "LOCAL_RANK" not in os.environ:
            # If CUDA_VISIBLE_DEVICES is set and LOCAL_RANK is not set,
            # then use it as LOCAL_RANK.

            # Unset CUDA_VISIBLE_DEVICES
            # because the other device must be visible to communicate
            return int(os.environ.pop("CUDA_VISIBLE_DEVICES"))
        else:
            return None
    else:
        return None


def get_master_addr(prior=None, launcher: str = None) -> Optional[str]:
    if prior is None:
        if launcher == "slurm":
            if not is_in_slurm_step():
                raise RuntimeError("This process seems not to be launched by 'srun'")

            # e.g nodelist = foo[1-10],bar[3-8] or foo4,bar[2-10]
            nodelist = os.environ["SLURM_STEP_NODELIST"]
            prior = nodelist.split(",")[0].split("-")[0].replace("[", "")

    if prior is not None:
        return str(prior)
    else:
        return os.environ.get("MASTER_ADDR")


def get_master_port(prior=None) -> Optional[int]:
    if prior is not None:
        return prior
    else:
        return _int_or_none(os.environ.get("MASTER_PORT"))


def get_node_rank(prior=None, launcher: str = None) -> Optional[int]:
    """Get Node Rank.

    Use for "multiprocessing distributed" mode.
    The initial RANK equals to the Node id in this case and
    the real Rank is set as (nGPU * NodeID) + LOCAL_RANK in torch.distributed.

    """
    if prior is not None:
        return prior
    elif launcher == "slurm":
        if not is_in_slurm_step():
            raise RuntimeError("This process seems not to be launched by 'srun'")

        # Assume ntasks_per_node == 1
        if os.environ["SLURM_STEP_NUM_NODES"] != os.environ["SLURM_NTASKS"]:
            raise RuntimeError(
                "Run with --ntasks_per_node=1 if mutliprocessing_distributed=true"
            )
        return int(os.environ["SLURM_NODEID"])
    elif launcher == "mpi":
        # Use mpi4py only for initialization and not using for communication
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        # Assume ntasks_per_node == 1 (We can't check whether it is or not)
        return comm.Get_rank()
    elif launcher is not None:
        raise RuntimeError(f"launcher='{launcher}' is not supported")
    else:
        return _int_or_none(os.environ.get("RANK"))


def get_num_nodes(prior=None, launcher: str = None) -> Optional[int]:
    """Get the number of nodes.

    Use for "multiprocessing distributed" mode.
    RANK equals to the Node id in this case and
    the real Rank is set as (nGPU * NodeID) + LOCAL_RANK in torch.distributed.

    """
    if prior is not None:
        return prior
    elif launcher == "slurm":
        if not is_in_slurm_step():
            raise RuntimeError("This process seems not to be launched by 'srun'")

        # Assume ntasks_per_node == 1
        if os.environ["SLURM_STEP_NUM_NODES"] != os.environ["SLURM_NTASKS"]:
            raise RuntimeError(
                "Run with --ntasks_per_node=1 if mutliprocessing_distributed=true"
            )
        return int(os.environ["SLURM_STEP_NUM_NODES"])
    elif launcher == "mpi":
        # Use mpi4py only for initialization and not using for communication
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        # Assume ntasks_per_node == 1 (We can't check whether it is or not)
        return comm.Get_size()
    elif launcher is not None:
        raise RuntimeError(f"launcher='{launcher}' is not supported")
    else:
        # prior is None -> NUM_NODES = 1
        return int(os.environ.get("WORLD_SIZE", 1))
