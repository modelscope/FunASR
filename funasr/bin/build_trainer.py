import os

import yaml


def update_dct(fin_configs, root):
    if root == {}:
        return {}
    for root_key, root_value in root.items():
        if not isinstance(root[root_key], dict):
            fin_configs[root_key] = root[root_key]
        else:
            if root_key in fin_configs.keys():
                result = update_dct(fin_configs[root_key], root[root_key])
                fin_configs[root_key] = result
            else:
                fin_configs[root_key] = root[root_key]
    return fin_configs


def parse_args(mode):
    if mode == "asr":
        from funasr.tasks.asr import ASRTask as ASRTask
    elif mode == "paraformer":
        from funasr.tasks.asr import ASRTaskParaformer as ASRTask
    elif mode == "paraformer_vad_punc":
        from funasr.tasks.asr import ASRTaskParaformer as ASRTask
    elif mode == "uniasr":
        from funasr.tasks.asr import ASRTaskUniASR as ASRTask
    elif mode == "mfcca":
        from funasr.tasks.asr import ASRTaskMFCCA as ASRTask
    elif mode == "tp":
        from funasr.tasks.asr import ASRTaskAligner as ASRTask
    else:
        raise ValueError("Unknown mode: {}".format(mode))
    parser = ASRTask.get_parser()
    args = parser.parse_args()
    return args, ASRTask


def build_trainer(modelscope_dict,
                  data_dir,
                  output_dir,
                  train_set="train",
                  dev_set="validation",
                  distributed=False,
                  dataset_type="small",
                  batch_bins=None,
                  max_epoch=None,
                  optim=None,
                  lr=None,
                  scheduler=None,
                  scheduler_conf=None,
                  specaug=None,
                  specaug_conf=None,
                  param_dict=None,
                  **kwargs):
    mode = modelscope_dict['mode']
    args, ASRTask = parse_args(mode=mode)
    # ddp related
    if args.local_rank is not None:
        distributed = True
    else:
        distributed = False
    args.local_rank = args.local_rank if args.local_rank is not None else 0
    local_rank = args.local_rank
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list[args.local_rank])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.local_rank)

    config = modelscope_dict['am_model_config']
    finetune_config = modelscope_dict['finetune_config']
    init_param = modelscope_dict['init_model']
    cmvn_file = modelscope_dict['cmvn_file']
    seg_dict_file = modelscope_dict['seg_dict']

    # overwrite parameters
    with open(config) as f:
        configs = yaml.safe_load(f)
    with open(finetune_config) as f:
        finetune_configs = yaml.safe_load(f)
        # set data_types
        if dataset_type == "large":
            if 'data_types' not in finetune_configs['dataset_conf']:
                finetune_configs["dataset_conf"]["data_types"] = "sound,text"
    finetune_configs = update_dct(configs, finetune_configs)
    for key, value in finetune_configs.items():
        if hasattr(args, key):
            setattr(args, key, value)

    # prepare data
    args.dataset_type = dataset_type
    if args.dataset_type == "small":
        args.train_data_path_and_name_and_type = [["{}/{}/wav.scp".format(data_dir, train_set), "speech", "sound"],
                                                  ["{}/{}/text".format(data_dir, train_set), "text", "text"]]
        args.valid_data_path_and_name_and_type = [["{}/{}/wav.scp".format(data_dir, dev_set), "speech", "sound"],
                                                  ["{}/{}/text".format(data_dir, dev_set), "text", "text"]]
    elif args.dataset_type == "large":
        args.train_data_file = None
        args.valid_data_file = None
    else:
        raise ValueError(f"Not supported dataset_type={args.dataset_type}")
    args.init_param = [init_param]
    args.cmvn_file = cmvn_file
    if os.path.exists(seg_dict_file):
        args.seg_dict_file = seg_dict_file
    else:
        args.seg_dict_file = None
    args.data_dir = data_dir
    args.train_set = train_set
    args.dev_set = dev_set
    args.output_dir = output_dir
    args.gpu_id = args.local_rank
    args.config = finetune_config
    if optim is not None:
        args.optim = optim
    if lr is not None:
        args.optim_conf["lr"] = lr
    if scheduler is not None:
        args.scheduler = scheduler
    if scheduler_conf is not None:
        args.scheduler_conf = scheduler_conf
    if specaug is not None:
        args.specaug = specaug
    if specaug_conf is not None:
        args.specaug_conf = specaug_conf
    if max_epoch is not None:
        args.max_epoch = max_epoch
    if batch_bins is not None:
        if args.dataset_type == "small":
            args.batch_bins = batch_bins
        elif args.dataset_type == "large":
            args.dataset_conf["batch_conf"]["batch_size"] = batch_bins
        else:
            raise ValueError(f"Not supported dataset_type={args.dataset_type}")
    if args.normalize in ["null", "none", "None"]:
        args.normalize = None
    if args.patience in ["null", "none", "None"]:
        args.patience = None
    args.local_rank = local_rank
    args.distributed = distributed
    ASRTask.finetune_args = args

    return ASRTask
