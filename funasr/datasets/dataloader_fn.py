import time
import torch
from funasr.datasets.dataset_jsonl import AudioDataset
from funasr.datasets.data_sampler import BatchSampler
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.tokenizer.build_tokenizer import build_tokenizer
from funasr.tokenizer.token_id_converter import TokenIDConverter
collate_fn = None
# collate_fn = collate_fn,

jsonl = "/Users/zhifu/funasr_github/test_local/aishell2_dev_ios/asr_task_debug_len.jsonl"

frontend = WavFrontend()
token_type = 'char'
bpemodel = None
delimiter = None
space_symbol = "<space>"
non_linguistic_symbols = None
g2p_type = None

tokenizer = build_tokenizer(
    token_type=token_type,
    bpemodel=bpemodel,
    delimiter=delimiter,
    space_symbol=space_symbol,
    non_linguistic_symbols=non_linguistic_symbols,
    g2p_type=g2p_type,
)
token_list = "/Users/zhifu/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.txt"
unk_symbol = "<unk>"

token_id_converter = TokenIDConverter(
    token_list=token_list,
    unk_symbol=unk_symbol,
)

dataset = AudioDataset(jsonl, frontend=frontend, tokenizer=tokenizer, token_id_converter=token_id_converter)
batch_sampler = BatchSampler(dataset)


def collator(samples: list = None):
    return samples

if __name__ == "__main__":
    
    dataloader_tr = torch.utils.data.DataLoader(dataset,
                                                collate_fn=dataset.collator,
                                                batch_sampler=batch_sampler,
                                                shuffle=False,
                                                num_workers=8,
                                                pin_memory=True)
    
    print(len(dataset))
    for i in range(3):
        print(i)
        beg = time.time()
        for j, data in enumerate(dataloader_tr):
            end = time.time()
            time_cost = end - beg
            beg = end
            print(j, time_cost)
    # data_iter = iter(dataloader_tr)
    # data = next(data_iter)
    pass

    
