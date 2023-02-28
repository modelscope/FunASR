from funasr.models.e2e_asr_paraformer import Paraformer, BiCifParaformer
from funasr.export.models.e2e_asr_paraformer import Paraformer as Paraformer_export
from funasr.export.models.e2e_asr_paraformer import BiCifParaformer as BiCifParaformer_export
from funasr.models.e2e_uni_asr import UniASR


def get_model(model, export_config=None):
    if isinstance(model, BiCifParaformer):
        return BiCifParaformer_export(model, **export_config)
    elif isinstance(model, Paraformer):
        return Paraformer_export(model, **export_config)
    else:
        raise "Funasr does not support the given model type currently."