from funasr.models.e2e_asr_paraformer import Paraformer
from funasr.export.models.e2e_asr_paraformer import Paraformer as Paraformer_export
from funasr.models.e2e_uni_asr import UniASR

def get_model(model, export_config=None):

    if isinstance(model, Paraformer):
        return Paraformer_export(model, **export_config)
    else:
        raise "The model is not exist!"