from funasr.models.e2e_asr_paraformer import Paraformer, BiCifParaformer
from funasr.export.models.e2e_asr_paraformer import Paraformer as Paraformer_export
from funasr.export.models.e2e_asr_paraformer import BiCifParaformer as BiCifParaformer_export
from funasr.models.e2e_vad import E2EVadModel
from funasr.export.models.e2e_vad import E2EVadModel as E2EVadModel_export
from funasr.punctuation.target_delay_transformer import TargetDelayTransformer
from funasr.export.models.target_delay_transformer import TargetDelayTransformer as TargetDelayTransformer_export
from funasr.punctuation.espnet_model import ESPnetPunctuationModel
from funasr.punctuation.vad_realtime_transformer import VadRealtimeTransformer
from funasr.export.models.vad_realtime_transformer import VadRealtimeTransformer as VadRealtimeTransformer_export

def get_model(model, export_config=None):
    if isinstance(model, BiCifParaformer):
        return BiCifParaformer_export(model, **export_config)
    elif isinstance(model, Paraformer):
        return Paraformer_export(model, **export_config)
    elif isinstance(model, E2EVadModel):
        return E2EVadModel_export(model, **export_config)
    elif isinstance(model, ESPnetPunctuationModel):
        if isinstance(model.punc_model, TargetDelayTransformer):
            return TargetDelayTransformer_export(model.punc_model, **export_config)
        elif isinstance(model.punc_model, VadRealtimeTransformer):
            return VadRealtimeTransformer_export(model.punc_model, **export_config)
    else:
        raise "Funasr does not support the given model type currently."
