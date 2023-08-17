from funasr.models.e2e_asr_paraformer import Paraformer, BiCifParaformer, ParaformerOnline
from funasr.export.models.e2e_asr_paraformer import Paraformer as Paraformer_export
from funasr.export.models.e2e_asr_paraformer import BiCifParaformer as BiCifParaformer_export
from funasr.export.models.e2e_asr_conformer import Conformer as Conformer_export

from funasr.models.e2e_vad import E2EVadModel
from funasr.export.models.e2e_vad import E2EVadModel as E2EVadModel_export
from funasr.models.target_delay_transformer import TargetDelayTransformer
from funasr.export.models.CT_Transformer import CT_Transformer as CT_Transformer_export
from funasr.train.abs_model import PunctuationModel
from funasr.models.vad_realtime_transformer import VadRealtimeTransformer
from funasr.export.models.CT_Transformer import CT_Transformer_VadRealtime as CT_Transformer_VadRealtime_export
from funasr.export.models.e2e_asr_paraformer import ParaformerOnline_encoder_predictor as ParaformerOnline_encoder_predictor_export
from funasr.export.models.e2e_asr_paraformer import ParaformerOnline_decoder as ParaformerOnline_decoder_export
from funasr.export.models.e2e_asr_contextual_paraformer import ContextualParaformer_backbone as ContextualParaformer_backbone_export
from funasr.export.models.e2e_asr_contextual_paraformer import ContextualParaformer_embedder as ContextualParaformer_embedder_export
from funasr.models.e2e_asr_contextual_paraformer import NeatContextualParaformer


def get_model(model, export_config=None):
    if isinstance(model, NeatContextualParaformer):
        backbone = ContextualParaformer_backbone_export(model, **export_config)
        embedder = ContextualParaformer_embedder_export(model, **export_config)
        return [embedder, backbone]
    elif isinstance(model, BiCifParaformer):
        return BiCifParaformer_export(model, **export_config)
    elif isinstance(model, ParaformerOnline):
        return (ParaformerOnline_encoder_predictor_export(model, model_name="model"),
                ParaformerOnline_decoder_export(model, model_name="decoder"))
    elif isinstance(model, Paraformer):
        return Paraformer_export(model, **export_config)
    elif isinstance(model, Conformer_export):
        return Conformer_export(model, **export_config)
    elif isinstance(model, E2EVadModel):
        return E2EVadModel_export(model, **export_config)
    elif isinstance(model, PunctuationModel):
        if isinstance(model.punc_model, TargetDelayTransformer):
            return CT_Transformer_export(model.punc_model, **export_config)
        elif isinstance(model.punc_model, VadRealtimeTransformer):
            return CT_Transformer_VadRealtime_export(model.punc_model, **export_config)
    else:
        raise "Funasr does not support the given model type currently."
