import logging
from typing import Union, Dict, List, Tuple, Optional

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.ctc.ctc import CTC
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.metrics.compute_acc import th_accuracy, compute_accuracy
# from funasr.models.e2e_asr_common import ErrorCalculator
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter
from funasr.register import tables


@tables.register("model_classes", "LLMASR")
class LLMASR(nn.Module):
    """ """
    
    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        decoder: str = None,
        decoder_conf: dict = None,
        ctc: str = None,
        ctc_conf: dict = None,
        ctc_weight: float = 0.5,
        llm: str = None,
        llm_conf: dict = None,
        adaptor: str = None,
        adaptor_conf: dict = None,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # extract_feats_in_collect_stats: bool = True,
        share_embedding: bool = False,
        # preencoder: Optional[AbsPreEncoder] = None,
        # postencoder: Optional[AbsPostEncoder] = None,
        **kwargs,
    ):
        
        super().__init__()
        
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        
        # audio encoder
        hub = encoder_conf.get("hub", None)
        if hub == "funasr":
            from funasr import AutoModel
            from funasr.models.scama.utils import sequence_mask
            init_param_path = encoder_conf.get("hub", "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
            model = AutoModel(model=init_param_path, model_revision="v2.0.4")
            frontend = model.kwargs.get("frontend")
            model.model.decoder = None
            
            self.model = model.model
            self.frontend = frontend
            self.mask_fn = sequence_mask
            
        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(encoder)
            encoder = encoder_class(input_size=input_size, **encoder_conf)
            encoder_output_size = encoder.output_size()

        # llm
        hub = llm_conf.get("hub", "hf")
        self.llm = None
        if hub == "hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

            init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
            model = AutoModelForCausalLM.from_pretrained(
                init_param_path,
                load_in_8bit=None,
                device_map=None,
                use_cache=None,
            )
            freeze_llm = llm_conf.get("freeze_llm", True)
            if freeze_llm:
                for name, param in model.named_parameters():
                    param.requires_grad = False
                model.eval()
            self.llm = model
        
        # adaptor
        adaptor_class = tables.adaptor_classes.get(adaptor)
        adaptor = adaptor_class(**adaptor_conf)
        
        self.adaptor = adaptor
        
        
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder


        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        #
        # if report_cer or report_wer:
        #     self.error_calculator = ErrorCalculator(
        #         token_list, sym_space, sym_blank, report_cer, report_wer
        #     )
        #
        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
    
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask:torch.Tensor,
        labels_ids:torch.Tensor,
        label_mask: torch.Tensor,
        audio_mask:torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # import pdb;
        # pdb.set_trace()
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]
        
        batch_size = speech.shape[0]
        
        # audio encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, audio_mask)
        
        # adaptor
        encoder_out = self.adaptor(encoder_out)

        if input_ids is not None:
            input_ids[input_ids == -1] = 0
            if hasattr(self.llm.model, "embed_tokens"):
                inputs_embeds = self.llm.model.embed_tokens(input_ids)
            elif hasattr(self.llm.model.model, "embed_tokens"):
                inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

            if audio_mask is not None:
                batch_size, token_num, dims = inputs_embeds.shape
                _, l, _ = encoder_out.shape
                encoder_outs_pad = F.pad(encoder_out, (0, 0, token_num-l-1, 1, 0, 0), value=0.0)
                inputs_embeds = encoder_outs_pad * audio_mask[:, :, None] + inputs_embeds * (~audio_mask[:, :, None])
                inputs_embeds = F.pad(inputs_embeds[:, 1:, :], (0, 0, 0, 1, 0, 0), value=0.0)

        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        loss = model_outputs.loss

        acc_att = -1
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(model_outputs.logits, -1)
                acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)

        stats = {}
        # Collect Attn branch stats
        stats["acc"] = acc_att.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
    
    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        audio_mask = kwargs.get("audio_mask")
        audio_token_lengths = audio_mask.sum(-1)

        batch = {"speech": speech, "speech_lengths": speech_lengths}
        enc, enc_lens = self.model.encode(**batch)
        enc_mask = self.mask_fn(enc_lens, enc.size(1), device=enc.device)[:, None, :]
        pre_acoustic_embeds, pre_token_length, _, _ = self.model.predictor(enc,
                                                                           mask=enc_mask,
                                                                           target_label_length=audio_token_lengths,
                                                                           )

        return pre_acoustic_embeds, pre_token_length
    
    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1
        
        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )
        
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        
        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
        
        return loss_att, acc_att, cer_att, wer_att
    

    def inference(self,
                  data_in,
                  data_lengths=None,
                  key: list = None,
                  tokenizer=None,
                  frontend=None,
                  **kwargs,
                  ):
        
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")
        
        # init beamsearch
        if self.beam_search is None:
            logging.info("enable beam_search")
            self.init_beam_search(**kwargs)
            self.nbest = kwargs.get("nbest", 1)
        
        meta_data = {}
        if isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank":  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(data_in, fs=frontend.fs, audio_fs=kwargs.get("fs", 16000),
                                                            data_type=kwargs.get("data_type", "sound"),
                                                            tokenizer=tokenizer)
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(audio_sample_list, data_type=kwargs.get("data_type", "sound"),
                                                   frontend=frontend)
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
        
        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        
        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=encoder_out[0], maxlenratio=kwargs.get("maxlenratio", 0.0), minlenratio=kwargs.get("minlenratio", 0.0)
        )
        
        nbest_hyps = nbest_hyps[: self.nbest]
        
        results = []
        b, n, d = encoder_out.size()
        for i in range(b):
            
            for nbest_idx, hyp in enumerate(nbest_hyps):
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx + 1}best_recog"]
                
                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()
                
                # remove blank symbol id, which is assumed to be 0
                token_int = list(filter(lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int))
                
                # Change integer-ids to tokens
                token = tokenizer.ids2tokens(token_int)
                text = tokenizer.tokens2text(token)
                
                text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
                result_i = {"key": key[i], "token": token, "text": text_postprocessed}
                results.append(result_i)
                
                if ibest_writer is not None:
                    ibest_writer["token"][key[i]] = " ".join(token)
                    ibest_writer["text"][key[i]] = text_postprocessed
        
        return results, meta_data
