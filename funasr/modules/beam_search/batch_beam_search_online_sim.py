"""Parallel beam search module for online simulation."""

import logging
from pathlib import Path
from typing import List

import yaml

import torch

from funasr.modules.beam_search.batch_beam_search import BatchBeamSearch
from funasr.modules.beam_search.beam_search import Hypothesis
from funasr.models.e2e_asr_common import end_detect


class BatchBeamSearchOnlineSim(BatchBeamSearch):
    """Online beam search implementation.

    This simulates streaming decoding.
    It requires encoded features of entire utterance and
    extracts block by block from it as it shoud be done
    in streaming processing.
    This is based on Tsunoo et al, "STREAMING TRANSFORMER ASR
    WITH BLOCKWISE SYNCHRONOUS BEAM SEARCH"
    (https://arxiv.org/abs/2006.14941).
    """

    def set_streaming_config(self, asr_config: str):
        """Set config file for streaming decoding.

        Args:
            asr_config (str): The config file for asr training

        """
        train_config_file = Path(asr_config)
        self.block_size = None
        self.hop_size = None
        self.look_ahead = None
        config = None
        with train_config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
            if "encoder_conf" in args.keys():
                if "block_size" in args["encoder_conf"].keys():
                    self.block_size = args["encoder_conf"]["block_size"]
                if "hop_size" in args["encoder_conf"].keys():
                    self.hop_size = args["encoder_conf"]["hop_size"]
                if "look_ahead" in args["encoder_conf"].keys():
                    self.look_ahead = args["encoder_conf"]["look_ahead"]
            elif "config" in args.keys():
                config = args["config"]
                if config is None:
                    logging.info(
                        "Cannot find config file for streaming decoding: "
                        + "apply batch beam search instead."
                    )
                    return
        if (
            self.block_size is None or self.hop_size is None or self.look_ahead is None
        ) and config is not None:
            config_file = Path(config)
            with config_file.open("r", encoding="utf-8") as f:
                args = yaml.safe_load(f)
            if "encoder_conf" in args.keys():
                enc_args = args["encoder_conf"]
            if enc_args and "block_size" in enc_args:
                self.block_size = enc_args["block_size"]
            if enc_args and "hop_size" in enc_args:
                self.hop_size = enc_args["hop_size"]
            if enc_args and "look_ahead" in enc_args:
                self.look_ahead = enc_args["look_ahead"]

    def set_block_size(self, block_size: int):
        """Set block size for streaming decoding.

        Args:
            block_size (int): The block size of encoder
        """
        self.block_size = block_size

    def set_hop_size(self, hop_size: int):
        """Set hop size for streaming decoding.

        Args:
            hop_size (int): The hop size of encoder
        """
        self.hop_size = hop_size

    def set_look_ahead(self, look_ahead: int):
        """Set look ahead size for streaming decoding.

        Args:
            look_ahead (int): The look ahead size of encoder
        """
        self.look_ahead = look_ahead

    def forward(
        self, x: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        self.conservative = True  # always true

        if self.block_size and self.hop_size and self.look_ahead:
            cur_end_frame = int(self.block_size - self.look_ahead)
        else:
            cur_end_frame = x.shape[0]
        process_idx = 0
        if cur_end_frame < x.shape[0]:
            h = x.narrow(0, 0, cur_end_frame)
        else:
            h = x

        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))
        minlen = int(minlenratio * x.size(0))
        logging.info("decoder input length: " + str(x.shape[0]))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(h)
        prev_hyps = []
        ended_hyps = []
        prev_repeat = False

        continue_decode = True

        while continue_decode:
            move_to_next_block = False
            if cur_end_frame < x.shape[0]:
                h = x.narrow(0, 0, cur_end_frame)
            else:
                h = x

            # extend states for ctc
            self.extend(h, running_hyps)

            while process_idx < maxlen:
                logging.debug("position " + str(process_idx))
                best = self.search(running_hyps, h)

                if process_idx == maxlen - 1:
                    # end decoding
                    running_hyps = self.post_process(
                        process_idx, maxlen, maxlenratio, best, ended_hyps
                    )
                n_batch = best.yseq.shape[0]
                local_ended_hyps = []
                is_local_eos = (
                    best.yseq[torch.arange(n_batch), best.length - 1] == self.eos
                )
                for i in range(is_local_eos.shape[0]):
                    if is_local_eos[i]:
                        hyp = self._select(best, i)
                        local_ended_hyps.append(hyp)
                    # NOTE(tsunoo): check repetitions here
                    # This is a implicit implementation of
                    # Eq (11) in https://arxiv.org/abs/2006.14941
                    # A flag prev_repeat is used instead of using set
                    elif (
                        not prev_repeat
                        and best.yseq[i, -1] in best.yseq[i, :-1]
                        and cur_end_frame < x.shape[0]
                    ):
                        move_to_next_block = True
                        prev_repeat = True
                if maxlenratio == 0.0 and end_detect(
                    [lh.asdict() for lh in local_ended_hyps], process_idx
                ):
                    logging.info(f"end detected at {process_idx}")
                    continue_decode = False
                    break
                if len(local_ended_hyps) > 0 and cur_end_frame < x.shape[0]:
                    move_to_next_block = True

                if move_to_next_block:
                    if (
                        self.hop_size
                        and cur_end_frame + int(self.hop_size) + int(self.look_ahead)
                        < x.shape[0]
                    ):
                        cur_end_frame += int(self.hop_size)
                    else:
                        cur_end_frame = x.shape[0]
                    logging.debug("Going to next block: %d", cur_end_frame)
                    if process_idx > 1 and len(prev_hyps) > 0 and self.conservative:
                        running_hyps = prev_hyps
                        process_idx -= 1
                        prev_hyps = []
                    break

                prev_repeat = False
                prev_hyps = running_hyps
                running_hyps = self.post_process(
                    process_idx, maxlen, maxlenratio, best, ended_hyps
                )

                if cur_end_frame >= x.shape[0]:
                    for hyp in local_ended_hyps:
                        ended_hyps.append(hyp)

                if len(running_hyps) == 0:
                    logging.info("no hypothesis. Finish decoding.")
                    continue_decode = False
                    break
                else:
                    logging.debug(f"remained hypotheses: {len(running_hyps)}")
                # increment number
                process_idx += 1

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logging.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logging.info(f"total log probability: {best.score:.2f}")
        logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logging.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        return nbest_hyps

    def extend(self, x: torch.Tensor, hyps: Hypothesis) -> List[Hypothesis]:
        """Extend probabilities and states with more encoded chunks.

        Args:
            x (torch.Tensor): The extended encoder output feature
            hyps (Hypothesis): Current list of hypothesis

        Returns:
            Hypothesis: The extended hypothesis

        """
        for k, d in self.scorers.items():
            if hasattr(d, "extend_prob"):
                d.extend_prob(x)
            if hasattr(d, "extend_state"):
                hyps.states[k] = d.extend_state(hyps.states[k])
