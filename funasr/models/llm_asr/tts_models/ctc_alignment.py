import torch
import logging
import torch.nn.functional as F


class CTC(torch.nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: builtin or warpctc
        reduce: reduce the CTC loss into a scalar
    """

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: bool = True,
        length_normalize: str = None,
    ):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        self.ignore_nan_grad = ignore_nan_grad
        self.length_normalize = length_normalize

        if self.ctc_type == "builtin":
            self.ctc_loss = torch.nn.CTCLoss(reduction="none")
        else:
            raise ValueError(
                f'ctc_type must be "builtin": {self.ctc_type}'
            )

        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        if self.ctc_type == "builtin":
            th_pred = th_pred.log_softmax(2)
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

            if loss.requires_grad and self.ignore_nan_grad:
                # ctc_grad: (L, B, O)
                ctc_grad = loss.grad_fn(torch.ones_like(loss))
                ctc_grad = ctc_grad.sum([0, 2])
                indices = torch.isfinite(ctc_grad)
                size = indices.long().sum()
                if size == 0:
                    # Return as is
                    logging.warning(
                        "All samples in this mini-batch got nan grad."
                        " Returning nan value instead of CTC loss"
                    )
                elif size != th_pred.size(1):
                    logging.warning(
                        f"{th_pred.size(1) - size}/{th_pred.size(1)}"
                        " samples got nan grad."
                        " These were ignored for CTC loss."
                    )

                    # Create mask for target
                    target_mask = torch.full(
                        [th_target.size(0)],
                        1,
                        dtype=torch.bool,
                        device=th_target.device,
                    )
                    s = 0
                    for ind, le in enumerate(th_olen):
                        if not indices[ind]:
                            target_mask[s : s + le] = 0
                        s += le

                    # Calc loss again using maksed data
                    loss = self.ctc_loss(
                        th_pred[:, indices, :],
                        th_target[target_mask],
                        th_ilen[indices],
                        th_olen[indices],
                    )
                    th_ilen, th_olen = th_ilen[indices], th_olen[indices]
            else:
                size = th_pred.size(1)

            if self.length_normalize is not None:
                if self.length_normalize == "olen":
                    loss = loss / th_olen
                else:
                    loss = loss / th_ilen

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "warpctc":
            # warpctc only supports float32
            th_pred = th_pred.to(dtype=torch.float32)

            th_target = th_target.cpu().int()
            th_ilen = th_ilen.cpu().int()
            th_olen = th_olen.cpu().int()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            if self.reduce:
                # NOTE: sum() is needed to keep consistency since warpctc
                # return as tensor w/ shape (1,)
                # but builtin return as tensor w/o shape (scalar).
                loss = loss.sum()
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

        if self.ctc_type == "gtnctc":
            # gtn expects list form for ys
            ys_true = [y[y != -1] for y in ys_pad]  # parse padded ys
        else:
            # ys_hat: (B, L, D) -> (L, B, D)
            ys_hat = ys_hat.transpose(0, 1)
            # (B, L) -> (BxL,)
            ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens).to(
            device=hs_pad.device, dtype=hs_pad.dtype
        )

        return loss

    def softmax(self, hs_pad):
        """softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


def ctc_forced_align(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    ignore_id: int = -1,
) -> torch.Tensor:
    """Align a CTC label sequence to an emission.

    Args:
        log_probs (Tensor): log probability of CTC emission output.
            Tensor of shape `(B, T, C)`. where `B` is the batch size, `T` is the input length,
            `C` is the number of characters in alphabet including blank.
        targets (Tensor): Target sequence. Tensor of shape `(B, L)`,
            where `L` is the target length.
        input_lengths (Tensor):
            Lengths of the inputs (max value must each be <= `T`). 1-D Tensor of shape `(B,)`.
        target_lengths (Tensor):
            Lengths of the targets. 1-D Tensor of shape `(B,)`.
        blank_id (int, optional): The index of blank symbol in CTC emission. (Default: 0)
        ignore_id (int, optional): The index of ignore symbol in CTC emission. (Default: -1)
    """
    targets[targets == ignore_id] = blank

    batch_size, input_time_size, _ = log_probs.size()
    bsz_indices = torch.arange(batch_size, device=input_lengths.device)

    _t_a_r_g_e_t_s_ = torch.cat(
        (
            torch.stack((torch.full_like(targets, blank), targets), dim=-1).flatten(start_dim=1),
            torch.full_like(targets[:, :1], blank),
        ),
        dim=-1,
    )
    diff_labels = torch.cat(
        (
            torch.as_tensor([[False, False]], device=targets.device).expand(batch_size, -1),
            _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2],
        ),
        dim=1,
    )

    neg_inf = torch.tensor(float("-inf"), device=log_probs.device, dtype=log_probs.dtype)
    padding_num = 2
    padded_t = padding_num + _t_a_r_g_e_t_s_.size(-1)
    best_score = torch.full((batch_size, padded_t), neg_inf, device=log_probs.device, dtype=log_probs.dtype)
    best_score[:, padding_num + 0] = log_probs[:, 0, blank]
    best_score[:, padding_num + 1] = log_probs[bsz_indices, 0, _t_a_r_g_e_t_s_[:, 1]]

    backpointers = torch.zeros((batch_size, input_time_size, padded_t), device=log_probs.device, dtype=targets.dtype)

    for t in range(1, input_time_size):
        prev = torch.stack(
            (best_score[:, 2:], best_score[:, 1:-1], torch.where(diff_labels, best_score[:, :-2], neg_inf))
        )
        prev_max_value, prev_max_idx = prev.max(dim=0)
        best_score[:, padding_num:] = log_probs[:, t].gather(-1, _t_a_r_g_e_t_s_) + prev_max_value
        backpointers[:, t, padding_num:] = prev_max_idx

    l1l2 = best_score.gather(
        -1, torch.stack((padding_num + target_lengths * 2 - 1, padding_num + target_lengths * 2), dim=-1)
    )

    path = torch.zeros((batch_size, input_time_size), device=best_score.device, dtype=torch.long)
    path[bsz_indices, input_lengths - 1] = padding_num + target_lengths * 2 - 1 + l1l2.argmax(dim=-1)

    for t in range(input_time_size - 1, 0, -1):
        target_indices = path[:, t]
        prev_max_idx = backpointers[bsz_indices, t, target_indices]
        path[:, t - 1] += target_indices - prev_max_idx

    alignments = _t_a_r_g_e_t_s_.gather(dim=-1, index=(path - padding_num).clamp(min=0))
    return alignments
