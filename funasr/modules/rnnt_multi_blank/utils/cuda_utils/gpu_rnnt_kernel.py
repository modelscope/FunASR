# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2018-2019, Mingkun Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
from numba import cuda

from funasr.modules.rnnt_multi_blank.utils import rnnt_helper

GPU_RNNT_THREAD_SIZE = 256


@cuda.jit(device=True, inline=True)
def logp(
    denom: torch.Tensor,
    acts: torch.Tensor,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    mb: int,
    t: int,
    u: int,
    v: int,
):
    """
    Compute the sum of log probability from the activation tensor and its denominator.

    Args:
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of the
            logprobs activation tensor across entire vocabulary.
        acts: Tensor of shape [B, T, U, V+1] flattened.
            Represents the logprobs activation tensor.
        maxT: The maximum possible acoustic sequence length.
            Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length.
            Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        mb: Batch indexer.
        t: Acoustic sequence timestep indexer.
        u: Target sequence timestep indexer.
        v: Vocabulary token indexer.

    Returns:
        The sum of logprobs[mb, t, u, v] + denom[mb, t, u]
    """

    col = (mb * maxT + t) * maxU + u
    return denom[col] + acts[col * alphabet_size + v]


@cuda.jit()
def compute_alphas_kernel(
    acts: torch.Tensor,
    denom: torch.Tensor,
    alphas: torch.Tensor,
    llForward: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B]
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
):
    """
    Compute alpha (forward variable) probabilities over the transduction step.

    Args:
        acts: Tensor of shape [B, T, U, V+1] flattened.
            Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of the
            logprobs activation tensor across entire vocabulary.
        alphas: Zero tensor of shape [B, T, U]. Will be updated inside the kernel
            with the forward variable probabilities.
        llForward: Zero tensor of shape [B]. Represents the log-likelihood of the
            forward pass. Returned as the forward pass loss that is reduced by
            the optimizer.
        xlen: Vector of length B which contains the actual acoustic sequence
            lengths in the padded activation tensor.
        ylen: Vector of length B which contains the actual target sequence
            lengths in the padded activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token
            - usually the RNNT blank). The matrix contains the padded target
            transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length.
            Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length.
            Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT blank token in the vocabulary.
            Generally the first or last token in the vocab.

    Updates:
        Kernel inplace updates the following inputs:
        -   alphas: forward variable scores.
        -   llForward: log-likelihood of forward variable.
    """

    # // launch B blocks, each block has U threads
    b = cuda.blockIdx.x  # // batch id
    u = cuda.threadIdx.x  # label id, u
    T = xlen[b]  # select AM length of current sample
    U = ylen[b] + 1  # select target length of current sample, +1 for the blank token

    labels: torch.Tensor = mlabels[
        b
    ]  # mb label start point, equivalent to mlabels + b * (maxU - 1)
    offset = b * maxT * maxU  # pointer indexing offset

    # alphas += offset # pointer offset, ignored since we explicitly add offset

    # Initilize alpha[b, t=0, u=0] for all b in B
    if u == 0:
        alphas[offset] = 0

    # sync until all alphas are initialized
    cuda.syncthreads()

    # Ordinary alpha calculations, broadcast across B=b and U=u
    # Look up forward variable calculation from rnnt_numpy.forward_pass()
    for n in range(1, T + U - 1):
        t = n - u

        if u == 0:
            # for t in range(1, T) step to initialize alphas[b, t, 0]
            if t > 0 and t < T:
                alphas[offset + t * maxU + u] = alphas[
                    offset + (t - 1) * maxU + u
                ] + logp(denom, acts, maxT, maxU, alphabet_size, b, t - 1, 0, blank_)
        elif u < U:
            # for u in range(1, U) step to initialize alphas[b, 0, u]
            if t == 0:
                alphas[offset + u] = alphas[offset + u - 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, 0, u - 1, labels[u - 1]
                )

            # for t in range(1, T) for u in range(1, U) step to compute alphas[b, t, u]
            elif t > 0 and t < T:
                no_emit = alphas[offset + (t - 1) * maxU + u] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t - 1, u, blank_
                )
                emit = alphas[offset + t * maxU + u - 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t, u - 1, labels[u - 1]
                )

                alphas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)

        # sync across all B=b and U=u
        cuda.syncthreads()

    # After final sync, alphas[b, T-1, U - 1] + logprobs[b, T-1, U-1, blank]
    # + denom[b, T-1, U-1] gives log-likelihood of forward pass.
    if u == 0:
        loglike = alphas[offset + (T - 1) * maxU + U - 1] + logp(
            denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_
        )
        llForward[b] = loglike


@cuda.jit()
def compute_betas_kernel(
    acts: torch.Tensor,
    denom: torch.Tensor,
    betas: torch.Tensor,
    llBackward: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B, U]
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
):
    """
    Compute beta (backward variable) probabilities over the transduction step.

    Args:
        acts: Tensor of shape [B, T, U, V+1] flattened.
            Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator
            of the logprobs activation tensor across entire vocabulary.
        betas: Zero tensor of shape [B, T, U]. Will be updated inside the kernel
            with the backward variable probabilities.
        llBackward: Zero tensor of shape [B]. Represents the log-likelihood
            of the backward pass. Returned as the backward pass loss that
            is reduced by the optimizer.
        xlen: Vector of length B which contains the actual acoustic
            sequence lengths in the padded activation tensor.
        ylen: Vector of length B which contains the actual target sequence
            lengths in the padded activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token
            - usually the RNNT blank). The matrix contains the padded target
            transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length.
            Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length.
            Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT blank token in the vocabulary.
            Generally the first or last token in the vocab.

    Updates:
        Kernel inplace updates the following inputs:
        -   betas: backward variable scores.
        -   llBackward: log-likelihood of backward variable.
    """

    # // launch B blocks, each block has U threads
    b = cuda.blockIdx.x  # // batch id
    u = cuda.threadIdx.x  # label id, u
    T = xlen[b]  # select AM length of current sample
    U = ylen[b] + 1  # select target length of current sample, +1 for the blank token

    labels: torch.Tensor = mlabels[
        b
    ]  # mb label start point, equivalent to mlabels + b * (maxU - 1)
    offset = b * maxT * maxU  # pointer indexing offset

    # betas += offset # pointer offset, ignored since we explicitly add offset

    # Initilize beta[b, t=T-1, u=U-1] for all b in B
    # with log_probs[b, t=T-1, u=U-1, blank]
    if u == 0:
        betas[offset + (T - 1) * maxU + U - 1] = logp(
            denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_
        )

    # sync until all betas are initialized
    cuda.syncthreads()

    # Ordinary beta calculations, broadcast across B=b and U=u
    # Look up backward variable calculation from rnnt_numpy.backward_pass()
    for n in range(T + U - 2, -1, -1):
        t = n - u

        if u == (U - 1):
            # for t in reversed(range(T - 1)) step to initialize betas[b, t, U-1]
            if t >= 0 and t < (T - 1):
                betas[offset + t * maxU + U - 1] = betas[
                    offset + (t + 1) * maxU + U - 1
                ] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, U - 1, blank_)
        elif u < U:
            if t == T - 1:
                # for u in reversed(range(U - 1)) step to initialize betas[b, T-1, u]
                betas[offset + (T - 1) * maxU + u] = betas[
                    offset + (T - 1) * maxU + u + 1
                ] + logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, u, labels[u])
            elif (t >= 0) and (t < T - 1):
                # for t in reversed(range(T - 1)) for u in reversed(range(U - 1))
                # step to compute betas[b, t, u]
                no_emit = betas[offset + (t + 1) * maxU + u] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_
                )
                emit = betas[offset + t * maxU + u + 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u]
                )
                betas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)

        # sync across all B=b and U=u
        cuda.syncthreads()

    # After final sync, betas[b, 0, 0] gives
    # log-likelihood of backward pass.
    if u == 0:
        llBackward[b] = betas[offset]


@cuda.jit()
def compute_grad_kernel(
    grads: torch.Tensor,
    acts: torch.Tensor,
    denom: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    logll: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B, U]
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
    fastemit_lambda: float,
    clamp: float,
):
    """
    Compute gradients over the transduction step.

    Args:
        grads: Zero Tensor of shape [B, T, U, V+1]. Is updated by this kernel to
            contain the gradients of this batch of samples.
        acts: Tensor of shape [B, T, U, V+1] flattened.
            Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator
            of the logprobs activation tensor across entire vocabulary.
        alphas: Alpha variable, contains forward probabilities.
            A tensor of shape [B, T, U].
        betas: Beta varoable, contains backward probabilities.
            A tensor of shape [B, T, U].
        logll: Log-likelihood of the forward variable, represented as a vector
            of shape [B]. Represents the log-likelihood of the forward pass.
        xlen: Vector of length B which contains the actual acoustic sequence
            lengths in the padded activation tensor.
        ylen: Vector of length B which contains the actual target sequence lengths
            in the padded activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token
            - usually the RNNT blank). The matrix contains the padded target
            transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length.
            Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length.
            Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT blank token in the vocabulary.
            Generally the first or last token in the vocab.
        fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
            FastEmit: Low-latency Streaming ASR with Sequence-level
            Emission Regularization.
        clamp: Float value. When set to value >= 0.0, will clamp the
            gradient to [-clamp, clamp].

    Updates:
        Kernel inplace updates the following inputs:
        -   grads: Gradients with respect to the log likelihood (logll).
    """

    # Kernel call:
    # blocks_per_grid = minibatch (b) * maxT (t) * maxU (u)
    # threads_per_block = constant buffer size of parallel threads (v :: Constant)
    tid = cuda.threadIdx.x  # represents v, taking steps of some constant size
    idx = tid  # index of v < V+1; in steps of constant buffer size
    col = cuda.blockIdx.x  # represents a fused index of b * t * u

    # Decompose original indices from fused `col`
    u = col % maxU  # (b * t * u) % u = u
    bt = (col - u) // maxU  # (b * t * u - u) // U = b * t
    t = bt % maxT  # (b * t) % t = t
    mb = (bt - t) // maxT  # (b * t - t) // T = b

    # constants
    T = xlen[mb]  # select AM length of current sample
    U = ylen[mb] + 1  # select target length of current sample, +1 for the blank token
    labels: torch.Tensor = mlabels[mb]  # labels = mlabels + mb * (maxU - 1);

    # Buffered gradient calculations, broadcast across B=b, T=t and U=u,
    # looped over V with some constant stride.
    # Look up gradient calculation from rnnt_numpy.compute_gradient()
    if t < T and u < U:
        # For cuda kernels, maximum number of threads per block is limited to some value
        # However, it may be the case that vocabulary size is larger than this limit
        # To work around this, an arbitrary thread buffer size is chosen such that,
        # 1) each element within the thread pool operates independently of the other
        # 2) An inner while loop moves the index of each buffer element by the size
        #    of the buffer itself, such that all elements of the vocabulary size are
        #    covered in (V + 1 // thread_buffer) number of steps.
        # As such, each thread will perform the while loop at least
        # (V + 1 // thread_buffer) number of times
        while idx < alphabet_size:
            # remember, `col` represents the tri-index [b, t, u]
            # therefore; logpk = denom[b, t, u] + acts[b, t, u, v]
            logpk = denom[col] + acts[col * alphabet_size + idx]
            # initialize the grad of the sample acts[b, t, u, v]
            grad = math.exp(alphas[col] + betas[col] + logpk - logll[mb])

            # If FastEmit regularization is enabled, calculate the gradeint of
            # probability of predicting the next label at the current timestep.
            # The formula for this is Equation 9 in https://arxiv.org/abs/2010.11148,
            # multiplied by the log probability of the current step (t, u),
            # normalized by the total log likelihood. Once the gradient has been
            # calculated, scale it by `fastemit_lambda`, as in Equation 10.
            if fastemit_lambda > 0.0 and u < U - 1:
                fastemit_grad = fastemit_lambda * math.exp(
                    alphas[col]  # alphas(t, u)
                    + (
                        denom[col] + acts[col * alphabet_size + labels[u]]
                    )  # y_hat(t, u)
                    + betas[col + 1]  # betas(t, u+1)
                    + logpk  # log Pr(k|t, u)
                    - logll[mb]  # total log likelihood for normalization
                )
            else:
                fastemit_grad = 0.0

            # Update the gradient of act[b, t, u, v] with the gradient from
            # FastEmit regularization
            grad = grad + fastemit_grad

            # // grad to last blank transition
            # grad[b, T-1, U-1, v=blank] -= exp(alphas[b, t, u) + logpk - logll[b])
            if (idx == blank_) and (t == T - 1) and (u == U - 1):
                grad -= math.exp(alphas[col] + logpk - logll[mb])

            # grad of blank across t < T;
            # grad[b, t<T-1, u, v=blank] -= exp(alphas[b, t, u]
            #     + logpk - logll[b] betas[b, t + 1, u])
            if (idx == blank_) and (t < T - 1):
                grad -= math.exp(alphas[col] + logpk - logll[mb] + betas[col + maxU])

            # grad of correct token across u < U;
            # grad[b, t, u<U-1, v=label[u]] -= exp(alphas[b, t, u]
            #     + logpk - logll[b] + betas[b, t, u+1])
            # Scale the gradient by (1.0 + FastEmit_lambda) in log space,
            # then exponentiate
            if (u < U - 1) and (idx == labels[u]):
                # exp(log(1 + fastemit_lambda) + ...) is numerically more stable than
                # multiplying (1.0 + fastemit_lambda) with result.
                grad -= math.exp(
                    math.log1p(fastemit_lambda)
                    + alphas[col]
                    + logpk
                    - logll[mb]
                    + betas[col + 1]
                )

            # update grads[b, t, u, v] = grad
            grads[col * alphabet_size + idx] = grad

            # clamp gradient (if needed)
            if clamp > 0.0:
                g = grads[col * alphabet_size + idx]
                g = min(g, clamp)
                g = max(g, -clamp)
                grads[col * alphabet_size + idx] = g

            # update internal index through the thread_buffer;
            # until idx < V + 1, such that entire vocabulary has been updated.
            idx += GPU_RNNT_THREAD_SIZE


@cuda.jit()
def compute_multiblank_alphas_kernel(
    acts: torch.Tensor,
    denom: torch.Tensor,
    sigma: float,
    alphas: torch.Tensor,
    llForward: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
    big_blank_duration: torch.Tensor,
    num_big_blanks: int,
):
    """
    Compute alpha (forward variable) probabilities for multi-blank transducuer loss
        (https://arxiv.org/pdf/2211.03541).

    Args:
        acts: Tensor of shape [B, T, U, V + 1 + num_big_blanks] flattened.
            Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of
            the logprobs activation tensor across entire vocabulary.
        sigma: Hyper-parameter for logit-undernormalization technique for training
            multi-blank transducers.
        alphas: Zero tensor of shape [B, T, U]. Will be updated inside the kernel
            with the forward variable probabilities.
        llForward: Zero tensor of shape [B]. Represents the log-likelihood of the
            forward pass. Returned as the forward pass loss that is
            reduced by the optimizer.
        xlen: Vector of length B which contains the actual acoustic sequence
            lengths in the padded activation tensor.
        ylen: Vector of length B which contains the actual target sequence
            lengths in the padded activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token
            - usually the RNNT blank). The matrix contains the padded target
            transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length.
            Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length.
            Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT standard blank token in the vocabulary.
        big_blank_durations: Vector of supported big blank durations of the model.
        num_big_blanks: Number of big blanks of the model.

    Updates:
        Kernel inplace updates the following inputs:
        -   alphas: forward variable scores.
        -   llForward: log-likelihood of forward variable.
    """

    # // launch B blocks, each block has U threads
    b = cuda.blockIdx.x  # // batch id
    u = cuda.threadIdx.x  # label id, u
    T = xlen[b]  # select AM length of current sample
    U = ylen[b] + 1  # select target length of current sample, +1 for the blank token

    labels: torch.Tensor = mlabels[
        b
    ]  # mb label start point, equivalent to mlabels + b * (maxU - 1)
    offset = b * maxT * maxU  # pointer indexing offset

    # Initilize alpha[b, t=0, u=0] for all b in B
    if u == 0:
        alphas[offset] = 0

    # sync until all alphas are initialized
    cuda.syncthreads()

    # Ordinary alpha calculations, broadcast across B=b and U=u
    # Look up forward variable calculation from rnnt_numpy.forward_pass()
    # Note: because of the logit under-normalization, everytime logp() is called,
    # it is always followed by a `-sigma` term.
    for n in range(1, T + U - 1):
        t = n - u

        if u == 0:
            # for t in range(1, T) step to initialize alphas[b, t, 0]
            if t > 0 and t < T:
                alphas[offset + t * maxU + u] = (
                    alphas[offset + (t - 1) * maxU + u]
                    + logp(denom, acts, maxT, maxU, alphabet_size, b, t - 1, 0, blank_)
                    - sigma
                )

                # Now add the weights for big blanks.
                for i in range(num_big_blanks):
                    if t >= big_blank_duration[i]:
                        alphas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(
                            alphas[offset + t * maxU + u],
                            alphas[offset + (t - big_blank_duration[i]) * maxU + u]
                            + logp(
                                denom,
                                acts,
                                maxT,
                                maxU,
                                alphabet_size,
                                b,
                                t - big_blank_duration[i],
                                0,
                                blank_ - 1 - i,
                            )
                            - sigma,
                        )

        elif u < U:
            # for u in range(1, U) step to initialize alphas[b, 0, u]
            if t == 0:
                alphas[offset + u] = (
                    alphas[offset + u - 1]
                    + logp(
                        denom,
                        acts,
                        maxT,
                        maxU,
                        alphabet_size,
                        b,
                        0,
                        u - 1,
                        labels[u - 1],
                    )
                    - sigma
                )

            # for t in range(1, T) for u in range(1, U) step to compute alphas[b, t, u]
            elif t > 0 and t < T:
                no_emit = (
                    alphas[offset + (t - 1) * maxU + u]
                    + logp(denom, acts, maxT, maxU, alphabet_size, b, t - 1, u, blank_)
                    - sigma
                )
                emit = (
                    alphas[offset + t * maxU + u - 1]
                    + logp(
                        denom,
                        acts,
                        maxT,
                        maxU,
                        alphabet_size,
                        b,
                        t,
                        u - 1,
                        labels[u - 1],
                    )
                    - sigma
                )

                alphas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)

                # Now add the weights for big blanks.
                for i in range(num_big_blanks):
                    if t >= big_blank_duration[i]:
                        # big-blank weight here is
                        # alpha(t - duration, u) * p(big-blank | t - duration, u)
                        #     / exp(sigma), in log domain
                        # do this all all big-blanks if the above condition is met
                        big_blank_no_emit = (
                            alphas[offset + (t - big_blank_duration[i]) * maxU + u]
                            + logp(
                                denom,
                                acts,
                                maxT,
                                maxU,
                                alphabet_size,
                                b,
                                t - big_blank_duration[i],
                                u,
                                blank_ - 1 - i,
                            )
                            - sigma
                        )
                        alphas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(
                            alphas[offset + t * maxU + u], big_blank_no_emit
                        )

        # sync across all B=b and U=u
        cuda.syncthreads()

    # After final sync, alphas[b, T-1, U - 1] + logprobs[b, T-1, U-1, blank]
    # + denom[b, T-1, U-1] gives log-likelihood of forward pass.
    if u == 0:
        loglike = (
            alphas[offset + (T - 1) * maxU + U - 1]
            + logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_)
            - sigma
        )

        # Now add the weights for big blanks for the final weight computation.
        for i in range(num_big_blanks):
            if T >= big_blank_duration[i]:
                big_blank_loglike = (
                    alphas[offset + (T - big_blank_duration[i]) * maxU + U - 1]
                    + logp(
                        denom,
                        acts,
                        maxT,
                        maxU,
                        alphabet_size,
                        b,
                        T - big_blank_duration[i],
                        U - 1,
                        blank_ - 1 - i,
                    )
                    - sigma
                )
                loglike = rnnt_helper.log_sum_exp(loglike, big_blank_loglike)

        llForward[b] = loglike


@cuda.jit()
def compute_multiblank_betas_kernel(
    acts: torch.Tensor,
    denom: torch.Tensor,
    sigma: float,
    betas: torch.Tensor,
    llBackward: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B, U]
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
    big_blank_duration: torch.Tensor,
    num_big_blanks: int,
):
    """
    Compute beta (backward variable) probabilities for multi-blank transducer loss
        (https://arxiv.org/pdf/2211.03541).

    Args:
        acts: Tensor of shape [B, T, U, V + 1 + num-big-blanks] flattened.
            Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator
            of the logprobs activation tensor across entire vocabulary.
        sigma: Hyper-parameter for logit-undernormalization technique for
            training multi-blank transducers.
        betas: Zero tensor of shape [B, T, U]. Will be updated inside the kernel
            with the backward variable probabilities.
        llBackward: Zero tensor of shape [B]. Represents the log-likelihood
            of the backward pass. Returned as the backward pass loss
            that is reduced by the optimizer.
        xlen: Vector of length B which contains the actual acoustic sequence
            lengths in the padded activation tensor.
        ylen: Vector of length B which contains the actual target sequence
            lengths in the padded activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token
            - usually the RNNT blank). The matrix contains the padded target
            transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length.
            Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length.
            Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT standard blank token in the vocabulary.
        big_blank_durations: Vector of supported big blank durations of the model.
        num_big_blanks: Number of big blanks of the model.

    Updates:
        Kernel inplace updates the following inputs:
        -   betas: backward variable scores.
        -   llBackward: log-likelihood of backward variable.
    """

    # // launch B blocks, each block has U threads
    b = cuda.blockIdx.x  # // batch id
    u = cuda.threadIdx.x  # label id, u
    T = xlen[b]  # select AM length of current sample
    U = ylen[b] + 1  # select target length of current sample, +1 for the blank token

    labels: torch.Tensor = mlabels[
        b
    ]  # mb label start point, equivalent to mlabels + b * (maxU - 1)
    offset = b * maxT * maxU  # pointer indexing offset

    # Note: just like the alphas, because of the logit under-normalization, everytime
    # logp() is called, it is always followed by a `-sigma` term.

    # Initilize beta[b, t=T-1, u=U-1] for all b in B with
    # log_probs[b, t=T-1, u=U-1, blank]
    if u == 0:
        betas[offset + (T - 1) * maxU + U - 1] = (
            logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_)
            - sigma
        )

    # sync until all betas are initialized
    cuda.syncthreads()

    # Ordinary beta calculations, broadcast across B=b and U=u
    # Look up backward variable calculation from rnnt_numpy.backward_pass()
    for n in range(T + U - 2, -1, -1):
        t = n - u

        if u == (U - 1):
            # for t in reversed(range(T - 1)) step to initialize betas[b, t, U-1]
            if t >= 0 and t < (T - 1):
                # beta[t, U - 1] = beta[t + 1, U - 1] * p(blank | t, U - 1) / exp(sigma)
                # this part is the same as regular RNN-T.
                betas[offset + t * maxU + U - 1] = (
                    betas[offset + (t + 1) * maxU + U - 1]
                    + logp(denom, acts, maxT, maxU, alphabet_size, b, t, U - 1, blank_)
                    - sigma
                )

                # now add the weights from big blanks
                for i in range(num_big_blanks):
                    if t + big_blank_duration[i] < T:
                        # adding to beta[t, U - 1] of weight (in log domain),
                        # beta[t + duration, U - 1] *
                        #     p(big-blank | t, U - 1) / exp(sigma)
                        betas[offset + t * maxU + U - 1] = rnnt_helper.log_sum_exp(
                            betas[offset + t * maxU + U - 1],
                            betas[offset + (t + big_blank_duration[i]) * maxU + U - 1]
                            + logp(
                                denom,
                                acts,
                                maxT,
                                maxU,
                                alphabet_size,
                                b,
                                t,
                                U - 1,
                                blank_ - 1 - i,
                            )
                            - sigma,
                        )
                    elif t + big_blank_duration[i] == T and big_blank_duration[i] != 1:
                        # adding to beta[T - duration, U - 1] of weight (in log domain),
                        # p(big-blank | T - duration, U - 1) / exp(sigma)
                        betas[offset + t * maxU + U - 1] = rnnt_helper.log_sum_exp(
                            betas[offset + t * maxU + U - 1],
                            logp(
                                denom,
                                acts,
                                maxT,
                                maxU,
                                alphabet_size,
                                b,
                                t,
                                U - 1,
                                blank_ - 1 - i,
                            )
                            - sigma,
                        )

        elif u < U:
            if t == T - 1:
                # for u in reversed(range(U - 1)) step to initialize betas[b, T-1, u]
                betas[offset + (T - 1) * maxU + u] = (
                    betas[offset + (T - 1) * maxU + u + 1]
                    + logp(
                        denom, acts, maxT, maxU, alphabet_size, b, T - 1, u, labels[u]
                    )
                    - sigma
                )
            elif (t >= 0) and (t < T - 1):
                # for t in reversed(range(T - 1)) for u in reversed(range(U - 1))
                # step to compute betas[b, t, u]
                no_emit = (
                    betas[offset + (t + 1) * maxU + u]
                    + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_)
                    - sigma
                )
                emit = (
                    betas[offset + t * maxU + u + 1]
                    + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u])
                    - sigma
                )
                betas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)

                # now add the weights from big blanks
                for i in range(num_big_blanks):
                    if t < T - big_blank_duration[i]:
                        # added weight for the big-blank,
                        # beta[t + duration, u] * p(big-blank | t, u) / exp(sigma)
                        big_blank_no_emit = (
                            betas[offset + (t + big_blank_duration[i]) * maxU + u]
                            + logp(
                                denom,
                                acts,
                                maxT,
                                maxU,
                                alphabet_size,
                                b,
                                t,
                                u,
                                blank_ - 1 - i,
                            )
                            - sigma
                        )
                        betas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(
                            betas[offset + t * maxU + u], big_blank_no_emit
                        )

        # sync across all B=b and U=u
        cuda.syncthreads()

    # After final sync, betas[b, 0, 0] gives
    # log-likelihood of backward pass.
    if u == 0:
        llBackward[b] = betas[offset]


@cuda.jit()
def compute_multiblank_grad_kernel(
    grads: torch.Tensor,
    acts: torch.Tensor,
    denom: torch.Tensor,
    sigma: float,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    logll: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B, U]
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
    big_blank_duration: torch.Tensor,
    num_big_blanks: int,
    fastemit_lambda: float,
    clamp: float,
):
    """
    Compute gradients for multi-blank transducer loss
        (https://arxiv.org/pdf/2211.03541).

    Args:
        grads: Zero Tensor of shape [B, T, U, V + 1 + num_big_blanks].
            Is updated by this kernel to contain the gradients of this batch of samples.
        acts: Tensor of shape [B, T, U, V + 1 + num_big_blanks] flattened.
            Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator
            of the logprobs activation tensor across entire vocabulary.
        sigma: Hyper-parameter for logit-undernormalization technique
            for training multi-blank transducers.
        alphas: Alpha variable, contains forward probabilities.
            A tensor of shape [B, T, U].
        betas: Beta varoable, contains backward probabilities.
            A tensor of shape [B, T, U].
        logll: Log-likelihood of the forward variable, represented as
            a vector of shape [B]. Represents the log-likelihood of the forward pass.
        xlen: Vector of length B which contains the actual acoustic
            sequence lengths in the padded activation tensor.
        ylen: Vector of length B which contains the actual target sequence
            lengths in the padded activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token
            - usually the RNNT blank). The matrix contains the padded target
            transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length.
            Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length.
            Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT blank token in the vocabulary.
            Generally the first or last token in the vocab.
        fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
            FastEmit: Low-latency Streaming ASR with Sequence-level
            Emission Regularization.
        clamp: Float value. When set to value >= 0.0, will clamp
            the gradient to [-clamp, clamp].
        big_blank_durations: Vector of supported big blank durations of the model.
        num_big_blanks: Number of big blanks of the model.

    Updates:
        Kernel inplace updates the following inputs:
        -   grads: Gradients with respect to the log likelihood (logll).
    """

    # Kernel call:
    # blocks_per_grid = minibatch (b) * maxT (t) * maxU (u)
    # threads_per_block = constant buffer size of parallel threads (v :: Constant)
    tid = cuda.threadIdx.x  # represents v, taking steps of some constant size
    idx = tid  # index of v < V+1; in steps of constant buffer size
    col = cuda.blockIdx.x  # represents a fused index of b * t * u

    # Decompose original indices from fused `col`
    u = col % maxU  # (b * t * u) % u = u
    bt = (col - u) // maxU  # (b * t * u - u) // U = b * t
    t = bt % maxT  # (b * t) % t = t
    mb = (bt - t) // maxT  # (b * t - t) // T = b

    # constants
    T = xlen[mb]  # select AM length of current sample
    U = ylen[mb] + 1  # select target length of current sample, +1 for the blank token
    labels: torch.Tensor = mlabels[mb]  # labels = mlabels + mb * (maxU - 1);

    # Buffered gradient calculations, broadcast across B=b, T=t and U=u, looped over
    # V with some constant stride. Look up gradient calculation from
    # rnnt_numpy.compute_gradient()
    if t < T and u < U:
        # For cuda kernels, maximum number of threads per block is limited to some value
        # However, it may be the case that vocabulary size is larger than this limit
        # To work around this, an arbitrary thread buffer size is chosen such that,
        # 1) each element within the thread pool operates independently of the other
        # 2) An inner while loop moves the index of each buffer element by the size
        #    of the buffer itself, such that all elements of the vocabulary size are
        #    covered in (V + 1 // thread_buffer) number of steps.
        # As such, each thread will perform the while loop at least
        # (V + 1 // thread_buffer) number of times
        while idx < alphabet_size:
            # remember, `col` represents the tri-index [b, t, u]
            # therefore; logpk = denom[b, t, u] + acts[b, t, u, v]
            logpk = denom[col] + acts[col * alphabet_size + idx]
            # initialize the grad of the sample acts[b, t, u, v]
            grad = math.exp(alphas[col] + betas[col] + logpk - logll[mb])

            # In all of the following computation, whenever logpk is used, we
            # need to subtract sigma based on our derivation of the gradient of
            # the logit under-normalization method.

            # If FastEmit regularization is enabled, calculate the gradeint of
            # probability of predicting the next label at the current timestep.
            # The formula for this is Equation 9 in https://arxiv.org/abs/2010.11148,
            # multiplied by the log probability of the current step (t, u), normalized
            # by the total log likelihood. Once the gradient has been calculated,
            # scale it by `fastemit_lambda`, as in Equation 10.
            if fastemit_lambda > 0.0 and u < U - 1:
                fastemit_grad = fastemit_lambda * math.exp(
                    alphas[col]  # alphas(t, u)
                    + (denom[col] + acts[col * alphabet_size + labels[u]])
                    + betas[col + 1]  # betas(t, u+1)
                    + logpk  # log Pr(k|t, u)
                    - sigma
                    - logll[mb]  # total log likelihood for normalization
                )
            else:
                fastemit_grad = 0.0

            # Update the gradient of act[b, t, u, v] with the gradient
            # from FastEmit regularization
            grad = grad + fastemit_grad

            # grad to last blank transition
            # grad[b, T-1, U-1, v=blank] -= exp(alphas[b, t, u)
            #     + logpk - sigma - logll[b])
            if (idx == blank_) and (t == T - 1) and (u == U - 1):
                grad -= math.exp(alphas[col] + logpk - sigma - logll[mb])
            else:
                # this is one difference of the multi-blank gradient from standard RNN-T
                # gradient -- basically, wherever the blank_ symbol is addressed in the
                # original code, we need to do similar things to big blanks, and we need
                # to change the if conditions to match the duration of the big-blank.
                # grad[b, T-duration, U-1, v=big-blank] -=
                #     exp(alphas[b, t, u) + logpk - sigma - logll[b])
                for i in range(num_big_blanks):
                    if (
                        (idx == blank_ - 1 - i)
                        and (t == T - big_blank_duration[i])
                        and (u == U - 1)
                    ):
                        grad -= math.exp(alphas[col] + logpk - sigma - logll[mb])

            # grad of blank across t < T;
            # grad[b, t<T-1, u, v=blank] -= exp(alphas[b, t, u] +
            #    logpk - sigma - logll[b] betas[b, t + 1, u])
            if (idx == blank_) and (t < T - 1):
                grad -= math.exp(
                    alphas[col] + logpk - sigma - logll[mb] + betas[col + maxU]
                )
            else:
                # This is another difference between multi-blank and RNN-T gradients.
                # Now we consider gradients for big-blanks.
                # grad[b, t<T-duration, u, v=big-blank] -=
                #     exp(alphas[b, t, u] + logpk - sigma - logll[b]
                #     + betas[b, t + duration, u])
                for i in range(num_big_blanks):
                    if (idx == blank_ - 1 - i) and (t < T - big_blank_duration[i]):
                        grad -= math.exp(
                            alphas[col]
                            + logpk
                            - sigma
                            - logll[mb]
                            + betas[col + big_blank_duration[i] * maxU]
                        )

            # grad of correct token across u < U;
            # grad[b, t, u<U-1, v=label[u]] -=
            #     exp(alphas[b, t, u] + logpk - sigma - logll[b] + betas[b, t, u+1])
            # Scale the gradient by (1.0 + FastEmit_lambda) in log space,
            # then exponentiate
            if (u < U - 1) and (idx == labels[u]):
                # exp(log(1 + fastemit_lambda) + ...) is numerically more stable than
                # multiplying (1.0 + fastemit_lambda) with result.
                grad -= math.exp(
                    math.log1p(fastemit_lambda)
                    + alphas[col]
                    + logpk
                    - sigma
                    - logll[mb]
                    + betas[col + 1]
                )

            # update grads[b, t, u, v] = grad
            grads[col * alphabet_size + idx] = grad

            # clamp gradient (if needed)
            if clamp > 0.0:
                g = grads[col * alphabet_size + idx]
                g = min(g, clamp)
                g = max(g, -clamp)
                grads[col * alphabet_size + idx] = g

            # update internal index through the thread_buffer;
            # until idx < V + 1, such that entire vocabulary has been updated.
            idx += GPU_RNNT_THREAD_SIZE
