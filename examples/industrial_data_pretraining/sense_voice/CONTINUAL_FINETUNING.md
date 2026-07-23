# Continual fine-tuning SenseVoiceSmall

This guide covers adding an accent, dialect, or domain to SenseVoiceSmall while
retaining its existing languages. Full-parameter, partial-parameter, adapter,
and replay setups cannot guarantee zero regression. Treat retention as an explicit
evaluation constraint.

## 1. Lock the evaluation sets first

Create one fixed validation set for every language or domain that must be
preserved and one for the new domain. The speakers in validation must be
speaker-disjoint from all training and replay data. Do not use a single mixed
CER to select checkpoints because a large Mandarin set can hide Cantonese
regression.

Record the original model's CER or WER on every set before generating
pseudo-labels. Define an acceptable relative regression for each retained
language before training:

```text
relative regression = (candidate CER - baseline CER) / baseline CER
```

The threshold is a product decision, not a FunASR guarantee. Select a checkpoint
only when it meets every retention threshold and the new-domain target.

A real starting point from [issue #3388](https://github.com/modelscope/FunASR/issues/3388)
is:

| Validation set | Utterances | Baseline CER |
| --- | ---: | ---: |
| Mandarin FLEURS test | 300 | 8.9% |
| Cantonese ASCEND dialogue | 314 | 9.1% |
| Teochew held-out, about 20 speakers in the source corpus | 300 | 93.3% |

Keep the Teochew held-out speakers out of the 8.45-hour training set.

## 2. Build a replay manifest

When manually labeled old-domain data is unavailable, decode owned or
appropriately licensed Mandarin and Cantonese audio with the original model.
Store the model revision and raw decoding output with each pseudo-label so the
manifest can be audited and regenerated.

Filter or review at least these cases before training:

- empty, repeated, or obviously truncated pseudo-labels;
- language mismatches;
- audio outside the duration range supported by the training setup;
- low decoder scores when a comparable score is available;
- a random sample from every speaker and language, including high-score samples.

A pseudo-label is not ground truth. Keep validation labels independent from
pseudo-label generation.

Sample a fixed replay budget for each training epoch instead of copying the
entire old-domain pool. Start with this small matrix:

| replay:new | New Teochew | Replay per epoch |
| --- | ---: | ---: |
| 1:1 | 8.45 h | 8.45 h |
| 2:1 | 8.45 h | 16.9 h |
| 3:1 | 8.45 h | 25.35 h |

For this example, start with 2:1. A practical first replay split is about
10.1 hours Mandarin and 6.8 hours Cantonese per epoch. The 60:40 split is an
experiment starting point, not an official ratio: it deliberately oversamples
Cantonese relative to a 100 h / 10 h source pool because Cantonese retention is
a separate requirement. Sample different records across epochs and keep a
minimum Cantonese share. Use the manifest as the source of truth when reported
pool totals disagree.

## 3. Use a supported language token

SenseVoiceSmall currently has a closed language-id set: `<|zh|>`, `<|en|>`,
`<|yue|>`, `<|ja|>`, and `<|ko|>`. For Teochew transcripts written as
standard Chinese, use `<|zh|>` for the first experiment.

Do not add `<|teochew|>` to the JSONL and assume it becomes a new language id.
The current tokenizer does not encode it as one supported special token. A real
custom language id requires coordinated tokenizer, vocabulary, language
embedding, checkpoint migration, export, and inference changes.

## 4. Train in two stages

Copy `finetune.sh` to an experiment directory, point it at the mixed training
manifest and fixed validation manifests, and replace the default long,
high-learning-rate run with a short first stage.

### Stage 1: head warm-up

Freeze the complete acoustic encoder and train the language-query embedding and
CTC/output parameters:

```bash
++train_conf.max_epoch=5 \
++freeze_param="encoder" \
++optim_conf.lr=0.00002 \
```

Three to five epochs are an initial budget, not a required value. Validate and
save often enough to observe all three curves. Do not wait for 50 epochs before
checking retention.

### Stage 2: release the later encoder blocks

If Teochew CER stalls while Mandarin and Cantonese remain inside their
retention gates, resume the best Stage 1 checkpoint and freeze only the input
block and the first three regular encoder blocks:

```bash
++freeze_param="encoder.encoders0,encoder.encoders.0,encoder.encoders.1,encoder.encoders.2" \
++optim_conf.lr=0.00002 \
```

The value is a comma-separated list of parameter-name prefixes. With the
current SenseVoiceSmall encoder, later `encoder.encoders.*` blocks remain
trainable. Lower the learning rate further, for example to `0.00001`, or stop
earlier if a retained language regresses.

At startup, verify that the log contains the expected lines:

```text
Setting encoder...requires_grad = False
```

Also inspect the model summary. A misspelled prefix silently leaves the intended
parameters trainable because no parameter name matches it.

## 5. Select the checkpoint by constraints

Evaluate every candidate checkpoint on Mandarin, Cantonese, and Teochew
separately.

1. Discard checkpoints that violate either retained-language threshold.
2. Among the remaining checkpoints, choose the lowest Teochew CER.
3. If none pass, increase replay from 1:1 to 2:1 or 3:1, increase the minimum
   Cantonese share, reduce the learning rate, or shorten training.
4. If all retained metrics pass but Teochew stalls, move from Stage 1 to the
   partial-freeze Stage 2 configuration.

Keep a table with the model revision, manifest revision, replay mix, frozen
prefixes, learning rate, step, and all three metrics. This makes the selected
checkpoint reproducible.

## 6. LoRA status

FunASR contains LoRA support for specific Paraformer/SANM paths, but
SenseVoiceSmall does not currently expose a complete, documented LoRA recipe.
Do not assume that setting `lora_only` creates trainable SenseVoice adapters.
Verify the trainable parameter list before relying on any adapter experiment.

## Launch checklist

- Validation speakers do not occur in train or replay data.
- Baseline CER/WER and acceptable relative regression are recorded per language.
- Pseudo-labels are filtered and manually sampled for audit.
- Replay is sampled to an explicit per-epoch budget with a Cantonese floor.
- Teochew uses the supported `<|zh|>` path unless the model is deliberately forked.
- Freeze logs and trainable parameters match the intended stage.
- Every saved checkpoint is evaluated on all retained and new domains.
