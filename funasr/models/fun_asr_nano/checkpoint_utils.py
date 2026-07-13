def normalize_checkpoint_state(state):
    """Unwrap common checkpoint containers and remove one DDP key prefix."""
    while isinstance(state, dict):
        wrapped = next(
            (
                state[key]
                for key in ("state_dict", "model_state_dict", "model")
                if isinstance(state.get(key), dict)
            ),
            None,
        )
        if wrapped is None or wrapped is state:
            break
        state = wrapped

    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def disable_incomplete_ctc(model, loaded_keys, *, log):
    """Disable timestamp inference unless every required CTC tensor was loaded."""
    if model.ctc_decoder is None or model.ctc is None:
        return []

    expected = {f"ctc_decoder.{key}" for key in model.ctc_decoder.state_dict()}
    expected.update(f"ctc.{key}" for key in model.ctc.state_dict())
    missing = sorted(expected.difference(loaded_keys))
    if not missing:
        return []

    preview = ", ".join(missing[:3])
    suffix = "" if len(missing) <= 3 else ", ..."
    log.warning(
        "Disabling CTC timestamps because the checkpoint did not initialize "
        "%d of %d required CTC tensors (%s%s). Text transcription remains available.",
        len(missing),
        len(expected),
        preview,
        suffix,
    )
    model.ctc_decoder = None
    model.ctc = None
    model.ctc_tokenizer = None
    model.blank_id = None
    return missing
