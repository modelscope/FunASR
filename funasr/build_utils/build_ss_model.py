from funasr.models.e2e_ss import MossFormer

def build_ss_model(args):
    model = MossFormer(
        in_channels=args.encoder_embedding_dim,
        out_channels=args.mossformer_sequence_dim,
        num_blocks=args.num_mossformer_layer,
        kernel_size=args.encoder_kernel_size,
        norm=args.norm,
        num_spks=args.num_spks,
        skip_around_intra=args.skip_around_intra,
        use_global_pos_enc=args.use_global_pos_enc,
        max_length=args.max_length)

    return model
