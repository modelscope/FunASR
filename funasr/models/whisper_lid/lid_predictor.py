from funasr.register import tables
from funasr.models.whisper_lid.eres2net.ResNet import (
    ERes2Net,
    BasicBlockERes2Net,
    BasicBlockERes2Net_diff_AFF,
)


@tables.register("lid_predictor_classes", "LidPredictor")
class LidPredictor(ERes2Net):
    def __init__(
        self,
        block=BasicBlockERes2Net,
        block_fuse=BasicBlockERes2Net_diff_AFF,
        num_blocks=[3, 4, 6, 3],
        m_channels=32,
        feat_dim=80,
        embedding_size=192,
        pooling_func="TSTP",
        two_emb_layer=False,
    ):
        super(LidPredictor, self).__init__(
            block=block,
            block_fuse=block_fuse,
            num_blocks=num_blocks,
            m_channels=m_channels,
            feat_dim=feat_dim,
            embedding_size=embedding_size,
            pooling_func=pooling_func,
            two_emb_layer=two_emb_layer,
        )
