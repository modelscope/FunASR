import torch
from torch.nn import functional as F
from funasr.models.decoder.abs_decoder import AbsDecoder


class DenseDecoder(AbsDecoder):
    def __init__(
        self,
        vocab_size,
        encoder_output_size,
        num_nodes_resnet1: int = 256,
        num_nodes_last_layer: int = 256,
        batchnorm_momentum: float = 0.5,
    ):
        super(DenseDecoder, self).__init__()
        self.resnet1_dense = torch.nn.Linear(encoder_output_size, num_nodes_resnet1)
        self.resnet1_bn = torch.nn.BatchNorm1d(
            num_nodes_resnet1, eps=1e-3, momentum=batchnorm_momentum
        )

        self.resnet2_dense = torch.nn.Linear(num_nodes_resnet1, num_nodes_last_layer)
        self.resnet2_bn = torch.nn.BatchNorm1d(
            num_nodes_last_layer, eps=1e-3, momentum=batchnorm_momentum
        )

        self.output_dense = torch.nn.Linear(num_nodes_last_layer, vocab_size, bias=False)

    def forward(self, features):
        embeddings = {}
        features = self.resnet1_dense(features)
        embeddings["resnet1_dense"] = features
        features = F.relu(features)
        features = self.resnet1_bn(features)

        features = self.resnet2_dense(features)
        embeddings["resnet2_dense"] = features
        features = F.relu(features)
        features = self.resnet2_bn(features)

        features = self.output_dense(features)
        return features, embeddings
