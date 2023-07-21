import os

import torch
import torch.nn as nn

class SequentialRNNLM(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.encoder = model.encoder
        self.rnn = model.rnn
        self.rnn_type = model.rnn_type
        self.decoder = model.decoder
        self.nlayers = model.nlayers
        self.nhid = model.nhid
        self.model_name = "seq_rnnlm"

    def forward(self, y, hidden1, hidden2=None):
        # batch_score function.
        emb = self.encoder(y)
        if self.rnn_type == "LSTM":
            output, (hidden1, hidden2) = self.rnn(emb, (hidden1, hidden2))
        else:
            output, hidden1 = self.rnn(emb, hidden1)

        decoded = self.decoder(
            output.contiguous().view(output.size(0) * output.size(1), output.size(2))
        )
        if self.rnn_type == "LSTM":
            return (
                decoded.view(output.size(0), output.size(1), decoded.size(1)),
                hidden1,
                hidden2,
            )
        else:
            return (
                decoded.view(output.size(0), output.size(1), decoded.size(1)),
                hidden1,
            )

    def get_dummy_inputs(self):
        tgt = torch.LongTensor([0, 1]).unsqueeze(0)
        hidden = torch.randn(self.nlayers, 1, self.nhid)
        if self.rnn_type == "LSTM":
            return (tgt, hidden, hidden)
        else:
            return (tgt, hidden)

    def get_input_names(self):
        if self.rnn_type == "LSTM":
            return ["x", "in_hidden1", "in_hidden2"]
        else:
            return ["x", "in_hidden1"]

    def get_output_names(self):
        if self.rnn_type == "LSTM":
            return ["y", "out_hidden1", "out_hidden2"]
        else:
            return ["y", "out_hidden1"]

    def get_dynamic_axes(self):
        ret = {
            "x": {0: "x_batch", 1: "x_length"},
            "y": {0: "y_batch"},
            "in_hidden1": {1: "hidden1_batch"},
            "out_hidden1": {1: "out_hidden1_batch"},
        }
        if self.rnn_type == "LSTM":
            ret.update(
                {
                    "in_hidden2": {1: "hidden2_batch"},
                    "out_hidden2": {1: "out_hidden2_batch"},
                }
            )
        return ret

    def get_model_config(self, path):
        return {
            "use_lm": True,
            "model_path": os.path.join(path, f"{self.model_name}.onnx"),
            "lm_type": "SequentialRNNLM",
            "rnn_type": self.rnn_type,
            "nhid": self.nhid,
            "nlayers": self.nlayers,
        }
