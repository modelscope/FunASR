import onnxruntime
import numpy as np


if __name__ == '__main__':
    onnx_path = "../damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/model.onnx"
    sess = onnxruntime.InferenceSession(onnx_path)
    input_name = [nd.name for nd in sess.get_inputs()]
    output_name = [nd.name for nd in sess.get_outputs()]

    def _get_feed_dict(text_length):
        return {'inputs': np.ones((1, text_length), dtype=np.int64), 'text_lengths': np.array([text_length,], dtype=np.int32)}

    def _run(feed_dict):
        output = sess.run(output_name, input_feed=feed_dict)
        for name, value in zip(output_name, output):
            print('{}: {}'.format(name, value))
    _run(_get_feed_dict(10))
