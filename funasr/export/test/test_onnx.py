import onnxruntime
import numpy as np


if __name__ == '__main__':
    onnx_path = "/mnt/workspace/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.onnx"
    sess = onnxruntime.InferenceSession(onnx_path)
    input_name = [nd.name for nd in sess.get_inputs()]
    output_name = [nd.name for nd in sess.get_outputs()]

    def _get_feed_dict(feats_length):
        return {'speech': np.zeros((1, feats_length, 560), dtype=np.float32), 'speech_lengths': np.array([feats_length,], dtype=np.int32)}

    def _run(feed_dict):
        output = sess.run(output_name, input_feed=feed_dict)
        for name, value in zip(output_name, output):
            print('{}: {}'.format(name, value.shape))

    _run(_get_feed_dict(100))
    _run(_get_feed_dict(200))