import onnxruntime
import numpy as np
import sys
import time
import pdb


if __name__ == '__main__':
    onnx_path = sys.argv[1]
    # pdb.set_trace()
    opts = onnxruntime.SessionOptions()
    # if len(sys.argv) > 2:
    #     opts.intra_op_num_threads = int(sys.argv[2])
    # sess = onnxruntime.InferenceSession(onnx_path, sess_options=opts)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_name = [nd.name for nd in sess.get_inputs()]
    output_name = [nd.name for nd in sess.get_outputs()]

    feed_dict = np.load('data/test_data_1x93x560.npy', allow_pickle=True, encoding='bytes').item()
    batch_size = int(sys.argv[2])
    feed_dict = {k: np.concatenate([v] * batch_size) for k, v in feed_dict.items()}

    def _run(feed_dict):
        output = sess.run(output_name, input_feed=feed_dict)
        """
        for name, value in zip(output_name, output):
            print('{}: {}'.format(name, value.shape))
        """

    for i in range(20):
        _run(feed_dict)

    all_rt = 0
    for i in range(100):
        tic = time.time()
        _run(feed_dict)
        toc = time.time()
        rt = toc - tic
        print(rt)
        all_rt += rt
    print('avg_time {}'.format(all_rt / 100.0))

    """
    opts.enable_profiling = True
    sess = onnxruntime.InferenceSession(onnx_path, sess_options=opts)
    output = sess.run(output_name, input_feed=feed_dict)
    prof_file = sess.end_profiling()
    """
