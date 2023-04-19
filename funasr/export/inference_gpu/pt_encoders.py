import sys
import time
import torch


def recognize(model_file, data_file, batch_size=1):
    if 'blade' in model_file:
        import torch_blade
    # """
    gpu = torch.device('cuda')
    module = torch.jit.load(model_file, map_location=gpu)
    # """
    # module = torch.jit.load(model_file)
    test_data = torch.load(data_file)
    _istuple = lambda x: isinstance(x, tuple)
    _mod = lambda x, func: tuple([func(i) for i in x]) if _istuple(x) else func(x)
    _batch = lambda x: torch.cat([x] * batch_size)
    test_data = tuple([_mod(d, _batch) for d in test_data])
    _cuda = lambda x: x.cuda()
    test_data = tuple([_mod(d, _cuda) for d in test_data])

    """
    import os
    fp16 = float(os.environ.get('FP16', False))
    if fp16:
        test_data = (test_data[0] / fp16, test_data[1])
        if 'blade' not in model_file:
            _half = lambda x: x.half()
            test_data = tuple([_mod(d, _half) for d in test_data])
            module.half()
    """


    # if torch.cuda.is_available():
    #     test_data = tuple([i.cuda() for i in test_data])

    with torch.no_grad():
        module.eval()
        def run():
            if isinstance(test_data, torch.Tensor):
                out = module(test_data)
            else:
                out = module(*test_data)
            torch.cuda.synchronize()
            return out

        # warmup
        for i in range(20):
            out = run()
            print(out[0])
            import pdb; pdb.set_trace()

        all_time = 0
        for i in range(1000):
            tic = time.time()
            out = run()
            cost_time = time.time() - tic
            print(cost_time)
            all_time += cost_time
        print('avg_time {}'.format(all_time / 100))


model_file = sys.argv[1]
data_file = sys.argv[2]
batch_size = int(sys.argv[3])
recognize(model_file, data_file, batch_size)
