import sys
import time
import torch
from torch.profiler import profile



def recognize(model_file, batch_size=1, tf32=-1):
    if 'blade' in model_file:
        import torch_blade
    # """
    gpu = torch.device('cuda')
    module = torch.jit.load(model_file, map_location=gpu)
    # """
    # module = torch.jit.load(model_file)
    ori_test_data = torch.load('data/test_data_1x93x560.pt')
    ori_test_data = [torch.cat([d] * batch_size) for d in ori_test_data]

    if torch.cuda.is_available():
        test_data = tuple([i.cuda() for i in ori_test_data])

    print(torch.backends.cuda.matmul.allow_tf32)
    print(torch.backends.cudnn.allow_tf32)
    if tf32 in [0, 1]:
        tf32 = tf32 != 0
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    print(torch.backends.cuda.matmul.allow_tf32)
    print(torch.backends.cudnn.allow_tf32)

    with torch.no_grad():
        module.eval()
        def run():
            # test_data = tuple([i.cuda() for i in ori_test_data])
            if isinstance(test_data, torch.Tensor):
                out = module(test_data)
            else:
                out = module(*test_data)
            torch.cuda.synchronize()
            return out

        # warmup
        for i in range(20):
            out = run()
            # import pdb; pdb.set_trace()

        save_timeline = False
        # save_timeline = True
        all_time = 0
        for i in range(10000):
            tic = time.time()
            if save_timeline:
                activities = [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                with profile(activities=activities) as prof:
                    out = run()
                print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=10))
                ts = int(time.time() * 1000)
                time_file = 'z_times/{}-{}-{}-timeline.json'.format(batch_size, i, ts)
                prof.export_chrome_trace(time_file)
            else:
                out = run()
            cost_time = time.time() - tic
            print(cost_time)
            all_time += cost_time
        print('avg_time {}'.format(all_time / 100))


model_file = sys.argv[1]
batch_size = int(sys.argv[2])
tf32 = int(sys.argv[3])
recognize(model_file, batch_size, tf32)
