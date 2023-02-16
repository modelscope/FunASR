# Get Started
To use this example, please execute the first stage of run.sh first to obtain the prepared data and pre-trained models:
```shell
sh run.sh --stage 0 --stop_stage 0
```
Then, you can execute unit_test.py to check the correctness of code:
```shell
python unit_test.py
# you will get the results:
[{'key': 'R8002_M8002_MS802-S0000_0000000_0001600', 'value': 'spk1 [(0.0, 8.88), (10.72, 11.92), (12.64, 15.2)]\nspk2 [(8.8, 9.76)]\nspk3 [(9.6, 10.96), (15.12, 15.68)]\nspk4 [(11.12, 12.72)]'}]
[{'key': 'R8002_M8002_MS802-S0000_0000000_0001600', 'value': 'spk1 [(0.0, 8.88), (10.72, 11.92), (12.64, 15.2)]\nspk2 [(8.8, 9.76)]\nspk3 [(9.6, 10.96), (15.12, 15.68)]\nspk4 [(11.12, 12.72)]'}]
[{'key': 'R8002_M8002_MS802-S0000_0000000_0001600', 'value': 'spk1 [(0.0, 8.88), (10.72, 11.92), (12.64, 15.2)]\nspk2 [(8.8, 9.76)]\nspk3 [(9.6, 10.88), (15.12, 15.68)]\nspk4 [(11.12, 12.72)]'}]
[{'key': 'test0', 'value': 'spk1 [(0.0, 8.88), (10.64, 15.2)]\nspk2 [(8.88, 9.84)]\nspk3 [(9.6, 11.04), (15.12, 15.68)]\nspk4 [(11.2, 11.76)]'}]
```
You can also execute run.sh to reproduce the diarization performance reported in [1]
```shell
sh run.sh --stage 1 --stop_stage 2
```

# Results
After executing "run.sh", you will get a DER about 4.21%, which is reported in [1], Table 6, line "SOND Oracle Profile".

# Reference
[1] Speaker Overlap-aware Neural Diarization for Multi-party Meeting Analysis, Zhihao Du, Shiliang Zhang, 
Siqi Zheng, Zhijie Yan. EMNLP 2022.