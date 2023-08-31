CUDA_VISIBLE_DEVICES=4,5,6,7 \
  python -m torch.distributed.launch --nproc_per_node 4 \
  --master_addr 127.0.0.4 --master_port 29503 \
  finetune.py