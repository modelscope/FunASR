serveraddr=localhost
manifest_path=/workspace/aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz
num_task=60
python3 client/decode_manifest_triton.py \
    --server-addr $serveraddr \
    --compute-cer \
    --model-name infer_pipeline \
    --num-tasks $num_task \
    --manifest-filename $manifest_path
