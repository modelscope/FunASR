from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import sys

# Define extraction pipeline
inference_sv_pipline = pipeline(
    task=Tasks.speaker_verification,
    model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch',
    output_dir=sys.argv[2],
)
# Extract speaker embedding
rec_result = inference_sv_pipline(
    audio_in=sys.argv[1],

)
