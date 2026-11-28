([简体中文](./README_zh.md)|[English](./README.md)|[日本語](./README_ja.md)|[한국어](./README_ko.md)|العربية)

<p align="center">
<a href="https://github.com/modelscope/FunASR"><img src="https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210" alt="FunASR"></a>
</p>

<div dir="rtl">

<p align="center">
  <strong>مجموعة أدوات صناعية للتعرف على الكلام، تدعم النشر في الوضع غير المتدفق (offline)، والبث المتدفق (streaming)، والأجهزة الطرفية (edge).</strong><br>
  <em>ASR · VAD · علامات الترقيم · خطوط معالجة المتحدثين · نماذج المشاعر والأحداث الصوتية · خدمة متوافقة مع OpenAI</em>
</p>

</div>

<p align="center">
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/v/funasr" alt="PyPI"></a>
  <a href="https://github.com/modelscope/FunASR"><img src="https://img.shields.io/github/stars/modelscope/FunASR?style=social" alt="Stars"></a>
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/dm/funasr" alt="Downloads"></a>
  <a href="https://modelscope.github.io/FunASR/"><img src="https://img.shields.io/badge/docs-online-blue" alt="Docs"></a>
  <a href="https://mcptoplist.com/server/io.github.modelscope%2Ffunasr-mcp"><img src="https://mcptoplist.com/badge/io.github.modelscope%2Ffunasr-mcp.svg" alt="MCP Toplist"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/10479" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10479" alt="modelscope%2FFunASR | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<div dir="rtl">

<p align="center">
  <a href="#quick-start">البدء السريع</a> · <a href="./docs/model_selection.md">اختيار النموذج</a> · <a href="#model-zoo">النماذج</a> · <a href="./docs/deployment_matrix.md">مصفوفة النشر</a> · <a href="https://www.funasr.com/">مركز النشر</a> · <a href="https://www.funasr.com/en/docs/">الوثائق</a> · <a href="#benchmark">قياس الأداء</a> · <a href="./CONTRIBUTING.md">المساهمة</a>
</p>

</div>

---

<div dir="rtl">

<a id="quick-start"></a>
## البدء السريع

### Transformers الأصلي

لنسخ الكلام بنموذج Fun-ASR-Nano عبر واجهة Hugging Face البرمجية، ابدأ من [دليل البدء السريع لـ Transformers 5.17.0 على CPU](./docs/transformers_native.md). لا حاجة إلى مجموعة أدوات FunASR ولا إلى شيفرة Python البعيدة.

[Space](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano) · [دفتر Notebook](https://colab.research.google.com/github/QwenAudio/Fun-ASR/blob/main/examples/colab/fun_asr_nano_transformers.ipynb) · [أمثلة Python / الدفعات](https://github.com/QwenAudio/Fun-ASR/tree/main/examples/transformers)

### مجموعة أدوات FunASR وخطوط المعالجة

</div>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)

<div dir="rtl">

لا تريد إعداد بيئة محلية؟ افتح [البدء السريع على Colab](./examples/colab/) لنسخ عيّنة عامة أو لرفع ملفك الصوتي الخاص من المتصفح.

هل وجدت FunASR مفيدًا؟ [امنح المشروع نجمة](https://github.com/modelscope/FunASR) ليصل إليه مزيد من المطورين.

</div>

```bash
# CPU-only installs can use the default PyPI wheels.
pip install torch torchaudio
pip install funasr
```

<div dir="rtl">

للبدء السريع على GPU، ثبّت حزم wheel الخاصة بـ PyTorch وtorchaudio المطابقة لبرنامج تشغيل NVIDIA لديك من [pytorch.org](https://pytorch.org/get-started/locally/) قبل تثبيت FunASR. وبعد التثبيت، تأكد من أن GPU ظاهر:

</div>

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

<div dir="rtl">

لا تستخدم `device="cuda"` إلا إذا طُبع `True`؛ وإلا فاستخدم `device="cpu"` أو أعد تثبيت PyTorch بحزمة CUDA wheel الصحيحة.

**مثال GPU لمجموعة أدوات FunASR: Fun-ASR-Nano** (الصينية والإنجليزية واليابانية ومجموعات اللهجات الصينية واللكنات الإقليمية؛ أما مسار Transformers الأصلي على CPU فهو مسار مستقل مرتبط أعلاه):

</div>

```python
from funasr import AutoModel

model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")
print(result[0]["text"])
```

<div dir="rtl">

لنقطة التحقق المستقلة التي تدعم 31 لغة، استخدم
[Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512).
تغطية اللغات تخص كل نقطة تحقق على حدة، لذا يجب التعامل مع Nano وMLT-Nano بوصفهما خيارين مختلفين.

لمثال موجَّه أساسًا للعمل على CPU، ويجمع التعرف على الكلام بخمس لغات مع وسوم المشاعر والأحداث الصوتية، استخدم **SenseVoiceSmall**.
يجمع خط المعالجة أدناه بينه وبين FSMN-VAD وCAM++ للحصول على مقاطع VAD تراعي المتحدثين (speaker-aware)؛
وهذه ليست مخرجات متحدثين أصلية في نقطة التحقق SenseVoiceSmall.
راجع [ورقة SenseVoice البحثية](https://arxiv.org/abs/2407.04051)،
و[نقطة التحقق على Hugging Face](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)،
و[نقطة التحقق GGUF للأجهزة الطرفية](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF).

</div>

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cpu")
result = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    batch_size_s=300,
)

# The AutoModel pipeline returns VAD segments with speaker ids and timestamps:
for seg in result[0]["sentence_info"]:
    print(f"[{seg['start']/1000:.1f}s] Speaker {seg['spk']}: {rich_transcription_postprocess(seg['sentence'])}")
```

<div dir="rtl">

يطبع هذا الكود لكل مقطع مُعاد: زمن بدايته بالثواني، ورقم المتحدث المجهول،
والنص بعد إزالة وسوم SenseVoice. يعتمد النص وحدود المقاطع على الصوت
ونقطة التحقق؛ ولا نؤكد هنا نصًا ناتجًا ثابتًا.

يستخرج CAM++ متجهات `spk_embedding`. ثم يجمّع `AutoModel` هذه المتجهات في عناقيد
ويُسند أرقام المتحدثين إلى مقاطع VAD. الأرقام محلية داخل التسجيل الواحد،
وليست هويات لأشخاص معروفين. راجع [عقد SDK](./docs/python_api.md) لمعرفة
حدود المكونات والنتائج. لا تنتقل إلى `device="cuda"` إلا بعد
التحقق من بيئة GPU متوافقة كما هو موضح أعلاه.

### التوسع والنشر للنموذج الرئيسي

عند العمل على نطاق واسع، سرّع Fun-ASR-Nano باستخدام vLLM (المعالجة الدفعية):

</div>

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(model="FunAudioLLM/Fun-ASR-Nano-2512", tensor_parallel_size=1)
results = model.generate(["audio1.wav", "audio2.wav"], language="auto")
```

<div dir="rtl">

> **النشر كخادم API:** [وصفة SenseVoice المحلية على CPU](#deploy) · [تقديم Nano على GPU وإعداد vLLM المثبَّت](./docs/vllm_guide.md)
>
> **الاستخدام مع وكلاء الذكاء الاصطناعي:** [MCP Server](examples/mcp_server/) لـ Claude/Cursor · [OpenAI API](examples/openai_api/) لـ LangChain/Dify/AutoGen
>
> **الاستخدام مع وكلاء الصوت:** [إضافة OpenClaw للزمن الحقيقي](integrations/openclaw/#compatibility) (تجريبية؛ تتطلب رقعة لم تُدمج بعد في SDK الأصلي (upstream)) · [تحويل الكلام إلى كلام من Hugging Face مع SenseVoice](./docs/community_projects.md#speech-to-speech) (مدمج في نسخة المصدر؛ وغير موجود في v1.0.0)

### لماذا FunASR؟

FunASR مجموعة أدوات: اختر المهمة ونقطة التحقق وبيئة التشغيل كلًّا على حدة.
دعم نموذج أو محوّل واحد لا يعني دعمه في كل خلفية تقديم (serving backend).

| المهمة | نقطة التحقق أو خط المعالجة | نقطة دخول التشغيل | قيد مهم |
|---|---|---|---|
| نسخ الملفات مع وسوم المشاعر/الأحداث | SenseVoiceSmall | Python `AutoModel`، على CPU أو GPU | نقطة تحقق بخمس لغات؛ الوسوم لا تحدد هوية المتحدثين. |
| نسخ الملفات بالاعتماد على نموذج لغوي كبير (LLM) | Fun-ASR-Nano | `AutoModel`؛ و`AutoModelVLLM` ذو المحرك المنفصل لمسار GPU الموثَّق | يغطي Nano الأساسي الصينية والإنجليزية واليابانية واللهجات/اللكنات الصينية؛ ويعتمد دعم الطوابع الزمنية على نقطة التحقق والمسار. |
| نسخ متعدد اللغات أوسع | Fun-ASR-MLT-Nano | Python `AutoModel` | نقطة تحقق مستقلة بـ 31 لغة؛ لا تنقل تغطيتها إلى Nano الأساسي. |
| نسخ مباشر على أجزاء (chunks) | Paraformer-zh-streaming | SDK البث المتدفق أو خدمة WebSocket في بيئة التشغيل | استخدم نقطة التحقق المتدفقة وذاكرة مؤقتة (cache) لكل جلسة، لا نقطة تحقق غير متدفقة (offline). |
| نسخ الملفات مع مراعاة المتحدثين (speaker-aware) | SenseVoiceSmall + FSMN-VAD + CAM++ | `AutoModel` مع VAD وتجميع المتجهات في عناقيد | أرقام مجهولة داخل التسجيل الواحد، وليست تعرّفًا على متحدثين مسجَّلين مسبقًا. |
| نص وطوابع زمنية ومتحدثون معًا | MOSS-Transcribe-Diarize، من OpenMOSS (طرف ثالث) | محوّل FunASR أو الخلفية الأصلية في دليل MOSS | غير متدفق (offline)، وتسميات مجهولة محصورة بالتسجيل؛ ولا يستخدم VAD خارجيًا ولا خط معالجة خارجيًا للمتحدثين في مساره الموحَّد. |
| نسخ أصلي على CPU/الأجهزة الطرفية | Fun-ASR-Nano أو SenseVoiceSmall بصيغة GGUF | بيئة تشغيل llama.cpp | يتطلب أوزانًا محوَّلة مطابقة؛ وGGUF ليست نقطة تحقق لـ Python `AutoModel`. |

راجع [Model Zoo](./model_zoo/readme.md) و[مصفوفة النشر](./docs/deployment_matrix.md)
لمعرفة حدود نقاط التحقق والواجهات والتراخيص. اختبر الأداء على صوتك الخاص
وعتادك قبل اختيار بيئة التشغيل.

هل تجرّب FunASR لأول مرة؟ استخدم [البدء السريع على Colab](./examples/colab/) قبل إعداد بيئة محلية. هل تختار أول نموذج؟ ابدأ من [دليل اختيار النموذج](./docs/model_selection.md). هل تخطط للانتقال من Whisper أو من مزوّد ASR سحابي؟ استخدم [دليل الانتقال](./docs/migration_from_whisper.md) و[مثال القياس](./examples/migration/) لاختبار صوت يمثل حالتك، ومطابقة الميزات، والإطلاق بأمان.

---

## التثبيت

</div>

```bash
pip install funasr
```

<details><summary dir="rtl">من المصدر والمتطلبات</summary>

```bash
git clone https://github.com/modelscope/FunASR.git && cd FunASR
pip install -e ./
```
<div dir="rtl">

المتطلبات: Python ≥ 3.8. ثبّت PyTorch وtorchaudio أولًا ([pytorch.org](https://pytorch.org/get-started/locally/))، ثم `pip install funasr`.

</div>

</details>

<div dir="rtl">

---

<a id="model-zoo"></a>
## مجموعة النماذج (Model Zoo)

تتضمن هذه القائمة نماذج من أطراف ثالثة. تنشر **OpenMOSS** نموذج MOSS-Transcribe-Diarize؛
وتوفر FunASR محوّلًا له ولا تملك أوزانه. مساره الموحَّد غير متدفق (offline)،
وتسمياته المجهولة محصورة بكل تسجيل، وهو ليس للزمن الحقيقي ولا للتعرف على هوية أشخاص معروفين.
تراخيص النماذج منفصلة عن ترخيص MIT الخاص بمجموعة الأدوات.

| النموذج | المهمة | اللغات | المعاملات | الروابط |
|-------|------|-----------|--------|-------|
| **Fun-ASR-Nano** | ASR | zh/en/ja + اللهجات واللكنات الصينية | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [HF / Transformers](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf) · [HF / FunASR](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) [GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) |
| **Fun-ASR-MLT-Nano** | ASR | 31 لغة | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) |
| **SenseVoiceSmall** | ASR + المشاعر + الأحداث | zh/en/ja/ko/yue | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) [GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) [الورقة البحثية](https://arxiv.org/abs/2407.04051) |
| **MOSS-Transcribe-Diarize** | طرف ثالث OpenMOSS: ASR غير متدفق + طوابع زمنية + متحدثون مجهولون | راجع البطاقة الرسمية | راجع البطاقة الرسمية | [🤗](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) [الدليل](./docs/moss_transcribe_diarize.md) |
| **Paraformer-zh** | ASR + طوابع زمنية | zh/en | 220M | [⭐](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Paraformer-zh-streaming | ASR متدفق | zh/en | 220M | [⭐](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming) |
| Qwen3-ASR | ASR، 52 لغة | متعدد اللغات | 1.7B | [الاستخدام](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | ASR، 17 لغة | متعدد اللغات | 1.5B | [الاستخدام](examples/industrial_data_pretraining/glm_asr) |
| Whisper-large-v3 | ASR + ترجمة | متعدد اللغات | 1550M | [الاستخدام](examples/industrial_data_pretraining/whisper) |
| Whisper-large-v3-turbo | ASR + ترجمة | متعدد اللغات | 809M | [الاستخدام](examples/industrial_data_pretraining/whisper) |
| ct-punc | علامات الترقيم | zh/en | 290M | [⭐](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗](https://huggingface.co/funasr/ct-punc) |
| fsmn-vad | VAD | zh/en | 0.4M | [⭐](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗](https://huggingface.co/funasr/fsmn-vad) |
| cam++ | تضمينات المتحدثين (مكوّن في خط المعالجة) | — | 7.2M | [⭐](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) [🤗](https://huggingface.co/funasr/campplus) |
| emotion2vec+large | التعرف على المشاعر | — | 300M | [⭐](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary) [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |

---

## الاستخدام

> [دليل Python التعليمي](./docs/tutorial/README.md) · [معاملات SDK ومخرجاته](./docs/python_api.md) · [التدريب](./docs/training.md) · [تسجيل النماذج](./docs/model_registration.md)

</div>

```python
from funasr import AutoModel

# Chinese production (VAD + ASR + punctuation + speaker)
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav", hotword="关键词 20")

# Optional Silero VAD (install first: python -m pip install "funasr[silero]")
model = AutoModel(
    model="paraformer-zh", vad_model="silero-vad", device="cuda",
    vad_kwargs={"silero_threshold": 0.5, "silero_min_silence_duration_ms": 100},
)
result = model.generate(input="audio.wav")

# Streaming real-time (feed audio chunk by chunk)
import soundfile as sf
model = AutoModel(model="paraformer-zh-streaming", device="cuda")
audio, sr = sf.read("speech.wav", dtype="float32")   # 16 kHz mono
chunk_size = [0, 10, 5]                               # 600 ms chunks
chunk_stride = chunk_size[1] * 960
cache = {}
n_chunks = (len(audio) - 1) // chunk_stride + 1
for i in range(n_chunks):
    chunk = audio[i * chunk_stride : (i + 1) * chunk_stride]
    res = model.generate(input=chunk, cache=cache, is_final=(i == n_chunks - 1),
                         chunk_size=chunk_size, encoder_chunk_look_back=4, decoder_chunk_look_back=1)
    if res[0]["text"]:
        print(res[0]["text"], end="", flush=True)

# Emotion recognition
model = AutoModel(model="emotion2vec_plus_large", device="cuda")
result = model.generate(input="audio.wav", granularity="utterance")
```

<div dir="rtl">

### واجهة سطر الأوامر CLI (ملائمة للوكلاء)

</div>

```bash
# Transcribe audio (simplest)
funasr audio.wav

# JSON output (for AI agents)
funasr audio.wav --output-format json

# SRT subtitles
funasr audio.wav --output-format srt --output-dir ./subs

# Speaker diarization + timestamps
funasr audio.wav --spk --timestamps -f json

# Choose model and language
funasr audio.wav --model paraformer --language zh

# Batch transcribe
funasr *.wav --output-format srt --output-dir ./output
```

<div dir="rtl">

النماذج المتاحة: `sensevoice` (الافتراضي)، `paraformer`، `paraformer-en`، `fun-asr-nano`

---

<a id="deploy"></a>
## النشر

شغّل خدمة SenseVoice محلية على CPU انطلاقًا من مجلد جديد في صدفة (shell) POSIX مع
Python 3.11. تثبّت الأوامر أدناه إصدار PyPI المنشور في بيئة منفصلة، لا في نسخة
المصدر هذه. أبقِ الخدمة غير المصادَق عليها على واجهة loopback؛ وراجع
[دليل الأمان](./examples/openai_api/SECURITY.md) قبل إتاحتها لعملاء آخرين.

</div>

```bash
python3.11 -m venv .venv-funasr-http
. .venv-funasr-http/bin/activate
python -m pip install torch torchaudio
python -m pip install funasr fastapi uvicorn python-multipart
python -m pip check
funasr-server --host 127.0.0.1 --port 8000 --model sensevoice --device cpu
```

<div dir="rtl">

انتظر حتى يكتمل تنزيل النموذج وتشغيل الخادم. في نافذة طرفية ثانية، ومن المجلد نفسه،
استخدم curl 7.76 أو أحدث لتنزيل عيّنة صوتية صينية عامة ونسخها.
يستخدم الطلب النموذج المحمَّل مسبقًا؛ ولا نَعِد بنص ثابت ولا بتسميات متحدثين.

</div>

```bash
curl --fail --location https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav && \
curl --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

<div dir="rtl">

للحصول على ASR وتسميات مجهولة للمتحدثين معًا بشكل غير متدفق (offline) (`moss-transcribe-diarize`)، جهّز البيئة المنفصلة
وفق [دليل خدمة MOSS وDocker وKubernetes وvLLM وSGLang وLocalAI وFunClip ←](./docs/moss_transcribe_diarize.md).
هذه خدمة بديلة، وليست أمرًا إضافيًا في بيئة CPU. أوقف
خدمة CPU قبل إعادة استخدام المنفذ 8000. ولتقديم Nano على GPU، اتبع
[دليل المحرك المنفصل المثبَّت](./docs/vllm_guide.md) وافحص سجلات الخلفية (backend) الفعلية؛
فاختيار النموذج وحده لا يثبت أن vLLM قد حُمّل.

</div>

```bash
# Docker streaming service
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
```

<div dir="rtl">

### CPU / الأجهزة الطرفية — llama.cpp / GGUF (دون GPU ودون Python)

شغّل **SenseVoice / Paraformer / Fun-ASR-Nano** على شكل **ملف تنفيذي واحد مكتفٍ بذاته** على CPU والأجهزة الطرفية — وهذا بالنسبة إلى FunASR ما يمثله [whisper.cpp](https://github.com/ggml-org/whisper.cpp) بالنسبة إلى Whisper، لكن بمعدل خطأ حروف (CER) أقل بنحو **3× من whisper.cpp على اللغة الصينية**. يتضمن FSMN-VAD مدمجًا، ولا يحتاج Python وقت التشغيل.

</div>

```bash
# Linux / macOS: run from the extracted release directory
bash download-funasr-model.sh sensevoice ./gguf        # or: paraformer | nano
./llama-funasr-sensevoice -m ./gguf/sensevoice-small-q8.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav
# → 欢迎大家来体验达摩院推出的语音识别模型
```

```powershell
# Windows PowerShell: run from the extracted archive root (with the `hf` CLI installed)
hf download FunAudioLLM/SenseVoiceSmall-GGUF sensevoice-small-q8.gguf --local-dir .\gguf
hf download FunAudioLLM/fsmn-vad-GGUF fsmn-vad.gguf --local-dir .\gguf
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav
# Use the windows-x64-vulkan package with a current AMD, Intel, or NVIDIA Vulkan driver:
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav --backend vulkan
# Use the windows-x64-cuda package on RTX 30-class GPUs:
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav --backend cuda
```

<div dir="rtl">

استخدم `funasr-llamacpp-linux-x64-vulkan.tar.gz` على أنظمة Linux ذات GPU
يتوفر فيها برنامج تشغيل/ICD لـ Vulkan يعمل بشكل سليم:

</div>

```bash
./llama-funasr-sensevoice -m ./gguf/sensevoice-small-q8.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav --backend vulkan
```

<div dir="rtl">

تستخدم حزمة Windows Vulkan ZIP محمّل Vulkan الخاص بالنظام الذي يوفره برنامج تشغيل GPU؛
ولا يلزم تثبيت Vulkan SDK إلا عند البناء من المصدر. وتسرّع حزمتا Vulkan
حاليًا SenseVoiceSmall.

تتيح الإصدارات الموسومة (tagged releases) حزمتَي Windows CUDA. فحزمة ZIP القياسية
`windows-x64-cuda` تستهدف معمارية CUDA رقم 86، بينما تستهدف
`windows-x64-cuda-blackwell` المعمارية 120 (`sm_120`) لبطاقات RTX 50 /
Blackwell. تضم الحزمتان ملفات cuBLAS DLL المطلوبة وتربطان مكتبة وقت تشغيل MSVC
ربطًا ساكنًا (static)، فيحتاج المستخدم إلى برنامج تشغيل NVIDIA متوافق دون حاجة إلى تثبيت
CUDA Toolkit منفصل. يتحقق CI من المعمارية وحدود الحزمة؛ لكنه لا يثبت
الاستدلال (inference) على عتاد Blackwell الفعلي.

**الملفات التنفيذية الجاهزة:** [الإصدارات](https://github.com/modelscope/FunASR/releases) · [v0.2.6](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) · [أرشيف Linux Vulkan](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-linux-x64-vulkan.tar.gz) · [ملف Windows Vulkan المضغوط](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-vulkan.zip) · [ملف Windows CUDA المضغوط](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda.zip) · [ملف Windows Blackwell CUDA المضغوط](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda-blackwell.zip) · **التنزيل والبدء السريع:** [funasr.com/deploy/llama-cpp](https://www.funasr.com/en/deploy/llama-cpp.html) · **نماذج GGUF:** [Hugging Face](https://huggingface.co/FunAudioLLM) · **الوثائق وقياسات الأداء:** [runtime/llama.cpp/](./runtime/llama.cpp/)

[مثال OpenAI API ←](./examples/openai_api/) · [عرض Gradio التجريبي ←](./examples/openai_api/GRADIO.md) · [وصفات العملاء ←](./examples/openai_api/CLIENTS.md) · [وصفات JavaScript/TypeScript ←](./examples/openai_api/JAVASCRIPT.md) · [قالب Kubernetes ←](./examples/openai_api/kubernetes/) · [وصفات سير العمل ←](./examples/openai_api/WORKFLOWS.md) · [مجموعة Postman ←](./examples/openai_api/POSTMAN.md) · [مواصفة OpenAPI ←](./examples/openai_api/OPENAPI.md) · [دليل الأمان ←](./examples/openai_api/SECURITY.md) · [مصفوفة النشر ←](./docs/deployment_matrix.md) · [وثائق النشر ←](./runtime/readme.md) · [التكامل مع الوكلاء ←](https://modelscope.github.io/FunASR/agent.html)

---

<a id="benchmark"></a>
## قياس الأداء

يحتفظ [تقرير القياس التاريخي](https://modelscope.github.io/FunASR/benchmark.html)
و[قياسات المحرك المنفصل](./docs/vllm_guide.md#benchmark) بنتائجهما
الأصلية. وهما سجلّان منفصلان، وليسا ترتيبًا شاملًا للسرعة
ولا ضمانًا للسعة الإنتاجية.

استخدم [ملاحظات RTFx وقابلية إعادة الإنتاج](./docs/benchmark/rtf_reproducibility.md)
لمقارنة نقطة التحقق/المراجعة (revision)، ومجموعة الصوت، والعتاد، وتجميع الدفعات (batching)، والإحماء (warmup)، ونطاق
قياس الزمن، وCER/WER. الإنتاجية في الوضع غير المتدفق (offline) ليست زمن الاستجابة في البث المتدفق. ويساعد
[مثال قياس الانتقال](./examples/migration/) على قياس تسجيلاتك الخاصة
بنطاق التقييم نفسه.

---

## المستجدات

- **MOSS-Transcribe-Diarize** يجلب التعرف على الكلام في التسجيلات الطويلة (long-form ASR) والطوابع الزمنية وتسميات المتحدثين المجهولة إلى خدمات FunASR وDocker وKubernetes وسير عمل vLLM/SGLang وFunClip. [نشر MOSS ->](./docs/moss_transcribe_diarize.md)
- **FunASR 1.4.29** يحافظ على إزالة تكرار أحداث SenseVoice عبر المقاطع النصية، ويستعيد طابعًا زمنيًا واحدًا لكل رمز (token) عندما يتعين على إطار CIF الأخير أن يُطلق أكثر من مرة. [#3931](https://github.com/modelscope/FunASR/pull/3931)، [#3935](https://github.com/modelscope/FunASR/pull/3935). `python -m pip install -U "funasr==1.4.29"`. لا يحدّث هذا الإصدار الخاص بالحزمة الجذرية حزم بيئات التشغيل المستقلة ONNX وTorch وWebSocket. [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.29)
  **FunASR 1.4.28** يحافظ على طول الخصائص (feature length) لكل مقطع كلام في دفعات الواجهة الأمامية لـ Whisper ذات الأطوال المختلطة، ويُبقي الطوابع الزمنية لرموز symbol-token محاذاةً صحيحة في المعالجة اللاحقة الأساسية للجمل. [#3918](https://github.com/modelscope/FunASR/pull/3918)، [#3929](https://github.com/modelscope/FunASR/pull/3929). `python -m pip install -U "funasr==1.4.28"`. لا يحدّث هذا الإصدار الخاص بالحزمة الجذرية حزمتَي ONNX وTorch المستقلتين. [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.28)
  **FunASR 1.4.27** يقبل مصفوفات NumPy الصوتية من نوع int16 ومتعددة القنوات، ويعالج نتائج VAD ذات المدة الصفرية، ويحافظ على مخرجات الطوابع الزمنية للجمل عند غياب علامات الترقيم، ويتجاهل الأسطر الفارغة في قاموس التقطيع (segmentation dictionary). [#3890](https://github.com/modelscope/FunASR/pull/3890)، [#3891](https://github.com/modelscope/FunASR/pull/3891)، [#3896](https://github.com/modelscope/FunASR/pull/3896)، [#3898](https://github.com/modelscope/FunASR/pull/3898)، [#3903](https://github.com/modelscope/FunASR/pull/3903). `python -m pip install -U "funasr==1.4.27"`. [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.27)
  **FunASR 1.4.26** يحافظ على المسافات بين الكلمات في مقاطع VAD الإنجليزية/الكورية، ويعيد نصًا عاديًا عند استخدام `response_format=text` في HTTP. [#3874](https://github.com/modelscope/FunASR/pull/3874)، [#3883](https://github.com/modelscope/FunASR/pull/3883). `python -m pip install -U "funasr==1.4.26"`. [الإصدار ونطاق التحقق](https://github.com/modelscope/FunASR/releases/tag/v1.4.26)
  **FunASR 1.4.25** يستعيد عمل تحليل المعاملات المنطقية (boolean) في Python 3.12 دون الحاجة إلى `distutils` ويحافظ على رسائل الخطأ المُطبَّعة؛ ويزيل المسافات البادئة المصطنعة من الطوابع الزمنية للجمل اللاتينية مع الحفاظ على الحدود والنص الخام. [إصلاح #3863](https://github.com/modelscope/FunASR/pull/3863)، [إصلاح #3866](https://github.com/modelscope/FunASR/pull/3866). التثبيت عبر `python -m pip install -U "funasr==1.4.25"`. [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.25)
  **FunASR 1.4.24** يتجاهل الرموز الفارغة غير اللغوية لمنع حلقات المحلل الرمزي (tokenizer)، ويصلح تطبيع التباين مع الحشو (padding)، ويمنع انهيارات CIF في الدفعات المختلطة التي تحتوي على عيّنات بلا إطلاق (zero-fire). [إصلاح #3852](https://github.com/modelscope/FunASR/pull/3852)، [إصلاح #3853](https://github.com/modelscope/FunASR/pull/3853)، [إصلاح #3854](https://github.com/modelscope/FunASR/pull/3854). نقطة التحقق المدرَّبة بالخيارين `norm_means=True` و`norm_vars=True` معًا تحتاج إلى إعادة تقييم جودة التعرف لأن الخصائص المُطبَّعة تتغير؛ ويبقى `norm_vars=False` هو الافتراضي. التثبيت عبر `python -m pip install -U "funasr==1.4.24"`. [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.24)
  **FunASR 1.4.23** يصلح مؤشر الواجهة الأمامية المتدفقة (online) عند `lfr_m=lfr_n=1` وأجزاء VAD النهائية الفارغة الزائدة في الوضع غير المتدفق (offline). عند تمرير `token_list`، تعيد `ids2tokens` في SentencePiece أجزاء النموذج؛ استخدم `tokens2text` للحصول على النص. وبدون `token_list` لا يتغير فك الترميز النصي القديم. [إصلاح #3849](https://github.com/modelscope/FunASR/pull/3849)، [إصلاح #3783](https://github.com/modelscope/FunASR/pull/3783). [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.23)
  **FunASR 1.4.22** يحترم `merge_length_s` لكل استدعاء، ويقيّم عتبات صمت VAD الديناميكية لكل إطار باستخدام مدة المقطع الحالي، مع الإبقاء على خيار العتبة الثابتة. [إصلاح #3841](https://github.com/modelscope/FunASR/pull/3841)، [إصلاح #3833](https://github.com/modelscope/FunASR/pull/3833). [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.22)
  **FunASR 1.4.21** يحافظ على ذاكرة علامات الترقيم المؤقتة في البث المتدفق عند الأجزاء الفارغة أو التي تحتوي مسافات فقط؛ ولا يتغير الاستدلال عند وجود محتوى. [إصلاح #3824](https://github.com/modelscope/FunASR/pull/3824). [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.21)
  **FunASR 1.4.20** يحافظ على سبب فشل استيراد التبعية الاختيارية الخاصة بالمتحدثين الأصلي ضمن `ImportError` متسلسل (chained) عند طلب `spk_model`، بينما يظل بناء نموذج ASR فقط متاحًا. [إصلاح #3819](https://github.com/modelscope/FunASR/pull/3819). [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.20)
  **FunASR 1.4.19** يعالج دفعات Paraformer ذات الرموز الفارغة، ويفرض `max_speakers` في التجميع التلقائي للمتحدثين، ويضيف دعم `--hf_overrides` المتحقَّق منه بصيغة كائن JSON إلى خدمة WebSocket للزمن الحقيقي. [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.19)
  الإصدار السابق **FunASR 1.4.18** يصلح الاستبدال الخاطئ للكلمات المفتاحية (hotword) عبر الأنظمة الكتابية المختلفة، ومحاذاة الطوابع الزمنية بعد تصفية الرموز (filtered tokens)، وانحراف الجمل/المتحدثين في VAD للغة الصينية. [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.18)
  **إصدار أقدم: FunASR 1.4.17** يضيف إدخال PCM16 الصريح ويحسّن معالجة الطوابع الزمنية والكلمات المفتاحية (hotword) ومعدل أخذ العينات في مكوّن المتحدثين والمقاطع الطويلة في البث المتدفق. [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.17)
  **إصدار أقدم: FunASR 1.4.16** يحافظ على التحديد الصريح للجهاز (device) لنماذج VAD وعلامات الترقيم والمتحدثين عبر استدعاءات الاستدلال، ويضيف أدلة اعتماد Transformers الأصلي المختبَرة. [الإصدار ونطاق التحقق ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.16)
- **Transformers الأصلي:** الإصدار **5.17.0** المنشور يدعم Fun-ASR-Nano عبر نقطة التحقق الرسمية `-hf`، مع أمثلة CPU ودفتر Notebook. [ابدأ من هنا ->](./docs/transformers_native.md)

> راجع [إصدارات GitHub](https://github.com/modelscope/FunASR/releases) للاطلاع على سجل التغييرات الكامل والملفات القابلة للتنزيل.

---

## المجتمع

ابدأ بـ[استكشاف الأخطاء وإصلاحها](./docs/troubleshooting.md) قبل الإبلاغ عن مشكلة.
أرفق النموذج الدقيق الذي تستخدمه، وبيئة التشغيل (runtime)، وبيئتك البرمجية (environment)، ومثالًا مصغّرًا لإعادة إنتاج المشكلة.

|  |  |
|---|---|
| 📖 [الوثائق](https://modelscope.github.io/FunASR/) | 🐛 [المشكلات (Issues)](https://github.com/modelscope/FunASR/issues) |
| 💬 [النقاشات](https://github.com/modelscope/FunASR/discussions) | 🤗 [HuggingFace](https://huggingface.co/funasr) |
| 🤝 [المساهمة](./CONTRIBUTING.md) | 🌐 [funasr.com](https://www.funasr.com) |
| 🗺️ [أدوار المستودعات وخارطة الطريق](./docs/repository_roles.md) | 📈 [خطة النمو](./docs/community_growth_20k.md) |
| 🧩 [مشاريع المجتمع](./docs/community_projects.md) | 💡 [معرض حالات الاستخدام](./docs/use_case_showcase.md) |

## تاريخ النجوم

<a href="https://star-history.com/#modelscope/FunASR&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" />
   <img alt="مخطط تاريخ النجوم" src="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" width="600" />
 </picture>
</a>

## الترخيص

- الشيفرة المصدرية لمجموعة أدوات FunASR في هذا المستودع: [ترخيص MIT](./LICENSE).
- أوزان النماذج المدرَّبة مسبقًا مرخَّصة بشكل منفصل. راجع الترخيص المعروض في بطاقة كل نموذج؛ وعندما تشير بطاقة النموذج إلى [اتفاقية ترخيص النماذج مفتوحة المصدر لـ FunASR](./MODEL_LICENSE) تسري تلك الشروط.

## الاستشهادات

</div>

```bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and others},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  booktitle={INTERSPEECH},
  year={2023}
}
```

ترجمة: فريق مُشَيِّد
