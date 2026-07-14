([English](README.md)|[简体中文](README_zh.md)|[日本語](README_ja.md)|한국어)

# FunASR OpenAI 호환 API 서버

FunASR OpenAI 호환 API는 `/v1/audio/transcriptions`를 제공합니다. OpenAI 스타일 SDK, 에이전트 프레임워크, Dify, n8n, HTTP 노드, 내부 업무 시스템에서 프라이빗 음성 인식 서비스로 사용할 수 있습니다.

## 빠른 시작

```bash
pip install funasr fastapi uvicorn python-multipart
python server.py --model sensevoice --device cuda --port 8000
```

모델 로드가 끝나면 서비스가 시작됩니다. 헬스 체크는 `GET /health`입니다.

바로 복사해서 쓸 수 있는 연동 예제가 필요하면 [클라이언트 레시피](CLIENTS.md), [JavaScript/TypeScript 레시피](JAVASCRIPT.md), [Gradio 브라우저 데모](GRADIO.md), [워크플로 레시피](WORKFLOWS.md), [Postman 컬렉션](POSTMAN.md), [OpenAPI 명세](OPENAPI.md), [보안 및 게이트웨이 가이드](SECURITY.md), [Kubernetes 배포 템플릿](kubernetes/README.md)을 참고하세요.

### 엔드투엔드 smoke test

다른 터미널에서 실행합니다.

```bash
bash smoke_test.sh
# curl/bash가 없는 환경을 위한 크로스 플랫폼 방식:
python smoke_test.py
```

동일한 수동 명령:

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/health
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## Gradio 브라우저 데모

로컬 브라우저에서 파일 업로드나 마이크 입력을 테스트하려면 먼저 API 서버를 시작한 뒤 선택 사항인 Gradio 프런트엔드를 실행합니다.

```bash
pip install gradio
python gradio_app.py --base-url http://localhost:8000
```

이 브라우저 데모는 smoke test와 동일한 OpenAI 호환 API 엔드포인트를 호출합니다. Docker, Kubernetes, 프로덕션 참고 사항은 [Gradio 브라우저 데모](GRADIO.md)를 확인하세요.

## OpenAI SDK로 사용하기

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

result = client.audio.transcriptions.create(
    model="sensevoice",  # "paraformer", "paraformer-en", "fun-asr-nano"도 사용할 수 있습니다
    file=open("meeting.wav", "rb"),
)
print(result.text)

verbose = client.audio.transcriptions.create(
    model="sensevoice",
    file=open("meeting.wav", "rb"),
    response_format="verbose_json",
)
print(verbose.segments)
```

## curl로 사용하기

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice

curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## 사용 가능한 모델

| Model | GPU 속도 | CPU 속도 | 언어 | 특징 |
|---|---|---|---|---|
| `sensevoice` | 170x realtime | 17x realtime | zh/en/ja/ko/yue | 감정 및 이벤트 태그 |
| `paraformer` | 120x realtime | 15x realtime | zh/en | 문장부호 복원 |
| `paraformer-en` | 120x realtime | 15x realtime | en | 영어 인식 |
| `fun-asr-nano` | 17x realtime | 3.6x realtime | 중영일 + 중국어 방언/지역 억양 | LLM-based, 타임스탬프 |

## API 엔드포인트

| Endpoint | Method | 설명 |
|---|---|---|
| `/v1/audio/transcriptions` | POST | OpenAI 호환 음성 전사 |
| `/v1/models` | GET | 모델 별칭 목록 |
| `/health` | GET | 헬스 체크, 로드된 모델, 사용 가능한 모델 |
| `/docs` | GET | FastAPI Swagger 문서 |

코드 작성 없이 확인하려면 [Gradio 브라우저 데모](GRADIO.md)로 로컬 업로드나 마이크 테스트를 진행하거나 [Postman 컬렉션](POSTMAN.md)을 가져오세요. API 게이트웨이, 개발자 포털, 내부 클라이언트 생성을 위해서는 [OpenAPI 명세](OPENAPI.md)를 사용할 수 있습니다.

## 에이전트 및 로우코드 워크플로

**LangChain**, **LlamaIndex**, **AutoGen**, **CrewAI**, **Semantic Kernel**, **Dify**, **n8n**, OpenAI audio API 또는 multipart HTTP를 지원하는 모든 시스템에서 사용할 수 있습니다.

- SDK, JavaScript/TypeScript, Agent tool 작성법은 [클라이언트 레시피](CLIENTS.md)와 [JavaScript/TypeScript 레시피](JAVASCRIPT.md)를 참고하세요.
- Dify, n8n, HTTP 노드, webhook worker는 [워크플로 레시피](WORKFLOWS.md)를 참고하세요.
- GUI smoke test는 [Postman 컬렉션](POSTMAN.md)을 참고하세요.
- schema 기반 가져오기는 [OpenAPI 명세](OPENAPI.md)를 사용할 수 있습니다.

## Docker 배포

기본 이미지는 CPU 모드로 시작하므로 재현 가능한 smoke test로 사용할 수 있습니다.

```bash
cd examples/openai_api
cp .env.example .env

docker compose up --build
```

동일한 `docker run` 명령:

```bash
docker build -t funasr-api .

docker run --rm -p 8000:8000 \
  -e FUNASR_DEVICE=cpu \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

GPU 호스트에서는 NVIDIA Container Toolkit과 CUDA 지원 PyTorch/FunASR 이미지가 필요합니다. CUDA 의존성에 맞게 이미지를 조정한 뒤 다음처럼 실행할 수 있습니다.

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e FUNASR_DEVICE=cuda \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

컨테이너 검증:

```bash
BASE_URL=http://localhost:8000 bash smoke_test.sh
python smoke_test.py --base-url http://localhost:8000
```

## Kubernetes 배포

팀에서 공유하거나 게이트웨이를 통해 노출하기 전에 [보안 및 게이트웨이 가이드](SECURITY.md)를 검토하고 TLS, 인증, 업로드 제한, 속도 제한, 로그 정책을 준비하세요.

영구 모델 캐시, 헬스 프로브, 프라이빗 `ClusterIP`를 갖춘 내부 클러스터 서비스가 필요하다면 [Kubernetes 배포 템플릿](kubernetes/README.md)에서 시작하세요. 예제 이미지를 빌드하고 push한 뒤 manifests를 적용하고, `kubectl port-forward`와 `python smoke_test.py --base-url http://localhost:8000`으로 검증합니다.

CUDA 지원 이미지와 GPU 스케줄링 설정이 준비되기 전까지는 기본 CPU 모드를 유지하세요.

## 설정

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--host` | `0.0.0.0` | 바인드 주소 |
| `--port` | `8000` | 포트 |
| `--device` | `cuda` | `cuda`, `cpu`, `mps` |
| `--model` | `sensevoice` | 시작 시 미리 로드할 모델 |

Docker 환경 변수:

| Env | 기본값 | 설명 |
|---|---|---|
| `FUNASR_PORT` | `8000` | `server.py`로 전달되는 컨테이너 포트 |
| `FUNASR_DEVICE` | `cpu` | 컨테이너 디바이스 모드. CUDA 지원 의존성이 포함된 이미지에서만 `cuda`로 설정하세요 |
| `FUNASR_MODEL` | `sensevoice` | 컨테이너 시작 시 로드할 모델 별칭 |

## 문제 해결

| 증상 | 해결 방법 |
|---|---|
| CUDA를 사용할 수 없음 | 먼저 `--device cpu`로 smoke test를 통과시키세요. |
| 8000 포트가 사용 중 | `--port 9000`으로 바꾸고 `BASE_URL=http://localhost:9000 bash smoke_test.sh` 또는 `python smoke_test.py --base-url http://localhost:9000`을 실행하세요. |
| 모델 다운로드가 느림 | 안정적인 네트워크에서 다시 시도하거나 ModelScope/Hugging Face에서 모델을 미리 다운로드하세요. |
| Dify/n8n 컨테이너에서 `localhost` 접속 실패 | 워크플로 런타임에서 접근 가능한 호스트명, Compose service name 또는 Kubernetes service name을 사용하세요. |
| 응답에 `segments`가 없음 | `response_format=verbose_json`를 설정하세요. |
