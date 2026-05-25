# FunASR Colab 빠른 시작

[English](README.md) | [简体中文](README_zh.md) | [日本語](README_ja.md) | 한국어

로컬 Python 환경을 준비하지 않고 브라우저에서 바로 FunASR을 실행할 수 있습니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)

## Notebook에서 확인할 수 있는 것

- Colab에 FunASR과 런타임 의존성을 설치합니다.
- Colab GPU가 있으면 자동으로 `cuda:0`을 사용하고, 없으면 CPU를 사용합니다.
- `paraformer-zh`, VAD, 문장부호 모델로 공개 샘플 오디오를 전사합니다.
- 직접 업로드한 오디오 파일을 같은 모델로 전사합니다.
- transcript JSON을 저장하여 공유, 비교, issue 보고에 활용할 수 있습니다.

## 사용 메모

- 첫 실행에서는 모델 파일을 다운로드하므로 몇 분이 걸릴 수 있습니다.
- CPU runtime은 짧은 smoke test에 적합합니다. 긴 오디오에는 GPU runtime이 더 빠릅니다.
- 프로덕션 배포를 검토하려면 notebook을 먼저 실행한 뒤 [deployment matrix](../../docs/deployment_matrix.md)를 확인하세요.
- OpenAI 호환 HTTP 서비스를 테스트하려면 [examples/openai_api](../openai_api/README_ko.md)를 사용하세요.

Notebook source: [funasr_quickstart.ipynb](https://github.com/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb).
