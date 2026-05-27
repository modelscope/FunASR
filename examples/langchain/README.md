# FunASR + LangChain Integration

Use FunASR as a speech-to-text tool in your LangChain agents. Since FunASR exposes an OpenAI-compatible API, integration is straightforward.

## Setup

```bash
# Start FunASR server
pip install torch torchaudio
pip install funasr vllm fastapi uvicorn python-multipart
funasr-server --device cuda

# Install LangChain
pip install langchain langchain-openai
```

## As a LangChain Tool

```python
from langchain.tools import tool
from openai import OpenAI

asr_client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

@tool
def speech_to_text(audio_path: str) -> str:
    """Transcribe an audio file to text using local FunASR.
    Supports wav, mp3, flac. Returns transcribed text with speaker IDs."""
    result = asr_client.audio.transcriptions.create(
        model="fun-asr-nano",
        file=open(audio_path, "rb"),
        response_format="verbose_json"
    )
    return result.text


# Use with any LangChain agent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")
tools = [speech_to_text]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can transcribe audio files."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "Please transcribe meeting.wav"})
```

## With Dify / AutoGen / CrewAI

Any framework supporting OpenAI audio API connects directly:

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",  # FunASR server
    api_key="unused"
)
result = client.audio.transcriptions.create(
    model="fun-asr-nano",
    file=open("audio.wav", "rb")
)
```

## Features

- 50+ languages (Chinese dialects, English, Japanese, Korean...)
- Speaker diarization (`spk=true`)
- Word-level timestamps (`response_format="verbose_json"`)
- Hotword boosting
- 170x realtime, fully local, MIT license

## Links

- [FunASR GitHub](https://github.com/modelscope/FunASR)
- [OpenAI API examples](../openai_api/)
- [Website](https://www.funasr.com)
