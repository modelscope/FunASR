# FunASR OpenAI 兼容 API Postman 集合

[English](POSTMAN.md)

当你希望先用图形界面验证私有 FunASR 语音 API，再接入 Dify、n8n、Agent 框架或内部服务时，可以导入 Postman collection。需要按 schema 导入时，可使用 [OpenAPI 规范](OPENAPI_zh.md)。

## 导入步骤

1. 启动服务：

   ```bash
   cd examples/openai_api
   python server.py --model sensevoice --device cuda --port 8000
   ```

2. 在 Postman 中导入 [`funasr-openai-api.postman_collection.json`](funasr-openai-api.postman_collection.json)。
3. 将 collection 变量 `FUNASR_BASE_URL` 改成可访问的服务地址，例如 `http://localhost:8000` 或 `http://funasr-api:8000`。
4. 第一次 smoke test 建议保持 `MODEL_ALIAS=sensevoice`；也可以先运行 `/v1/models`，再复制返回的模型别名。

## 请求列表

| Request | 用途 |
|---|---|
| `Health check` | 确认服务可访问并返回 JSON。 |
| `List model aliases` | 查看服务暴露的 OpenAI 兼容模型别名。 |
| `Transcribe audio - verbose JSON` | 上传音频并返回 `text`、`segments` 和耗时信息。 |
| `Transcribe audio - text only` | 最小转写请求，适合验证 OpenAI 兼容客户端。 |

发送转写请求前，请打开 `Body` tab，在 `file` form-data 字段选择本地音频文件。

## 故障排查

| 现象 | 处理方式 |
|---|---|
| `ECONNREFUSED` | 确认服务已启动，并且 Postman 能访问 `FUNASR_BASE_URL`。 |
| Docker 服务正常但 Postman 连不上 | 使用 Docker Compose 暴露到宿主机的端口，例如 `http://localhost:8000`。 |
| `422` 或提示缺少文件 | 确认 `file` form-data 行已启用，并指向本地音频文件。 |
| 模型别名未知 | 先运行 `List model aliases`，再把返回的别名填入 `MODEL_ALIAS`。 |
| 响应中没有 `segments` | 设置 `RESPONSE_FORMAT=verbose_json`。 |
