# FunASR OpenAI 兼容 API 安全与网关指南

当你准备把示例 OpenAI 兼容 API 提供给团队、工作流引擎、浏览器 UI 或笔记本之外的服务时，请先看这份指南。示例服务刻意保持轻量，主要关注 `/v1/audio/transcriptions` 兼容性，本身不内置鉴权策略。

## 推荐拓扑

```text
OpenAI SDK / Dify / n8n / 浏览器 UI
        |
        v
TLS + 鉴权 + 上传限制 + 日志
(反向代理、API 网关、Ingress 或 Service Mesh)
        |
        v
FunASR OpenAI 兼容 API
(私有主机、虚机、容器或 Kubernetes ClusterIP)
```

只要条件允许，就让 FunASR 保持在私有网络中。公网 TLS、身份认证、请求限制和审计日志应放在团队已有的边界组件上。

## 对团队开放前的最低控制项

| 控制项 | 为什么重要 | 建议在哪里实现 |
|---|---|---|
| TLS | 音频通常包含隐私信息。 | 反向代理、API 网关或 Ingress。 |
| 鉴权 | 本地示例会接受任意 SDK `api_key` 占位符。 | 网关 Bearer token、Basic auth、OAuth/OIDC 或内部 SSO。 |
| 上传大小限制 | 避免误传超大文件导致内存和磁盘压力。 | 网关 request body limit 和应用侧检查。 |
| 超时 | 长音频需要更长 HTTP timeout，但异常客户端不能无限挂住。 | 客户端、代理和服务进程管理器。 |
| 限流 | 防止突发请求打满 GPU/CPU。 | 网关、Ingress controller 或队列 worker。 |
| 私有 `/health` | 健康信息是运维数据，不应作为公开产品端点。 | 网络 allowlist 或私有监控路径。 |
| 日志与留存 | 请求元数据有价值，但原始音频可能敏感。 | 集中日志策略和存储生命周期。 |

## NGINX 反向代理示例

下面只是起点，不是完整生产策略。证书、身份系统和密钥管理需要按你的环境补齐。

```nginx
server {
    listen 443 ssl http2;
    server_name funasr.example.com;

    client_max_body_size 200m;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;

    location / {
        # 在这里增加 auth_request、Basic auth、mTLS 或 API 网关策略。
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Authorization $http_authorization;
    }

    location = /health {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://127.0.0.1:8000/health;
    }
}
```

## Caddy 反向代理示例

使用 `caddy hash-password` 生成密码 hash，真实凭据不要写入仓库。

```caddyfile
funasr.example.com {
    request_body {
        max_size 200MB
    }

    basicauth /* {
        team_user <hashed-password>
    }

    reverse_proxy 127.0.0.1:8000 {
        transport http {
            read_timeout 600s
            write_timeout 600s
        }
    }
}
```

生产团队优先使用已有 SSO/OIDC 网关，而不是共享密码。

## Kubernetes 注意事项

Kubernetes 模板默认使用私有 `ClusterIP`。在增加 Ingress 或 LoadBalancer 前，请先完成：

- 使用 Ingress controller 或 API 网关强制 TLS、鉴权、上传大小限制和限流。
- 模型缓存卷只暴露给拥有该服务的 namespace 或 node pool。
- 使用 `NetworkPolicy` 限制可调用服务的 namespace。
- 第一次验证先用 `kubectl port-forward` 加 `smoke_test.py`，再开放路由。
- 如果增加 GPU，固定调度规则，并在部署说明中记录镜像 tag、CUDA runtime 和模型 alias。

## 客户端配置

OpenAI SDK 通常要求传入 API key 字符串，即使本地 FunASR 不检查它。加上网关后，请使用网关发放的 token 作为 SDK key：

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://funasr.example.com/v1",
    api_key=os.environ["FUNASR_API_KEY"],
)
```

内部 HTTP worker 应从环境变量或密钥系统读取 token。不要把 token 提交到工作流定义、notebook、截图或 Postman 导出文件里。

## 数据处理清单

- 先决定原始音频是否允许存储、保存多久、谁可以访问。
- 日志建议记录 request ID、音频时长、模型 alias、状态、延迟和错误类型；除非策略允许，不要记录原始转写文本。
- 如果转写文本可能包含个人信息，请在接入用户前写清留存和删除规则。
- 写 benchmark 报告时，把公开样例和客户/员工私有音频分开。
- 打开 GitHub issue 前，先脱敏 header、token、文件名和说话人姓名。

## 上线检查清单

1. 先在本地启动，并运行 `bash smoke_test.sh` 或 `python smoke_test.py`。
2. 增加网关后，通过网关 URL 验证 `/health`、`/v1/models` 和 `/v1/audio/transcriptions`。
3. 分别测试小文件、允许范围内的大文件，以及超过上传限制的文件。
4. 确认未授权请求会在到达 FunASR 前失败。
5. 确认长音频和慢客户端的 timeout 行为。
6. 记录模型 alias、设备、镜像 tag、FunASR 版本和网关策略。

相关文档：[OpenAI API README](README_zh.md)、[客户端配方](CLIENTS.md)、[工作流配方](WORKFLOWS_zh.md)、[Gradio 浏览器 Demo](GRADIO_zh.md)、[Kubernetes 模板](kubernetes/README_zh.md) 和仓库 [安全策略](../../SECURITY.md)。
