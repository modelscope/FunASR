# FunASR OpenAI 兼容 API Kubernetes 部署

这个目录提供 `examples/openai_api` 服务的 CPU-first Kubernetes 模板。适合在集群内部为 Agent、Web 后端、工作流引擎或批处理 worker 提供 OpenAI 兼容语音转写接口。

模板默认比较保守：

- 默认使用 `ClusterIP`，不直接暴露公网 `LoadBalancer`。
- 默认 `FUNASR_DEVICE=cpu`，与便携 Dockerfile 匹配。
- 在 `/root/.cache` 挂载持久化缓存卷，避免 Pod 重启后重复下载模型。
- 使用 `/health` 做 startup、readiness 和 liveness probe。
- 挂载内存型 `/dev/shm`，便于 PyTorch 和音频预处理使用。

## 1. 构建并推送镜像

在 `examples/openai_api` 目录中执行：

```bash
docker build -t registry.example.com/speech/funasr-api:cpu-latest .
docker push registry.example.com/speech/funasr-api:cpu-latest
```

如果使用不同的 registry、repo 或 tag，请修改 `kustomization.yaml`。

## 2. 部署

```bash
kubectl create namespace speech --dry-run=client -o yaml | kubectl apply -f -
kubectl -n speech apply -k .
kubectl -n speech rollout status deploy/funasr-api --timeout=15m
```

模型下载和首次加载可能需要几分钟。`startupProbe` 默认允许最多 10 分钟，超过后 Kubernetes 才会重启容器。

## 3. Smoke test

建议先保持服务内网私有，通过 `port-forward` 验证：

```bash
kubectl -n speech port-forward svc/funasr-api 8000:8000
```

在另一个终端进入 `examples/openai_api` 后运行：

```bash
python smoke_test.py --base-url http://localhost:8000
```

集群内客户端可以使用 `http://funasr-api.speech.svc.cluster.local:8000` 作为直接 HTTP base URL，使用 `http://funasr-api.speech.svc.cluster.local:8000/v1` 作为 OpenAI SDK base URL。

## 4. 根据集群调整配置

| 配置 | 默认值 | 什么时候调整 |
|---|---|---|
| `FUNASR_MODEL` | `sensevoice` | 调用 `/v1/models` 后按需要切换模型别名。 |
| `FUNASR_DEVICE` | `cpu` | 只有在镜像已适配 CUDA 且集群 GPU 调度已配置后才改成 `cuda`。 |
| PVC 大小 | `20Gi` | 缓存多个模型或较大模型版本时增大。 |
| 内存 request | `8Gi` | 根据启动过程和真实音频负载观测结果调整。 |
| Startup probe | 10 分钟 | registry、模型下载或存储后端较慢时增大。 |

## GPU 说明

示例 Dockerfile 默认面向 CPU。GPU 集群需要先把镜像改成 CUDA-capable PyTorch/FunASR 依赖，再根据集群增加 GPU 调度字段，例如：

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
nodeSelector:
  nvidia.com/gpu.present: "true"
```

不同 Kubernetes 发行版的 GPU label、runtime class 和 device plugin 配置并不相同。服务对外开放前，请先补齐鉴权、TLS、上传大小限制和限流。

## 运维检查

- 使用 `/health` 做就绪检查，使用 `/v1/models` 确认模型别名。
- 记录模型别名、设备、音频时长、响应格式、延迟和错误文本。
- 由于缓存 PVC 是 `ReadWriteOnce`，建议先从 1 个副本开始；横向扩容前先评估镜像、每 Pod 缓存或共享只读模型缓存方案。
- 服务暴露到可信 namespace 之外前，先加鉴权和网络策略。
- Dify、n8n 或 Web 后端在同一集群内访问时，应使用 Kubernetes service name，不要使用 `localhost`。
