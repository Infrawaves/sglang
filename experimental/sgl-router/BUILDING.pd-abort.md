# PD abort

基线：`online_base_0908`（`5930959b923fc7ed9a6f63b4b2dc1e5bad5e6358`）。

Prefill 失败时及时返回错误，并向本次选中的 Decode 发送携带同一 `rid` 的
`POST /abort_request`，避免继续等待 KV bootstrap 超时。Prefill 4xx 保持原状态码，
5xx 映射为 502。Decode 先返回非 2xx 时保留原响应。
`sgl_router_pd_decode_abort_requests_total{outcome="success|failed"}` 记录取消 RPC 结果；
success 表示 Worker 接受 RPC，不代表已经验证 GPU 内存释放。

原有 `--no-tokenizer`、`--worker-urls`、`--max-chat-body-bytes`、访问日志和错误体截断保持基线行为。

## 从本分支的提交构建

在完整 SGLang 仓库根目录执行。`git archive` 只发送指定提交里的 Router 源码，避免混入旧目录或未提交文件。

```bash
git switch feat/router-pd-abort-online-0908
git merge-base --is-ancestor 5930959b923fc7ed9a6f63b4b2dc1e5bad5e6358 HEAD
REV=$(git rev-parse HEAD)
IMAGE=harbor.local.clusters/sglang/sglang:sgl-router-venus-online0908-pd-abort-${REV:0:8}

set -o pipefail
git archive --format=tar "$REV" experimental/sgl-router | docker build \
  --network=host \
  -f experimental/sgl-router/Dockerfile.pd-abort \
  --build-arg SOURCE_REVISION="$REV" \
  --build-arg HTTP_PROXY=http://172.18.1.65:7890 \
  --build-arg HTTPS_PROXY=http://172.18.1.65:7890 \
  --build-arg ALL_PROXY=socks5://172.18.1.65:7890 \
  --build-arg NO_PROXY=localhost,127.0.0.1,172.18.0.0/16,10.0.0.0/8,.svc,.cluster.local \
  -t "$IMAGE" -
docker push "$IMAGE"
```

镜像的 `/usr/local/bin/sgl-router` 即新二进制；创建持久容器后手动运行 `sgl-router` 使用同一版本。
镜像标签 `org.opencontainers.image.revision` 记录源码提交。运行时无需增加 PD abort 开关。
两个 Worker 都无法读取同一图片时，Decode 可能先返回 500；应使用独立 Mock Worker 的
Prefill 失败、Decode 等待场景验证 abort，避免用线上 Worker 注入故障。
