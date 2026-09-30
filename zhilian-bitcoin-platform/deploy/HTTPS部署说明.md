# 智链公网 HTTPS 部署说明

本目录提供比赛演示用的 HTTPS 反向代理配置。应用容器只暴露给 Docker 内部网络，公网流量统一由 Caddy 接收，证书由 Caddy 自动申请和续期。

## 1. 前置条件

- 一台具有公网 IPv4 或 IPv6 的 Linux 云主机；
- 一个已经解析到该主机的域名，例如 `demo.example.com`；
- 云主机安全组和防火墙放行 TCP 80、443，以及可选的 UDP 443；
- 已安装 Docker Engine 和 Docker Compose Plugin；
- 不直接对公网开放应用端口 8765。

公网链接不能仅由项目代码生成。域名、DNS 和云主机属于部署环境，需由参赛者提供或在所选云平台创建。

## 2. 配置

在项目根目录复制 `.env.example` 为 `.env`，至少填写：

```text
PUBLIC_HOST=demo.example.com
BITCOIN_LIVE_ENABLED=1
BITCOIN_INFERENCE_DEVICE=cpu
DEMO_RATE_LIMIT_PER_MINUTE=120
DEMO_MAX_BODY_BYTES=16384
```

如启用在线大模型，只把密钥放在服务器 `.env` 中，不要提交到代码仓库，也不要写入浏览器端文件。

## 3. 启动

在项目根目录执行：

```bash
docker compose -f deploy/docker-compose.https.yml up -d --build
```

Caddy 首次启动后会自动为 `PUBLIC_HOST` 申请公开可信证书。证书申请成功后，演示地址为：

```text
https://demo.example.com/
```

检查服务：

```bash
curl -fsS https://demo.example.com/api/health
docker compose -f deploy/docker-compose.https.yml ps
docker compose -f deploy/docker-compose.https.yml logs --tail=100 caddy
```

## 4. 安全约束

- 只对外开放 80/443，应用容器端口 8765 仅使用 `expose`；
- 应用服务以非 root 用户运行，并启用只读根文件系统；
- 服务端限制 JSON 请求体大小和每个客户端的 API 请求频率；
- 服务端不会把地址、交易哈希或查询参数写入访问日志；
- 应用和 Caddy 都设置安全响应头；
- 在线大模型密钥只在服务端读取；
- 正式演示前建议在云主机层再增加防火墙、备份和访问审计；
- 如果希望限制评委之外的访问，建议在 Caddy 或云平台增加 Basic Auth、Cloudflare Access 或 VPN，不要把访问口令硬编码进前端。

## 5. 下线与更新

```bash
docker compose -f deploy/docker-compose.https.yml pull
docker compose -f deploy/docker-compose.https.yml up -d --build
docker compose -f deploy/docker-compose.https.yml down
```

Caddy 的 `caddy_data` 卷保存证书和自动续期状态，删除该卷会导致证书重新申请。
