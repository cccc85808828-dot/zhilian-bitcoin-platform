# GitHub 上传版说明

请将本目录中的全部内容上传到 GitHub 仓库根目录，而不是只上传某一个子目录。

仓库根目录上传完成后，应能直接看到：

- `README.md`
- `启动竞赛演示.cmd`
- `run_offline.ps1`
- `portable_offline_server.ps1`
- `demo/`
- `src/`
- `scripts/`
- `artifacts/`

评审运行时，下载仓库后进入根目录，双击 `启动竞赛演示.cmd` 即可开始离线演示。

不要上传 `.env` 或任何真实 API 密钥；仓库只保留 `.env.example`。在线研判是可选模式，默认竞赛演示不需要 Python、Docker 或联网。
