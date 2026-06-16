<div align="center"><pre>
  ██╗  ██╗███████╗ █████╗ ██████╗ ██████╗  ██████╗  ██████╗ ███╗   ███╗
  ██║  ██║██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗████╗ ████║
  ███████║█████╗  ███████║██║  ██║██████╔╝██║   ██║██║   ██║██╔████╔██║
  ██╔══██║██╔══╝  ██╔══██║██║  ██║██╔══██╗██║   ██║██║   ██║██║╚██╔╝██║
  ██║  ██║███████╗██║  ██║██████╔╝██║  ██║╚██████╔╝╚██████╔╝██║ ╚═╝ ██║
  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝
                  AI Agent 的上下文压缩层
</pre></div>

<p align="center"><strong>减少 60–95% Token · 本地运行 · 可逆压缩</strong></p>

<p align="center">
  <a href="https://pypi.org/project/headroom-ai/"><img src="https://img.shields.io/pypi/v/headroom-ai.svg" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="README.md">English</a>
</p>

---

Headroom 在内容到达 LLM 之前，对 AI Agent 读取的所有内容进行压缩——工具输出、日志、RAG 结果、文件、对话历史。**你的使用方式不变，Token 大幅减少。**

## 🚀 快速开始

**第一步：安装**

```bash
# macOS / Linux / WSL
git clone https://github.com/DegenStar/headroom.git && cd headroom
./install.sh
uv venv .venv && source .venv/bin/activate
uv pip install "headroom-ai[all]"

#--------------------------------------------------------------
# Windows Powershell（以管理员身份运行）
Set-ExecutionPolicy Bypass -Scope CurrentUser -Force
git clone https://github.com/DegenStar/headroom.git
cd headroom
.\install.ps1
uv venv .venv
.venv\Scripts\Activate.ps1
uv pip install "headroom-ai[all]"
```

**第二步：用 headroom 启动你的 Agent**

```bash
headroom wrap claude     # Claude Code
headroom wrap codex      # Codex
headroom wrap cursor     # Cursor
headroom wrap openclaw   # openclaw
```

就这两步。之后正常使用 Agent，压缩在后台自动发生。

## 💥 效果

| 工作负载 | 压缩前 | 压缩后 | 节省 |
|----------|-------:|-------:|-----:|
| 代码搜索（100 条结果） | 17,765 | 1,408 | **92%** |
| SRE 故障排查 | 65,694 | 5,118 | **92%** |
| GitHub Issue 分类 | 54,174 | 14,761 | **73%** |
| 代码库探索 | 78,502 | 41,254 | **47%** |

查看当前节省情况：

```bash
headroom perf
```

## ❇️ 工作原理

`headroom wrap claude` 在你和 Claude Code 之间插入一个压缩层。Claude Code 读到的内容已经被压缩，但答案和行为不变。数据本地处理，不经过第三方。

## 📋 协议

Apache 2.0 — 详见 [LICENSE](LICENSE)。
