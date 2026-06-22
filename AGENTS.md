# headroom

Headroom — 开源 agent 工程平台。Rust + Python 混合项目，提供 agent 编排、内存管理、插件系统。

## 技术栈

- Rust (Cargo.toml, crates/)
- Python (pyproject.toml, uv)
- Docker (Dockerfile, docker-compose.yml)
- MkDocs (mkdocs.yml)

## 目录结构

| 目录 | 用途 |
|------|------|
| `crates/` | Rust crate 源码 |
| `headroom/` | Python 包 |
| `sdk/` | SDK |
| `plugins/` | 插件 |
| `benchmarks/` | 性能基准测试 |
| `e2e/` | 端到端测试 |
| `tests/` | 测试 |
| `examples/` | 示例 |
| `docs/` | 文档 |
| `wiki/` | Wiki |
| `scripts/` | 脚本 |
| `sql/` | SQL 文件 |
| `docker/` | Docker 配置 |

## 关键文件

- `README.md` — 项目说明
- `CONTRIBUTING.md` — 贡献指南
- `CHANGELOG.md` — 变更日志
- `Makefile` — 构建命令
- `llms.txt` — LLM 友好说明

## 清理提示

- `target/` (Rust 构建缓存) 可清理，体积大
- `.venv/` 可清理
- `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/` 可清理
- `headroom_memory.db`, `headroom_memory_vectors.db` 是运行时数据库
