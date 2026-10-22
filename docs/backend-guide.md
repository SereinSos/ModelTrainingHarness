# 通用接口与可替换后端说明

本文档解释 harness 系统中 5 个通用接口各自的作用，以及每个接口可替换的具体后端是什么。

---

## 1. VectorStore — 向量数据库

**作用**：存储 Embedding 模型生成的向量，并提供相似性检索能力。
流水线中负责"存储侧分数"：把文档向量灌进去，再用查询向量检索，衡量检索质量。

| 后端 | 是什么 | 部署方式 | 适用场景 |
|------|--------|---------|---------|
| `memory` | 纯内存字典，无持久化 | 零依赖，进程内 | 开发调试、MVP 验证 |
| `milvus` | 开源向量数据库，Go 编写，存算分离架构 | 需独立部署服务（Docker/K8s） | 大规模生产（亿级向量），国内生态好 |
| `qdrant` | 开源向量搜索引擎，Rust 编写，过滤能力突出 | Docker 或嵌入式模式 | 需要复杂标量+向量组合过滤 |
| `weaviate` | 开源向量数据库，GraphQL 接口，内置向量化模块 | Docker/K8s，支持多租户 | 企业级场景，需要 RBAC 权限控制 |
| `chromadb` | 轻量嵌入式向量数据库，API 极简（4 个核心函数） | pip 安装即用，也可部署 Server | 快速原型、LangChain 生态默认搭配 |
| `pgvector` | PostgreSQL 扩展，向量和业务数据同表存储 | 装 Postgres + pgvector 扩展 | 已有 PG 基础设施，不想多引入一个数据库 |
| `seekdb` | AI-Native 搜索数据库，支持向量+全文+标量混合检索 | Docker 或嵌入式 | 需要多模态检索（向量+全文+GIS） |

**SeekDB 适配器详情**：
- `SeekDBVectorStore` — 向量检索（实现 VectorStore Protocol）
- `SeekDBFullTextSearch` — 全文检索（MATCH...AGAINST）
- `SeekDBScalarSearch` — 普通 SQL 查询
- `SeekDBGISSearch` — GIS 地理检索（ST_Distance_Sphere, ST_Within）
- `SeekDBHybridSearch` — 混合检索（向量+全文+标量过滤组合）

**切换方式**：修改 `config/harness.yaml` 中 `vector_store.type`，并填写对应的 `connection` 字段。

---

## 2. HPCBackend — 超算/集群后端

**作用**：把训练任务打包、上传到远程计算节点执行、拉回结果。
流水线中负责"超算训练"环节。

| 后端 | 是什么 | 作业调度命令 | 适用场景 |
|------|--------|------------|---------|
| `local` | 本地执行，不上传远程 | 直接 subprocess 运行 | 开发调试、单机训练 |
| `slurm` | SLURM 作业调度系统，全球超算中心最主流 | `sbatch` 提交 / `squeue` 查询 / `scancel` 取消 | 大多数超算集群（清华、国家超算等） |
| `pbs` | PBS/Torque 作业调度系统，较老但仍有大量集群在用 | `qsub` 提交 / `qstat` 查询 / `qdel` 取消 | 部分高校和科研院所的老集群 |
| `ssh` | 通过 SSH 连接远程服务器直接执行 | `scp` 传文件 + `ssh` 运行命令 | 没有调度系统的裸机服务器 |

**容器运行时**：超算环境通常用 **Apptainer**（原 Singularity）运行容器，因为：
- 不需要 root 权限（Docker 需要）
- 原生支持 GPU 直通（`--nv` 参数）
- 与 SLURM/PBS 无缝集成

**切换方式**：修改 `hpc.backend`，填写 `host`、`user`、`ssh_key` 等连接信息。

---

## 3. EmbeddingModel — 向量化模型

**作用**：把文本转换为高维向量（Embedding）。
流水线中负责"应用侧算向量"环节。

| 后端 | 是什么 | 依赖 | 适用场景 |
|------|--------|------|---------|
| `stub` | 确定性随机向量，不真正编码语义 | 无 | 开发调试 |
| `sentence_transformer` | 基于 sentence-transformers 库加载本地模型 | `pip install sentence-transformers` | 本地推理、离线环境、自定义微调模型 |
| `openai` | 调用 OpenAI Embedding API | `pip install openai` + API Key | 不想管模型部署、直接用云端服务 |
| `custom_api` | 调用自定义 HTTP 接口 | 无（标准 HTTP） | 你们有自己的 Embedding 服务 |

**常见模型举例**（sentence-transformer 可用）：
- `BAAI/bge-small-zh-v1.5` — 中文，轻量，512 维
- `BAAI/bge-large-zh` — 中文，高精度，1024 维
- `sentence-transformers/all-MiniLM-L6-v2` — 英文，轻量，384 维

**切换方式**：修改 `embedding.type` 和 `embedding.model_name_or_path`。

---

## 4. DatasetProvider — 数据集加载

**作用**：提供评测用的数据集（文本 + 标签/查询 + 语料库）。
流水线中负责"本地快评"环节的数据来源。

| 后端 | 是什么 | 依赖 | 适用场景 |
|------|--------|------|---------|
| `stub` | 内置硬编码的小数据集 | 无 | 开发调试 |
| `mteb` | MTEB（Massive Text Embedding Benchmark），标准 Embedding 评测基准 | `pip install mteb` | 跑标准评测分数、与其他模型对比 |
| `huggingface` | 从 HuggingFace Hub 加载任意数据集 | `pip install datasets` | 使用自定义/私有数据集 |
| `custom` | 从本地 JSON/CSV 加载 | 无 | 你们自己的业务数据 |

**MTEB 支持的任务类型**：
- `STS` — 语义文本相似性
- `Retrieval` — 文本检索
- `Classification` — 文本分类
- `Clustering` — 文本聚类
- `Reranking` — 重排序
- `PairClassification` — 句对分类

**切换方式**：修改 `evaluation.provider` 和 `evaluation.tasks`。

---

## 5. ExperimentTracker — 实验追踪

**作用**：记录每次实验的参数、指标、产物，便于对比和复现。
流水线中负责"MLflow 存档"环节。

| 后端 | 是什么 | 依赖 | 适用场景 |
|------|--------|------|---------|
| `none` | 只打印到终端，不持久化 | 无 | 开发调试 |
| `mlflow` | 开源 ML 实验管理平台，支持参数/指标/产物/模型注册 | `pip install mlflow` | 标准 MLOps 流程，团队协作 |
| `wandb` | Weights & Biases，云端实验追踪+可视化 | `pip install wandb` + API Key | 需要漂亮的可视化仪表盘 |
| `tensorboard` | Google 的训练可视化工具 | `pip install tensorboard` | 已在用 TensorBoard 的团队 |

**切换方式**：修改 `tracking.backend`，填写 `tracking_uri` 等连接信息。

---

## 总结：当前已接入的真实实现

```yaml
vector_store.type: chromadb          # 真实嵌入式向量数据库
hpc.backend: local                   # 本地，零依赖
embedding.type: sentence_transformer # 真实 Embedding 模型 (all-MiniLM-L6-v2)
evaluation.provider: mteb            # 真实 MTEB 数据集 (BIOSSES, 100 条)
tracking.backend: none               # 终端打印，零依赖
```

流水线已使用真实后端跑通端到端：
- **ChromaDB** 持久化存储向量到 `./data/chroma_db/`
- **SentenceTransformer** 生成 384 维真实语义向量
- **MTEB BIOSSES** 提供 100 条生物医学句子相似性评测数据
- 存储侧分数 `avg_top1_similarity` 反映真实检索质量
