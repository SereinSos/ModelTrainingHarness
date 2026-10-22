# Model Training Harness 使用指南

## 快速开始

### 1. 安装

```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync

# 安装可选依赖（按需）
uv add sentence-transformers  # 真实 Embedding 模型
uv add mteb                   # MTEB 评测数据集
uv add chromadb               # ChromaDB 向量库
uv add mlflow                 # MLflow 实验追踪
```

### 2. 配置

编辑 `config/harness.yaml`，选择你要使用的后端：

```yaml
# 向量数据库
vector_store:
  type: chromadb              # memory | chromadb | milvus | qdrant | ...
  connection:
    path: ./data/chroma_db

# Embedding 模型
embedding:
  type: sentence_transformer  # stub | sentence_transformer | openai | ...
  model_name_or_path: sentence-transformers/all-MiniLM-L6-v2

# 数据集
evaluation:
  provider: mteb              # stub | mteb | huggingface | ...
  tasks:
    - name: BIOSSES
      languages: [en]
```

### 3. 运行

```bash
# 预检
make doctor

# 查看已注册的后端
make registry

# 运行完整流水线
make run

# 运行基线（跳过超算）
make baseline

# 本地评测
make eval
```

---

## 核心概念

### 流水线流程

```
doctor → eval → HPC → vector → scoring → tracking → report
  │        │      │       │         │           │          │
  │        │      │       │         │           │          └─ 生成 text+json 报告
  │        │      │       │         │           └─ 记录实验到 MLflow/WandB
  │        │      │       │         └─ 计算模型侧+存储侧分数
  │        │      │       └─ 向量灌入 ChromaDB/Milvus
  │        │      └─ 打包上传超算训练（可选）
  │        └─ MTEB 评测 Embedding 质量
  └─ 检查容器/数据库/超算连通性
```

### 两层分数

- **模型侧分数**：MTEB 评测结果，衡量 Embedding 质量（如 STS、Classification 任务得分）
- **存储侧分数**：向量检索质量，衡量数据库检索效果（如 avg_top1_similarity）

### 通用接口

所有外部依赖都通过 `Protocol` 抽象，可在 `config/harness.yaml` 中切换：

| 接口 | 作用 | 可选后端 |
|------|------|---------|
| `VectorStore` | 向量存储与检索 | memory, chromadb, milvus, qdrant, seekdb, ... |
| `HPCBackend` | 超算任务提交 | local, slurm, pbs, ssh |
| `EmbeddingModel` | 文本向量化 | stub, sentence_transformer, openai, custom_api |
| `DatasetProvider` | 评测数据集加载 | stub, mteb, huggingface, custom |
| `ExperimentTracker` | 实验追踪记录 | none, mlflow, wandb, tensorboard |

---

## CLI 命令详解

### `harness doctor`

预检系统状态，检查：
- 容器镜像是否存在
- 向量数据库是否可连接
- 超算后端是否可达
- 评测数据集是否可用

```bash
$ make doctor
  [PASS] container_image: Found: /opt/containers/skdb_latest.sif
  [PASS] vector_store: Connected (chromadb)
  [PASS] hpc_connectivity: Backend: local
  [PASS] datasets: All 1 tasks available (mteb)

All checks passed.
```

### `harness eval`

运行本地快速评测，输出 MTEB 分数：

```bash
$ make eval
Model: sentence-transformers/all-MiniLM-L6-v2
Elapsed: 2.3s
  BIOSSES: 0.8734
```

### `harness run`

运行完整流水线（doctor → eval → HPC → vector → scoring → report）：

```bash
$ make run
[tracker:stub] start_run: full-20261022-110306
[tracker:stub] metric: model/BIOSSES=0.8734
[tracker:stub] metric: storage/avg_top1_similarity=0.6840
Pipeline complete. Report: [reports/report_20261022_110306.txt, reports/report_20261022_110306.json]
```

### `harness baseline`

运行基线流程（跳过 HPC 超算训练）：

```bash
$ make baseline
```

### `harness registry`

查看当前已注册的所有后端实现：

```bash
$ make registry
vector_store: memory, chromadb
hpc_backend: local
embedding: stub, sentence_transformer
dataset_provider: stub, mteb
tracker: none
```

---

## 配置详解

### 向量数据库配置

#### ChromaDB（嵌入式，推荐入门）

```yaml
vector_store:
  type: chromadb
  connection:
    path: ./data/chroma_db    # 本地持久化路径
  collection:
    name: harness_vectors
    dimension: 384            # 与 Embedding 模型维度匹配
    metric: cosine            # cosine | l2 | ip
```

#### Milvus（分布式，生产级）

```yaml
vector_store:
  type: milvus
  connection:
    host: localhost
    port: 19530
  collection:
    name: harness_vectors
    dimension: 1024
    metric: cosine
```

#### InMemory（开发调试）

```yaml
vector_store:
  type: memory
  connection: {}
  collection:
    name: test
    dimension: 384
```

### Embedding 模型配置

#### SentenceTransformer（本地模型，推荐）

```yaml
embedding:
  type: sentence_transformer
  model_name_or_path: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 32
  device: cpu                 # cpu | cuda:0 | cuda:1
```

常用模型：
- `sentence-transformers/all-MiniLM-L6-v2` — 英文，384 维，轻量
- `BAAI/bge-small-zh-v1.5` — 中文，512 维
- `BAAI/bge-large-zh` — 中文，1024 维，高精度

#### Stub（开发调试）

```yaml
embedding:
  type: stub
  model_name_or_path: dummy
```

### 数据集配置

#### MTEB（标准评测基准）

```yaml
evaluation:
  provider: mteb
  tasks:
    - name: BIOSSES           # 100 条样本，最小
      languages: [en]
    - name: STS22
      languages: [en, zh]
  dataset_cache_dir: ./data/mteb_cache
  output_dir: ./results/eval
```

常用任务（按样本量排序）：
- `BIOSSES` — 100 条，STS 任务
- `STS12` — 3108 条，STS 任务
- `Banking77Classification` — 分类任务

#### Stub（开发调试）

```yaml
evaluation:
  provider: stub
  tasks:
    - name: STS22
```

### 超算后端配置

#### Local（本地模拟）

```yaml
hpc:
  backend: local
  remote_dir: /tmp/hpc_work
  resources:
    nodes: 1
    gpus: 0
```

#### SLURM（生产超算）

```yaml
hpc:
  backend: slurm
  host: hpc.cluster.example.com
  user: researcher
  ssh_key: ~/.ssh/hpc_ed25519
  remote_dir: /scratch/researcher/training
  resources:
    nodes: 4
    ntasks_per_node: 8
    gpus: 8
    memory: "256G"
    time_limit: "72:00:00"
    partition: gpu
  container_bind: /data:/mnt
```

### 实验追踪配置

#### MLflow

```yaml
tracking:
  backend: mlflow
  tracking_uri: http://localhost:5000
  experiment_name: model-training-harness
```

启动 MLflow 服务：

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

#### None（不追踪）

```yaml
tracking:
  backend: none
```

---

## 报告解读

### 文本报告（reports/report_*.txt）

```
============================================================
Model Training Evaluation Report
============================================================
Run ID:    stub-run-000
Timestamp: 2026-10-22T11:03:06.419378

── Model-side Scores ──────────────────────────────
  BIOSSES                                  0.8734    # MTEB 评测分数

── Storage-side Scores ────────────────────────────
  avg_top1_similarity                      0.6840    # 向量检索质量
  total_vectors                            50.0000   # 存储的向量数
```

### JSON 报告（reports/report_*.json）

```json
{
  "run_id": "stub-run-000",
  "timestamp": "2026-10-22T11:03:06.419378",
  "model_scores": [
    {
      "task": "BIOSSES",
      "score": 0.8734,
      "details": {"source": "mteb"}
    }
  ],
  "storage_scores": [
    {
      "metric": "avg_top1_similarity",
      "score": 0.6840,
      "details": {"num_queries": 2, "store_count": 50}
    }
  ]
}
```

---

## 常见问题

### Q: 如何切换向量数据库？

修改 `config/harness.yaml` 中的 `vector_store.type`，然后运行 `make doctor` 验证连通性。

### Q: 如何使用中文 Embedding 模型？

```yaml
embedding:
  type: sentence_transformer
  model_name_or_path: BAAI/bge-small-zh-v1.5
```

注意同步修改 `vector_store.collection.dimension` 为 512。

### Q: 如何添加新的评测任务？

```yaml
evaluation:
  provider: mteb
  tasks:
    - name: BIOSSES
    - name: STS12
    - name: Banking77Classification
```

### Q: 如何跳过超算训练？

使用 `make baseline` 而非 `make run`。

### Q: 报告在哪里？

`./reports/` 目录，每次运行生成带时间戳的 text 和 json 文件。

### Q: 如何清理运行时产物？

```bash
make clean
```

---

## 开发指南

### 添加新的后端实现

1. 在对应目录创建文件（如 `src/harness/vector/mydb.py`）
2. 实现接口方法
3. 用装饰器注册：

```python
from harness.factories import register_vector_store

@register_vector_store("mydb")
class MyDBVectorStore:
    def connect(self) -> None: ...
    def store(self, vectors, metadata) -> None: ...
    def search(self, query, top_k) -> list[dict]: ...
    # ... 其他方法
```

4. 在 `factories.py` 的 `_ensure_loaded()` 中添加 import
5. 更新 `config/harness.yaml` 使用新后端

### 运行测试

```bash
make test
# 或
uv run pytest -v
```

### 代码格式化

```bash
make format
make lint
```

---

## 架构文档

- [MVP 架构图](./mvp-architecture.md)
- [后端接口说明](./backend-guide.md)

---

## 许可

MIT
