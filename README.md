# minibrain

最小知识中台。**两条范式不同的产品检索链路，各自独立存储和判权限；一个带持久化会话的 Agent 统一编排。**

## 文档导航

| 想了解什么 | 从这里开始 |
|---|---|
| 项目定位、安装和评测命令 | 当前 README |
| 完整架构、数据流、模块职责和技术取舍 | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| 高频 RAG 面试问题与项目化答案 | [`docs/INTERVIEW.md`](docs/INTERVIEW.md) |
| 2 分钟介绍、5 分钟演示和面试准备计划 | [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) |
| 扩展到真实规模时的边界 | [`SCALING.md`](SCALING.md) |
| 实验过程和完整结果 | [`eval/RESULTS.md`](eval/RESULTS.md) |

第一次阅读建议按 `README → ARCHITECTURE → INTERVIEW → DEMO_GUIDE` 的顺序。功能状态和运行方式放 README；架构文档只讲当前真实实现；面试文档用于回答问题；历史探索和候选功能仍放在 `docs/BACKLOG.md`，不混入面试主线。

不是"又一个 RAG demo"——一条链路的 RAG 没有边界需要验证。这个项目要证明的是边界本身。

```text
用户提问（FastAPI + htmx，SSE 状态/心跳/最终安全答案）
  └─ LangGraph Agent（PostgreSQL checkpointer，会话锁 + 请求幂等 + 知识域路由 + 关系链多跳）
       ├─ vector-rag  Markdown 标题 → 800 child 召回 → pgvector + BM25 → RRF
       │              索引 manifest → 文档内章节检索 → parent_id 去重 → token budget → 权限安全短缓存
       └─ table-rag   CSV/XLSX → 物理表 → 受限只读 SQL             "华东区销售额合计是多少？"
       └─ Claim 引用  E1.1 → 结构化 claims → ID / 数字 / 编号确定性校验 → 安全拒答
       └─ Run Trace   候选取舍 / 证据 / claims / 指标 / 延迟 / token → PostgreSQL
```

### 当前主路径与实验边界

这个仓库刻意保留了三种实现角色，但它们不是三条同等级的产品链路：

| 角色 | 实现 | 当前用途 |
|---|---|---|
| **产品主路径** | `agent/graph.py` + `modules/vector_rag/` + `modules/table_rag/` | Web 问答；LangGraph 会话、LlamaIndex 混合检索、表格 SQL |
| **手写基线** | `handwritten/` | 讲清 tool loop / BM25 / RRF 的底层机制，并给框架迁移提供可比基线 |
| **GraphRAG 实验** | `modules/graph_rag/` | 在锁定语料和 embedding 后做受控对比；未接入 gateway / Web，不是生产链路 |

向量链路默认走全库向量 + 持久化 BM25 的 RRF 混合检索，并做确定性去重；MMR 与
cross-encoder 都保留为显式实验开关、默认关闭。MMR 关闭是公开集实测结论，不是省事。
CE 关闭同样是消融结论：排序指标提升，但 40 候选下检索 P50 增加约 1.45 秒且 context 变长。
LLM listwise rerank 的历史实验保留在评测报告中，但运行时代码已移除，避免在最终回答前再调用一次大模型。
切分采用结构感知的 small-to-big：Markdown 标题层级形成章节边界并写入 metadata，
800 字 child 负责 dense/sparse 召回，1600 字 parent 独立持久化且不做 embedding。
58 题受控消融证明收益来自 parent-aware child 去重（Recall@5 0.847→0.886，token 不增加），
不是扩大正文；完整 parent 展开增加 469 token 且关键答案支持零提升，因此默认关闭、保留实验开关。
GraphRAG 当前使用本地持久化图，尚未解决多用户图权限和并发写，
简历和演示中应称为 **GraphRAG 对照实验**。

产品路径还包含四个工程边界：同一文件重复上传用内容 hash 直接复用，内容变化时原
`document_id` 升版本并清理旧索引；连续 Markdown 标题不再产生标题空块；命中长文档后
可用精确 `filename` 在权限过滤内做章节二次检索；检索缓存键绑定用户、管理员身份、
可见知识版本和全部检索参数。检索片段统一标为不可信数据，疑似 prompt injection 只作
事实材料而不执行。以上都不增加额外 Planner / evidence-selector LLM 调用。

第二条链路是表格而不是图谱，是刻意的：**一张报表切碎再向量召回回来，算不出正确的合计。**
这是"两条链路不能合并"最无可辩驳的证据，而且它不需要 LLM 抽取、不需要图数据库、不需要第二种语言。

> 第一次读这个仓库，先跑这个——它把两条链路每一步的中间产物原样打印出来，
> 包括"把表格误传进文档链路会被切成什么样"：
>
> ```sh
> uv run --no-sync python scripts/explain.py
> ```

## 跑起来

前置：PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) 扩展（`brew install pgvector`）、[uv](https://docs.astral.sh/uv/)。

```sh
uv sync
cp .env.example .env          # 填 DATABASE_URL；两个 API key 先不填也能跑通表格链路
uv run minibrain-init-db      # 建库 + 跑 schema.sql，幂等
uv run minibrain-create-user alice alice123   # 第一个用户自动是管理员
uv run minibrain-seed-demo alice              # 可选：幂等加载内置演示文档和销售表
```

再开两个终端。Web 只负责登记上传，worker 负责可恢复的异步入库：

```sh
uv run minibrain-worker
uv run uvicorn minibrain.web.app:app --reload --port 8000
```

首页中的 ready 文档可以点击“生成 Wiki”。每次编译会保留一篇带 source、文件名和版本的
来源摘要页，并最多新建或增量更新 3 个可复用主题页；主题页通过多对多关系记录全部原文来源，
还可以建立显式主题关联。解析原文按版本不可变保存，Wiki 可以打开编译时的确切版本；
原文更新后，来源页和受影响主题会变成 `stale`，不再参与问答。
Wiki 页的“向 Wiki 提问”会先用本地 BM25 在可见且版本最新的来源页与主题页中选择 Top 3，
再调用回答模型综合并列出 W 编号引用。首页同时确定性检查过期来源、失败页面、无来源主题和
孤立主题，并记录编译、查询与显式 lint。这是一条独立 MVP 链路，不会影响原有 RAG 入库与问答。

已有数据库从旧版升级时，应用新 schema 并为现有节点回填 sparse 索引：

```sh
uv run minibrain-init-db
uv run minibrain-reindex-lexical  # 不重新切分，不调用 embedding
```

升级前已入库的节点没有 `parent_context_id`，检索时会自动按旧 chunk 返回。
要让旧文档使用结构感知父子切分，在知识库页面点该文档的“重建”（会重新调用 embedding）；
`minibrain-reindex-lexical` 只重建 BM25，不会改变切分结构。

打开 http://127.0.0.1:8000 。上传 `.csv` / `.xlsx` 走表格链路；纯文本、Markdown、HTML、
DOCX、PPTX 和带文字层的 PDF 走文档链路。DOCX 会按原始块顺序转成 Markdown，保留
Heading 1–6、列表和表格；HTML 会删除脚本/样式并提取可见正文；PPTX 按幻灯片顺序提取
文字。旧 `.doc`、图片、扫描 PDF 和纯图片 PPTX 会明确落 `failed`。XLSX 按 OOXML 内容自动路由，
每个可见 Sheet 建一张独立物理表；隐藏 Sheet 不导入，未计算公式明确失败。

不填 API key 时：表格链路完整可用（不需要模型），文档链路的文档会落 `failed` 并写明原因，
提问会提示 `agent_not_configured`。**这是设计好的降级，不是坏掉了。**

### Docker Compose 一键启动

机器上只需 Docker。先准备密钥配置；`DATABASE_URL` 在容器内会被 Compose 自动覆盖：

```sh
cp .env.example .env
docker compose up --build -d
docker compose exec web minibrain-create-user alice alice123
```

打开 http://127.0.0.1:8000 。Compose 会先启动带 pgvector 的 PostgreSQL，等待健康后
幂等执行 `minibrain-init-db`，成功后才启动 Web 和 Worker，避免 Worker 抢在建表前领取任务。
运行状态和日志可用下面两条命令检查：

```sh
docker compose ps
docker compose logs -f web worker
```

PostgreSQL 数据保存在具名卷 `minibrain_postgres`；普通 `docker compose down` 不会删除它。
`docker compose down -v` 会永久删除知识库和用户数据，不应作为日常停止命令。

## 测试

```sh
uv run pytest            # 集成测试需要 PostgreSQL + pgvector
uv run pytest -k 权限     # 或 guardrail / cleanup
```

**不 mock 数据库**——要验证的恰恰是 SQL 里的权限过滤和只读事务，
mock 掉数据库等于把被测对象本身删了。所以测试建的是真实用户和真实数据，
session fixture 结束时统一清理，`test_cleanup.py` 专门验证清理本身。

测试数量会随功能增长，以 `uv run pytest --collect-only` 和 CI 结果为准，不在 README 固化
`passed / skipped` 快照。配了 key 与没配 key 的两条分支都要覆盖：没配 key 时文档必须落
`failed` 并写明原因，不许假装 ready。CI 刻意不注入密钥，所以那条降级路径由 CI 守门
（本地有 key 反而测不到）。

测试有没有牙，是验证过的：把 `_visibility_clause()` 的非管理员分支改成永真
（模拟"漏一个分支"这类典型越权 bug），3 个权限测试立刻失败。

公开检索报告还经过独立 CI 门禁。修改检索策略后先本地重跑 NanoBEIR，并将新报告作为
candidate 提交；CI 检查 NanoNQ / NanoHotpotQA 的 MRR、Recall@5、Complete Recall@5、
NDCG@10 与检索 P50，超过预算直接失败。CI 本身不调用付费 embedding API：

```sh
uv run --extra public-eval python scripts/eval_nanobeir.py \
  --output eval/results/nanobeir-global-bm25.json
uv run python scripts/check_eval_regression.py
```

如果异常中断留下了残渣：

```sh
uv run minibrain-purge --list     # 先看会删什么
uv run minibrain-purge            # 清掉 smoke_ / live_ 前缀的用户及其全部数据
```

索引一致性审计默认只读；确认后才显式清理孤儿节点：

```sh
uv run minibrain-index-audit
uv run minibrain-index-audit --prune-orphans
```

Web 页右上角的“运行记录”按用户隔离展示每次问答的工具轨迹、证据、延迟、token
和稳定错误信息；管理员可查看全局记录。它不保存模型内部推理。

每次成功回答可提交有帮助/待改进反馈；差评支持原因标签和备注，并可从运行记录页导出
带问题、答案、证据、引用和模型版本的回归候选 JSON。导出项刻意标记为需人工审核，
补齐期望答案后才应进入正式评测集。右上角“监控指标”按权限聚合成功率、无证据率、
p50/p95 延迟、token、引用覆盖、差评和失败类型。完整会话仍持久化在 PostgreSQL，
但单次模型请求只携带最近 6 轮且受 8000-token 预算约束；裁剪不额外调用摘要模型。
问答页面使用 POST SSE 发送 started、heartbeat、completed、failed 事件。当前不直接流出原始
模型 token：最终结构化 claims 必须先通过引用有效性与 grounding 检查，浏览器才收到答案，
避免已展示的无依据内容无法撤回。

右上角的“检索实验室”用于解释单次文档检索：它直接执行正式检索链路，但不调用
Agent/生成模型，逐阶段展示 Dense、BM25、RRF、确定性去重、可选 cross-encoder、
可选 MMR 和 parent 展开的候选排名、分数与耗时。结果仍使用当前用户的权限过滤；
轨迹只在显式 `explain=True` 时构造，不进入 LLM 上下文。

## 开发流程：改动必须走 PR

```sh
git config core.hooksPath .githooks    # 一次性，启用 pre-push 拦截

git switch -c fix/你的改动
git push -u origin fix/你的改动
gh pr create --fill
gh pr checks --watch                   # 等 CI 绿
gh pr merge --squash --delete-branch
```

**为什么不直接推 main**——这不是形式主义，是实测出来的：

项目在 AI 辅助开发环境中持续迭代，关键架构、测试与评测均以可执行结果验收。
一次版本推送前本地 87 个测试全绿，但 **CI 首次运行仍连续发现两个本地环境未覆盖的问题：**

| # | 问题 | 为什么本地发现不了 |
|---|---|---|
| 1 | `url.replace(f"/{dbname}", "/postgres")` 把连接串改烂 | `str.replace` 替换所有匹配。CI 的连接串里**用户名恰好等于库名**，本地的不是 |
| 2 | 三个测试在无 API key 时断言恒真 | CI 不带 key → 文档落 `failed` → 清单为空 → `assert X not in 空` 恒真。**其中一个是权限测试，等于从来没测过** |

第二个尤其危险：**假绿比红更糟**——红了你会去修，假绿你以为有覆盖，其实没有。

所以规则是：**本地绿不等于对，CI 绿了才能合。**
CI 的价值不是重复跑本地已经绿的测试，是跑本地跑不到的那条路径。

> 服务端分支保护对免费私有仓库不开放，所以用 `.githooks/pre-push` 客户端兜底。
> 它挡不住 `--no-verify`，但挡得住手滑——而手滑正是实际会发生的那种。

## 评测（结论是测出来的，不是声称的）

这个项目的每个主张都要有数字撑着。评测分两块，结果都在版本库里：

```sh
uv run --no-sync python scripts/probe_vector_rag.py    # 向量链路在什么问题上失效
uv run --no-sync python scripts/probe_multihop.py      # 多跳失败是召回率还是召回时机
uv run --no-sync python scripts/probe_silent_error.py  # 召回不全时模型会不会静默答错
uv run --no-sync python scripts/eval_routing.py        # 43 题路由准确率
uv run --no-sync python scripts/eval_answer_quality.py # 8 题端到端质量 smoke（会调用回答模型 + Judge）
uv run --no-sync python scripts/eval_answer_quality.py --case-id mh-01 --skip-calibration # 单题低成本回归
# 锁定同一语料/切分/58题，对比 Embedding 的质量、维度与 API 延迟
uv run --no-sync python scripts/compare_embeddings.py --config eval/embedding_models.json
# 逐档扩语料，并报告 20 并发下的 P50/P95、QPS 与失败率
uv run --no-sync python scripts/bench_latency.py --repeat 3 --workers 20 --concurrent-rounds 3
# 复用旧答案，只用独立模型重判；同模型会直接中止
JUDGE_MODEL=qwen/qwen3-235b-a22b-2507 uv run --no-sync python scripts/eval_answer_quality.py --rejudge eval/results/answer-quality-rejudged.json --require-independent-judge --output eval/results/answer-quality-independent-qwen3-235b.json
# 生成人工盲标模板；填完后计算 Judge ↔ 人工一致率
uv run --no-sync python scripts/eval_judge_agreement.py --report eval/results/answer-quality-independent-qwen3-235b.json --init-labels eval/human-labels/answer-quality-independent-qwen3-235b.json
uv run --no-sync python scripts/eval_judge_agreement.py --report eval/results/answer-quality-independent-qwen3-235b.json --labels eval/human-labels/answer-quality-independent-qwen3-235b.json --output eval/results/judge-human-agreement.json
```

`eval_answer_quality.py` 同时报告 Answer Correctness、Faithfulness、Citation Correctness、
Answer Relevance 和拒答正确率。它先用 4 条人工标注样例校准 Judge，校准失败就中止；
判定保留逐条事实、证据编号和理由，不把单个分数冒充真值。Judge 响应按完整输入缓存，
生成与 Judge 的延迟/token 分开记录。完整套件需显式传 `--suite grounding`。
正式重判建议加 `--require-independent-judge`：它会对规范化后的模型名做最低限度检查，
同名立即失败；不同别名或同系列模型是否真正独立仍需人工确认。人工模板不包含 Judge
判定和分数，填写后报告分类一致率、Cohen's κ、连续分数 MAE 与 ±0.2 一致率。

另有两个固定版本的小型公开检索集，用来防止只在自造企业语料上调出漂亮数字：

```sh
uv sync --extra public-eval
uv run --no-sync python scripts/eval_nanobeir.py --download-only  # 只下载和校验，零 API 成本
uv run --no-sync python scripts/eval_nanobeir.py                  # 完整检索评测
```

- `NanoNQRetrieval`：5035 篇文档 / 50 题，主要做单证据通用检索的 no-regression guard。
- `NanoHotpotQARetrieval`：5090 篇文档 / 50 题，每题恰好 2 篇相关文档，主要看多文档召回和 MMR。
- 两者都按官方 commit 固定版本。Minibrain 返回 chunk，但 qrels 是 document-level，所以评分前会把同一原文的多个 chunk 确定性聚合。
- 三个默认 arm 共用 40 条候选池，业务 metadata 关闭，避免把候选数和规则过滤混进 MMR 的收益。未使用官方预生成的“强制含正例”候选表。
- 输出会分别写两个 task；它们只是选定的 NanoBEIR 子集，不冒充完整 NanoBEIR 总分。自造企业评测仍然保留，因为公开集测不了权限、metadata 和路由。

首次全量结果见 [`eval/results/nanobeir.md`](eval/results/nanobeir.md)：HotpotQA 上 hybrid 的
Complete Recall@5 从 0.72 升到 0.82，但当前 MMR 又降回 0.72；NanoNQ 上则是纯向量最好。
因此不把 hybrid / MMR 写成“默认必然提升”，只按具体任务和消融数据做取舍。

| 报告 | 结论摘要 |
|---|---|
| [`eval/results/answer-quality-targeted-final.json`](eval/results/answer-quality-targeted-final.json) | 最新代码针对 `hal-10/hal-11/mh-08` 的低成本回归为 3/3：Correctness、Faithfulness、Citation、Relevance、拒答均为 1.000。过程抓到两类真实问题：制度限额被历史流水覆盖、无证据年份触发安全门；也抓到 Judge 把数据集设计备注误当事实证据。生产提示词已细分“政策金额/历史统计”，截断结构化正文可由已提交 claims 确定性收口；Judge 不再接收设计备注，缓存键包含 rubric。 |
| [`eval/results/answer-quality-independent-qwen3-235b.json`](eval/results/answer-quality-independent-qwen3-235b.json) | 用 Qwen3-235B 独立重判 DeepSeek 生成的同一批 8 个答案，校准 4/4、运行 8/8 成功：Correctness 0.850、Faithfulness 1.000、Citation 1.000、Relevance 0.975、拒答 1.000。`mh-01/mh-08` 各 0.5，暴露第二跳证据缺失；人工盲标尚未填写，所以仍把分数视作模型评估，不冒充人工真值。Judge 含校准约 37.6k token，按本次配置价格估算约 $0.0067。 |
| [`eval/results/embedding-models.md`](eval/results/embedding-models.md) | 同一 130 篇 / 719 chunks / 58 题的纯 Dense 对比：BGE-M3 相比 Qwen3-Embedding-8B@1024，Recall@5 0.796 vs 0.743、NDCG@5 0.791 vs 0.726，本次语料编码约 40s vs 128s；但尚未跑公开集与端到端答案回归，所以先列为迁移候选，不直接换线上索引。 |
| [`eval/results/bench_latency.json`](eval/results/bench_latency.json) | 本机 Compose、130 篇、20 并发 × 3 轮：60/60 成功，约 5.19 QPS；并发请求 P50 829ms、P95 10.13s。证明脚本和链路能承压，不外推成生产 SLO；高尾延迟主要受外部 Embedding API 波动影响。 |
| [`eval/results/answer-quality-smoke.md`](eval/results/answer-quality-smoke.md) | 历史同模型 smoke：Answer Correctness 0.738、Faithfulness 0.938、Citation Correctness 0.929、Relevance 1.000、拒答正确率 1.000。它抓到 `mh-01/mh-08` 两个多跳缺口，也证明子串答案检查会把否定句误判为正确；正式解读应以上一行独立重判为准。 |
| [`eval/results/ablate_rerank_ce.md`](eval/results/ablate_rerank_ce.md) | **cross-encoder 确实会排得更准，但没有省下 context**：企业集固定 40 候选后 Complete Recall@5 0.707→0.741、NDCG@5 0.886→0.912，同时 P50 14ms→1467ms、前五 context 1062→2655 字；两个公开集的冻结 top-10 候选内复核也提升。因此保留实验开关，默认关闭。 |
| [`eval/results/ablate_parent_child.md`](eval/results/ablate_parent_child.md) | **父子检索拆成三组后，真正有效的是 parent-aware child 去重，不是扩大正文**：Recall@5 0.847→0.886、NDCG@5 0.869→0.894，context token 不增加；再展开 1600 字 parent 后 Recall 反降 0.004、token 中位数 +469、关键答案支持零提升。因此默认只去重，完整 parent 展开保留为关闭的实验开关。 |
| [`eval/RESULTS.md`](eval/RESULTS.md) | **pgvector 的收益来自 `LIMIT`，不是来自 HNSW 索引**——这两件事常被混为一谈，本项目一度也混了。900 片段实测：索引确实走上了（逐档 `EXPLAIN` 核对），但**延迟和精确检索落在同一个噪声区间，`ef` 从 10 调到 200 毫无区别**；真正兑现的收益是「排序截断在数据库里做、只发回 k 行」，省掉 3.7MB 数据搬运（101ms → 2.2ms）。这一节推翻重写过两次，藏着三个测量错误，其中最值得看的是第三个：**HNSW 索引因反复 upload/purge 膨胀到 1125MB（表仅 18MB），静默退化**——我把这个运维问题当成「近似检索的召回代价」写进了文档，而它「正好符合理论预期」，所以差点没被发现。 |
| [`eval/RESULTS.md`](eval/RESULTS.md)（倒排索引） | **给 BM25 建倒排索引：197.00ms → 2.25ms，快 87.6 倍，而 MRR/Recall/Hit Rate 一位小数都没变**。根因是原实现每次查询都把 900 篇文档（34 万字）重新分词——分词是片段的固有属性，只该入库做一次。**只索引标识符不索引中文二元组，省 99% 存储且结果完全一致**。BM25 从占本地开销 28.2% 降到 0.4%，新的第一瓶颈变成 SQL 拉取（99.3%）。 |
| [`eval/RESULTS.md`](eval/RESULTS.md)（扩语料） | **扩语料到 900 片段（切分第一次真正运行），重测延迟发现瓶颈换人了**：BM25 增长 **40.5 倍**（超线性，根因是每次查询都把全部文档重新分词），而**余弦计算只占本地开销的 0.1%**。整个项目的性能叙事一直是「内存全量算余弦会慢」——**实测错了对象**。优化顺序因此完全反转，pgvector 从第一位掉到最后一位。 |
| [`eval/RESULTS.md`](eval/RESULTS.md)（延迟基线） | **延迟基线：测完推翻了下一步的计划前提**。原打算上 pgvector + HNSW 加速检索，实测发现**检索只占 1.8%，98.2% 是等 embedding API 返回**；而且本地计算的大头是 **SQL 拉取数据（90%）不是余弦计算（0.2%）**——和文档里写了很久的假设相反。线性外推要到约 **4,700 个片段**本地计算才追平网络延迟，当前 84 个，**差 56 倍**。于是计划改成：先扩语料到几千片段，再上 pgvector。 |
| [`eval/RESULTS.md`](eval/RESULTS.md)（幻觉） | **幻觉评测：12 道诱导题 × 3 轮，一次真幻觉都没有**（拒答正确率 100%，对照组答案正确率也 100%——没把系统改哑巴）。最重要的发现是**指标自己出错了三次，比系统还多**；其中一条是方法上限：**确定性字符串匹配能可靠地测「有没有出处」，测不好「有没有拒答」**——数字和编号是离散可穷举的，而「我不知道」在自然语言里有无穷多种说法。这正是 Faithfulness 要用 LLM 当裁判的原因。 |
| [`eval/RESULTS.md`](eval/RESULTS.md)（Agentic RAG） | **Agentic RAG 测完发现不用做——能力早就在且是自适应的**（多跳题续查触发率 71~89%，单点事实 0%，每题正好 1 次调用）。真正的缺口是**不会停**：构造 10 道压力题后，「链条到顶」类 50% 耗尽轮数，连查 6~8 次换着措辞问同一件事。加停止判定 + 让兜底基于已有证据作答后，**有结论率全组 100%**，多跳·原有耗尽率 33%→0%、答案正确率 67%→100%、还更便宜了（3.44→2.22 次调用）。 |
| [`eval/RESULTS.md`](eval/RESULTS.md)（混合检索） | **混合检索：全局 MRR 0.855 → 0.950，原有题目零损伤**。过程翻了两次车，翻车比结果值钱：① 朴素 RRF 把普通问题的 Recall@5 从 0.803 打到 0.610，根因是中文二元组的「偶然稀有」（`天年` 是「三年年假」切出的伪词）——所以关键词路只负责标识符；② 项目编号纹丝不动，是**结构性平局**（两边名次都是 {1,2}，调 k 也没用），改破平规则后 0.750 → 1.000。 |
| [`eval/RESULTS.md`](eval/RESULTS.md)（扩语料） | **扩语料到 84 篇后，同样 10 道题指标掉了约 10 个点**（Complete Recall@10 从 1.000 掉到 0.800）——玩具规模语料上的检索指标不能外推。另测出真正的盲区不是「编号」而是**「前缀相同、只差几位数字的编号」**：问 `MTG-20260617-02`，前三名全是别的会议纪要，正确答案排第 8。而且问题在**排序不在召回**（Recall@5=1.000，MRR 只有 0.489）——这正是混合检索的确切用武之地。 |
| [`eval/RESULTS.md`](eval/RESULTS.md)（初版） | **标准 IR 指标全都很漂亮**（MRR=1.000、Hit@k=1.000、Recall@5=0.930、NDCG@5=0.945），但同一份数据换成 Complete Recall@5 只有 0.800，全局聚合类 k=3 时 **0/3**——**标准指标在"必须召回全"的场景下会系统性高估**。多跳失败的根因是"召回时机"不是"召回率"：同样 k=3，单次检索漏掉目标文档（实测排第 6），两步检索命中。 |
| [`eval/ROUTING.md`](eval/ROUTING.md) | 严格路由准确率 **83.7% → 95.3%**（各跑 3 轮）。失败原本集中在"文档·组织事实" 1/6 且三轮一致，根因是 system prompt 只注入了表结构没注入文档清单；改后该类 **6/6**，且三轮零波动、过度调用率反而从 9.3% 降到 4.7%。过程中暴露并修好了一处回归（`tbl-18`）：schema 给了列名没给取值，模型只能猜枚举值。 |
| [`SCALING.md`](SCALING.md) | **这套做法在真实规模下会怎样**：文档清单那段是 `O(文档数)`，1000 万篇文档时不可行。消融数据显示贡献更大的权威来源规则是 `O(1)`——需要替换的只是清单。业界替代方案（语义路由 / 分层索引 / 元数据过滤 / 语义层）与不做的理由（15 篇语料上 MRR 已是 1.000，测不出差别）。 |
| [`eval/PROMPT_ABLATION.md`](eval/PROMPT_ABLATION.md) | 消融实验（43 题 × 4 版本 × 3 轮）验证根因：目标类别 39% → 61% → 78% → **100%** 单调改善。结论是**给资料 ≠ 给判断依据**——光列出文档修不好 `doc-08`，必须明确"冲突时听谁的"。 |

两条最该被记住的：

**1. 向量链路会静默答错。** 问"公司总共多少人"，top_k=5 漏掉财务部，模型答"合计约 73 人"
（真值 84，少算 13%），而回答列了明细、标了来源、语气自然，用户无从察觉。
**检索的缺陷被生成层完美掩盖了**——这是"必须有评测体系"最直接的论据。

**2. 路由错了但答案对了，比路由错更危险。** 三轮答案正确率 100%，路由准确率只有 83.7%，
差额全是"走错链路但因数据重叠碰巧答对"。`doc-08` 是运气用完的地方：走表格链路查出
4 个小组（多了一个记账行），文档里写的是 3 个——**而子串匹配的答案检查没抓到**。
子串匹配只能抓"少答"，抓不到"多答"。

### embedding 维度

`qwen/qwen3-embedding-8b` 实际输出 **4096** 维。代码按 `EMBEDDING_DIMENSIONS` 做
MRL 截断 + 重新归一化（默认降到 1024，与 ff-companybrain 同口径），省 4 倍内存和存储。

query 和 chunk 走同一个函数、同一套截断口径——这两边一旦不同，检索会悄悄变差且极难查。
改 `EMBEDDING_DIMENSIONS` 必须重新索引全部文档。

## 结构

```
schema.sql                  identity + 两个模块 + observability schema。改结构就改这里
eval/
  baselines/                NanoBEIR 冻结基线，供 CI 判断检索质量/延迟回归
  corpus/                   基础语料与扩展虚构公司文档（非结构化）
  corpus_table/             3 张 CSV（花名册/销售/报销），和文档是同一家公司
  probes*.json              检索 / 盲区 / 幻觉探针，每题标注真值或必需来源
  routing.json              43 道路由用例，每题标注"该调哪几个工具"
  RESULTS.md / ROUTING.md   评测结论，保留失败过程与测量修正
tests/                      pytest 单元/集成测试（数量以 collect-only 为准）
scripts/
  explain.py                跑一遍看懂两条产品链路（中间产物全打印）
  probe_*.py                检索、性能、幻觉与多跳探针
  eval_*.py / ablate_*.py  路由、轨迹评测与组件消融
  check_eval_regression.py  对候选公开集报告执行确定性回归门禁
src/minibrain/
  config.py                 环境变量，一次读取一次校验
  contracts.py              UserContext / ModuleId / Evidence，薄契约
  db.py                     模块连接池，各自锁死 search_path
  gateway.py                模块分发。唯一允许 import modules/ 的地方
  identity/                 用户名密码 + bcrypt + 会话
  modules/
    vector_rag/             LlamaIndex + pgvector + BM25/RRF + metadata/MMR + 可选 CE
    table_rag/              CSV/XLSX Sheet 建表 + 只读 SQL + 四层护栏
    graph_rag/              PropertyGraphIndex 对照实验（未接产品 gateway）
  agent/                    LangGraph Agent + PostgreSQL checkpointer
  handwritten/              手写 Agent / 检索基线，不是产品路径
  web/                      FastAPI + Jinja2 + htmx，无构建步骤
```

仓库规模会随实验快速变化，不把 LOC 当成功能完成度。当前应从上表判断哪些是产品路径、
哪些是对照实验，并以测试收集结果和评测产物判断是否真的生效。

## 四条不将就的规矩

尺寸可以小，这四条不能松——它们事后返工的成本极高，而现在遵守它们的成本几乎为零。

**1. 权限过滤写在 SQL 的 WHERE 里，不是查完再筛。**
`_visibility_clause()` 是两个模块各自唯一的可见性判定，永远出现在 WHERE 里。
`chunks.source_id` 是从 `documents` 冗余下来的，就是为了让过滤不必 join。
应用层后过滤是最容易长出越权 bug 的地方：漏一个分支就是数据泄露。

**2. `core.py` 里永远不 import fastapi。**
模块只收 `UserContext` 和普通参数，只返回 dataclass。
做到这点，后面加 MCP、加 CLI、加定时任务都是白送的。

**3. 业务逻辑不进 `web/app.py`。**
路由只做解析、鉴权、调 gateway、渲染。一旦开了"就这一处先放这儿"的口子，
它会长成一个上千行、几十个分支的路由文件，然后再也搬不回去。

**4. 状态机区分 `failed` 与 `dead_letter`，失败不伪装成 ready。**
上传接口只登记就立刻返回，持久化 worker 用 lease 领取任务并在处理中续租；进程崩溃后
任务可由其他 worker 重新领取，临时错误耗尽重试进入 `dead_letter`，永久解析错误进入
`failed`，二者都支持人工重试。前端轮询状态，启动时还会归档超时的问答 run。
向量化是分钟级的，同步阻塞在生产上会被反代掐断——这个坑要在第一天就避开。

## 表格链路的四层护栏

LLM 会写 SQL，所以护栏必须是纵深的，任何一层单独都不够：

1. **表名白名单** —— 先用 SQL 算出当前用户看得见哪些表，权限的真正落点
2. **只允许单条 SELECT / WITH**
3. **强制外层 LIMIT**（把语句包进子查询）
4. **只读事务 + statement_timeout** —— 在连接层面，不依赖上面三层的正确性

第 4 层用 `SET default_transaction_read_only = on` 实现，不需要建 PG 角色，
也就不需要超级用户权限。物理表名由模块生成（`t_<8位hex>`），用户输入永远不进标识符。

## 留好的缝

这些都是设计好的升级路径，不是欠的债：

| 现在 | 将来 | 改哪里 |
|---|---|---|
| 单进程、gateway 是 dispatch table | 模块拆成独立 HTTP 服务 | `gateway.call` 改成带 `x-ff-*` header 的 fetch，调用方零改动 |
| 一库多 schema | 模块拆成独立数据库 | `db.py` 的 conninfo；每个模块 schema 一次 pg_dump |
| PostgreSQL 实体队列 + 单独 worker | Redis/Celery 或云队列 | 保持 `gateway.claim_next/process` 契约，替换调度实现 |
| PostgreSQL 全历史 + 无额外 LLM 的短期窗口 | 超长任务再启用摘要；同 thread 加分布式并发锁 | `agent/graph.py` |
| SSE 传执行状态与最终校验后答案 | 若未来改成 token 流，先设计可撤回/缓冲的 grounding 协议 | `web/app.py` |
| GraphRAG 单用户实验 | 带来源级权限、增量更新和并发写的第三条产品链路 | 权限模型成立后再注册进 `gateway._REGISTRY` |

加第三条链路是最能验证边界的时刻——前两条已经把 gateway 和 UserContext 的形状压出来了。

## 已知边界

- 表名白名单靠正则提取 `FROM` / `JOIN` 后的标识符。只读事务和 search_path 锁定是兜底，
  但足够刁钻的构造可能绕过白名单这一层。真要上生产，这里应该换成真正的 SQL 解析。
- 会话存明文 token，没有轮换和刷新。
- 入库任务已持久化在 `documents/datasets`：`FOR UPDATE SKIP LOCKED` 防重复领取，
  lease 过期可恢复且处理线程会周期续租，临时错误最多重试 3 次后进 `dead_letter`。
- Router 只接收 `O(source 数)` 的可见知识域摘要，确定域时把 source 作为权限内 metadata
  过滤条件；不确定时省略过滤，避免错误路由造成漏召回。超大规模 source 数仍需按需域检索，
  触发条件见 [`SCALING.md`](SCALING.md)。
- 主路径已有持久化全局 BM25，能从向量候选池之外独立救回节点；中文按现有消融只索引
  拉丁/数字标识符。纯英文语料会索引普通单词，目前还没有停用词表或词干化。
- 列取值只对**低基数文本列**注入（≤12 个取值、≤40 字符）。高基数列模型仍然只能猜——
  大规模下正确做法是按需检索（value retrieval），见 [`SCALING.md`](SCALING.md)。
- PostgreSQL 保存完整会话；送入模型的临时视图按完整 turn 保留最近 6 轮并受 8000-token
  预算限制，不会拆断 tool call 协议，也不增加摘要模型调用。同一 thread 用 PostgreSQL
  advisory lock 跨进程串行，request UUID + 数据库唯一索引提供成功重试幂等；冲突返回 409，
  不在 Web 进程里做不可靠的内存锁。确实需要跨几十轮任务状态时再升级为摘要。
- 主向量索引 manifest 锁定 embedding endpoint / model / dimensions 与 MRL+L2 变换版本；
  同维度换模型也会 fail-fast。迁移前的非空旧索引必须显式执行
  `minibrain-index-manifest --adopt-current`，来源不确定时必须重建，不能用 adopt 猜历史。
- GraphRAG 是单用户受控实验，未解决图上来源级权限、并发写和增量建图，未接入 Web 产品路径。
- 文档链路支持纯文本 / Markdown / HTML / DOCX / PPTX / PDF 文字层。DOCX 表格作为
  Markdown 证据保留，不自动建 table-rag 数据集；仍没有旧 `.doc`、OCR、图片型 PPTX
  和 PDF 版面还原。
- 主路径已有 claim 级结构化引用，计算 citation coverage / validity，并确定性检查数字和
  业务编号是否出现在对应证据中；失败会由零额外 LLM 的安全门替换为保守拒答，并保留
  原答案供 Run Trace 审计。它不能证明完整语义蕴含，因此不冒充 Faithfulness。
- 检索实验室展示 Dense 顶分与分差、BM25 顶分、两路 Top-5 重合度、RRF 顶分和三态
  决策。58 道可答题 + 8 道无答案题的 held-out 校准中，Dense 阈值误拒 26.7% 且无答案
  召回为 0%，因此分数硬门保持 report-only；只对空结果和精确业务编号不存在提前拒答。
- Agent 证据上下文使用 `cl100k_base` 固定口径执行 4000-token / 12-evidence 硬预算：
  先保留最高相关候选，再优先覆盖不同文档，最后补同文档片段；跨工具重复内容会被丢弃。
  这是可复现的组装口径，不等于 DeepSeek/OpenRouter 最终账单 token（供应商 tokenizer 可能不同）。
- Run Trace 已记录成功/失败、工具、证据、延迟和供应商返回的 token；worker 启动时会将
  超时仍为 `running` 的记录归档成稳定的 `stale_run_archived` 错误。
- 答案检查用子串匹配，只能抓"少答"，抓不到"多答"（`doc-08` 就是漏网的例子）。
