-- minibrain schema。启动时整份跑一遍，幂等。
-- 没上线之前不要迁移框架：改结构就改这个文件，drop database 重建比幂等 ALTER 便宜。
--
-- 边界：三个 schema 各归一个模块，每个模块用自己的连接池并锁死 search_path。
-- 将来要拆成独立数据库/独立服务时，每个 schema 一次 pg_dump 就搬走了。

-- 扩展统一放在专用 schema，不放 public。
--
-- 为什么：每个模块的连接池都把 search_path 锁死在自己的 schema
-- （db.py，这是"模块够不着别人的数据"这条边界的物理落点）。
-- 而 vector 这类扩展提供的是**类型和运算符**，所有模块都要能解析。
--
-- 放 public 也能用，但那样就得把 public 加进 search_path ——
-- public 是任何人都能建表的地方，等于给边界开了个口子。
-- 放进只含扩展、不含任何数据表的 extensions schema，边界仍然成立。
create schema if not exists extensions;

create extension if not exists "pgcrypto" with schema extensions;   -- gen_random_uuid()

-- pgvector：把向量检索搬进数据库。
--
-- 为什么现在才加（实测依据见 eval/RESULTS.md 探针七/十/十一）：
--   一开始以为它是来"加速余弦计算"的 —— 错了。实测余弦只占本地开销的 0.1%，
--   numpy 算 900×1024 的矩阵乘只要 0.1ms。
--   真正的开销是 _visible_chunks 每次把 900 行 × 1024 维浮点数（约 3.7MB）
--   全搬进 Python 内存（101ms，占本地 99.3%），而最后只用了 top-5。
--
--   所以 pgvector 解决的是**数据搬运**，不是向量运算：
--   让数据库自己排序，只把前 k 行发回来。
create extension if not exists "vector" with schema extensions;

-- 建表时也要能解析 vector 类型和 gen_random_uuid()。
-- apply_schema 用的是不锁 search_path 的裸连接（它要建 schema 本身），
-- 所以这里显式加上。只影响执行 schema.sql 这一次会话。
set search_path to public, extensions;

create schema if not exists identity;
create schema if not exists mod_vector;
create schema if not exists mod_vector_li;
create schema if not exists mod_table;
create schema if not exists observability;


-- ============================================================
-- identity：平台身份。只管"你是谁""是不是管理员"。
-- 明确不管任何模块内部权限 —— 那是模块自己的事。
-- ============================================================

create table if not exists identity.users (
  id            uuid primary key default gen_random_uuid(),
  username      text not null unique,
  password_hash text not null,
  is_admin      boolean not null default false,
  created_at    timestamptz not null default now()
);

create table if not exists identity.sessions (
  token      text primary key,
  user_id    uuid not null references identity.users(id) on delete cascade,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null
);

create index if not exists sessions_user_idx on identity.sessions (user_id);
create index if not exists sessions_expires_idx on identity.sessions (expires_at);


-- ============================================================
-- mod_vector：向量检索链路。文档 → chunk → embedding → 语义召回。
-- ============================================================

create table if not exists mod_vector.sources (
  id         uuid primary key default gen_random_uuid(),
  name       text not null,
  visibility text not null check (visibility in ('private', 'public')),
  owner_id   uuid not null,                      -- 逻辑外键指向 identity.users，跨 schema 不建约束
  created_at timestamptz not null default now(),
  unique (owner_id, name)
);

create table if not exists mod_vector.documents (
  id         uuid primary key default gen_random_uuid(),
  source_id  uuid not null references mod_vector.sources(id) on delete cascade,
  filename   text not null,
  content    text not null default '',          -- 原文。落库而不是留在内存里，进程重启后仍可重跑处理
  -- 真状态机。失败就是失败，不许伪装成 ready。
  status     text not null default 'uploaded'
             check (status in ('uploaded', 'processing', 'ready', 'failed')),
  error      jsonb,                              -- 结构化错误，status='failed' 时必填
  char_count integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists documents_source_idx on mod_vector.documents (source_id);

-- LLM Wiki MVP：一个文档对应一个可追溯的整理页。它是原文的派生视图，
-- 不参与检索索引；原文删除时自动删除，原文升级时由应用标记 stale。
create table if not exists mod_vector.wiki_pages (
  id               uuid primary key default gen_random_uuid(),
  document_id      uuid not null unique references mod_vector.documents(id) on delete cascade,
  document_version integer not null,
  title            text not null,
  content          text not null default '',
  status           text not null default 'building'
                   check (status in ('building', 'ready', 'failed', 'stale')),
  error             text,
  model             text,
  created_at        timestamptz not null default now(),
  updated_at        timestamptz not null default now()
);

create index if not exists wiki_pages_status_idx on mod_vector.wiki_pages (status, updated_at desc);

-- Karpathy-style LLM Wiki 的最小增量层。来源摘要页继续由 wiki_pages 管理；
-- 主题页可以由多篇原文共同支持，因此不能复用“一文档一页”的唯一关系。
create table if not exists mod_vector.wiki_topics (
  id         uuid primary key default gen_random_uuid(),
  source_id  uuid not null references mod_vector.sources(id) on delete cascade,
  slug       text not null,
  title      text not null,
  content    text not null default '',
  status     text not null default 'ready'
             check (status in ('ready', 'failed', 'stale')),
  error      text,
  model      text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (source_id, slug)
);

create index if not exists wiki_topics_status_idx
  on mod_vector.wiki_topics (source_id, status, updated_at desc);

create table if not exists mod_vector.wiki_topic_documents (
  topic_id         uuid not null references mod_vector.wiki_topics(id) on delete cascade,
  document_id      uuid not null references mod_vector.documents(id) on delete cascade,
  document_version integer not null,
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now(),
  primary key (topic_id, document_id)
);

create index if not exists wiki_topic_documents_document_idx
  on mod_vector.wiki_topic_documents (document_id);

create table if not exists mod_vector.wiki_topic_links (
  from_topic_id uuid not null references mod_vector.wiki_topics(id) on delete cascade,
  to_topic_id   uuid not null references mod_vector.wiki_topics(id) on delete cascade,
  created_at    timestamptz not null default now(),
  primary key (from_topic_id, to_topic_id),
  check (from_topic_id <> to_topic_id)
);

create table if not exists mod_vector.wiki_events (
  id          uuid primary key default gen_random_uuid(),
  source_id   uuid not null references mod_vector.sources(id) on delete cascade,
  action      text not null,
  target_type text not null check (target_type in ('source_page', 'topic', 'query', 'lint')),
  target_id   uuid,
  detail      jsonb not null default '{}'::jsonb,
  created_at  timestamptz not null default now()
);

create index if not exists wiki_events_source_idx
  on mod_vector.wiki_events (source_id, created_at desc);

create table if not exists mod_vector.chunks (
  id              uuid primary key default gen_random_uuid(),
  document_id     uuid not null references mod_vector.documents(id) on delete cascade,
  -- source_id 冗余一份：让权限过滤能在同一个 WHERE 里完成，不必 join。
  -- 这是"权限过滤写进 SQL"这条规矩的物理前提。
  source_id       uuid not null references mod_vector.sources(id) on delete cascade,
  ordinal         integer not null,
  content         text not null,
  -- vector(N) 而不是 real[]：real[] 只能全搬进内存自己算，
  -- vector 类型才能用 <=> 距离运算符和 HNSW 索引，让排序在数据库里完成。
  --
  -- ★ 维度写死 1024，必须和 EMBEDDING_DIMENSIONS 一致。
  --   这是 pgvector 的硬要求（索引需要固定维度）。
  --   改 EMBEDDING_DIMENSIONS 要连这里一起改，并重新索引全部文档 ——
  --   .env.example 里那条警告现在多了一个必须同步的地方。
  embedding       vector(1024) not null,
  embedding_model text not null,
  embedding_dim   integer not null
);

create index if not exists chunks_document_idx on mod_vector.chunks (document_id);
create index if not exists chunks_source_idx on mod_vector.chunks (source_id);

-- HNSW：向量的近似最近邻索引（Hierarchical Navigable Small World）。
--
-- ★ 它是**近似**的 —— 用召回率换速度。所以加它必须同时测两件事：
--   快了多少（延迟）+ 丢了多少（Recall/MRR）。
--   本项目有 58 道标注题，正好能量出这个交换。数字见 eval/RESULTS.md 探针十二。
--
-- 两个建索引参数（查询参数 ef_search 是会话级的，在代码里设）：
--   m               = 每个节点的最大连接数。大 → 召回高、索引大、建得慢
--   ef_construction = 建索引时的候选集大小。大 → 质量高、建得慢
-- 这里用 pgvector 的默认值（16 / 64），没有调参 ——
-- 调参需要独立验证集，拿测试集调出来的参数是自欺。
--
-- vector_cosine_ops：按余弦距离建索引，和检索时用的 <=> 必须一致。
-- 用错距离函数索引会失效（而且不报错，只是悄悄变慢变差）。
create index if not exists chunks_embedding_hnsw_idx
  on mod_vector.chunks using hnsw (embedding vector_cosine_ops);

-- term_count：这个片段一共有多少个 token（含中文二元组）。
-- BM25 的长度归一化要用它做分母，所以必须是**全量**分词的计数，
-- 不能只数下面 chunk_terms 里的标识符。
alter table mod_vector.chunks
  add column if not exists term_count integer not null default 0;


-- ============================================================
-- chunk_terms：BM25 的倒排索引。词 → 出现在哪些片段、出现几次。
--
-- 为什么需要：原来 bm25_scores 每次查询都把**全部文档重新分词、重算 IDF**，
-- 是 O(总字符数) 不是 O(片段数)。实测 84 片段时 4.98ms，
-- 900 片段（34 万字）时涨到 201.65ms —— 增长 40.5 倍，超线性。
-- 见 eval/RESULTS.md 探针十。
--
-- ★ 只索引**标识符**（字母数字 token），不索引中文二元组。
-- 因为查询侧走 identifier_tokens()，中文 token 永远不会被查到。
-- 实测：全量索引约 112,000 行，只索引标识符 759 行 —— 省 99% 且结果完全一致。
--
-- source_id 从 chunks 冗余下来，理由和 chunks.source_id 一样：
-- 让权限过滤能在同一个 WHERE 里完成，不必 join 回去。
-- ============================================================

create table if not exists mod_vector.chunk_terms (
  chunk_id  uuid    not null references mod_vector.chunks(id) on delete cascade,
  source_id uuid    not null references mod_vector.sources(id) on delete cascade,
  term      text    not null,
  freq      integer not null,
  primary key (chunk_id, term)
);

-- 查询按 term 找片段，这个索引是整件事的关键
create index if not exists chunk_terms_term_idx on mod_vector.chunk_terms (term);
create index if not exists chunk_terms_source_idx on mod_vector.chunk_terms (source_id);


-- ============================================================
-- mod_vector_li：LlamaIndex 主路径的持久化 sparse 索引。
--
-- PGVectorStore 自己维护 data_nodes；下面两张表只维护 BM25 所需统计，
-- 让关键词路能从全库独立召回，而不是先被向量候选池截断。
-- node_id 不建到 data_nodes 的 FK：框架表的列约束不由本项目控制，
-- 删除一致性由 chain.delete_source_nodes 显式维护并有清理测试兜底。
-- ============================================================

-- small-to-big 的父块只用于生成上下文，不参与 dense/sparse 召回。
-- 子节点通过 metadata.parent_context_id 指向这里；不设跨框架表外键，删除由生命周期层负责。
create table if not exists mod_vector_li.parent_contexts (
  id            text primary key,
  document_id   text,
  source_name   text not null,
  owner_id      text not null,
  visibility    text not null check (visibility in ('private', 'public')),
  filename      text not null,
  ordinal       integer not null,
  heading_path  text not null default '',
  content       text not null,
  content_hash  text not null,
  created_at    timestamptz not null default now()
);
create index if not exists parent_contexts_document_idx
  on mod_vector_li.parent_contexts (document_id);
create index if not exists parent_contexts_source_idx
  on mod_vector_li.parent_contexts (owner_id, source_name);

-- PGVectorStore 只约束维度，无法阻止“同为 1024 维但来自不同模型”的向量混用。
-- 主链路首次读写前校验这份生成签名；不兼容时 fail-fast，要求全量重建。
create table if not exists mod_vector_li.index_manifest (
  index_name           text primary key,
  embedding_endpoint   text not null,
  embedding_model      text not null,
  embedding_dimensions integer not null,
  transform_version    text not null,
  provenance           text not null check (
    provenance in ('initialized_empty', 'operator_adopted')
  ),
  created_at           timestamptz not null default now(),
  updated_at           timestamptz not null default now()
);

create table if not exists mod_vector_li.node_lexical_stats (
  node_id      text primary key,
  document_id  text,
  source_name  text not null,
  owner_id     text not null,
  visibility   text not null check (visibility in ('private', 'public')),
  term_count   integer not null,
  metadata_    jsonb not null default '{}'::jsonb
);

-- 旧库兼容：历史节点没有稳定 document_id，只能按 source 清理；新入库节点
-- 从这里开始支持文档级删除/重建。nullable 是为了不伪造历史关联。
alter table mod_vector_li.node_lexical_stats
  add column if not exists document_id text;

create index if not exists node_lexical_owner_idx
  on mod_vector_li.node_lexical_stats (owner_id);
create index if not exists node_lexical_source_idx
  on mod_vector_li.node_lexical_stats (source_name);
create index if not exists node_lexical_document_idx
  on mod_vector_li.node_lexical_stats (document_id);

create table if not exists mod_vector_li.node_terms (
  node_id text not null references mod_vector_li.node_lexical_stats(node_id) on delete cascade,
  term    text not null,
  freq    integer not null check (freq > 0),
  primary key (node_id, term)
);

create index if not exists node_terms_term_idx on mod_vector_li.node_terms (term);


-- ============================================================
-- mod_table：结构化表格链路。CSV → 物理表 → 受限只读 SQL。
--
-- 为什么不是"切分 + embedding"：一张报表切碎再向量召回回来，
-- 算不出正确的合计。这是两条链路不能合并的最硬证据。
-- ============================================================

create table if not exists mod_table.sources (
  id         uuid primary key default gen_random_uuid(),
  name       text not null,
  visibility text not null check (visibility in ('private', 'public')),
  owner_id   uuid not null,
  created_at timestamptz not null default now(),
  unique (owner_id, name)
);

-- XLSX 原文只存一份；每个可见 Sheet 仍各自对应一条 dataset 和一张物理表。
-- 这样生命周期保持“一 dataset 一表”，又不会按 Sheet 重复存整份工作簿。
create table if not exists mod_table.workbooks (
  id         uuid primary key default gen_random_uuid(),
  source_id  uuid not null references mod_table.sources(id) on delete cascade,
  filename   text not null,
  content    bytea not null,
  created_at timestamptz not null default now()
);

-- 每份 CSV / XLSX Sheet 在 mod_table schema 里落成一张真实物理表，
-- 这张登记表记录元信息和权限归属。LLM 生成的 SQL 只允许碰登记在册且当前用户可见的表。
create table if not exists mod_table.datasets (
  id         uuid primary key default gen_random_uuid(),
  source_id  uuid not null references mod_table.sources(id) on delete cascade,
  filename   text not null,
  content    text not null default '',           -- CSV 原文（上传时已解码）。同样落库而不是留内存
  table_name text not null unique,               -- 已消毒的物理表名，形如 t_<8位hex>
  columns    jsonb not null default '[]'::jsonb, -- [{"name": "...", "type": "text|numeric"}]
  row_count  integer not null default 0,
  status     text not null default 'uploaded'
             check (status in ('uploaded', 'processing', 'ready', 'failed')),
  error      jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists datasets_source_idx on mod_table.datasets (source_id);

alter table mod_table.datasets add column if not exists workbook_id uuid
  references mod_table.workbooks(id) on delete cascade;
alter table mod_table.datasets add column if not exists sheet_name text;
alter table mod_table.datasets add column if not exists parsed_as text not null default 'csv';
create index if not exists datasets_workbook_idx on mod_table.datasets (workbook_id);

-- 文档是怎么被解析成文本的：text / text(gbk) / pdf。
-- ★ 排查「这篇怎么检索不到」时，第一个要看的就是它当初怎么被解析的。
--   传了 PDF 却显示 text，说明走了错误的分支。
alter table mod_vector.documents add column if not exists parsed_as text not null default 'text';

-- 注册表级内容指纹与版本。相同 source/filename 重复上传时用它跳过 embedding；
-- 内容变化则复用 document_id、version + 1，并在重新入库前删除旧节点。
alter table mod_vector.documents add column if not exists content_hash text not null default '';
alter table mod_vector.documents add column if not exists version integer not null default 1;

-- 原始资料的不可变版本层。documents 指向当前版本；这里保留每次用于检索和
-- Wiki 编译的解析文本，使旧 Wiki 的来源仍可审计。
create table if not exists mod_vector.document_revisions (
  document_id uuid not null references mod_vector.documents(id) on delete cascade,
  version integer not null check (version > 0),
  content text not null,
  content_hash text not null,
  parsed_as text not null,
  char_count integer not null,
  created_at timestamptz not null default now(),
  primary key (document_id, version)
);

insert into mod_vector.document_revisions
  (document_id, version, content, content_hash, parsed_as, char_count, created_at)
select id, version, content, content_hash, parsed_as, char_count, updated_at
from mod_vector.documents
on conflict (document_id, version) do nothing;

-- 旧数据库中的约束不会被 create table if not exists 更新，显式迁移一次。
alter table mod_vector.wiki_events drop constraint if exists wiki_events_target_type_check;
alter table mod_vector.wiki_events add constraint wiki_events_target_type_check
  check (target_type in ('source_page', 'topic', 'query', 'lint'));

-- 切出来多少个片段。主路径（LlamaIndex）把节点存在自己的表里，
-- 所以这里缓存一个计数，供文档列表显示；手写版走 chunks 表，两者不冲突。
alter table mod_vector.documents add column if not exists chunk_count integer not null default 0;

-- documents / datasets 自身就是持久化任务记录：上传事务提交时任务已经存在，
-- 不会出现「文档登记成功但另一个队列表写失败」的双写窗口。
alter table mod_vector.documents add column if not exists attempt_count integer not null default 0;
alter table mod_vector.documents add column if not exists available_at timestamptz not null default now();
alter table mod_vector.documents add column if not exists lease_until timestamptz;
alter table mod_vector.documents add column if not exists processing_started_at timestamptz;
alter table mod_vector.documents add column if not exists finished_at timestamptz;
create index if not exists documents_queue_idx
  on mod_vector.documents (status, available_at, created_at);

alter table mod_table.datasets add column if not exists attempt_count integer not null default 0;
alter table mod_table.datasets add column if not exists available_at timestamptz not null default now();
alter table mod_table.datasets add column if not exists lease_until timestamptz;
alter table mod_table.datasets add column if not exists processing_started_at timestamptz;
alter table mod_table.datasets add column if not exists finished_at timestamptz;
create index if not exists datasets_queue_idx
  on mod_table.datasets (status, available_at, created_at);

-- retryable 异常耗尽重试后进入 dead_letter；格式/权限等永久错误仍是 failed。
-- 旧库的匿名 check constraint 名由 PostgreSQL 按列名生成，幂等升级时显式替换。
alter table mod_vector.documents drop constraint if exists documents_status_check;
alter table mod_vector.documents add constraint documents_status_check
  check (status in ('uploaded', 'processing', 'ready', 'failed', 'dead_letter'));
alter table mod_table.datasets drop constraint if exists datasets_status_check;
alter table mod_table.datasets add constraint datasets_status_check
  check (status in ('uploaded', 'processing', 'ready', 'failed', 'dead_letter'));


-- ============================================================
-- observability：平台级问答运行记录。只记录跨模块轨迹，不读取模块内部表。
-- ============================================================

create table if not exists observability.runs (
  id            uuid primary key default gen_random_uuid(),
  user_id       uuid not null,
  session_id    text,
  request_id    text,
  question      text not null,
  answer        text,
  status        text not null check (status in ('running', 'succeeded', 'failed')),
  model         text not null,
  started_at    timestamptz not null default now(),
  finished_at   timestamptz,
  latency_ms    bigint,
  input_tokens  integer not null default 0,
  output_tokens integer not null default 0,
  total_tokens  integer not null default 0,
  tool_calls    jsonb not null default '[]'::jsonb,
  evidence      jsonb not null default '[]'::jsonb,
  claims        jsonb not null default '[]'::jsonb,
  citation_metrics jsonb not null default '{}'::jsonb,
  context_decisions jsonb not null default '[]'::jsonb,
  context_metrics jsonb not null default '{}'::jsonb,
  error         jsonb
);

-- 旧数据库幂等升级：create table if not exists 不会给现有表补列。
alter table observability.runs
  add column if not exists claims jsonb not null default '[]'::jsonb;
alter table observability.runs
  add column if not exists citation_metrics jsonb not null default '{}'::jsonb;
alter table observability.runs
  add column if not exists context_decisions jsonb not null default '[]'::jsonb;
alter table observability.runs
  add column if not exists context_metrics jsonb not null default '{}'::jsonb;
alter table observability.runs
  add column if not exists raw_answer text;
alter table observability.runs
  add column if not exists confidence_report jsonb not null default '{}'::jsonb;
alter table observability.runs
  add column if not exists request_id text;

-- 相同会话内的客户端请求键只允许执行一次。NULL 表示 CLI/旧调用，不参与唯一约束。
create unique index if not exists runs_session_request_uidx
  on observability.runs (user_id, session_id, request_id)
  where request_id is not null;

create index if not exists runs_user_started_idx
  on observability.runs (user_id, started_at desc);
create index if not exists runs_status_started_idx
  on observability.runs (status, started_at desc);

-- 用户反馈只关联 run，不复制问题/答案/证据。一个用户对一次运行只有一条当前反馈，
-- 重复点击采用 upsert；这样可以纠正误点，也不会把一次回答重复计入回归集。
create table if not exists observability.run_feedback (
  id          uuid primary key default gen_random_uuid(),
  run_id      uuid not null references observability.runs(id) on delete cascade,
  user_id     uuid not null,
  rating      smallint not null check (rating in (-1, 1)),
  reason      text check (reason is null or reason in (
                'incorrect', 'missing_evidence', 'bad_citation',
                'irrelevant', 'too_verbose', 'other'
              )),
  note        text not null default '',
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  unique (run_id, user_id)
);

create index if not exists run_feedback_user_updated_idx
  on observability.run_feedback (user_id, updated_at desc);
create index if not exists run_feedback_rating_updated_idx
  on observability.run_feedback (rating, updated_at desc);
