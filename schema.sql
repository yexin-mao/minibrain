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
create schema if not exists mod_table;


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

-- 每份 CSV 在 mod_table schema 里落成一张真实物理表，
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
