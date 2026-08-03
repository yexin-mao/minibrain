-- minibrain schema。启动时整份跑一遍，幂等。
-- 没上线之前不要迁移框架：改结构就改这个文件，drop database 重建比幂等 ALTER 便宜。
--
-- 边界：三个 schema 各归一个模块，每个模块用自己的连接池并锁死 search_path。
-- 将来要拆成独立数据库/独立服务时，每个 schema 一次 pg_dump 就搬走了。

create extension if not exists "pgcrypto";   -- gen_random_uuid()

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
  embedding       real[] not null,
  embedding_model text not null,
  embedding_dim   integer not null
);

create index if not exists chunks_document_idx on mod_vector.chunks (document_id);
create index if not exists chunks_source_idx on mod_vector.chunks (source_id);


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
