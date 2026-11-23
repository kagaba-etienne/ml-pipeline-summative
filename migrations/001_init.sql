-- 001_init.sql: tables for training runs, uploads and predictions.
-- Run once: psql "$DIRECT_DB_URL" -f migrations/001_init.sql

begin;

-- One row per retraining run. A promoted run is a model version.
create table if not exists training_runs (
    id                 serial primary key,
    status             text not null default 'running'
                       check (status in ('running', 'done', 'failed')),
    n_new              int,            -- uploaded images used
    n_rejected         int,            -- uploaded images rejected during preprocessing
    n_replay           int,            -- original training images mixed in
    champion_metrics   jsonb,          -- current model on the test sample
    challenger_metrics jsonb,          -- retrained model on the test sample
    promoted           boolean not null default false,
    model_version      int unique,     -- set only when promoted
    model_key          text,           -- R2 key of the .h5 (null = file bundled in the repo)
    error              text,
    started_at         timestamptz not null default now(),
    finished_at        timestamptz
);

-- Only one retraining run at a time.
create unique index if not exists one_running_run on training_runs ((true)) where status = 'running';

-- One row per uploaded image. The label comes from the folder name.
create table if not exists uploads (
    id              uuid primary key default gen_random_uuid(),
    batch_id        uuid not null,
    relative_path   text not null,     -- <root>/<class>/<file> as picked in the browser
    label           text not null,
    r2_key          text not null unique,
    content_type    text not null,
    size_bytes      int,
    md5             text,
    status          text not null default 'requested'
                    check (status in ('requested', 'pending', 'used', 'rejected')),
    reason          text,              -- why it was rejected
    training_run_id int references training_runs (id),
    created_at      timestamptz not null default now()
);

create index if not exists uploads_status_idx on uploads (status);

-- One row per prediction (monitoring).
create table if not exists predictions (
    id            bigserial primary key,
    label         text not null,
    confidence    real not null,
    latency_ms    real not null,
    model_version int,
    created_at    timestamptz not null default now()
);

-- Supabase exposes the public schema through its REST API. Enabling RLS with
-- no policies blocks that access; the app connects as the table owner and is unaffected.
alter table training_runs enable row level security;
alter table uploads       enable row level security;
alter table predictions   enable row level security;

commit;
