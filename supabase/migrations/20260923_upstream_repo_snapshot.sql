-- Proyecto: Proyectovisual2
-- Snapshot del repositorio upstream del profesor para trazabilidad y sincronización

CREATE TABLE IF NOT EXISTS upstream_repo_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    repo_name TEXT NOT NULL,
    repo_url TEXT NOT NULL,
    default_branch TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    short_sha TEXT NOT NULL,
    author_name TEXT,
    author_email TEXT,
    commit_message TEXT,
    committed_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    readme_excerpt TEXT,
    file_count INTEGER NOT NULL DEFAULT 0,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (repo_url, commit_sha)
);

CREATE INDEX IF NOT EXISTS idx_upstream_repo_snapshots_repo_time
    ON upstream_repo_snapshots (repo_url, committed_at DESC);
