CREATE TABLE IF NOT EXISTS cache_entries (
    key TEXT PRIMARY KEY,
    payload JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries (expires_at);

CREATE TABLE IF NOT EXISTS research_sessions (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_used TEXT[] NOT NULL DEFAULT '{}',
    sources_failed TEXT[] NOT NULL DEFAULT '{}',
    duration_ms INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);