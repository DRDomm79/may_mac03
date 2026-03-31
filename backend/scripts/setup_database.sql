-- ─── DAC HealthPrice v2 — Database Setup ────────────────────────────────────
-- Run in Supabase SQL Editor

-- Quote log
CREATE TABLE IF NOT EXISTS hp_quote_log (
    id              SERIAL PRIMARY KEY,
    quote_ref       VARCHAR(30) UNIQUE NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW(),
    input_json      JSONB NOT NULL,
    result_json     JSONB NOT NULL,
    model_version   VARCHAR(30)
);
CREATE INDEX IF NOT EXISTS idx_hp_quotes_created ON hp_quote_log (created_at DESC);

-- User behavior (what people choose — for behavioral analysis, NOT claims retraining)
CREATE TABLE IF NOT EXISTS hp_user_behavior (
    id              SERIAL PRIMARY KEY,
    quote_ref       VARCHAR(30),
    created_at      TIMESTAMP DEFAULT NOW(),
    age             INT,
    gender          VARCHAR(10),
    country         VARCHAR(20),
    region          VARCHAR(30),
    smoking         VARCHAR(20),
    exercise        VARCHAR(20),
    occupation      VARCHAR(30),
    preexist_count  INT DEFAULT 0,
    ipd_tier        VARCHAR(20),
    include_opd     BOOLEAN DEFAULT FALSE,
    include_dental  BOOLEAN DEFAULT FALSE,
    include_maternity BOOLEAN DEFAULT FALSE,
    family_size     INT DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_hp_behavior_created ON hp_user_behavior (created_at DESC);

-- Historical claims (for frequency-severity model retraining)
CREATE TABLE IF NOT EXISTS hp_claims (
    id              SERIAL PRIMARY KEY,
    coverage_type   VARCHAR(20) NOT NULL,
    age             INT,
    gender          VARCHAR(10),
    smoking         VARCHAR(20),
    exercise        VARCHAR(20),
    occupation      VARCHAR(30),
    region          VARCHAR(30),
    preexist_count  INT DEFAULT 0,
    claim_count     INT NOT NULL DEFAULT 0,
    claim_amount    DECIMAL(10,2) DEFAULT 0,
    ingested_at     TIMESTAMP DEFAULT NOW(),
    batch_id        VARCHAR(50)
);
CREATE INDEX IF NOT EXISTS idx_hp_claims_coverage ON hp_claims (coverage_type);
CREATE INDEX IF NOT EXISTS idx_hp_claims_batch ON hp_claims (batch_id);

-- Model registry
CREATE TABLE IF NOT EXISTS hp_model_registry (
    id              SERIAL PRIMARY KEY,
    version         VARCHAR(30) UNIQUE NOT NULL,
    coverage_type   VARCHAR(20),
    deployed_at     TIMESTAMP DEFAULT NOW(),
    freq_model_type VARCHAR(50),
    sev_model_type  VARCHAR(50),
    sev_r2          DECIMAL(5,4),
    training_rows   INT,
    status          VARCHAR(20) DEFAULT 'active',
    notes           TEXT
);

SELECT 'Schema created successfully' AS result;
