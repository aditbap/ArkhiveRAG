CREATE TABLE IF NOT EXISTS api_keys (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  created_by uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name varchar(120) NOT NULL,
  key_prefix varchar(24) NOT NULL UNIQUE,
  key_hash varchar(64) NOT NULL,
  role varchar(20) NOT NULL DEFAULT 'member',
  expires_at timestamptz,
  revoked_at timestamptz,
  last_used_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_api_keys_org_active
  ON api_keys (org_id, created_at DESC)
  WHERE revoked_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_prefix_active
  ON api_keys (key_prefix)
  WHERE revoked_at IS NULL;
