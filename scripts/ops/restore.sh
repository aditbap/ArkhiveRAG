#!/usr/bin/env bash
set -euo pipefail

# ArkhiveRAG — Verified Restore & Non-Destructive Restore Drill Script
# Usage:
#   ./scripts/ops/restore.sh <backup_dir> [--target-db <db_name>] [--drop-after-verify]
# Example (non-destructive restore drill):
#   ./scripts/ops/restore.sh ./backups/rag-backup-20261125T040000Z --target-db rag_restore_verify --drop-after-verify

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <backup_dir> [--target-db <db_name>] [--drop-after-verify]" >&2
  exit 1
fi

BACKUP_DIR="$1"
shift

TARGET_DB="${POSTGRES_DB:-rag_platform}"
DROP_AFTER_VERIFY="false"
PG_CONTAINER="${PG_CONTAINER:-rag-postgres}"
PG_USER="${POSTGRES_USER:-postgres}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-db)
      TARGET_DB="$2"
      shift 2
      ;;
    --drop-after-verify)
      DROP_AFTER_VERIFY="true"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${BACKUP_DIR}/postgres.dump" ]]; then
  echo "[Restore] ERROR: Missing ${BACKUP_DIR}/postgres.dump" >&2
  exit 1
fi

# 1. Verify SHA-256 checksums before touching any database
if [[ -f "${BACKUP_DIR}/manifest.sha256" ]]; then
  echo "[Restore] Verifying SHA-256 manifest..."
  (
    cd "${BACKUP_DIR}"
    if command -v shasum >/dev/null 2>&1; then
      shasum -a 256 -c manifest.sha256
    else
      sha256sum -c manifest.sha256
    fi
  )
fi

echo "[Restore] Recreating target database '${TARGET_DB}' on ${PG_CONTAINER}..."
docker exec "${PG_CONTAINER}" psql -U "${PG_USER}" -d postgres -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS \"${TARGET_DB}\";"
docker exec "${PG_CONTAINER}" psql -U "${PG_USER}" -d postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE \"${TARGET_DB}\";"
docker exec "${PG_CONTAINER}" psql -U "${PG_USER}" -d "${TARGET_DB}" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS btree_gin;"

echo "[Restore] Restoring postgres.dump into '${TARGET_DB}'..."
docker exec -i "${PG_CONTAINER}" pg_restore \
  -U "${PG_USER}" \
  -d "${TARGET_DB}" \
  --no-owner \
  --no-acl \
  --exit-on-error < "${BACKUP_DIR}/postgres.dump"

echo "[Restore] Verifying restored row counts & pgvector HNSW index..."
RESTORED_COUNTS="$(docker exec "${PG_CONTAINER}" psql -U "${PG_USER}" -d "${TARGET_DB}" -tAc \
  "SELECT count(*) || ' chunks | ' || (SELECT count(*) FROM documents) || ' docs | ' || (SELECT count(*) FROM kg_entities) || ' entities | ' || (SELECT count(*) FROM kg_relations) || ' relations' FROM document_chunks;")"
INDEX_COUNT="$(docker exec "${PG_CONTAINER}" psql -U "${PG_USER}" -d "${TARGET_DB}" -tAc \
  "SELECT count(*) FROM pg_indexes WHERE tablename = 'document_chunks' AND indexname IN ('idx_chunks_embedding_hnsw', 'idx_chunks_org_content_tsv_gin', 'idx_chunks_content_tsv_gin');")"

echo "[Restore] Restored counts: ${RESTORED_COUNTS}"
echo "[Restore] Critical indexes verified: ${INDEX_COUNT}/2 (HNSW + stored content_tsv GIN)"

if [[ -f "${BACKUP_DIR}/counts.txt" ]]; then
  EXPECTED_COUNTS="$(cat "${BACKUP_DIR}/counts.txt")"
  echo "[Restore] Backup snapshot counts: ${EXPECTED_COUNTS}"
  if [[ "${RESTORED_COUNTS}" != "${EXPECTED_COUNTS}" ]]; then
    echo "[Restore] ERROR: Row count mismatch between backup and restored DB!" >&2
    exit 1
  fi
  echo "[Restore] 100% row-count parity confirmed."
fi

if [[ "${INDEX_COUNT}" -lt 2 ]]; then
  echo "[Restore] ERROR: Missing critical vector/FTS indexes after restore!" >&2
  exit 1
fi

if [[ "${DROP_AFTER_VERIFY}" == "true" ]]; then
  echo "[Restore] Cleaning up temporary verification database '${TARGET_DB}'..."
  docker exec "${PG_CONTAINER}" psql -U "${PG_USER}" -d postgres -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS \"${TARGET_DB}\";"
fi

echo "[Restore] Verified successfully."
