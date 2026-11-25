#!/usr/bin/env bash
set -euo pipefail

# ArkhiveRAG — Production Backup Script (PostgreSQL + MinIO + SHA-256 Manifest)
# Usage:
#   DATABASE_URL="postgresql://..." ./scripts/ops/backup.sh [backup_root_dir]
# Optional env vars:
#   PG_CONTAINER=rag-postgres   (uses docker exec when pg_dump isn't installed locally)
#   MINIO_CONTAINER=rag-minio   (snapshots /data from the MinIO container when mc isn't installed)

BACKUP_ROOT="${1:-./backups}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${BACKUP_ROOT}/rag-backup-${TIMESTAMP}"
PG_CONTAINER="${PG_CONTAINER:-rag-postgres}"
MINIO_CONTAINER="${MINIO_CONTAINER:-rag-minio}"
PG_USER="${POSTGRES_USER:-postgres}"
PG_DB="${POSTGRES_DB:-rag_platform}"

mkdir -p "${OUT_DIR}"
echo "[Backup] Creating backup in ${OUT_DIR}..."

# 1. PostgreSQL custom-format compressed dump (preserves pgvector vectors, HNSW indexes, KG, & ACLs)
if command -v pg_dump >/dev/null 2>&1 && [[ -n "${DATABASE_URL:-}" ]]; then
  echo "[Backup] Dumping PostgreSQL via local pg_dump..."
  pg_dump "${DATABASE_URL}" --format=custom --no-owner --no-acl --file="${OUT_DIR}/postgres.dump"
else
  echo "[Backup] Dumping PostgreSQL via docker exec (${PG_CONTAINER})..."
  docker exec "${PG_CONTAINER}" pg_dump -U "${PG_USER}" -d "${PG_DB}" --format=custom --no-owner --no-acl > "${OUT_DIR}/postgres.dump"
fi

# 2. MinIO object storage snapshot (uses `docker cp`, so it works even on distroless Chainguard images)
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "${MINIO_CONTAINER}"; then
  echo "[Backup] Archiving MinIO /data volume from ${MINIO_CONTAINER}..."
  TMP_MINIO_DIR="${OUT_DIR}/minio-data-tmp"
  docker cp "${MINIO_CONTAINER}:/data" "${TMP_MINIO_DIR}"
  tar -czf "${OUT_DIR}/minio-data.tar.gz" -C "${TMP_MINIO_DIR}" .
  rm -rf "${TMP_MINIO_DIR}"
else
  echo "[Backup] Skipping MinIO snapshot (container ${MINIO_CONTAINER} not running locally)."
fi

# 3. Record schema migration state & SHA-256 integrity manifest
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "${PG_CONTAINER}"; then
  docker exec "${PG_CONTAINER}" psql -U "${PG_USER}" -d "${PG_DB}" -tAc \
    "SELECT count(*) || ' chunks | ' || (SELECT count(*) FROM documents) || ' docs | ' || (SELECT count(*) FROM kg_entities) || ' entities | ' || (SELECT count(*) FROM kg_relations) || ' relations' FROM document_chunks;" \
    > "${OUT_DIR}/counts.txt" || true
fi

(
  cd "${OUT_DIR}"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 postgres.dump $( [[ -f minio-data.tar.gz ]] && echo minio-data.tar.gz ) > manifest.sha256
  else
    sha256sum postgres.dump $( [[ -f minio-data.tar.gz ]] && echo minio-data.tar.gz ) > manifest.sha256
  fi
)

echo "[Backup] Completed successfully:"
ls -lh "${OUT_DIR}"
cat "${OUT_DIR}/manifest.sha256"
