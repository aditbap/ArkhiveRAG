export const CONSTANTS = {
  // Chunking (LightRAG-aligned sizes + overlap)
  DEFAULT_CHUNK_SIZE: 1200,
  MAX_CHUNK_SIZE: 2000,
  MIN_CHUNK_SIZE: 64,
  CHUNK_OVERLAP: 150,
  PARENT_CHUNK_MAX: 2048,
  MAX_FILE_SIZE_BYTES: 100 * 1024 * 1024, // 100 MB

  // Retrieval & reranking
  DEFAULT_SEARCH_LIMIT: 20,
  DEFAULT_RERANK_LIMIT: 8,
  SCORE_THRESHOLD: 0.2,
  MAX_CONVERSATION_MESSAGES: 50,

  // LightRAG Knowledge Graph & dual-level retrieval defaults
  KG_FIELD_SEP: '<SEP>',
  KG_TUPLE_DELIMITER: '<|#|>',
  KG_COMPLETION_DELIMITER: '<|COMPLETE|>',
  KG_ENTITY_NAME_MAX_LENGTH: 256,
  KG_MAX_EXTRACTION_ENTITIES: 40,
  KG_MAX_EXTRACTION_RECORDS: 100,
  KG_DEFAULT_MAX_GLEANING: 1,
  KG_FORCE_LLM_SUMMARY_ON_MERGE: 8,
  KG_SUMMARY_MAX_TOKENS: 1200,
  KG_SUMMARY_TARGET_TOKENS: 500,
  KG_MAX_SOURCE_IDS_PER_RECORD: 200,
  KG_DEFAULT_MAX_ASYNC: 4,

  // LightRAG Dual-Level Retrieval & Token Budget defaults
  DEFAULT_RAG_MODE: 'mix' as const,
  DEFAULT_KG_TOP_K: 20,
  DEFAULT_CHUNK_TOP_K: 20,
  DEFAULT_KG_COSINE_THRESHOLD: 0.2,
  DEFAULT_RELATED_CHUNKS_PER_ITEM: 5,
  DEFAULT_MAX_ENTITY_TOKENS: 4000,
  DEFAULT_MAX_RELATION_TOKENS: 4000,
  DEFAULT_MAX_TOTAL_TOKENS: 16000,

  // LightRAG i18n-aware recursive chunk separators (paragraph > line > CJK sentence/clause > space > char)
  DEFAULT_CHUNK_SEPARATORS: [
    '\n\n',
    '\n',
    '。',
    '！',
    '？',
    '；',
    '，',
    ' ',
    '',
  ] as readonly string[],
};
