export type RagQueryMode = 'auto' | 'mix' | 'hybrid' | 'local' | 'global' | 'naive';

export type ConcreteRagQueryMode = Exclude<RagQueryMode, 'auto'>;

export const RAG_QUERY_MODES: readonly RagQueryMode[] = ['auto', 'mix', 'hybrid', 'local', 'global', 'naive'] as const;

export function isRagQueryMode(value: unknown): value is RagQueryMode {
  return typeof value === 'string' && (RAG_QUERY_MODES as readonly string[]).includes(value);
}

export type KgExtractFormat = 'auto' | 'tuple' | 'json';

export type LLMRole = 'query' | 'extract' | 'keywords' | 'summary';

export interface ExtractedEntity {
  name: string;
  canonicalName: string;
  type: string;
  description: string;
}

export interface ExtractedRelation {
  source: string;
  target: string;
  sourceCanonical: string;
  targetCanonical: string;
  keywords: string[];
  description: string;
  weight: number;
}

export interface ChunkExtractionResult {
  entities: ExtractedEntity[];
  relations: ExtractedRelation[];
}

export interface MergedEntityInput {
  entityName: string;
  canonicalName: string;
  entityType: string;
  description: string;
  sourceChunkIds: string[];
  documentIds: string[];
  embedding?: number[];
}

export interface MergedRelationInput {
  sourceEntity: string;
  targetEntity: string;
  sourceCanonical: string;
  targetCanonical: string;
  keywords: string;
  description: string;
  weight: number;
  sourceChunkIds: string[];
  documentIds: string[];
  embedding?: number[];
}

export interface RetrievedEntity {
  id: string;
  entityName: string;
  canonicalName: string;
  entityType: string;
  description: string;
  degree: number;
  sourceChunkIds: string[];
  documentIds: string[];
  similarity: number;
}

export interface RetrievedRelation {
  id: string;
  sourceEntity: string;
  targetEntity: string;
  sourceCanonical: string;
  targetCanonical: string;
  keywords: string;
  description: string;
  weight: number;
  sourceChunkIds: string[];
  documentIds: string[];
  similarity: number;
}

export interface DualKeywords {
  highLevel: string[];
  lowLevel: string[];
}

export interface QueryAnalysisResult {
  standaloneQuery: string;
  highLevelKeywords: string[];
  lowLevelKeywords: string[];
  usedFallback: boolean;
}

export interface TokenBudgetConfig {
  maxEntityTokens: number;
  maxRelationTokens: number;
  maxTotalTokens: number;
}

export interface ContextChunkInput {
  id: string;
  documentId: string;
  documentTitle: string;
  fileName: string;
  fileType: string;
  content: string;
  sectionHeading?: string | null;
  pageNumber?: number | null;
  parentChunkId?: string | null;
  chunkRole?: 'parent' | 'child' | 'standalone';
  effectiveAt?: Date | string | null;
  expiresAt?: Date | string | null;
  supersedesDocumentId?: string | null;
  authorityTier?: number | null;
  temporalBadge?: string;
  similarity: number;
  source?: 'vector' | 'fts' | 'entity' | 'relation' | 'ppr' | 'community' | 'maxsim' | 'hybrid';
}

export interface AssembledContext {
  contextText: string;
  entitiesUsed: RetrievedEntity[];
  relationsUsed: RetrievedRelation[];
  chunksUsed: ContextChunkInput[];
  estimatedTokens: {
    entities: number;
    relations: number;
    chunks: number;
    total: number;
  };
}
