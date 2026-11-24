export interface ChunkResult {
  text: string;
  pageNumber?: number;
  sectionHeading?: string;
  chunkRole?: 'parent' | 'child' | 'standalone';
  parentIndex?: number;
  metadata?: Record<string, unknown>;
}

export interface ParentChildChunkBatch {
  parents: ChunkResult[];
  children: ChunkResult[];
  standalone: ChunkResult[];
}

export interface ChunkerProvider {
  readonly name: string;

  chunk(text: string, metadata?: Record<string, unknown>): Promise<ChunkResult[]>;
}
