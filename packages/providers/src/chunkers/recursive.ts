import {
  ChunkerProvider,
  ChunkResult,
  CONSTANTS,
  estimateTokens,
  ParsedDocument,
  type ParentChildChunkBatch,
} from '@rag/core';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

interface HeadingAnchor {
  offset: number;
  level: number;
  title: string;
}

/**
 * Extracts Markdown-style and numbered-section headings with their character
 * offsets so every chunk can carry a LightRAG-style `H1 → H2 → H3` breadcrumb.
 */
function extractHeadingAnchors(text: string): HeadingAnchor[] {
  const anchors: HeadingAnchor[] = [];
  const lines = text.split('\n');
  let offset = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    const mdMatch = trimmed.match(/^(#{1,6})\s+(.+?)\s*#*$/);
    if (mdMatch?.[1] && mdMatch[2]) {
      anchors.push({
        offset,
        level: mdMatch[1].length,
        title: mdMatch[2].trim().slice(0, 80),
      });
    } else if (/^(?:Chapter|Section|Bab|Bagian|Pasal)\s+[\dIVXLCM]+[.:]?\s+\S+/i.test(trimmed) && trimmed.length <= 100) {
      anchors.push({
        offset,
        level: 1,
        title: trimmed.slice(0, 80),
      });
    } else if (/^\d+(?:\.\d+){0,2}\s+[A-Z][\p{L}\p{N}\s,&()/-]{2,70}$/u.test(trimmed)) {
      const depth = trimmed.split(/\s+/)[0]?.split('.').length ?? 1;
      anchors.push({
        offset,
        level: Math.min(3, depth),
        title: trimmed.slice(0, 80),
      });
    }
    offset += line.length + 1;
  }

  return anchors;
}

function resolveHeadingPath(anchors: HeadingAnchor[], offset: number): string | undefined {
  if (anchors.length === 0) return undefined;
  const stack: Array<{ level: number; title: string }> = [];

  for (const anchor of anchors) {
    if (anchor.offset > offset) break;
    while (stack.length > 0 && (stack[stack.length - 1]?.level ?? 0) >= anchor.level) {
      stack.pop();
    }
    stack.push({ level: anchor.level, title: anchor.title });
  }

  if (stack.length === 0) return undefined;
  // Collapse very deep hierarchies to `top → … → leaf` (mirrors LightRAG's breadcrumb rule)
  if (stack.length > 3) {
    return `${stack[0]?.title} → … → ${stack[stack.length - 1]?.title}`;
  }
  return stack.map((s) => s.title).join(' → ');
}

export class RecursiveChunker implements ChunkerProvider {
  public readonly name = 'recursive-character-i18n';
  private splitter: RecursiveCharacterTextSplitter;

  constructor(
    chunkSize: number = CONSTANTS.DEFAULT_CHUNK_SIZE,
    chunkOverlap: number = CONSTANTS.CHUNK_OVERLAP,
    separators: readonly string[] = CONSTANTS.DEFAULT_CHUNK_SEPARATORS,
  ) {
    this.splitter = new RecursiveCharacterTextSplitter({
      chunkSize,
      chunkOverlap,
      separators: [...separators],
    });
  }

  async chunk(text: string, metadata?: Record<string, unknown>): Promise<ChunkResult[]> {
    const anchors = extractHeadingAnchors(text);
    const docs = await this.splitter.createDocuments([text], [metadata || {}]);

    let searchCursor = 0;
    return docs.map((doc) => {
      const content = doc.pageContent;
      const snippet = content.slice(0, 80);
      let pos = text.indexOf(snippet, searchCursor);
      if (pos === -1) pos = text.indexOf(snippet);
      if (pos !== -1) searchCursor = Math.max(searchCursor, pos);

      const sectionHeading = pos !== -1 ? resolveHeadingPath(anchors, pos) : undefined;
      return {
        text: content,
        sectionHeading,
        metadata: {
          ...doc.metadata,
          ...(sectionHeading ? { sectionHeading } : {}),
        },
      };
    });
  }

  /**
   * Structure-aware document chunking: preserves per-page provenance (`pageNumber`)
   * when the parser returned page boundaries (PDFs, OCR'd docs) AND attaches
   * `sectionHeading` breadcrumbs to every chunk.
   */
  async chunkDocument(parsed: ParsedDocument): Promise<ChunkResult[]> {
    if (parsed.pages && parsed.pages.length > 1) {
      const allChunks: ChunkResult[] = [];
      let lastHeading: string | undefined;

      for (const page of parsed.pages) {
        const trimmed = page.text?.trim();
        if (!trimmed) continue;
        const pageChunks = await this.chunk(trimmed, { pageNumber: page.number });
        for (const c of pageChunks) {
          if (c.sectionHeading) lastHeading = c.sectionHeading;
          allChunks.push({
            ...c,
            pageNumber: page.number,
            sectionHeading: c.sectionHeading ?? lastHeading,
          });
        }
      }
      if (allChunks.length > 0) return allChunks;
    }

    const singlePage = parsed.pages?.[0]?.number;
    const chunks = await this.chunk(parsed.text);
    return chunks.map((c) => ({
      ...c,
      ...(singlePage != null ? { pageNumber: singlePage } : {}),
    }));
  }

  /**
   * Semantic + Parent-Child Document Chunking (#7):
   * - Short documents (<= parentThresholdTokens, default 380 tokens) stay as `'standalone'`
   *   chunks with zero extra storage or latency overhead.
   * - Longer documents are split into coherent parent sections (~700-900 tokens, preserving
   *   heading breadcrumbs & page numbers) and tight semantic child chunks (~250-300 tokens)
   *   pointing to their parent via `parentIndex`.
   *
   * Child chunks are embedded for pinpoint precision; winning children expand to their
   * deduplicated parent section at generation time within strict per-intent token budgets.
   */
  async chunkDocumentParentChild(
    parsed: ParsedDocument,
    options?: {
      parentSizeChars?: number;
      parentOverlapChars?: number;
      childSizeChars?: number;
      childOverlapChars?: number;
      minDocTokensForHierarchy?: number;
    },
  ): Promise<ParentChildChunkBatch> {
    const totalTokens = estimateTokens(parsed.text);
    const minTokens = options?.minDocTokensForHierarchy ?? 380;

    if (totalTokens <= minTokens) {
      const standalone = (await this.chunkDocument(parsed)).map((c) => ({
        ...c,
        chunkRole: 'standalone' as const,
      }));
      return { parents: [], children: [], standalone };
    }

    // Parent chunks capped at ~750-900 tokens (~3000-3400 chars) to keep prompt tokens tight
    const parentSize = options?.parentSizeChars ?? 3000;
    const parentOverlap = options?.parentOverlapChars ?? 250;
    const childSize = options?.childSizeChars ?? 950;
    const childOverlap = options?.childOverlapChars ?? 120;

    const parentChunker = new RecursiveChunker(parentSize, parentOverlap);
    const childSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: childSize,
      chunkOverlap: childOverlap,
      separators: [...CONSTANTS.DEFAULT_CHUNK_SEPARATORS],
    });

    const rawParents = await parentChunker.chunkDocument(parsed);
    const parents: ChunkResult[] = [];
    const children: ChunkResult[] = [];
    const standalone: ChunkResult[] = [];

    for (const parent of rawParents) {
      // If a parent section is already compact (<= 320 tokens), keep it as standalone instead of duplicating
      if (estimateTokens(parent.text) <= 320) {
        standalone.push({
          ...parent,
          chunkRole: 'standalone',
        });
        continue;
      }

      const parentIndex = parents.length;
      parents.push({
        ...parent,
        chunkRole: 'parent',
      });

      const subDocs = await childSplitter.createDocuments([parent.text], [parent.metadata || {}]);
      for (const sub of subDocs) {
        const text = sub.pageContent.trim();
        if (!text) continue;
        children.push({
          text,
          pageNumber: parent.pageNumber,
          sectionHeading: parent.sectionHeading,
          chunkRole: 'child',
          parentIndex,
          metadata: {
            ...parent.metadata,
            chunkRole: 'child',
          },
        });
      }
    }

    return { parents, children, standalone };
  }
}
