export type RagMode = 'auto' | 'mix' | 'hybrid' | 'local' | 'global' | 'naive'

export interface Citation {
  index?: number
  documentId?: string
  document?: string
  documentTitle?: string
  fileName?: string
  fileType?: string
  score?: number
  pageNumber?: number
  sectionHeading?: string
  snippet?: string
}

export interface RetrievalMetadata {
  mode: RagMode
  requestedMode?: RagMode
  entitiesCount?: number
  relationsCount?: number
  chunksCount?: number
  estimatedTokens?: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  retrieval?: RetrievalMetadata
}
