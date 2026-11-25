'use client'

import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { X, Loader2, Copy, Check, Trash2, KeyRound, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { getAccessToken } from '@/lib/auth-client'

const API_URL = process.env.NEXT_PUBLIC_API_URL

interface ApiKeyItem {
  id: string
  name: string
  keyPrefix: string
  role: 'admin' | 'member' | 'viewer'
  expiresAt: string | null
  revokedAt: string | null
  lastUsedAt: string | null
  createdAt: string
}

async function apiKeysFetch<T = unknown>(path = '', init: RequestInit = {}): Promise<T> {
  const token = await getAccessToken()
  if (!token) throw new Error('Not signed in')
  const res = await fetch(`${API_URL}/api-keys${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(init.body ? { 'Content-Type': 'application/json' } : {}),
      ...init.headers,
    },
  })
  const body = await res.json().catch(() => null)
  if (!res.ok) throw new Error(body?.error?.message ?? `Request failed (${res.status})`)
  return body as T
}

export function ApiKeysModal({
  libraryName,
  callerRole,
  onClose,
}: {
  libraryName: string
  callerRole: 'owner' | 'admin'
  onClose: () => void
}) {
  const [keys, setKeys] = useState<ApiKeyItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [role, setRole] = useState<'admin' | 'member' | 'viewer'>('member')
  const [expiresInDays, setExpiresInDays] = useState<string>('90')
  const [creating, setCreating] = useState(false)
  const [mintedSecret, setMintedSecret] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [reloadTick, setReloadTick] = useState(0)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await apiKeysFetch<{ data: ApiKeyItem[] }>('')
        if (!cancelled) {
          setKeys(res.data)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load API keys')
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [reloadTick])

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    setCreating(true)
    setError(null)
    try {
      const days = expiresInDays.trim() ? Number(expiresInDays) : null
      const res = await apiKeysFetch<{ data: ApiKeyItem & { key: string } }>('', {
        method: 'POST',
        body: JSON.stringify({
          name: name.trim(),
          role,
          ...(days ? { expiresInDays: days } : {}),
        }),
      })
      setMintedSecret(res.data.key)
      setName('')
      setReloadTick((t) => t + 1)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create API key')
    } finally {
      setCreating(false)
    }
  }

  const handleRevoke = async (id: string) => {
    setError(null)
    try {
      await apiKeysFetch(`/${id}`, { method: 'DELETE' })
      setReloadTick((t) => t + 1)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to revoke API key')
    }
  }

  const copySecret = async () => {
    if (!mintedSecret) return
    await navigator.clipboard.writeText(mintedSecret)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-xs"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl rounded-xl border border-zinc-200 bg-white p-5 text-zinc-900 shadow-2xl dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-100"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3 border-b border-zinc-200 pb-3 dark:border-zinc-800">
          <div>
            <h2 className="flex items-center gap-2 text-base font-semibold">
              <KeyRound className="h-4 w-4 text-[#00A6CE]" />
              API Keys — {libraryName}
            </h2>
            <p className="mt-1 text-xs text-zinc-500">
              Scoped credentials (<code className="font-mono">ak_live_...</code>) for REST APIs, SDKs, and MCP clients.
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <form onSubmit={handleCreate} className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-12">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Key label (e.g. Production Backend)"
            maxLength={120}
            className="sm:col-span-5 rounded-md border border-zinc-300 bg-zinc-50 px-2.5 py-1.5 text-sm outline-none focus:border-zinc-500 dark:border-zinc-700 dark:bg-zinc-900"
          />
          <select
            value={role}
            onChange={(e) => setRole(e.target.value as 'admin' | 'member' | 'viewer')}
            className="sm:col-span-3 rounded-md border border-zinc-300 bg-zinc-50 px-2 py-1.5 text-sm outline-none dark:border-zinc-700 dark:bg-zinc-900"
          >
            <option value="viewer">viewer</option>
            <option value="member">member</option>
            {callerRole === 'owner' || callerRole === 'admin' ? <option value="admin">admin</option> : null}
          </select>
          <input
            type="number"
            min={1}
            max={365}
            value={expiresInDays}
            onChange={(e) => setExpiresInDays(e.target.value)}
            title="Expires in days"
            className="sm:col-span-2 rounded-md border border-zinc-300 bg-zinc-50 px-2 py-1.5 text-sm outline-none dark:border-zinc-700 dark:bg-zinc-900"
          />
          <Button type="submit" disabled={creating || !name.trim()} className="sm:col-span-2">
            {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Create'}
          </Button>
        </form>

        {mintedSecret && (
          <div className="mt-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs">
            <div className="flex items-center gap-1.5 font-medium text-amber-600 dark:text-amber-400">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
              Copy this secret now — it will not be shown again.
            </div>
            <div className="mt-2 flex items-center gap-2">
              <code className="flex-1 overflow-x-auto rounded bg-zinc-900 px-2.5 py-1.5 font-mono text-[11px] text-zinc-100">
                {mintedSecret}
              </code>
              <Button type="button" size="sm" variant="outline" onClick={copySecret}>
                {copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}
              </Button>
            </div>
          </div>
        )}

        {error && <p className="mt-3 text-xs text-red-500">{error}</p>}

        <div className="mt-4 max-h-64 overflow-y-auto divide-y divide-zinc-200 dark:divide-zinc-800">
          {loading ? (
            <div className="flex justify-center py-6">
              <Loader2 className="h-5 w-5 animate-spin text-zinc-400" />
            </div>
          ) : keys.length === 0 ? (
            <p className="py-6 text-center text-xs text-zinc-500">No API keys created for this workspace yet.</p>
          ) : (
            keys.map((k) => {
              const revoked = Boolean(k.revokedAt)
              return (
                <div key={k.id} className="flex items-center justify-between gap-3 py-2.5 text-sm">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`font-medium truncate ${revoked ? 'line-through text-zinc-400' : ''}`}>
                        {k.name}
                      </span>
                      <span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[11px] font-mono text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                        {k.role}
                      </span>
                      {revoked && (
                        <span className="rounded bg-red-500/10 px-1.5 py-0.5 text-[11px] text-red-500">revoked</span>
                      )}
                    </div>
                    <div className="mt-0.5 text-[11px] text-zinc-500 font-mono">
                      {k.keyPrefix}…
                      {k.lastUsedAt ? ` · last used ${new Date(k.lastUsedAt).toLocaleDateString()}` : ' · never used'}
                      {k.expiresAt ? ` · expires ${new Date(k.expiresAt).toLocaleDateString()}` : ''}
                    </div>
                  </div>
                  {!revoked && (
                    <button
                      onClick={() => handleRevoke(k.id)}
                      className="rounded p-1.5 text-zinc-500 hover:bg-zinc-100 hover:text-red-500 dark:hover:bg-zinc-800"
                      title="Revoke key"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  )}
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}
