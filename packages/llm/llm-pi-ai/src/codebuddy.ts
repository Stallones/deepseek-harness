/**
 * CodeBuddy 官方 Desktop「craft」主对话形态的出站头/请求体派生。
 *
 * 目的：让 cb 路由直连 copilot.tencent.com 时，请求长得像官方 Desktop，
 * 用于防风控/限流。全部逻辑是纯函数 + 极小的会话计时态，不 fork pi-ai。
 *
 * 关联文档：my-wb2api/docs/cb-direct-plan.md
 * @module dsh-llm-pi-ai/codebuddy
 */

import { createHash } from 'node:crypto'
import { appendFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import type { Message } from '@deepseek-ai/dsh-llm'

/** 官方 Desktop 静态头（抓包实录，live-body §4）。 */
const CODEBUDDY_STATIC_HEADERS: Readonly<Record<string, string>> = {
  // 必须覆盖 DSH 的 attribution User-Agent（deepseek-harness/…），否则
  // CodeBuddy 安全策略返回 11128 / 400 拦截。
  'User-Agent': 'CodeBuddyIDE/4.11.3 CodeBuddy/4.11.3',
  'X-IDE-Type': 'CodeBuddyIDE',
  'X-IDE-Name': 'CodeBuddyIDE',
  'X-IDE-Version': '4.11.3',
  'X-Product-Version': '4.11.3',
  'X-Env-ID': 'production',
  'X-Product': 'SaaS',
  'X-Requested-With': 'XMLHttpRequest',
}

/** 会话内恒定的确定性派生（见 cb-direct-plan §2）。 */
const DERIVED_KEYS = {
  conversation: ':conversation',
  request: ':request',
  requestTrace: ':request-trace',
  trace: ':trace',
  parent: ':parent',
} as const

/** md5 成 32hex（官方 conversation/request/trace/parent 均为 32hex）。 */
function md5Hex(input: string): string {
  return createHash('md5').update(input).digest('hex')
}

/** 每请求唯一的 32hex（message-id / span；官方同款随机）。 */
function randomHex(): string {
  const bytes = globalThis.crypto.getRandomValues(new Uint8Array(16))
  return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('')
}

/**
 * 会话级计时快照。纯内存、重启失效；与官方 Desktop 进程内计时语义一致。
 * 首轮 firstByte/streamEnd 缺省（官方首轮这两个值同样为空）。
 */
export interface CodeBuddyMonitorState {
  /** 会话首轮 prompt 组装时间（毫秒），后续轮复用。 */
  promptPrepareStart?: number
  /** 上一响应头到达时刻（毫秒）。 */
  firstByte?: number
  /** 上一响应流结束时刻（毫秒）。 */
  streamEnd?: number
}

/**
 * 出站 craft 头（会话头 + 静态头 + trace/b3 + monitor）。
 * @param sessionId - DSH 会话 UUID（`options.sessionId`）。
 * @param modelId - 目标模型 id。
 * @param monitor - 会话计时快照（首轮可为空）。
 */
export function buildCodeBuddyHeaders(
  sessionId: string,
  modelId: string,
  monitor: CodeBuddyMonitorState | undefined,
): Record<string, string> {
  const base = sessionId
  const trace = md5Hex(base + DERIVED_KEYS.trace)
  const parent = md5Hex(base + DERIVED_KEYS.parent)
  const span = randomHex()
  const messageId = randomHex()

  const headers: Record<string, string> = {
    'X-Agent-Intent': 'craft',
    'X-Conversation-ID': md5Hex(base + DERIVED_KEYS.conversation),
    'X-Conversation-Request-ID': md5Hex(base + DERIVED_KEYS.request),
    'X-Conversation-Message-ID': messageId,
    'X-Request-ID': md5Hex(base + DERIVED_KEYS.request),
    'X-Model-ID': modelId,
    ...CODEBUDDY_STATIC_HEADERS,
    'X-Request-Trace-Id': md5Hex(base + DERIVED_KEYS.requestTrace),
    'X-Trace-ID': trace,
    b3: `${trace}-${span}-1-${parent}`,
    'X-B3-TraceId': trace,
    'X-B3-SpanId': span,
    'X-B3-ParentSpanId': parent,
    'X-B3-Sampled': '1',
  }

  if (monitor?.promptPrepareStart !== undefined) {
    headers['monitor_promptPrepareStartTime'] = String(monitor.promptPrepareStart)
  }
  headers['monitor_httpSendTime'] = String(Date.now())
  // firstByte / streamEnd 是「上一响应」的测量值，首轮缺省（官方同款）。
  if (monitor?.firstByte !== undefined) {
    headers['monitor_firstByteTime'] = String(monitor.firstByte)
  }
  if (monitor?.streamEnd !== undefined) {
    headers['monitor_streamEndTime'] = String(monitor.streamEnd)
  }
  return headers
}

/**
 * 从历史消息里取上一轮 assistant 的 responseId（== 上游 SSE chunk id ==
 * 官方 message-id），作为本轮 body 的 previous_response_id。
 *
 * 只在 replayState 是 pi-ai v2 envelope 且带 responseId 时返回，否则 undefined
 * （首轮 / compaction 断链 / 状态不可用均回退缺省，全量 messages 已证明 200）。
 */
export function derivePreviousResponseId(messages: readonly Message[]): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i]
    if (message === undefined || message.role !== 'assistant') continue
    const source = (message as { source?: unknown }).source
    if (source === null || typeof source !== 'object') continue
    const replayState = (source as { replayState?: unknown }).replayState
    if (replayState === null || typeof replayState !== 'object') continue
    const response = (replayState as { response?: unknown }).response
    if (response === null || typeof response !== 'object') continue
    const r = response as { kind?: unknown; version?: unknown; responseId?: unknown }
    if (r.kind !== 'pi-ai' || r.version !== 2) continue
    if (typeof r.responseId === 'string' && r.responseId.length > 0) return r.responseId
  }
  return undefined
}

/**
 * 出站 wire 日志（仅 cb 路由）：把 pi-ai/OpenAI SDK 真正发出的
 * URL/headers/body 写成一个 JSONL 文件，便于核对 craft 形态。
 *
 * 开关：环境变量 `DSH_CB_LOG`。
 * - 设成 `1` 或 `true`：写到 `<DSH_HOME>/my-dsh/cb-outbound.log`（缺省
 *   `~/.dsh/my-dsh/cb-outbound.log`）。
 * - 设成其它路径：写到该路径。
 * - 不设置：完全不开启（无文件写入开销）。
 */
function resolveCbLogPath(): string | undefined {
  const value = process.env.DSH_CB_LOG
  if (value === undefined || value === '') return undefined
  if (value === '1' || value === 'true') {
    const home = process.env.DSH_HOME ?? join(homedir(), '.dsh')
    return join(home, 'my-dsh', 'cb-outbound.log')
  }
  return value
}

/** 日志里隐藏 Authorization/Cookie，避免明文 token 落盘。 */
function redactHeaders(headers: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {}
  for (const [name, value] of Object.entries(headers)) {
    const lower = name.toLowerCase()
    out[name] = lower === 'authorization' || lower === 'cookie' || lower === 'set-cookie'
      ? value.slice(0, 20) + '…(redacted)'
      : value
  }
  return out
}

async function appendCbLog(entry: unknown): Promise<void> {
  const path = resolveCbLogPath()
  if (path === undefined) return
  try {
    await appendFile(path, JSON.stringify(entry) + '\n', 'utf8')
  } catch {
    // 日志失败绝不能影响 LLM 请求。
  }
}

/**
 * 为 cb 路由构造一个 fetch 包装器；它先记录最终请求，再调用真实 fetch。
 * @param sessionId - DSH 会话 id（只进日志）。
 * @param modelId - 请求模型 id（只进日志）。
 * @returns 包装 fetch；未开启日志时返回 undefined。
 */
export function codeBuddyDebugFetch(
  sessionId: string,
  modelId: string,
): typeof globalThis.fetch | undefined {
  if (resolveCbLogPath() === undefined) return undefined
  const realFetch = globalThis.fetch
  return async (input, init) => {
    try {
      const isRequest = typeof Request !== 'undefined' && input instanceof Request
      const url = typeof input === 'string'
        ? input
        : isRequest
          ? input.url
          : String(input)
      const method = init?.method ?? (isRequest ? input.method : 'POST')
      const headersRecord: Record<string, string> = {}
      const headerSource = init?.headers ?? (isRequest ? input.headers : undefined)
      if (headerSource instanceof Headers) {
        for (const [name, value] of headerSource.entries()) headersRecord[name] = value
      } else if (Array.isArray(headerSource)) {
        for (const [name, value] of headerSource) headersRecord[String(name)] = String(value)
      } else if (typeof headerSource === 'object' && headerSource !== null) {
        for (const [name, value] of Object.entries(headerSource)) headersRecord[name] = String(value)
      }
      let bodyText = ''
      if (init?.body !== undefined && init.body !== null) {
        bodyText = typeof init.body === 'string'
          ? init.body
          : await new Response(init.body).text()
      } else if (isRequest && input.body !== null) {
        bodyText = await new Response(input.body).text()
      }
      void appendCbLog({
        time: new Date().toISOString(),
        type: 'cb-request',
        sessionId,
        modelId,
        method,
        url,
        headers: redactHeaders(headersRecord),
        bodyLength: bodyText.length,
        body: bodyText,
      })
    } catch {
      // 记录失败不影响请求。
    }
    return realFetch(input, init)
  }
}

/**
 * 会话计时态容器：`sessionId → 快照`。为避免无限增长，超过上限淘汰最久未用。
 * 官方 Desktop 同样是进程内计时，本实现以最小 LRU 兜底内存泄漏。
 */
export class CodeBuddyMonitorStore {
  private readonly state = new Map<string, CodeBuddyMonitorState>()
  private readonly order = new Map<string, number>()
  private cursor = 0

  constructor(private readonly limit = 1000) {}

  /** 取（不存在则建）一个会话的计时快照，并刷新其最近使用位次。 */
  snapshot(sessionId: string): CodeBuddyMonitorState {
    const existing = this.state.get(sessionId)
    if (existing !== undefined) {
      this.touch(sessionId)
      return existing
    }
    const fresh: CodeBuddyMonitorState = {}
    this.state.set(sessionId, fresh)
    this.touch(sessionId)
    this.evict()
    return fresh
  }

  private touch(sessionId: string): void {
    this.order.set(sessionId, ++this.cursor)
  }

  private evict(): void {
    if (this.state.size <= this.limit) return
    let oldest: string | undefined
    let oldestOrder = Number.POSITIVE_INFINITY
    for (const [id, ord] of this.order) {
      if (ord < oldestOrder) {
        oldestOrder = ord
        oldest = id
      }
    }
    if (oldest !== undefined) {
      this.state.delete(oldest)
      this.order.delete(oldest)
    }
  }
}
