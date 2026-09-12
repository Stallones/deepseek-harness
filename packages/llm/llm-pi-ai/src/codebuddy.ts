/**
 * CodeBuddy 官方 Desktop「craft」主对话形态的出站头/请求体派生。
 *
 * 目的：让 cb 路由直连 copilot.tencent.com 时，请求长得像官方 Desktop，
 * 用于防风控/限流。全部逻辑是纯函数 + 极小的会话计时态，不 fork pi-ai。
 *
 * 形态基线：my-wb2api/docs/headers.md（逐头实测表 + ID 生成规则）、
 * my-wb2api/docs/body.md §10（DSH cb 路由 vs 官方 逐项核对）。
 * @module dsh-llm-pi-ai/codebuddy
 */

import { createHash } from 'node:crypto'
import { appendFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import type { Message } from '@deepseek-ai/dsh-llm'

/**
 * 官方 Desktop 版本。静态头里的 UA / X-IDE-Version / X-Product-Version 三处由它派生，
 * 升级时只改这里。
 *
 * 4.11.3 → 4.12.0 的形态差异（my-wb2api/docs/headers.md §9）：`X-Agent-Intent` 的
 * `custom` 形态消失（改 `web-fetch`）；模型改 `deepseek-v4.1-flash`；`reasoning_effort`
 * 上限由 `xhigh` 改 `max`；`max_tokens` 改 128000；**新增必带 `X-Device-Token-Error`**。
 */
const CODEBUDDY_VERSION = '4.12.0'

/** 官方 Desktop 静态头（抓包实录，my-wb2api/docs/headers.md §2）。 */
const CODEBUDDY_STATIC_HEADERS: Readonly<Record<string, string>> = {
  // 必须覆盖 DSH 的 attribution User-Agent（deepseek-harness/…），否则
  // CodeBuddy 安全策略返回 11128 / 400 拦截。
  'User-Agent': `CodeBuddyIDE/${CODEBUDDY_VERSION} CodeBuddy/${CODEBUDDY_VERSION}`,
  'X-IDE-Type': 'CodeBuddyIDE',
  'X-IDE-Name': 'CodeBuddyIDE',
  'X-IDE-Version': CODEBUDDY_VERSION,
  'X-Product-Version': CODEBUDDY_VERSION,
  'X-Env-ID': 'production',
  'X-Product': 'SaaS',
  'X-Domain': 'www.codebuddy.cn',
  'X-Requested-With': 'XMLHttpRequest',
  'sec-fetch-mode': 'cors',
  'accept-language': '*',
  // 4.12.0 起**每条 chat 必带**（4.11.3 完全不发此头）。实测当日 28/28。
  // 语义是「SDK 未初始化」的客户端故障标记——官方真实客户端就是带着它跑的。
  'X-Device-Token-Error': 'sdk_not_initialized',
  // 官方链级「产品码」，实测唯一取值。注意：官方**部分链不带**此头，产生规则
  // 至今未解（headers.md §3.2）；此处按多数形态恒发。
  'X-Product-Code': 'codebuddy',
}

/** 会话内恒定的确定性派生（见 cb-direct-plan §2）。 */
const DERIVED_KEYS = {
  conversation: ':conversation',
  request: ':request',
  requestTrace: ':request-trace',
  trace: ':trace',
  parent: ':parent',
} as const

/**
 * md5 成 32hex。只用于**非 ID 形态**的确定性派生种子：
 * 官方 `x-trace-id` 就是「纯随机 32hex，不是 UUID」（第 13 位不固定为 4），
 * md5 输出正好同形，且天然满足「链内恒定」。
 */
function md5Hex(input: string): string {
  return createHash('md5').update(input).digest('hex')
}

/** n 字节随机 → 2n 位 hex。官方 `x-b3-spanid` / `x-b3-parentspanid` = 8 字节 = **16hex**。 */
function randomHex(bytes: number): string {
  const buf = globalThis.crypto.getRandomValues(new Uint8Array(bytes))
  return Array.from(buf, b => b.toString(16).padStart(2, '0')).join('')
}

/**
 * 把 32hex 掰成 UUIDv4 形态：第 13 位强制 `4`（version），第 17 位落 `89ab`（variant）。
 *
 * 官方 `x-conversation-id` / `-request-id` / `-message-id` 都是**随机 UUIDv4 去横线**，
 * 实测 151/151 第 13 位恒为 `4`（headers.md §4）。直接拿 md5 当这些值会露馅 ——
 * md5 的第 13 位是自由的，与官方形态不符。
 */
function asUuidV4(hex32: string): string {
  const chars = hex32.split('')
  chars[12] = '4'
  chars[16] = '89ab'[Number.parseInt(chars[16] ?? '0', 16) % 4] ?? '8'
  return chars.join('')
}

/** 会话内确定的 **UUIDv4 去横线（32hex）**——官方 conversation / request / message-id 形态。 */
function seededUuidV4Hex(seed: string): string {
  return asUuidV4(md5Hex(seed))
}

/** 会话内确定的 **UUIDv4 保留横线（36 位）**——官方 `x-request-trace-id` 形态。 */
function seededUuidV4(seed: string): string {
  const h = seededUuidV4Hex(seed)
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`
}

/** 每请求唯一的 **UUIDv4 去横线（32hex）**——官方 `x-conversation-message-id` 形态。 */
function randomUuidV4Hex(): string {
  return asUuidV4(randomHex(16))
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
 * 从 CodeBuddy access token（JWT）里取账号 uid —— 官方 `X-User-Id` 的值（`sub` 声明）。
 * 解析不了就返回 undefined，调用方据此省略该头（它是惰性头，缺了不影响功能）。
 */
function userIdFromToken(apiKey: string | undefined): string | undefined {
  if (apiKey === undefined || apiKey === '') return undefined
  const parts = apiKey.split('.')
  if (parts.length !== 3) return undefined
  try {
    const payload: unknown = JSON.parse(Buffer.from(parts[1] ?? '', 'base64url').toString('utf8'))
    if (payload === null || typeof payload !== 'object') return undefined
    const sub = (payload as { sub?: unknown }).sub
    return typeof sub === 'string' && sub.length > 0 ? sub : undefined
  } catch {
    return undefined
  }
}

/**
 * 出站 craft 头（会话头 + 静态头 + trace/b3 + monitor）。
 *
 * ID 形态严格对齐官方（my-wb2api/docs/headers.md §4）：
 * - 会话/请求/message-id：**UUIDv4 去横线**（第 13 位 = `4`）
 * - `x-trace-id` / `x-b3-traceid`：**纯随机 32hex，非 UUID**
 * - `x-b3-spanid` / `x-b3-parentspanid`：**16hex**
 * - `x-request-trace-id`：**UUIDv4 保留横线（36）**
 *
 * @param sessionId - DSH 会话 UUID（`options.sessionId`）。
 * @param modelId - 目标模型 id。
 * @param monitor - 会话计时快照（首轮可为空）。
 * @param apiKey - 该路由的 access token（JWT），用于派生 `X-User-Id`。
 */
export function buildCodeBuddyHeaders(
  sessionId: string,
  modelId: string,
  monitor: CodeBuddyMonitorState | undefined,
  apiKey?: string,
): Record<string, string> {
  const base = sessionId
  // trace 族：官方是纯随机 32hex 且**链内恒定** —— md5 派生同形且稳定。
  const trace = md5Hex(base + DERIVED_KEYS.trace)
  // b3 的 span / parent 官方是 8 字节 = 16hex（此前误用 32hex）。
  const parent = md5Hex(base + DERIVED_KEYS.parent).slice(0, 16)
  const span = randomHex(8)
  // message-id 是**每请求**新的 UUIDv4 去横线；它同时是上游响应的 id 来源
  // （官方把它原样回显为 response.id，下一轮再作为 previous_response_id 发回）。
  const messageId = randomUuidV4Hex()

  const headers: Record<string, string> = {
    'X-Agent-Intent': 'craft',
    'X-Conversation-ID': seededUuidV4Hex(base + DERIVED_KEYS.conversation),
    'X-Conversation-Request-ID': seededUuidV4Hex(base + DERIVED_KEYS.request),
    'X-Conversation-Message-ID': messageId,
    'X-Request-ID': seededUuidV4Hex(base + DERIVED_KEYS.request),
    'X-Model-ID': modelId,
    ...CODEBUDDY_STATIC_HEADERS,
    'X-Request-Trace-Id': seededUuidV4(base + DERIVED_KEYS.requestTrace),
    'X-Trace-ID': trace,
    b3: `${trace}-${span}-1-${parent}`,
    'X-B3-TraceId': trace,
    'X-B3-SpanId': span,
    'X-B3-ParentSpanId': parent,
    'X-B3-Sampled': '1',
  }

  const userId = userIdFromToken(apiKey)
  if (userId !== undefined) headers['X-User-Id'] = userId

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
