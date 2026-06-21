/**
 * Type definitions for OAP LangChain Integration
 * @module @aporthq/aport-agent-guardrails-langchain
 */

/**
 * Agent context passed during tool execution in LangChain / LangGraph
 */
export interface LangChainAgentContext {
  /** Unique identifier for the agent instance */
  agentId: string;
  /** Session identifier for tracking */
  sessionId?: string;
  /** User identifier if available */
  userId?: string;
  /** LangChain run ID */
  runId?: string;
  /** LangGraph thread/state ID */
  threadId?: string;
  /** Conversation history or state */
  state?: Record<string, unknown>;
  /** Additional context properties */
  [key: string]: unknown;
}

/**
 * OAP API verification request payload
 */
export interface OAPVerifyRequest {
  /** Passport identifier for the agent */
  passport: string;
  /** Policy identifier to evaluate against */
  policy: string;
  /** Name of the tool being called */
  tool: string;
  /** Arguments passed to the tool */
  args?: Record<string, unknown>;
  /** Agent execution context */
  agent_context?: LangChainAgentContext;
}

/**
 * OAP API verification response
 */
export interface OAPVerifyResponse {
  /** Whether the action is allowed */
  allow: boolean;
  /** Reason for the decision */
  reason?: string;
  /** Receipt ID for audit trail */
  receipt_id?: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * OAP Receipt for audit trails
 */
export interface OAPReceipt {
  /** Receipt ID */
  id: string;
  /** Timestamp of the decision */
  timestamp: string;
  /** Agent ID */
  agentId: string;
  /** Tool name */
  tool: string;
  /** Decision result */
  decision: 'approved' | 'denied';
  /** Reason for the decision */
  reason?: string;
  /** Arguments passed to the tool */
  args?: Record<string, unknown>;
  /** Passport used */
  passport?: string;
  /** Policy evaluated */
  policy?: string;
}

/**
 * Error thrown when OAP authorization fails
 */
export class OAPAuthorizationError extends Error {
  /** Receipt for the denied authorization */
  receipt?: OAPReceipt;
  /** Tool that was blocked */
  tool: string;
  /** Decision reason */
  reason: string;
  /** HTTP status code if from API */
  statusCode?: number;

  constructor(
    message: string,
    options: {
      receipt?: OAPReceipt;
      tool?: string;
      reason?: string;
      statusCode?: number;
    }
  ) {
    super(message);
    this.name = 'OAPAuthorizationError';
    this.receipt = options.receipt;
    this.tool = options.tool || 'unknown';
    this.reason = options.reason || 'Authorization denied by OAP policy';
    this.statusCode = options.statusCode;

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, OAPAuthorizationError);
    }
  }
}

/**
 * Configuration options for OAPToolMiddleware
 */
export interface OAPToolMiddlewareOptions {
  /** Passport identifier for the agent (required) */
  passport: string;
  /** Policy identifier to evaluate against (required) */
  policy: string;
  /** APort API endpoint URL */
  apiEndpoint?: string;
  /** API key for APort service */
  apiKey?: string;
  /** Behavior when the APort API is unreachable or errors */
  fallbackOnFailure?: 'deny' | 'allow' | 'error';
  /** Request timeout in milliseconds */
  timeoutMs?: number;
  /** Whether to emit receipt events via callback */
  emitReceipts?: boolean;
  /** Optional callback for receipt events */
  onReceipt?: (receipt: OAPReceipt) => void;
}

/**
 * Represents a LangChain-compatible tool that can be wrapped.
 * Supports both modern Runnable tools (.invoke) and legacy BaseTool (._call).
 */
export interface LangChainToolLike {
  /** Tool name */
  name: string;
  /** Tool description */
  description?: string;
  /** Modern LangChain invoke method */
  invoke?: (input: unknown, config?: unknown) => Promise<unknown>;
  /** Legacy LangChain _call method */
  _call?: (input: unknown, config?: unknown) => Promise<unknown>;
  /** Additional properties */
  [key: string]: unknown;
}

/**
 * Represents a LangGraph node function that can be wrapped.
 */
export type LangGraphNodeFn = (
  state: Record<string, unknown>,
  config?: Record<string, unknown>
) => Promise<Record<string, unknown>>;
