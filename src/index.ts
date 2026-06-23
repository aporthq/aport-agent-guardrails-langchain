/**
 * OAP (Open Agent Protocol) Guardrails for LangChain
 *
 * Deterministic pre-action authorization for LangChain and LangGraph AI agents.
 * Enforces policy-based tool access control via the APort guardrail API.
 *
 * @packageDocumentation
 * @module @aporthq/aport-agent-guardrails-langchain
 * @example
 * ```typescript
 * import { OAPToolMiddleware } from '@aporthq/aport-agent-guardrails-langchain';
 *
 * const middleware = new OAPToolMiddleware({
 *   passport: 'passport-agent-123',
 *   policy: 'policy-sandbox-456',
 *   apiKey: process.env.APORT_API_KEY,
 * });
 *
 * const guardedTool = middleware.wrapTool(webSearchTool);
 * const result = await guardedTool.invoke({ query: 'AI safety' });
 * ```
 */

// Main middleware
export { OAPToolMiddleware, OAPReceiptGenerator } from './oap-tool-middleware';

// Types
export {
  LangChainAgentContext,
  OAPVerifyRequest,
  OAPVerifyResponse,
  OAPReceipt,
  OAPAuthorizationError,
  OAPToolMiddlewareOptions,
  LangChainToolLike,
  LangGraphNodeFn,
} from './types';

// Re-export for convenience
export { OAPToolMiddleware as default } from './oap-tool-middleware';
