/**
 * OAP Tool Middleware for LangChain
 *
 * Implements pre-action authorization for LangChain and LangGraph agents using OAP (Open Agent Protocol).
 * Intercepts every tool call via wrapped `.invoke()` or `._call()` methods and verifies against the APort API.
 *
 * @module @aporthq/aport-agent-guardrails-langchain
 */


import {
  LangChainAgentContext,
  OAPVerifyRequest,
  OAPVerifyResponse,
  OAPReceipt,
  OAPAuthorizationError,
  OAPToolMiddlewareOptions,
  LangChainToolLike,
  LangGraphNodeFn,
} from './types';

/**
 * Receipt generator for audit trails
 */
export class OAPReceiptGenerator {
  /**
   * Generate an approval receipt
   */
  static approved(params: {
    agentId: string;
    tool: string;
    args?: Record<string, unknown>;
    passport?: string;
    policy?: string;
    reason?: string;
  }): OAPReceipt {
    return {
      id: this.generateId(),
      timestamp: new Date().toISOString(),
      agentId: params.agentId,
      tool: params.tool,
      decision: 'approved',
      reason: params.reason,
      args: params.args,
      passport: params.passport,
      policy: params.policy,
    };
  }

  /**
   * Generate a denial receipt
   */
  static denied(params: {
    agentId: string;
    tool: string;
    args?: Record<string, unknown>;
    passport?: string;
    policy?: string;
    reason?: string;
  }): OAPReceipt {
    return {
      id: this.generateId(),
      timestamp: new Date().toISOString(),
      agentId: params.agentId,
      tool: params.tool,
      decision: 'denied',
      reason: params.reason,
      args: params.args,
      passport: params.passport,
      policy: params.policy,
    };
  }

  /**
   * Generate unique receipt ID
   */
  private static generateId(): string {
    return `oap_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }
}

/**
 * OAP Tool Middleware for LangChain
 *
 * Provides deterministic pre-action authorization for tool calls
 * by verifying against the APort guardrail API.
 *
 * Supports:
 * - LangChain tools (via `.invoke()` or `._call()` wrapping)
 * - LangGraph nodes (via function wrapping)
 * - Custom tool-like objects with duck-typed invocation
 *
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
export class OAPToolMiddleware {
  private readonly passport: string;
  private readonly policy: string;
  private readonly apiEndpoint: string;
  private readonly apiKey?: string;
  private readonly fallbackOnFailure: 'deny' | 'allow' | 'error';
  private readonly timeoutMs: number;
  private readonly emitReceipts: boolean;
  private readonly onReceipt?: (receipt: OAPReceipt) => void;

  constructor(options: OAPToolMiddlewareOptions) {
    if (!options.passport || !options.policy) {
      throw new Error('Both passport and policy are required');
    }

    this.passport = options.passport;
    this.policy = options.policy;
    this.apiEndpoint = options.apiEndpoint || 'https://api.aport.io/v1/verify';
    this.apiKey = options.apiKey;
    this.fallbackOnFailure = options.fallbackOnFailure || 'deny';
    this.timeoutMs = options.timeoutMs || 5000;
    this.emitReceipts = options.emitReceipts !== false;
    this.onReceipt = options.onReceipt;
  }

  /**
   * Wrap a LangChain tool to enforce OAP authorization on every invocation.
   *
   * Supports both modern Runnable-style tools (`.invoke()`) and legacy
   * BaseTool instances (`._call()`). The returned wrapper preserves all
   * original tool properties (name, description, schema, etc.).
   *
   * @param tool - The LangChain tool to wrap
   * @param context - Optional agent context to include in verification
   * @returns A guarded tool with the same interface as the original
   */
  wrapTool<T extends LangChainToolLike>(
    tool: T,
    context?: LangChainAgentContext
  ): T {
    const toolName = tool.name || 'unknown-tool';
    const agentContext = context || this.inferAgentContext(tool);

    // Create the wrapper that intercepts invocations
    const wrapper: LangChainToolLike = {
      ...tool,
    };

    // Wrap .invoke() if present (modern Runnable tools)
    if (typeof tool.invoke === 'function') {
      wrapper.invoke = async (input: unknown, config?: unknown): Promise<unknown> => {
        const args = this.inputToArgs(input);
        await this.authorizeToolCall(toolName, args, agentContext);
        return tool.invoke!(input, config);
      };
    }

    // Wrap ._call() if present (legacy BaseTool)
    if (typeof tool._call === 'function') {
      wrapper._call = async (input: unknown, config?: unknown): Promise<unknown> => {
        const args = this.inputToArgs(input);
        await this.authorizeToolCall(toolName, args, agentContext);
        return tool._call!(input, config);
      };
    }

    // If neither invoke nor _call, attempt to make the wrapper callable
    // by proxying function calls (for tool() decorator results)
    if (typeof tool === 'function') {
      const callableWrapper = async (input: unknown, config?: unknown): Promise<unknown> => {
        const args = this.inputToArgs(input);
        await this.authorizeToolCall(toolName, args, agentContext);
        return (tool as unknown as (input: unknown, config?: unknown) => Promise<unknown>)(input, config);
      };
      // Copy enumerable properties from original tool to callable wrapper
      Object.keys(tool).forEach((key) => {
        try {
          (callableWrapper as unknown as Record<string, unknown>)[key] = (tool as unknown as Record<string, unknown>)[key];
        } catch {
          // Ignore read-only property errors
        }
      });
      // Preserve non-enumerable .name explicitly
      const originalName = (tool as unknown as Record<string, unknown>).name;
      try {
        Object.defineProperty(callableWrapper as unknown as Record<string, unknown>, 'name', {
          value: originalName,
          configurable: true,
        });
      } catch {
        // Ignore if name cannot be set
      }
      return callableWrapper as unknown as T;
    }

    return wrapper as unknown as T;
  }

  /**
   * Wrap a LangGraph node function to enforce OAP authorization.
   *
   * LangGraph nodes receive the current graph state and return updates.
   * This wrapper intercepts the node execution, extracts any tool call
   * intent from the state, and verifies authorization before proceeding.
   *
   * @param nodeFn - The LangGraph node function to wrap
   * @param nodeName - Name of the node (used as the tool name in OAP)
   * @param context - Optional agent context to include in verification
   * @returns A guarded node function with the same interface
   */
  wrapNode(
    nodeFn: LangGraphNodeFn,
    nodeName: string,
    context?: LangChainAgentContext
  ): LangGraphNodeFn {
    const agentContext = context || this.inferAgentContext({ name: nodeName });

    return async (
      state: Record<string, unknown>,
      config?: Record<string, unknown>
    ): Promise<Record<string, unknown>> => {
      // Extract tool call arguments from state if available
      const args = this.extractStateArgs(state, nodeName);
      await this.authorizeToolCall(nodeName, args, agentContext);
      return nodeFn(state, config);
    };
  }

  /**
   * Authorize a single tool call via the APort API.
   *
   * @param tool - Name of the tool being called
   * @param args - Arguments passed to the tool
   * @param agentContext - Agent execution context
   * @throws {OAPAuthorizationError} If the tool call is denied by policy
   */
  private async authorizeToolCall(
    tool: string,
    args: Record<string, unknown>,
    agentContext: LangChainAgentContext
  ): Promise<void> {
    const agentId = agentContext?.agentId || 'unknown-agent';

    let apiResult: OAPVerifyResponse | undefined;
    let apiError: Error | undefined;

    try {
      apiResult = await this.verifyWithAPI({
        tool,
        args,
        agentContext,
      });
    } catch (error) {
      apiError = error instanceof Error ? error : new Error(String(error));
    }

    // Handle API failure with fallback strategy
    if (!apiResult) {
      return this.handleApiFailure(apiError, { agentId, tool, args });
    }

    // Handle explicit denial
    if (!apiResult.allow) {
      const receipt = OAPReceiptGenerator.denied({
        agentId,
        tool,
        args,
        passport: this.passport,
        policy: this.policy,
        reason: apiResult.reason || 'Denied by OAP policy',
      });

      this.emitReceipt(receipt);

      throw new OAPAuthorizationError(
        `Tool "${tool}" blocked by OAP policy: ${apiResult.reason || 'Authorization denied'}`,
        {
          receipt,
          tool,
          reason: apiResult.reason || 'Authorization denied by OAP policy',
        }
      );
    }

    // Handle approval
    const receipt = OAPReceiptGenerator.approved({
      agentId,
      tool,
      args,
      passport: this.passport,
      policy: this.policy,
      reason: apiResult.reason,
    });

    this.emitReceipt(receipt);
  }

  /**
   * Verify a tool call against the APort API
   */
  private async verifyWithAPI(params: {
    tool: string;
    args: Record<string, unknown>;
    agentContext: LangChainAgentContext;
  }): Promise<OAPVerifyResponse> {
    const payload: OAPVerifyRequest = {
      passport: this.passport,
      policy: this.policy,
      tool: params.tool,
      args: params.args,
      agent_context: params.agentContext,
    };

    return this.postJSON<OAPVerifyResponse>(this.apiEndpoint, payload);
  }

  /**
   * Handle API failure based on fallback configuration
   */
  private handleApiFailure(
    error: Error | undefined,
    params: {
      agentId: string;
      tool: string;
      args?: Record<string, unknown>;
    }
  ): void {
    const { agentId, tool, args } = params;

    switch (this.fallbackOnFailure) {
      case 'allow': {
        const receipt = OAPReceiptGenerator.approved({
          agentId,
          tool,
          args,
          passport: this.passport,
          policy: this.policy,
          reason: `Allowed after API failure: ${error?.message || 'Unknown error'}`,
        });
        this.emitReceipt(receipt);
        return;
      }

      case 'error': {
        const receipt = OAPReceiptGenerator.denied({
          agentId,
          tool,
          args,
          passport: this.passport,
          policy: this.policy,
          reason: `API error: ${error?.message || 'Unknown error'}`,
        });
        this.emitReceipt(receipt);
        throw new OAPAuthorizationError(
          `Tool "${tool}" blocked due to OAP API error: ${error?.message || 'Unknown error'}`,
          {
            receipt,
            tool,
            reason: `API error: ${error?.message || 'Unknown error'}`,
          }
        );
      }

      case 'deny':
      default: {
        const receipt = OAPReceiptGenerator.denied({
          agentId,
          tool,
          args,
          passport: this.passport,
          policy: this.policy,
          reason: `Denied after API failure: ${error?.message || 'Unknown error'}`,
        });
        this.emitReceipt(receipt);
        throw new OAPAuthorizationError(
          `Tool "${tool}" blocked by OAP policy (fail-closed): ${error?.message || 'API unreachable'}`,
          {
            receipt,
            tool,
            reason: `Denied after API failure: ${error?.message || 'Unknown error'}`,
          }
        );
      }
    }
  }

  /**
   * Emit a receipt via callback if configured
   */
  private emitReceipt(receipt: OAPReceipt): void {
    if (this.emitReceipts && this.onReceipt) {
      try {
        this.onReceipt(receipt);
      } catch {
        // Ignore receipt callback errors
      }
    }
  }

  /**
   * Convert tool input to a serializable args record
   */
  private inputToArgs(input: unknown): Record<string, unknown> {
    if (input && typeof input === 'object') {
      return input as Record<string, unknown>;
    }
    return { input };
  }

  /**
   * Infer agent context from tool metadata if not explicitly provided
   */
  private inferAgentContext(tool: LangChainToolLike): LangChainAgentContext {
    return {
      agentId: 'langchain-agent',
      sessionId: undefined,
      toolName: tool.name,
    };
  }

  /**
   * Extract arguments from LangGraph state for authorization
   */
  private extractStateArgs(
    state: Record<string, unknown>,
    nodeName: string
  ): Record<string, unknown> {
    // Common LangGraph state keys that contain tool arguments
    if (state.messages && Array.isArray(state.messages)) {
      const lastMessage = state.messages[state.messages.length - 1];
      if (lastMessage && typeof lastMessage === 'object') {
        const msg = lastMessage as Record<string, unknown>;
        if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
          const relevant = msg.tool_calls.find(
            (tc: unknown) =>
              tc &&
              typeof tc === 'object' &&
              (tc as Record<string, unknown>).name === nodeName
          );
          if (relevant && typeof relevant === 'object') {
            return ((relevant as Record<string, unknown>).args as Record<string, unknown>) || {};
          }
        }
        if (msg.additional_kwargs && typeof msg.additional_kwargs === 'object') {
          return (msg.additional_kwargs as Record<string, unknown>) || {};
        }
      }
    }
    return { stateKeys: Object.keys(state) };
  }

  /**
   * Perform an HTTP POST with JSON payload using native fetch.
   * Compatible with Node 18+, Cloudflare Workers, Deno, Bun, and browsers.
   */
  private async postJSON<T>(url: string, payload: unknown): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const data = await response.text();

      if (!response.ok) {
        throw new Error(
          `OAP API returned ${response.status}: ${data || 'No response body'}`
        );
      }

      try {
        return JSON.parse(data) as T;
      } catch {
        throw new Error(`Invalid JSON response: ${data}`);
      }
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timed out after ${this.timeoutMs}ms`);
      }

      throw error;
    }
  }

  /**
   * Get middleware configuration (non-sensitive)
   */
  getConfig(): Omit<OAPToolMiddlewareOptions, 'apiKey' | 'onReceipt'> {
    return {
      passport: this.passport,
      policy: this.policy,
      apiEndpoint: this.apiEndpoint,
      fallbackOnFailure: this.fallbackOnFailure,
      timeoutMs: this.timeoutMs,
      emitReceipts: this.emitReceipts,
    };
  }
}

