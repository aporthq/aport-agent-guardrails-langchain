/**
 * OAP Tool Middleware Tests for LangChain
 *
 * Comprehensive test suite for the OAP LangChain integration.
 * Tests cover allow/deny flows, network failures, timeouts, receipt emission,
 * fallback strategies, LangChain tool wrapping, and LangGraph node wrapping.
 */

import {
  OAPToolMiddleware,
  OAPReceiptGenerator,
  OAPAuthorizationError,
  OAPReceipt,
  LangChainToolLike,
  LangGraphNodeFn,
} from '../src';

describe('OAPToolMiddleware', () => {
  let mockFetch: jest.Mock;
  let emittedReceipts: OAPReceipt[];

  const createMockTool = (name: string): LangChainToolLike => ({
    name,
    description: `A mock tool named ${name}`,
    invoke: jest.fn(async (input: unknown) => ({ result: `invoked ${name} with ${JSON.stringify(input)}` })),
    _call: jest.fn(async (input: unknown) => ({ result: `called ${name} with ${JSON.stringify(input)}` })),
  });

  const createMockNodeFn = (): LangGraphNodeFn =>
    jest.fn(async (state: Record<string, unknown>) => ({
      ...state,
      nodeExecuted: true,
    }));

  const setupMockFetchResponse = (statusCode: number, body: unknown) => {
    mockFetch.mockResolvedValue({
      ok: statusCode >= 200 && statusCode < 300,
      status: statusCode,
      text: async () => JSON.stringify(body),
    });
  };

  const setupMockFetchError = (errorMessage: string) => {
    mockFetch.mockRejectedValue(new Error(errorMessage));
  };

  const setupMockFetchTimeout = () => {
    mockFetch.mockImplementation(() => {
      return new Promise((_resolve, reject) => {
        setTimeout(() => {
          const abortError = new Error('The operation was aborted');
          abortError.name = 'AbortError';
          reject(abortError);
        }, 50);
      });
    });
  };

  beforeEach(() => {
    jest.clearAllMocks();
    emittedReceipts = [];
    mockFetch = jest.fn();
    global.fetch = mockFetch;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Test 1: Allow Flow - Tool call permitted when API returns allow', () => {
    it('should allow tool execution when API returns allow: true via invoke()', async () => {
      setupMockFetchResponse(200, { allow: true, reason: 'Allowed by policy' });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        apiKey: 'test-api-key',
      });

      const tool = createMockTool('web_search');
      const guarded = middleware.wrapTool(tool);
      const result = await guarded.invoke!({ query: 'test' });

      expect(result).toEqual({ result: 'invoked web_search with {"query":"test"}' });
      expect(tool.invoke).toHaveBeenCalledWith({ query: 'test' }, undefined);
    });

    it('should allow tool execution via _call() for legacy tools', async () => {
      setupMockFetchResponse(200, { allow: true, reason: 'Allowed by policy' });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const tool = createMockTool('legacy_tool');
      const guarded = middleware.wrapTool(tool);
      const result = await guarded._call!({ input: 'test' });

      expect(result).toEqual({ result: 'called legacy_tool with {"input":"test"}' });
    });
  });

  describe('Test 2: Deny Flow - Tool call blocked when API returns deny', () => {
    it('should throw OAPAuthorizationError when API returns allow: false', async () => {
      setupMockFetchResponse(200, { allow: false, reason: 'Tool not permitted' });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const tool = createMockTool('bash');
      const guarded = middleware.wrapTool(tool);

      await expect(guarded.invoke!({ command: 'rm -rf /' })).rejects.toThrow(OAPAuthorizationError);
    });

    it('should include denial reason in error message', async () => {
      setupMockFetchResponse(200, { allow: false, reason: 'Dangerous tool blocked' });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const tool = createMockTool('delete_all');
      const guarded = middleware.wrapTool(tool);

      try {
        await guarded.invoke!({});
        fail('Should have thrown OAPAuthorizationError');
      } catch (error) {
        expect(error).toBeInstanceOf(OAPAuthorizationError);
        const authError = error as OAPAuthorizationError;
        expect(authError.reason).toBe('Dangerous tool blocked');
        expect(authError.tool).toBe('delete_all');
        expect(authError.receipt).toBeDefined();
        expect(authError.receipt?.decision).toBe('denied');
      }
    });
  });

  describe('Test 3: Network Failure Fallback - Deny (default)', () => {
    it('should deny tool call when API is unreachable and fallback is deny', async () => {
      setupMockFetchError('ECONNREFUSED');

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        fallbackOnFailure: 'deny',
      });

      const tool = createMockTool('web_search');
      const guarded = middleware.wrapTool(tool);

      await expect(guarded.invoke!({})).rejects.toThrow('fail-closed');
    });

    it('should emit denial receipt on network failure with deny fallback', async () => {
      setupMockFetchError('ECONNREFUSED');

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        fallbackOnFailure: 'deny',
        onReceipt: (receipt) => emittedReceipts.push(receipt),
      });

      const tool = createMockTool('web_search');
      const guarded = middleware.wrapTool(tool);

      try {
        await guarded.invoke!({});
      } catch {
        // Expected
      }

      expect(emittedReceipts).toHaveLength(1);
      expect(emittedReceipts[0].decision).toBe('denied');
      expect(emittedReceipts[0].reason).toContain('ECONNREFUSED');
    });
  });

  describe('Test 4: Network Failure Fallback - Allow', () => {
    it('should allow tool call when API is unreachable and fallback is allow', async () => {
      setupMockFetchError('ECONNREFUSED');

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        fallbackOnFailure: 'allow',
      });

      const tool = createMockTool('web_search');
      const guarded = middleware.wrapTool(tool);
      const result = await guarded.invoke!({});

      expect(result).toBeDefined();
    });

    it('should emit approval receipt on network failure with allow fallback', async () => {
      setupMockFetchError('ECONNREFUSED');

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        fallbackOnFailure: 'allow',
        onReceipt: (receipt) => emittedReceipts.push(receipt),
      });

      const tool = createMockTool('web_search');
      const guarded = middleware.wrapTool(tool);
      await guarded.invoke!({});

      expect(emittedReceipts).toHaveLength(1);
      expect(emittedReceipts[0].decision).toBe('approved');
      expect(emittedReceipts[0].reason).toContain('Allowed after API failure');
    });
  });

  describe('Test 5: Network Failure Fallback - Error', () => {
    it('should throw error when API fails and fallback is error', async () => {
      setupMockFetchError('ETIMEDOUT');

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        fallbackOnFailure: 'error',
      });

      const tool = createMockTool('web_search');
      const guarded = middleware.wrapTool(tool);

      await expect(guarded.invoke!({})).rejects.toThrow('API error');
    });
  });

  describe('Test 6: Invalid Passport Handling', () => {
    it('should throw when passport is missing', () => {
      expect(() => {
        new OAPToolMiddleware({
          passport: '',
          policy: 'policy-456',
        });
      }).toThrow('Both passport and policy are required');
    });

    it('should throw when policy is missing', () => {
      expect(() => {
        new OAPToolMiddleware({
          passport: 'passport-123',
          policy: '',
        });
      }).toThrow('Both passport and policy are required');
    });
  });

  describe('Test 7: Timeout Handling', () => {
    it('should reject when request times out', async () => {
      setupMockFetchTimeout();

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        timeoutMs: 10,
        fallbackOnFailure: 'deny',
      });

      const tool = createMockTool('web_search');
      const guarded = middleware.wrapTool(tool);

      await expect(guarded.invoke!({})).rejects.toThrow('timed out');
    });
  });

  describe('Test 8: Async Verification Correctness', () => {
    it('should handle concurrent tool calls independently', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          text: async () => JSON.stringify({ allow: true, reason: 'test' }),
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          text: async () => JSON.stringify({ allow: false, reason: 'test' }),
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          text: async () => JSON.stringify({ allow: true, reason: 'test' }),
        });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const toolA = createMockTool('tool_a');
      const toolB = createMockTool('tool_b');
      const toolC = createMockTool('tool_c');

      const results = await Promise.allSettled([
        middleware.wrapTool(toolA).invoke!({}),
        middleware.wrapTool(toolB).invoke!({}),
        middleware.wrapTool(toolC).invoke!({}),
      ]);

      expect(results[0].status).toBe('fulfilled');
      expect(results[1].status).toBe('rejected');
      expect(results[2].status).toBe('fulfilled');
    });
  });

  describe('Test 9: API Key Authentication', () => {
    it('should include Authorization header when apiKey is provided', async () => {
      setupMockFetchResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        apiKey: 'secret-key-xyz',
      });

      const tool = createMockTool('web_search');
      await middleware.wrapTool(tool).invoke!({});

      const requestInit = mockFetch.mock.calls[0][1] as { headers: Record<string, string> };
      expect(requestInit.headers['Authorization']).toBe('Bearer secret-key-xyz');
    });

    it('should not include Authorization header when apiKey is omitted', async () => {
      setupMockFetchResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const tool = createMockTool('web_search');
      await middleware.wrapTool(tool).invoke!({});

      const requestInit = mockFetch.mock.calls[0][1] as { headers: Record<string, string> };
      expect(requestInit.headers['Authorization']).toBeUndefined();
    });
  });

  describe('Test 10: Receipt Emission via Callback', () => {
    it('should emit receipt on allowed tool call', async () => {
      setupMockFetchResponse(200, { allow: true, reason: 'Permitted' });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        onReceipt: (receipt) => emittedReceipts.push(receipt),
      });

      const tool = createMockTool('web_search');
      await middleware.wrapTool(tool).invoke!({ query: 'AI' });

      expect(emittedReceipts).toHaveLength(1);
      expect(emittedReceipts[0].decision).toBe('approved');
      expect(emittedReceipts[0].tool).toBe('web_search');
      expect(emittedReceipts[0].args).toEqual({ query: 'AI' });
      expect(emittedReceipts[0].passport).toBe('passport-123');
      expect(emittedReceipts[0].policy).toBe('policy-456');
      expect(emittedReceipts[0].id).toMatch(/^oap_\d+_[a-z0-9]+$/);
    });

    it('should emit receipt on denied tool call', async () => {
      setupMockFetchResponse(200, { allow: false, reason: 'Blocked' });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        onReceipt: (receipt) => emittedReceipts.push(receipt),
      });

      const tool = createMockTool('bash');
      try {
        await middleware.wrapTool(tool).invoke!({});
      } catch {
        // Expected
      }

      expect(emittedReceipts).toHaveLength(1);
      expect(emittedReceipts[0].decision).toBe('denied');
      expect(emittedReceipts[0].reason).toBe('Blocked');
    });
  });

  describe('Test 11: Receipt Suppression', () => {
    it('should not emit receipts when emitReceipts is false', async () => {
      setupMockFetchResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        emitReceipts: false,
        onReceipt: (receipt) => emittedReceipts.push(receipt),
      });

      const tool = createMockTool('web_search');
      await middleware.wrapTool(tool).invoke!({});

      expect(emittedReceipts).toHaveLength(0);
    });
  });

  describe('Test 12: Non-JSON API Response', () => {
    it('should fallback to deny when API returns invalid JSON', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
        text: async () => 'not valid json',
      });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        fallbackOnFailure: 'deny',
      });

      const tool = createMockTool('web_search');
      await expect(middleware.wrapTool(tool).invoke!({})).rejects.toThrow('fail-closed');
    });
  });

  describe('Test 13: Non-2xx API Response', () => {
    it('should fallback when API returns 500', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        text: async () => 'Internal Server Error',
      });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        fallbackOnFailure: 'error',
      });

      const tool = createMockTool('web_search');
      await expect(middleware.wrapTool(tool).invoke!({})).rejects.toThrow('OAP API returned 500');
    });
  });

  describe('Test 14: Missing Agent Context', () => {
    it('should handle missing agentId in context gracefully', async () => {
      setupMockFetchResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const tool = createMockTool('web_search');
      const result = await middleware.wrapTool(tool, { agentId: '' }).invoke!({});

      expect(result).toBeDefined();
    });
  });

  describe('Test 15: getConfig Non-Sensitive Data', () => {
    it('should return config without apiKey or onReceipt', () => {
      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        apiKey: 'secret',
        apiEndpoint: 'https://custom.aport.io/v1/verify',
        fallbackOnFailure: 'allow',
        timeoutMs: 10000,
        emitReceipts: true,
        onReceipt: () => {},
      });

      const config = middleware.getConfig();

      expect(config.passport).toBe('passport-123');
      expect(config.policy).toBe('policy-456');
      expect(config.apiEndpoint).toBe('https://custom.aport.io/v1/verify');
      expect(config.fallbackOnFailure).toBe('allow');
      expect(config.timeoutMs).toBe(10000);
      expect(config.emitReceipts).toBe(true);
      expect(config).not.toHaveProperty('apiKey');
      expect(config).not.toHaveProperty('onReceipt');
    });
  });

  describe('Test 16: Custom API Endpoint', () => {
    it('should use custom apiEndpoint when provided', async () => {
      setupMockFetchResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        apiEndpoint: 'https://staging.aport.io/v1/verify',
      });

      const tool = createMockTool('web_search');
      await middleware.wrapTool(tool).invoke!({});

      const url = mockFetch.mock.calls[0][0] as string;
      expect(url).toBe('https://staging.aport.io/v1/verify');
    });
  });

  describe('Test 17: HTTP Support (non-HTTPS)', () => {
    it('should allow non-https URLs via fetch', async () => {
      setupMockFetchResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        apiEndpoint: 'http://localhost:8080/verify',
      });

      const tool = createMockTool('web_search');
      await middleware.wrapTool(tool).invoke!({});

      const url = mockFetch.mock.calls[0][0] as string;
      expect(url).toBe('http://localhost:8080/verify');
    });
  });

  describe('Test 18: Callable Tool Wrapping', () => {
    it('should wrap callable tools (tool() decorator results)', async () => {
      setupMockFetchResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      // Simulate a tool() decorator result: callable function with metadata
      const callableTool = async function callable_tool(input: unknown) {
        return { result: `callable result: ${JSON.stringify(input)}` };
      } as unknown as LangChainToolLike;
      callableTool.description = 'A callable tool';

      const guarded = middleware.wrapTool(callableTool);
      const result = await (guarded as unknown as (input: unknown) => Promise<unknown>)({ query: 'test' });

      expect(result).toEqual({ result: 'callable result: {"query":"test"}' });
    });

    it('should block callable tools when denied', async () => {
      setupMockFetchResponse(200, { allow: false, reason: 'Blocked' });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const callableTool = async function blocked_tool(input: unknown) {
        return { result: `callable result: ${JSON.stringify(input)}` };
      } as unknown as LangChainToolLike;
      callableTool.description = 'A blocked tool';

      const guarded = middleware.wrapTool(callableTool);
      await expect((guarded as unknown as (input: unknown) => Promise<unknown>)({})).rejects.toThrow(OAPAuthorizationError);
    });
  });

  describe('Test 19: LangGraph Node Wrapping', () => {
    it('should allow LangGraph node execution when API returns allow', async () => {
      setupMockFetchResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const nodeFn = createMockNodeFn();
      const guardedNode = middleware.wrapNode(nodeFn, 'research_node');
      const result = await guardedNode({ query: 'AI' }, { configurable: { thread_id: 't1' } });

      expect(result).toEqual({ query: 'AI', nodeExecuted: true });
      expect(nodeFn).toHaveBeenCalled();
    });

    it('should deny LangGraph node execution when API returns deny', async () => {
      setupMockFetchResponse(200, { allow: false, reason: 'Node blocked' });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const nodeFn = createMockNodeFn();
      const guardedNode = middleware.wrapNode(nodeFn, 'dangerous_node');

      await expect(guardedNode({ query: 'hack' }, {})).rejects.toThrow(OAPAuthorizationError);
    });

    it('should extract tool call args from LangGraph state with messages', async () => {
      setupMockFetchResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const nodeFn = createMockNodeFn();
      const guardedNode = middleware.wrapNode(nodeFn, 'search_node');

      const state = {
        messages: [
          {
            tool_calls: [
              { name: 'search_node', args: { query: 'LangChain' } },
            ],
          },
        ],
      };

      await guardedNode(state, {});

      const requestInit = mockFetch.mock.calls[0][1] as { body: string };
      const payload = JSON.parse(requestInit.body);
      expect(payload.args.query).toBe('LangChain');
    });
  });

  describe('Test 20: Tool properties preserved after wrapping', () => {
    it('should preserve original tool name and description', async () => {
      setupMockFetchResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const tool = createMockTool('preserved_tool');
      tool.description = 'My important tool';

      const guarded = middleware.wrapTool(tool);
      expect(guarded.name).toBe('preserved_tool');
      expect(guarded.description).toBe('My important tool');
    });
  });
});

describe('OAPReceiptGenerator', () => {
  it('should generate unique receipt IDs', () => {
    const receipt1 = OAPReceiptGenerator.approved({
      agentId: 'agent1',
      tool: 'tool1',
    });
    const receipt2 = OAPReceiptGenerator.approved({
      agentId: 'agent2',
      tool: 'tool2',
    });

    expect(receipt1.id).not.toBe(receipt2.id);
    expect(receipt1.id).toMatch(/^oap_/);
  });

  it('should include current timestamp', () => {
    const before = Date.now();
    const receipt = OAPReceiptGenerator.denied({
      agentId: 'agent1',
      tool: 'tool1',
    });
    const after = Date.now();

    const receiptTime = new Date(receipt.timestamp).getTime();
    expect(receiptTime).toBeGreaterThanOrEqual(before);
    expect(receiptTime).toBeLessThanOrEqual(after);
  });

  it('should include passport and policy when provided', () => {
    const receipt = OAPReceiptGenerator.approved({
      agentId: 'agent1',
      tool: 'tool1',
      passport: 'pass-123',
      policy: 'pol-456',
    });

    expect(receipt.passport).toBe('pass-123');
    expect(receipt.policy).toBe('pol-456');
  });
});
