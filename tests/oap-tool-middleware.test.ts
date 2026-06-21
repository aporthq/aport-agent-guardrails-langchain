/**
 * OAP Tool Middleware Tests for LangChain
 *
 * Comprehensive test suite for the OAP LangChain integration.
 * Tests cover allow/deny flows, network failures, timeouts, receipt emission,
 * fallback strategies, LangChain tool wrapping, and LangGraph node wrapping.
 */

import * as https from 'https';
import * as http from 'http';
import {
  OAPToolMiddleware,
  OAPReceiptGenerator,
  OAPAuthorizationError,
  OAPReceipt,
  LangChainToolLike,
  LangGraphNodeFn,
} from '../src';

// Mock https and http modules
jest.mock('https');
jest.mock('http');

describe('OAPToolMiddleware', () => {
  let mockRequest: jest.Mock;
  let mockReq: {
    on: jest.Mock;
    write: jest.Mock;
    end: jest.Mock;
    destroy: jest.Mock;
  };
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

  const setupMockResponse = (statusCode: number, body: unknown) => {
    const mockRes = {
      statusCode,
      on: jest.fn((event: string, handler: (chunk?: string) => void) => {
        if (event === 'data') {
          handler(JSON.stringify(body));
        }
        if (event === 'end') {
          handler();
        }
      }),
    };

    mockRequest.mockImplementation((_options: unknown, callback: (res: unknown) => void) => {
      callback(mockRes);
      return mockReq;
    });
  };

  const setupMockError = (errorMessage: string) => {
    mockRequest.mockImplementation(() => {
      setTimeout(() => {
        const errorHandler = mockReq.on.mock.calls.find(
          (call: [string, (...args: unknown[]) => void]) => call[0] === 'error'
        )?.[1];
        if (errorHandler) {
          errorHandler(new Error(errorMessage));
        }
      }, 10);
      return mockReq;
    });
  };

  const setupMockTimeout = () => {
    mockRequest.mockImplementation(() => {
      setTimeout(() => {
        const timeoutHandler = mockReq.on.mock.calls.find(
          (call: [string, (...args: unknown[]) => void]) => call[0] === 'timeout'
        )?.[1];
        if (timeoutHandler) {
          timeoutHandler();
        }
      }, 10);
      return mockReq;
    });
  };

  beforeEach(() => {
    jest.clearAllMocks();
    emittedReceipts = [];

    mockReq = {
      on: jest.fn(),
      write: jest.fn(),
      end: jest.fn(),
      destroy: jest.fn(),
    };

    mockRequest = jest.fn(() => mockReq);
    jest.spyOn(https, 'request').mockImplementation(mockRequest as unknown as typeof https.request);
    jest.spyOn(http, 'request').mockImplementation(mockRequest as unknown as typeof http.request);
  });

  describe('Test 1: Allow Flow - Tool call permitted when API returns allow', () => {
    it('should allow tool execution when API returns allow: true via invoke()', async () => {
      setupMockResponse(200, { allow: true, reason: 'Allowed by policy' });

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
      setupMockResponse(200, { allow: true, reason: 'Allowed by policy' });

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
      setupMockResponse(200, { allow: false, reason: 'Tool not permitted' });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const tool = createMockTool('bash');
      const guarded = middleware.wrapTool(tool);

      await expect(guarded.invoke!({ command: 'rm -rf /' })).rejects.toThrow(OAPAuthorizationError);
    });

    it('should include denial reason in error message', async () => {
      setupMockResponse(200, { allow: false, reason: 'Dangerous tool blocked' });

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
      setupMockError('ECONNREFUSED');

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
      setupMockError('ECONNREFUSED');

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
      setupMockError('ECONNREFUSED');

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
      setupMockError('ECONNREFUSED');

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
      setupMockError('ETIMEDOUT');

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
      setupMockTimeout();

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        timeoutMs: 100,
        fallbackOnFailure: 'deny',
      });

      const tool = createMockTool('web_search');
      const guarded = middleware.wrapTool(tool);

      await expect(guarded.invoke!({})).rejects.toThrow('timed out');
    });

    it('should destroy request on timeout', async () => {
      setupMockTimeout();

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        timeoutMs: 100,
        fallbackOnFailure: 'deny',
      });

      const tool = createMockTool('web_search');
      const guarded = middleware.wrapTool(tool);

      try {
        await guarded.invoke!({});
      } catch {
        // Expected
      }

      expect(mockReq.destroy).toHaveBeenCalled();
    });
  });

  describe('Test 8: Async Verification Correctness', () => {
    it('should handle concurrent tool calls independently', async () => {
      let callCount = 0;
      mockRequest.mockImplementation((_options: unknown, callback: (res: unknown) => void) => {
        callCount++;
        const mockRes = {
          statusCode: 200,
          on: jest.fn((event: string, handler: (chunk?: string) => void) => {
            if (event === 'data') {
              handler(JSON.stringify({ allow: callCount % 2 === 1, reason: 'test' }));
            }
            if (event === 'end') {
              handler();
            }
          }),
        };
        callback(mockRes);
        return mockReq;
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
      setupMockResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        apiKey: 'secret-key-xyz',
      });

      const tool = createMockTool('web_search');
      await middleware.wrapTool(tool).invoke!({});

      const requestOptions = mockRequest.mock.calls[0][0] as { headers: Record<string, string> };
      expect(requestOptions.headers['Authorization']).toBe('Bearer secret-key-xyz');
    });

    it('should not include Authorization header when apiKey is omitted', async () => {
      setupMockResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const tool = createMockTool('web_search');
      await middleware.wrapTool(tool).invoke!({});

      const requestOptions = mockRequest.mock.calls[0][0] as { headers: Record<string, string> };
      expect(requestOptions.headers['Authorization']).toBeUndefined();
    });
  });

  describe('Test 10: Receipt Emission via Callback', () => {
    it('should emit receipt on allowed tool call', async () => {
      setupMockResponse(200, { allow: true, reason: 'Permitted' });

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
      setupMockResponse(200, { allow: false, reason: 'Blocked' });

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
      setupMockResponse(200, { allow: true });

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
      const mockRes = {
        statusCode: 200,
        on: jest.fn((event: string, handler: (chunk?: string) => void) => {
          if (event === 'data') {
            handler('not valid json');
          }
          if (event === 'end') {
            handler();
          }
        }),
      };

      mockRequest.mockImplementation((_options: unknown, callback: (res: unknown) => void) => {
        callback(mockRes);
        return mockReq;
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
      const mockRes = {
        statusCode: 500,
        on: jest.fn((event: string, handler: (chunk?: string) => void) => {
          if (event === 'data') {
            handler('Internal Server Error');
          }
          if (event === 'end') {
            handler();
          }
        }),
      };

      mockRequest.mockImplementation((_options: unknown, callback: (res: unknown) => void) => {
        callback(mockRes);
        return mockReq;
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
      setupMockResponse(200, { allow: true });

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
      setupMockResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        apiEndpoint: 'https://staging.aport.io/v1/verify',
      });

      const tool = createMockTool('web_search');
      await middleware.wrapTool(tool).invoke!({});

      const requestOptions = mockRequest.mock.calls[0][0] as {
        hostname: string;
        path: string;
      };
      expect(requestOptions.hostname).toBe('staging.aport.io');
      expect(requestOptions.path).toBe('/v1/verify');
    });
  });

  describe('Test 17: HTTP Support (non-HTTPS)', () => {
    it('should use http module for non-https URLs', async () => {
      setupMockResponse(200, { allow: true });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
        apiEndpoint: 'http://localhost:8080/verify',
      });

      const tool = createMockTool('web_search');
      await middleware.wrapTool(tool).invoke!({});

      expect(http.request).toHaveBeenCalled();
      expect(https.request).not.toHaveBeenCalled();
    });
  });

  describe('Test 18: Callable Tool Wrapping', () => {
    it('should wrap callable tools (tool() decorator results)', async () => {
      setupMockResponse(200, { allow: true });

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
      setupMockResponse(200, { allow: false, reason: 'Blocked' });

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
      setupMockResponse(200, { allow: true });

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
      setupMockResponse(200, { allow: false, reason: 'Node blocked' });

      const middleware = new OAPToolMiddleware({
        passport: 'passport-123',
        policy: 'policy-456',
      });

      const nodeFn = createMockNodeFn();
      const guardedNode = middleware.wrapNode(nodeFn, 'dangerous_node');

      await expect(guardedNode({ query: 'hack' }, {})).rejects.toThrow(OAPAuthorizationError);
    });

    it('should extract tool call args from LangGraph state with messages', async () => {
      setupMockResponse(200, { allow: true });

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

      const writeCall = mockReq.write.mock.calls[0]?.[0];
      const payload = JSON.parse(writeCall);
      expect(payload.args.query).toBe('LangChain');
    });
  });

  describe('Test 20: Tool properties preserved after wrapping', () => {
    it('should preserve original tool name and description', async () => {
      setupMockResponse(200, { allow: true });

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
