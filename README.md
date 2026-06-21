# APort Agent Guardrails for LangChain

[![npm version](https://badge.fury.io/js/@aporthq%2Faport-agent-guardrails-langchain.svg)](https://badge.fury.io/js/@aporthq%2Faport-agent-guardrails-langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deterministic pre-action authorization for [LangChain](https://js.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/) AI agents using the Open Agent Protocol (OAP).

## Overview

`@aporthq/aport-agent-guardrails-langchain` wraps LangChain tools and LangGraph nodes to enforce policy-based access control. Every tool invocation is intercepted, sent to the APort guardrail API for verification, and either allowed to proceed or blocked with a detailed denial reason.

**Key features:**
- 🔒 **Pre-action authorization** — Block dangerous tool calls before they execute
- 🛡️ **Fail-closed by default** — If the APort API is unreachable, tool calls are denied
- 📊 **Audit receipts** — Every decision generates an immutable receipt for compliance
- ⚡ **Async-native** — Non-blocking verification that respects LangChain's event loop
- 🔄 **LangGraph compatible** — Works with node-level tool calls in state graphs
- 🧩 **Zero framework lock-in** — Duck-typed wrapping works with any tool-like object

## Installation

```bash
npm install @aporthq/aport-agent-guardrails-langchain
```

## Quick Start

```typescript
import { OAPToolMiddleware } from '@aporthq/aport-agent-guardrails-langchain';
import { tool } from '@langchain/core/tools';

// Create the middleware
const middleware = new OAPToolMiddleware({
  passport: 'passport-agent-123',
  policy: 'policy-sandbox-456',
  apiKey: process.env.APORT_API_KEY,
});

// Wrap any LangChain tool
const searchTool = tool(async ({ query }) => {
  return `Results for: ${query}`;
}, {
  name: 'web_search',
  description: 'Search the web',
});

const guardedSearch = middleware.wrapTool(searchTool);

// Use in your agent — authorization happens automatically
const result = await guardedSearch.invoke({ query: 'AI safety' });
```

## Configuration

```typescript
const middleware = new OAPToolMiddleware({
  passport: 'passport-agent-123',      // Required: Agent passport ID
  policy: 'policy-sandbox-456',        // Required: Policy ID to evaluate
  apiEndpoint: 'https://api.aport.io/v1/verify', // Optional: custom API endpoint
  apiKey: process.env.APORT_API_KEY,   // Optional: API key for authentication
  fallbackOnFailure: 'deny',           // Optional: 'deny' | 'allow' | 'error'
  timeoutMs: 5000,                     // Optional: Request timeout (default: 5000ms)
  emitReceipts: true,                  // Optional: Enable receipt callbacks
  onReceipt: (receipt) => console.log(receipt), // Optional: Receipt handler
});
```

### Fallback Strategies

| Strategy | Behavior when API fails |
|----------|------------------------|
| `deny` (default) | Block the tool call and throw `OAPAuthorizationError` |
| `allow` | Allow the tool call and log the failure |
| `error` | Throw an `OAPAuthorizationError` with the API error details |

## LangGraph Integration

Wrap LangGraph nodes to enforce authorization at the graph level:

```typescript
import { StateGraph } from '@langchain/langgraph';

const researchNode = async (state: typeof StateAnnotation.State) => {
  const results = await searchTool.invoke(state.query);
  return { results };
};

// Guard the node
const guardedResearchNode = middleware.wrapNode(researchNode, 'research_node');

const workflow = new StateGraph(StateAnnotation)
  .addNode('research', guardedResearchNode)
  .addEdge('__start__', 'research');
```

## API Reference

### `OAPToolMiddleware`

#### `wrapTool<T>(tool: T, context?: LangChainAgentContext): T`

Wraps a LangChain tool to enforce OAP authorization on every invocation.

Supports:
- Modern Runnable-style tools (`.invoke()`)
- Legacy BaseTool instances (`._call()`)
- Callable tool functions (`tool()` decorator results)

#### `wrapNode(nodeFn: LangGraphNodeFn, nodeName: string, context?: LangChainAgentContext): LangGraphNodeFn`

Wraps a LangGraph node function to enforce authorization before execution.

#### `getConfig(): OAPToolMiddlewareConfig`

Returns the middleware configuration (excluding sensitive fields like `apiKey`).

### Types

```typescript
interface LangChainAgentContext {
  agentId: string;
  sessionId?: string;
  userId?: string;
  runId?: string;
  threadId?: string;
  state?: Record<string, unknown>;
}

class OAPAuthorizationError extends Error {
  receipt?: OAPReceipt;
  tool: string;
  reason: string;
}
```

## Error Handling

```typescript
try {
  await guardedTool.invoke({ command: 'rm -rf /' });
} catch (error) {
  if (error instanceof OAPAuthorizationError) {
    console.log('Tool blocked:', error.tool);
    console.log('Reason:', error.reason);
    console.log('Receipt:', error.receipt);
  }
}
```

## Testing

```bash
npm test
```

## License

MIT © LiftRails Inc.
