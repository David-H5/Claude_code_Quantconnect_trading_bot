# Agent Trace - View Execution Traces

View detailed execution traces for debugging agent workflows.

## Arguments
- `$ARGUMENTS`: Trace ID to view, or `--recent [count]` to list recent traces

## Usage
```
/agent-trace --recent           # List last 5 traces
/agent-trace --recent 10        # List last 10 traces
/agent-trace code_review_20251205_143022   # View specific trace
```

## Instructions

### List Recent Traces

Show recent workflow execution traces:

```bash
python3 -c "
import sys
sys.path.insert(0, '.claude/hooks')
from agent_orchestrator import Tracer, format_trace

tracer = Tracer()
traces = tracer.list_traces(limit=10)

if not traces:
    print('No traces found')
else:
    print('\n## Recent Traces\n')
    print('| Status | Trace ID | Workflow | Duration | Agents |')
    print('|--------|----------|----------|----------|--------|')
    for t in traces:
        status = '✅' if t['status'] == 'success' else '⚠️'
        print(f'| {status} | {t[\"trace_id\"]} | {t[\"workflow\"]} | {t.get(\"duration_ms\", 0):.0f}ms | {t[\"spans\"]} |')
"
```

### View Specific Trace

If `$ARGUMENTS` is a trace ID (not `--recent`), display detailed trace:

```bash
python3 -c "
import sys
sys.path.insert(0, '.claude/hooks')
from agent_orchestrator import Tracer, format_trace

tracer = Tracer()
trace = tracer.get_trace('$ARGUMENTS')

if trace:
    print(format_trace(trace))
else:
    print('Trace not found: $ARGUMENTS')
"
```

## Trace Output Format

```markdown
## Trace: code_review_20251205_143022
Workflow: code_review
Duration: 4532ms

### Spans
| Agent | Status | Duration | Tokens |
|-------|--------|----------|--------|
| SecurityScanner | ✅ | 1234ms | 500/1200 |
| TypeChecker | ✅ | 987ms | 400/800 |
| TestAnalyzer | ❌ | 2100ms | 600/0 |
| Architect | ✅ | 211ms | 800/2000 |

### Errors
- **TestAnalyzer**: Timeout after 2000ms
```

## Trace Storage

Traces are stored in `.claude/traces/` as JSON files:
- Filename: `{workflow}_{timestamp}.json`
- Auto-cleanup: Keep last 100 traces

## Understanding Traces

### Status Icons
- ✅ `success` - Agent completed successfully
- ❌ `failed` - Agent encountered an error
- ⏱️ `timeout` - Agent timed out
- ⚠️ `partial` - Workflow had some failures

### Tokens Column
Format: `input/output`
- Example: `500/1200` = 500 input tokens, 1200 output tokens

### Duration
Time in milliseconds for agent execution.

## Debugging Tips

1. **High duration + timeout**: Agent is hitting context limits
2. **Failed with 0 output tokens**: Agent crashed before generating output
3. **All agents failed**: Check circuit breaker status with `/agent-status`
4. **Partial success**: Check which agents failed and why

## Related Commands
- `/agent-status` - View circuit breaker and overall status
- `/agents auto` - Run workflow with automatic tracing
