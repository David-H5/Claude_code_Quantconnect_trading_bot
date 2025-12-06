# Error Recovery Protocol (P1 - UPGRADE-013)

When you encounter errors during autonomous operation, follow this recovery protocol.

## Error Categories

### Category 1: Import/Module Errors

**Detection**: `ImportError`, `ModuleNotFoundError`, `No module named`

**Recovery Actions**:

1. Check if module is in `requirements.txt`
2. If missing: Add to requirements.txt, run `pip install`
3. If present: Check for typos in import statement
4. If circular import: Refactor to lazy import or move import inside function
5. If still failing: Check Python path and virtual environment

**Decision**: Recoverable in most cases

---

### Category 2: Syntax Errors

**Detection**: `SyntaxError`, `IndentationError`, `TabError`

**Recovery Actions**:

1. Read the error line number carefully
2. Check for:
   - Missing colons after `if/for/def/class`
   - Unmatched parentheses/brackets/braces
   - Mixed tabs and spaces
   - Unclosed strings
3. Fix the specific syntax issue
4. Run `python -m py_compile <file>` to verify

**Decision**: Always recoverable - syntax errors have deterministic fixes

---

### Category 3: Test Failures

**Detection**: `pytest` failures, `AssertionError`, test count mismatch

**Recovery Actions**:

1. **Analyze failure type**:
   - Logic error → Fix the code
   - Test expectation wrong → Fix the test
   - Missing fixture → Add fixture
   - Flaky test → Add retry or fix race condition

2. **Fix vs Rollback Decision**:
   - Single test failing → Fix forward
   - Multiple tests failing after change → Consider rollback
   - All tests failing → Likely environment issue, check dependencies

3. **After fix**: Run the specific test, then full suite

**Decision**: Recoverable, but watch for cascading failures

---

### Category 4: API/Network Errors

**Detection**: `ConnectionError`, `TimeoutError`, `HTTPError`, rate limit messages

**Recovery Actions**:

1. **Transient errors** (timeout, connection reset):
   - Wait 5-30 seconds
   - Retry with exponential backoff
   - Max 3 retries

2. **Rate limits** (429, "rate limit exceeded"):
   - Check rate limit headers for reset time
   - Wait until reset
   - Consider reducing request frequency

3. **Auth errors** (401, 403):
   - Check API key configuration
   - Verify credentials not expired
   - **ESCALATE** if auth issue persists

4. **Server errors** (500, 502, 503):
   - Wait and retry (transient)
   - If persistent, document and move on

**Decision**: Transient errors recoverable; auth issues require escalation

---

### Category 5: File System Errors

**Detection**: `FileNotFoundError`, `PermissionError`, `IsADirectoryError`

**Recovery Actions**:

1. **File not found**:
   - Verify path is correct (absolute vs relative)
   - Check if file was deleted or moved
   - Create file if it should exist

2. **Permission errors**:
   - Check file permissions
   - Verify not trying to write to read-only location
   - **ESCALATE** if system permissions issue

3. **Directory errors**:
   - Create missing directories with `os.makedirs`
   - Check path construction logic

**Decision**: Usually recoverable with path fixes

---

### Category 6: Runtime Errors

**Detection**: `TypeError`, `ValueError`, `KeyError`, `AttributeError`, `IndexError`

**Recovery Actions**:

1. **TypeError/ValueError**:
   - Check function arguments and types
   - Add input validation
   - Cast types if appropriate

2. **KeyError/AttributeError**:
   - Check for None values before accessing
   - Use `.get()` with defaults for dicts
   - Verify object has expected attributes

3. **IndexError**:
   - Check list/array bounds
   - Handle empty collections

4. **General approach**:
   - Add defensive checks
   - Log the problematic value
   - Consider whether data validation needed

**Decision**: Recoverable with defensive coding

---

### Category 7: Git/Version Control Errors

**Detection**: `git` command failures, merge conflicts, detached HEAD

**Recovery Actions**:

1. **Merge conflicts**:
   - Examine conflicting files
   - Choose resolution strategy
   - Mark as resolved and commit

2. **Detached HEAD**:
   - Create branch from current state if needed
   - Or checkout to proper branch

3. **Push failures**:
   - Pull first, resolve conflicts
   - Never force push to main/develop

4. **Hook failures**:
   - Check pre-commit hook output
   - Fix formatting/lint issues
   - Retry commit

**Decision**: Recoverable, but be careful with force operations

---

## Recovery vs Escalation Decision Tree

```text
Error Encountered
       │
       ├── Is error message clear?
       │   ├── YES → Attempt fix
       │   └── NO → Search for error pattern
       │
       ├── After fix attempt, is error resolved?
       │   ├── YES → Continue work
       │   └── NO → Count attempts
       │
       ├── Attempts < 3?
       │   ├── YES → Try alternative approach
       │   └── NO → Enter fallback protocol
       │
       └── Is this blocking critical path?
           ├── YES → Document and escalate
           └── NO → Skip, document, continue with other tasks
```

---

## Error Logging Template

When encountering errors, document them:

```markdown
## Error Encountered

**Time**: [timestamp]
**File**: [file path]
**Error Type**: [category from above]
**Error Message**: [full error message]

**Recovery Attempt 1**:
- Action: [what you tried]
- Result: [success/failure]

**Recovery Attempt 2** (if needed):
- Action: [what you tried]
- Result: [success/failure]

**Resolution**: [how it was fixed OR why it was escalated]
```

---

## Anti-Patterns to Avoid

### DON'T: Silent Failure

```python
# BAD
try:
    risky_operation()
except:
    pass  # Swallowing all errors
```

### DO: Log and Handle Appropriately

```python
# GOOD
try:
    risky_operation()
except SpecificError as e:
    logger.warning(f"Expected error handled: {e}")
    fallback_operation()
```

### DON'T: Infinite Retry

```python
# BAD
while True:
    try:
        api_call()
        break
    except:
        time.sleep(1)  # Infinite loop risk
```

### DO: Bounded Retry with Backoff

```python
# GOOD
for attempt in range(3):
    try:
        api_call()
        break
    except TransientError:
        time.sleep(2 ** attempt)
else:
    raise MaxRetriesExceeded()
```

---

## Integration with Other Protocols

- **After 3 failed recovery attempts** → Enter `stuck_detection.md` protocol
- **When blocked by external factor** → Enter `graceful_fallback.md` protocol
- **When error reveals design flaw** → Document as P0/P1 insight for next RIC iteration
