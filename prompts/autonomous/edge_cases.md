# Edge Case Library (P2 - UPGRADE-013)

Known failure patterns and their solutions. Reference this when encountering unexpected behavior.

---

## Pattern 1: Circular Import Loop

**Detection Signs**:

- `ImportError: cannot import name X from partially initialized module`
- Import works in isolation but fails in context
- Stack trace shows same module multiple times

**Root Cause**: Module A imports Module B, which imports Module A

**Recovery**:

1. Identify the circular dependency chain
2. Solutions (in order of preference):
   - Move shared code to a third module
   - Use lazy imports (import inside function)
   - Use `TYPE_CHECKING` for type hints only
3. Verify with `python -c "import module_a; import module_b"`

**Prevention**: Check import graph before adding new imports

---

## Pattern 2: Test Pollution / State Leakage

**Detection Signs**:

- Tests pass in isolation but fail when run together
- Test order affects results
- `pytest tests/test_a.py` passes but `pytest tests/` fails

**Root Cause**: Shared state between tests (globals, singletons, class variables)

**Recovery**:

1. Identify the polluted state:
   - Class variables not reset
   - Global variables modified
   - Singleton patterns
   - File system artifacts
2. Add proper cleanup in `teardown` or fixture `yield`
3. Use `pytest --randomly-seed=X` to find order-dependent tests

**Prevention**: Always use fixtures with proper cleanup

---

## Pattern 3: Pre-Commit Hook Infinite Loop

**Detection Signs**:

- Pre-commit modifies files
- Modified files trigger pre-commit again
- Commit never completes

**Root Cause**: Formatters/linters modify files that then fail checks

**Recovery**:

1. Abort with Ctrl+C
2. Stage the modified files: `git add -A`
3. Commit with `--no-verify` ONCE: `git commit --no-verify -m "WIP"`
4. Then fix the root cause

**Prevention**: Run pre-commit manually before committing: `pre-commit run --all-files`

---

## Pattern 4: Mock Not Applied Correctly

**Detection Signs**:

- Mock appears correct but original function still called
- `@patch` seems ignored
- Side effects occur despite mocking

**Root Cause**: Patching wrong import path

**Recovery**:

1. **Golden Rule**: Patch where it's USED, not where it's DEFINED
   ```python
   # If module_a.py does: from utils import helper
   # Patch: @patch('module_a.helper')  # NOT @patch('utils.helper')
   ```
2. Check import style (from vs import)
3. Verify patch is applied before function call

**Prevention**: Always trace the import path from the module being tested

---

## Pattern 5: Context Window Exhaustion

**Detection Signs**:

- Responses become truncated
- Claude seems to "forget" earlier context
- Repetitive questions about already-discussed topics

**Root Cause**: Approaching 200K token limit

**Recovery**:

1. Create checkpoint immediately: `git commit -m "WIP: context checkpoint"`
2. Update `claude-session-notes.md` with:
   - Current goal
   - Key decisions made
   - Next steps
3. Use `/compact` if available
4. For overnight: watchdog should trigger session restart

**Prevention**: Monitor context usage, checkpoint regularly

---

## Pattern 6: QuantConnect API Method Case Mismatch

**Detection Signs**:

- Method works locally but fails in QuantConnect
- `AttributeError` on algorithm methods
- IDE shows method but runtime doesn't

**Root Cause**: Python API uses snake_case, some properties use PascalCase

**Recovery**:

1. Check CLAUDE.md for correct method names:
   - Methods: `self.set_start_date()` (snake_case)
   - Properties: `self.Portfolio[symbol].Invested` (PascalCase)
   - Framework methods: `data.ContainsKey()` (PascalCase)
2. When in doubt, check QuantConnect docs or LEAN source

**Prevention**: Always verify against CLAUDE.md QuantConnect section

---

## Pattern 7: Async/Await Missing

**Detection Signs**:

- `coroutine was never awaited` warning
- Function returns coroutine object instead of result
- Async operations not completing

**Root Cause**: Missing `await` keyword

**Recovery**:

1. Find all async function calls
2. Add `await` before each
3. Ensure calling function is also `async`
4. If in sync context, use `asyncio.run()` or event loop

**Prevention**: Enable async linting rules

---

## Pattern 8: Git Detached HEAD After Checkout

**Detection Signs**:

- `HEAD detached at abc1234`
- Commits not visible on any branch
- `git push` fails

**Root Cause**: Checked out specific commit instead of branch

**Recovery**:

1. If you have uncommitted work to save:
   ```bash
   git stash
   git checkout main
   git stash pop
   ```
2. If you made commits in detached state:
   ```bash
   git branch temp-branch  # Save the commits
   git checkout main
   git merge temp-branch   # Or cherry-pick
   ```

**Prevention**: Always `git checkout branch-name`, not commit hash

---

## Pattern 9: Environment Variable Not Loaded

**Detection Signs**:

- `KeyError` when accessing env variable
- Feature works locally but not in CI
- `.env` file exists but values are None

**Root Cause**: `.env` not loaded or wrong file

**Recovery**:

1. Check if `python-dotenv` is installed and called:
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Must be called before accessing env vars
   ```
2. Verify `.env` file path (relative to working directory)
3. Check for `.env.example` vs `.env` confusion
4. In CI: Check secrets/environment configuration

**Prevention**: Always use explicit `load_dotenv()` call early in startup

---

## Pattern 10: Race Condition in Tests

**Detection Signs**:

- Tests pass/fail randomly
- "Flaky" test label
- Works with delays but not without

**Root Cause**: Async operations completing out of order

**Recovery**:

1. Add explicit waits/synchronization
2. Use `pytest-asyncio` properly
3. Mock time-dependent operations
4. For file operations, use temp directories with cleanup

**Prevention**: Avoid time.sleep() in tests; use proper async patterns

---

## Quick Reference: Pattern → Solution

| Pattern | Quick Fix |
|---------|-----------|
| Circular import | Move to third module or lazy import |
| Test pollution | Add teardown/cleanup fixtures |
| Pre-commit loop | `git commit --no-verify` then fix |
| Mock not working | Patch where USED, not defined |
| Context exhaustion | Checkpoint, update notes, compact |
| QC method case | Check CLAUDE.md snake_case vs PascalCase |
| Missing await | Add await, make caller async |
| Detached HEAD | `git checkout branch-name` |
| Env not loaded | Call `load_dotenv()` early |
| Flaky tests | Mock time, add synchronization |

---

## Adding New Patterns

When you encounter a new edge case:

1. Document the detection signs
2. Identify the root cause
3. Write the recovery steps
4. Add prevention advice
5. Update this file
