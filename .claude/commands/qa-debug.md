# QA Debug Phase

Run comprehensive debugging and integrity checks. Use this during RIC Loop Phase 4 (Double-Check) and Phase 5 (Introspection).

## Usage

```
/qa-debug [upgrade-id]
```

## What Gets Checked

### Debug Diagnostics
- `debug_statements` - pdb/ipdb imports and set_trace() calls
- `todo_fixme` - TODO, FIXME, BUG, HACK, XXX comments
- `print_statements` - Excessive print() calls
- `breakpoints` - breakpoint() calls
- `incomplete_code` - NotImplementedError, WIP markers

### Code Integrity
- `corrupt_files` - Null bytes, encoding errors, truncation
- `missing_imports` - Undefined names (via pyflakes)
- `circular_imports` - Star imports in __init__.py
- `orphan_functions` - Unused functions/classes
- `init_files` - Missing __init__.py in packages

### Cross-References
- `broken_imports` - Import paths to non-existent modules
- `config_refs` - Config keys that don't exist
- `class_refs` - Type hints to unknown classes
- `doc_links` - Broken links in documentation

### RIC Compliance
- `ric_phases` - Required phases in upgrade docs
- `upgrade_docs` - Upgrade doc completeness
- `iteration_tracking` - [ITERATION X/5] markers

## RIC Loop Integration

### Phase 4: Double-Check
```bash
python scripts/qa_validator.py --check debug --check integrity
```

### Phase 5: Introspection
```bash
python scripts/qa_validator.py --check xref --verbose
```

### Phase 6: Metacognition
```bash
python scripts/qa_validator.py --check ric --upgrade UPGRADE-XXX
```

### Phase 7: Integration (Exit Check)
```bash
python scripts/qa_validator.py --verbose
```

## Examples

```bash
# Full debug check
/qa-debug

# Focus on specific upgrade
/qa-debug UPGRADE-014

# Just integrity checks
python scripts/qa_validator.py --check integrity

# Cross-reference validation
python scripts/qa_validator.py --check xref --verbose
```

---

Run debug validation now:

```bash
python3 scripts/qa_validator.py --check debug --check integrity --check xref $ARGUMENTS --verbose
```
