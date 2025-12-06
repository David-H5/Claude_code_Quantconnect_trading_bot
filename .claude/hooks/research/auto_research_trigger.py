#!/usr/bin/env python3
"""
Auto Research Trigger Hook - Detects research keywords and suggests thorough research.

This hook runs on UserPromptSubmit and checks if the user's prompt contains
research-related keywords. If detected, it suggests using thorough research
with fast parallel agents.

Hook Type: UserPromptSubmit
"""

import json
import os
import sys


# Import from thorough_research module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thorough_research import (
    extract_urls_from_text,
    should_auto_trigger,
)


def main():
    """Check user prompt for research keywords."""
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        # No input or invalid JSON - pass through
        print(json.dumps({"continue": True}))
        return

    # Get the user's prompt
    prompt = hook_input.get("prompt", "")
    if not prompt:
        print(json.dumps({"continue": True}))
        return

    # Check for research keywords
    should_trigger, keyword = should_auto_trigger(prompt)

    if not should_trigger:
        # No research keywords - continue normally
        print(json.dumps({"continue": True}))
        return

    # Research keywords detected - add suggestion
    urls = extract_urls_from_text(prompt)

    if urls:
        suggestion = f"""---
**🔬 THOROUGH RESEARCH RECOMMENDED** (keyword: "{keyword}")

Detected research intent with {len(urls)} URL(s). For comprehensive extraction:

```bash
python3 .claude/hooks/thorough_research.py auto "{prompt[:50]}..."
```

Or spawn fast haiku agents in parallel for multi-pass extraction.
---
"""
    else:
        suggestion = f"""---
**🔬 THOROUGH RESEARCH RECOMMENDED** (keyword: "{keyword}")

Detected research intent. After finding URLs via WebSearch, use:

```bash
python3 .claude/hooks/thorough_research.py plan <url1> <url2> --topic "research"
```

Benefits: 4x more info per URL, parallel haiku agents, immediate persistence.
---
"""

    # Return with suggestion appended (informational, doesn't block)
    result = {"continue": True, "stdout": suggestion}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
