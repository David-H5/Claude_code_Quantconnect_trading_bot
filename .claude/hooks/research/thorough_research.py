#!/usr/bin/env python3
"""
Thorough Research Helper - Multi-pass URL fetching with fast agents.

This script automates thorough web research by:
1. Multiple fetch passes per URL with different extraction prompts
2. Generating fast agent (haiku) Task calls for parallel fetching
3. Immediate file persistence to avoid context compaction loss

Usage:
    # Generate Task calls for thorough research
    python3 .claude/hooks/thorough_research.py generate <url> [--topic "topic"]

    # Generate multi-URL parallel research plan
    python3 .claude/hooks/thorough_research.py plan <url1> <url2> ... [--topic "topic"]

    # Generate research file template
    python3 .claude/hooks/thorough_research.py template <topic> [--output path]

    # Show fetch prompts for a category
    python3 .claude/hooks/thorough_research.py prompts [category]

Part of RIC v5.0+ research automation.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


# Fetch prompt categories - each extracts different aspects
FETCH_PROMPTS = {
    "overview": {
        "name": "Overview & Concepts",
        "prompt": "Extract the main concepts, purpose, and high-level architecture. Include any diagrams or flowcharts described. List key terminology with definitions.",
        "priority": 1,
    },
    "code": {
        "name": "Code Examples",
        "prompt": "Extract ALL code examples, code snippets, and implementation samples VERBATIM. Preserve exact formatting, comments, and variable names. Include the context/explanation for each code block.",
        "priority": 1,
    },
    "api": {
        "name": "API & Signatures",
        "prompt": "Extract ALL function signatures, method definitions, class interfaces, API endpoints, and their parameters. Include return types, default values, and parameter descriptions.",
        "priority": 2,
    },
    "config": {
        "name": "Configuration & Parameters",
        "prompt": "Extract ALL configuration options, settings, parameters, environment variables, and their default values. Include valid ranges, types, and examples.",
        "priority": 2,
    },
    "examples": {
        "name": "Usage Examples & Patterns",
        "prompt": "Extract ALL usage examples, best practices, common patterns, and recommended approaches. Include anti-patterns or things to avoid if mentioned.",
        "priority": 2,
    },
    "troubleshooting": {
        "name": "Troubleshooting & Errors",
        "prompt": "Extract ALL error messages, troubleshooting steps, common issues, FAQs, and their solutions. Include edge cases and gotchas.",
        "priority": 3,
    },
    "changelog": {
        "name": "Changes & Version Info",
        "prompt": "Extract version information, changelog entries, breaking changes, deprecations, and migration guides. Note publication/update dates.",
        "priority": 3,
    },
}

# Topic-specific prompt sets for common research areas
TOPIC_PRESETS = {
    "quantconnect": ["overview", "code", "api", "config", "examples"],
    "trading": ["overview", "code", "examples", "config"],
    "api": ["api", "code", "config", "examples", "troubleshooting"],
    "library": ["overview", "api", "code", "examples", "config"],
    "tutorial": ["overview", "code", "examples"],
    "documentation": ["overview", "api", "config", "examples"],
    "research": ["overview", "code", "examples"],
    "default": ["overview", "code", "api", "config"],
}

# Keywords that auto-trigger thorough research mode
RESEARCH_TRIGGER_KEYWORDS = [
    # Primary triggers
    "research",
    "investigate",
    "explore",
    "deep dive",
    "thorough",
    "comprehensive",
    # Documentation triggers
    "documentation",
    "docs",
    "learn about",
    "understand",
    "how does",
    "how to",
    # Analysis triggers
    "analyze",
    "study",
    "examine",
    "review",
    # Comparison triggers
    "compare",
    "difference between",
    "pros and cons",
    "best practices",
]


def should_auto_trigger(text: str) -> tuple[bool, str]:
    """
    Check if text contains keywords that should auto-trigger thorough research.

    Returns:
        Tuple of (should_trigger, matched_keyword)
    """
    text_lower = text.lower()

    for keyword in RESEARCH_TRIGGER_KEYWORDS:
        if keyword in text_lower:
            return True, keyword

    return False, ""


def extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from text."""
    import re

    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def auto_research_suggestion(text: str) -> str | None:
    """
    Generate auto-research suggestion if keywords detected.

    Returns suggestion text or None if no trigger.
    """
    should_trigger, keyword = should_auto_trigger(text)

    if not should_trigger:
        return None

    urls = extract_urls_from_text(text)

    suggestion = f"""
**🔬 THOROUGH RESEARCH AUTO-TRIGGERED** (keyword: "{keyword}")

Detected research intent. Recommended approach:

"""

    if urls:
        suggestion += f"""**URLs Found**: {len(urls)} URLs detected in your request.

**Recommended Action**: Spawn fast agents for multi-pass extraction:
```bash
python3 .claude/hooks/thorough_research.py plan {' '.join(urls[:3])} --topic "research"
```

Or execute directly with parallel Task calls (haiku model).
"""
    else:
        suggestion += """**No URLs Found**: First search for relevant sources:
1. Use WebSearch to find URLs
2. Then run thorough research on discovered URLs:
   ```bash
   python3 .claude/hooks/thorough_research.py plan <url1> <url2> --topic "research"
   ```
"""

    suggestion += """
**Benefits of Thorough Research**:
- 4x more information per URL (multi-pass extraction)
- Parallel haiku agents (~30-60 seconds)
- Immediate file persistence (prevents context loss)
"""

    return suggestion


def get_prompts_for_topic(topic: str | None) -> list[str]:
    """Get appropriate fetch prompt categories for a topic."""
    if not topic:
        return TOPIC_PRESETS["default"]

    topic_lower = topic.lower()
    for key, prompts in TOPIC_PRESETS.items():
        if key in topic_lower:
            return prompts

    return TOPIC_PRESETS["default"]


def generate_task_call(
    url: str,
    prompt_category: str,
    topic: str,
    model: str = "haiku",
) -> dict:
    """Generate a Task tool call for fetching a URL."""
    prompt_info = FETCH_PROMPTS[prompt_category]

    task_prompt = f"""Research Task: Fetch and extract information from URL.

**URL**: {url}
**Topic**: {topic}
**Extraction Focus**: {prompt_info['name']}

**Instructions**:
1. Use WebFetch to fetch the URL with this prompt:
   "{prompt_info['prompt']}"

2. Format the extracted content as markdown with clear headers.

3. Return the extracted content in this format:

   ## {prompt_info['name']} - {urlparse(url).netloc}

   **Source**: {url}
   **Fetched**: [current timestamp]

   [Extracted content here]

4. If the fetch fails or content is empty, report the error.

Be thorough - extract ALL relevant information for this category."""

    return {
        "tool": "Task",
        "params": {
            "subagent_type": "general-purpose",
            "model": model,
            "description": f"Fetch {prompt_category} from URL",
            "prompt": task_prompt,
        },
    }


def generate_parallel_research_plan(
    urls: list[str],
    topic: str,
    thorough: bool = True,
) -> list[dict]:
    """Generate Task calls for parallel URL research."""
    tasks = []
    prompt_categories = get_prompts_for_topic(topic)

    if not thorough:
        # Quick mode: only overview for each URL
        prompt_categories = ["overview"]

    for url in urls:
        for category in prompt_categories:
            tasks.append(generate_task_call(url, category, topic))

    return tasks


def generate_research_template(topic: str, urls: list[str] | None = None) -> str:
    """Generate a research document template."""
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p %Z")

    url_section = ""
    if urls:
        url_list = "\n".join(f"- {url}" for url in urls)
        url_section = f"""
## Sources

{url_list}
"""

    return f"""# {topic} Research - {now.strftime("%B %Y")}

## Research Overview

**Date**: {date_str}
**Time Started**: {time_str}
**Topic**: {topic}
**Status**: In Progress

## Research Objectives

1. [Objective 1]
2. [Objective 2]
3. [Objective 3]
{url_section}
## Findings

### Overview & Concepts

[Content from overview fetch]

### Code Examples

[Content from code fetch]

### API & Configuration

[Content from api/config fetch]

### Usage Patterns

[Content from examples fetch]

## Key Discoveries

| Discovery | Impact | Source |
|-----------|--------|--------|
| | | |

## Action Items

- [ ] [Action 1]
- [ ] [Action 2]

## Change Log

| Date | Change | Impact |
|------|--------|--------|
| {date_str} | Initial research | - |
"""


def format_task_calls_for_claude(tasks: list[dict]) -> str:
    """Format Task calls for Claude to execute."""
    output = []
    output.append("## Parallel Research Tasks")
    output.append("")
    output.append(
        f"Execute these {len(tasks)} Task calls **in parallel** (single message with multiple Task tool uses):"
    )
    output.append("")
    output.append("```")

    for i, task in enumerate(tasks, 1):
        params = task["params"]
        output.append(f"Task {i}:")
        output.append(f"  subagent_type: {params['subagent_type']}")
        output.append(f"  model: {params['model']}")
        output.append(f"  description: {params['description']}")
        output.append("  prompt: |")
        for line in params["prompt"].split("\n"):
            output.append(f"    {line}")
        output.append("")

    output.append("```")
    output.append("")
    output.append("**After all agents return**: Consolidate findings into the research file immediately.")

    return "\n".join(output)


def format_task_calls_as_json(tasks: list[dict]) -> str:
    """Format Task calls as JSON for programmatic use."""
    return json.dumps(tasks, indent=2)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Thorough Research Helper - Multi-pass URL fetching with fast agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate research tasks for a single URL
    python3 .claude/hooks/thorough_research.py generate https://docs.example.com --topic "API Integration"

    # Plan parallel research for multiple URLs
    python3 .claude/hooks/thorough_research.py plan url1 url2 url3 --topic "QuantConnect"

    # Create research file template
    python3 .claude/hooks/thorough_research.py template "MCP Server Research" --output docs/research/

    # List available fetch prompts
    python3 .claude/hooks/thorough_research.py prompts
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate Task calls for a URL")
    gen_parser.add_argument("url", help="URL to research")
    gen_parser.add_argument("--topic", "-t", default="general", help="Research topic")
    gen_parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (overview only)")
    gen_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Plan parallel research for multiple URLs")
    plan_parser.add_argument("urls", nargs="+", help="URLs to research")
    plan_parser.add_argument("--topic", "-t", default="general", help="Research topic")
    plan_parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (overview only)")
    plan_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # Template command
    tmpl_parser = subparsers.add_parser("template", help="Generate research document template")
    tmpl_parser.add_argument("topic", help="Research topic")
    tmpl_parser.add_argument("--output", "-o", help="Output directory or file path")
    tmpl_parser.add_argument("--urls", "-u", nargs="*", help="URLs to include in template")

    # Prompts command
    prompts_parser = subparsers.add_parser("prompts", help="Show available fetch prompts")
    prompts_parser.add_argument("category", nargs="?", help="Specific category to show")

    # Check command - auto-detect research intent
    check_parser = subparsers.add_parser("check", help="Check text for research keywords (auto-trigger)")
    check_parser.add_argument("text", nargs="+", help="Text to check for research keywords")
    check_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # Auto command - check + generate if triggered
    auto_parser = subparsers.add_parser("auto", help="Auto-detect and generate research tasks")
    auto_parser.add_argument("text", nargs="+", help="Text to analyze (will extract URLs and generate tasks)")
    auto_parser.add_argument("--topic", "-t", default="research", help="Research topic")

    # Keywords command - show trigger keywords
    subparsers.add_parser("keywords", help="Show auto-trigger keywords")

    # Help command
    subparsers.add_parser("help", help="Show this help message")

    args = parser.parse_args()

    if args.command == "generate":
        categories = get_prompts_for_topic(args.topic)
        if args.quick:
            categories = ["overview"]

        tasks = [generate_task_call(args.url, cat, args.topic) for cat in categories]

        if args.json:
            print(format_task_calls_as_json(tasks))
        else:
            print(format_task_calls_for_claude(tasks))

    elif args.command == "plan":
        tasks = generate_parallel_research_plan(args.urls, args.topic, thorough=not args.quick)

        if args.json:
            print(format_task_calls_as_json(tasks))
        else:
            print(format_task_calls_for_claude(tasks))
            print(f"\n**Total agents to spawn**: {len(tasks)} (haiku model)")
            print("**Estimated parallel time**: ~30-60 seconds")

    elif args.command == "template":
        template = generate_research_template(args.topic, args.urls)

        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir():
                # Generate filename from topic
                safe_name = args.topic.upper().replace(" ", "-").replace("_", "-")
                output_path = output_path / f"{safe_name}-RESEARCH.md"

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(template)
            print(f"Created: {output_path}")
        else:
            print(template)

    elif args.command == "prompts":
        if args.category:
            if args.category in FETCH_PROMPTS:
                info = FETCH_PROMPTS[args.category]
                print(f"## {info['name']} (Priority {info['priority']})")
                print()
                print(f"**Prompt**: {info['prompt']}")
            else:
                print(f"Unknown category: {args.category}")
                print(f"Available: {', '.join(FETCH_PROMPTS.keys())}")
                sys.exit(1)
        else:
            print("## Available Fetch Prompt Categories")
            print()
            for key, info in sorted(FETCH_PROMPTS.items(), key=lambda x: x[1]["priority"]):
                print(f"### {key} - {info['name']} (Priority {info['priority']})")
                print(f"  {info['prompt'][:80]}...")
                print()

            print("## Topic Presets")
            print()
            for topic, categories in TOPIC_PRESETS.items():
                print(f"  {topic}: {', '.join(categories)}")

    elif args.command == "check":
        text = " ".join(args.text)
        should_trigger, keyword = should_auto_trigger(text)

        if args.json:
            result = {
                "triggered": should_trigger,
                "keyword": keyword,
                "urls": extract_urls_from_text(text),
            }
            print(json.dumps(result, indent=2))
        else:
            if should_trigger:
                suggestion = auto_research_suggestion(text)
                print(suggestion)
            else:
                print("No research keywords detected.")
                print(f"\nTrigger keywords: {', '.join(RESEARCH_TRIGGER_KEYWORDS[:10])}...")

    elif args.command == "auto":
        text = " ".join(args.text)
        should_trigger, keyword = should_auto_trigger(text)
        urls = extract_urls_from_text(text)

        if not should_trigger and not urls:
            print("No research keywords or URLs detected.")
            print("Use 'check' command to see trigger keywords.")
            sys.exit(0)

        if urls:
            # Generate tasks for found URLs
            print(f"**🔬 AUTO-RESEARCH** (keyword: '{keyword}' | URLs: {len(urls)})\n")
            tasks = generate_parallel_research_plan(urls, args.topic, thorough=True)
            print(format_task_calls_for_claude(tasks))
            print(f"\n**Total agents to spawn**: {len(tasks)} (haiku model)")
        else:
            # No URLs, suggest WebSearch first
            print(f"**🔬 AUTO-RESEARCH** (keyword: '{keyword}')\n")
            print("No URLs found in text. Suggested workflow:")
            print("1. WebSearch for relevant URLs")
            print("2. Run: python3 .claude/hooks/thorough_research.py plan <urls> --topic research")

    elif args.command == "keywords":
        print("## Auto-Trigger Keywords")
        print()
        print("When these keywords are detected, thorough research is recommended:\n")
        for kw in RESEARCH_TRIGGER_KEYWORDS:
            print(f"  - {kw}")
        print()
        print("Use: python3 .claude/hooks/thorough_research.py check '<your text>'")

    elif args.command == "help" or args.command is None:
        parser.print_help()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
