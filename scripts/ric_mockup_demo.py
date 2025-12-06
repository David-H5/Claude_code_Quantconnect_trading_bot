#!/usr/bin/env python3
"""
RIC Loop Mockup Demo - Agent Orchestration Integration

Demonstrates how the agent orchestrator integrates with RIC phases.
This is a mockup/simulation - no actual Task tools are called.

Usage:
    python scripts/ric_mockup_demo.py [--phase PHASE] [--topic TOPIC]

Examples:
    python scripts/ric_mockup_demo.py
    python scripts/ric_mockup_demo.py --phase RESEARCH --topic "circuit breaker patterns"
    python scripts/ric_mockup_demo.py --phase VERIFY
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "hooks"))

from agent_orchestrator import (
    AGENT_TEMPLATES,
    WORKFLOW_TEMPLATES,
    AgentCircuitBreaker,
    FallbackRouter,
    TokenTracker,
    Tracer,
    WorkflowGenerator,
    detect_ric_phase,
    get_ric_recommended_agents,
    get_ric_recommended_workflow,
)


# =============================================================================
# Mockup Simulation
# =============================================================================


class MockAgentResult:
    """Simulated agent result for demonstration."""

    def __init__(self, agent_name: str, success: bool = True):
        self.agent_name = agent_name
        self.success = success
        self.output = f"[MOCKUP] {agent_name} completed analysis"
        self.tokens_in = 500 + hash(agent_name) % 500
        self.tokens_out = 1000 + hash(agent_name) % 1500
        self.duration_ms = 1000 + hash(agent_name) % 3000


def simulate_agent_execution(agent_name: str, model: str, tracer: Tracer, tracker: TokenTracker) -> MockAgentResult:
    """Simulate agent execution with tracing and token tracking."""

    # Start span
    span_id = tracer.start_span(agent_name)

    # Simulate execution
    import random
    import time

    time.sleep(0.1)  # Brief pause for realism

    # 90% success rate simulation
    success = random.random() > 0.1
    result = MockAgentResult(agent_name, success)

    # Track tokens
    tracker.record(agent_name, model, result.tokens_in, result.tokens_out)

    # End span
    tracer.end_span(
        span_id,
        status="success" if success else "failed",
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        error=None if success else "Simulated failure",
    )

    return result


def run_ric_phase(phase: str, topic: str = "default topic", verbose: bool = True) -> dict[str, Any]:
    """
    Run a simulated RIC phase with agent orchestration.

    Args:
        phase: RIC phase name (RESEARCH, PLAN, BUILD, VERIFY, REFLECT)
        topic: Topic/context for the phase
        verbose: Print detailed output

    Returns:
        Results dictionary with agents run, costs, traces
    """
    print(f"\n{'='*60}")
    print(f"  RIC PHASE: {phase}")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    # Get recommendations
    agents = get_ric_recommended_agents(phase)
    workflow = get_ric_recommended_workflow(phase)

    print(f"📋 Recommended Agents: {', '.join(agents)}")
    if workflow:
        print(f"🔄 Recommended Workflow: {workflow}")
    print()

    # Initialize tracking
    tracer = Tracer()
    tracker = TokenTracker()
    circuit_breaker = AgentCircuitBreaker()
    fallback_router = FallbackRouter()

    # Start trace
    trace_id = tracer.start_trace(f"ric_{phase.lower()}")
    print(f"🔍 Started trace: {trace_id}\n")

    results = []

    # Execute agents
    for agent_name in agents:
        # Check circuit breaker
        if not circuit_breaker.can_execute(agent_name):
            print(f"  ⚠️  {agent_name}: Circuit OPEN, trying fallback...")
            fallback = fallback_router.get_fallback(agent_name, 0)
            if fallback and circuit_breaker.can_execute(fallback):
                agent_name = fallback
                print(f"      → Using fallback: {fallback}")
            else:
                print("      ❌ No available fallback")
                continue

        # Get agent spec
        if agent_name in AGENT_TEMPLATES:
            agent = AGENT_TEMPLATES[agent_name]
            model = agent.model.value
        else:
            model = "haiku"

        print(f"  🤖 Running {agent_name} ({model})...")

        # Simulate execution
        result = simulate_agent_execution(agent_name, model, tracer, tracker)
        results.append(result)

        # Record success/failure
        if result.success:
            circuit_breaker.record_success(agent_name)
            print(f"      ✅ Success ({result.tokens_in}→{result.tokens_out} tokens)")
        else:
            circuit_breaker.record_failure(agent_name)
            print("      ❌ Failed (circuit breaker updated)")

    # End trace
    tracer.end_trace()

    # Print summary
    print(f"\n{'─'*40}")
    print("📊 Phase Summary")
    print(f"{'─'*40}")

    successful = sum(1 for r in results if r.success)
    print(f"  Agents: {successful}/{len(results)} successful")
    print(f"  Total cost: ${tracker.get_total_cost():.4f}")

    # Generate Task calls for reference
    if workflow and workflow in WORKFLOW_TEMPLATES:
        print(f"\n📝 Generated Task Calls for {workflow}:")
        print(f"{'─'*40}")
        wf = WORKFLOW_TEMPLATES[workflow]
        calls = WorkflowGenerator.generate_parallel(wf, {"topic": topic})
        for call in calls[:2]:  # Show first 2 for brevity
            lines = call.split("\n")
            for line in lines[:5]:
                print(f"  {line}")
            print("  ...")
        if len(calls) > 2:
            print(f"  ... and {len(calls)-2} more agents")

    return {
        "phase": phase,
        "agents_run": len(results),
        "successful": successful,
        "cost_usd": tracker.get_total_cost(),
        "trace_id": trace_id,
    }


def run_full_ric_loop(topic: str = "feature implementation"):
    """
    Run a complete mockup RIC loop through all 5 phases.

    Demonstrates:
    - Phase detection and recommendations
    - Agent spawning with tracing
    - Circuit breaker and fallback
    - Cost tracking
    - Execution tracing
    """
    print("\n" + "=" * 70)
    print("   RIC LOOP MOCKUP DEMONSTRATION - Agent Orchestration v1.5")
    print("=" * 70)
    print(f"\n🎯 Topic: {topic}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    phases = ["RESEARCH", "PLAN", "BUILD", "VERIFY", "REFLECT"]
    all_results = []

    for i, phase in enumerate(phases):
        print(f"\n[Iteration 1/{len(phases)}] Phase {i}: {phase}")
        results = run_ric_phase(phase, topic)
        all_results.append(results)

    # Final summary
    print("\n" + "=" * 70)
    print("   RIC LOOP COMPLETE - Summary")
    print("=" * 70)

    total_agents = sum(r["agents_run"] for r in all_results)
    total_successful = sum(r["successful"] for r in all_results)
    total_cost = sum(r["cost_usd"] for r in all_results)

    print(f"""
📊 Overall Results:
   Phases completed: {len(all_results)}/5
   Total agents spawned: {total_agents}
   Successful: {total_successful} ({100*total_successful/max(total_agents,1):.0f}%)
   Estimated cost: ${total_cost:.4f}

🔍 Traces:
""")
    for r in all_results:
        print(f"   - {r['trace_id']}")

    print("""
📝 Key Insights:
   - P0 RESEARCH: Use haiku agents for fast parallel exploration
   - P1 PLAN: Architect agent (sonnet) for design decisions
   - P2 BUILD: Implementer + refactorer (sonnet) for code
   - P3 VERIFY: Parallel haiku agents for testing + security
   - P4 REFLECT: deep_architect (opus) only if critical decisions needed

💡 Commands to Use:
   /agents run ric_research topic="your topic"
   /agents run ric_verify target="files to verify"
   /ric-agents          # Get phase recommendations
   /agent-trace         # View execution traces
""")


def main():
    parser = argparse.ArgumentParser(description="RIC Loop Mockup Demo - Agent Orchestration Integration")
    parser.add_argument(
        "--phase",
        choices=["RESEARCH", "PLAN", "BUILD", "VERIFY", "REFLECT"],
        help="Run single phase (default: full loop)",
    )
    parser.add_argument("--topic", default="circuit breaker implementation", help="Topic for the RIC session")
    parser.add_argument("--detect", action="store_true", help="Detect current phase from progress file and run")

    args = parser.parse_args()

    if args.detect:
        phase = detect_ric_phase()
        if phase:
            print(f"🔍 Detected phase from progress file: {phase}")
            run_ric_phase(phase, args.topic)
        else:
            print("❌ Could not detect RIC phase. Check claude-progress.txt")
            sys.exit(1)
    elif args.phase:
        run_ric_phase(args.phase, args.topic)
    else:
        run_full_ric_loop(args.topic)


if __name__ == "__main__":
    main()
