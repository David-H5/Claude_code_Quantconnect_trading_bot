"""
Evaluation framework for autonomous AI agent trading system.

Layer: 3 (Domain Logic)
May import from: Layers 0-2 (utils, observability, config, models, compliance)
May be imported by: Layer 4 (algorithms, api, ui)

This module implements multiple evaluation methodologies:
- STOCKBENCH: Contamination-free testing with 2024-2025 market data
- CLASSic Framework: Cost, Latency, Accuracy, Stability, Security evaluation
- Walk-Forward Analysis: Out-of-sample validation
- Advanced Trading Metrics: Expectancy, Profit Factor, Omega Ratio
- Component-Level Evaluation: Individual subsystem testing
- Overfitting Prevention: QuantConnect best practices
- QuantConnect Integration: Backtesting integration
- PSI Drift Detection: Population Stability Index monitoring (2025)
- 2025 Benchmarks: Updated thresholds based on latest AI bot research
- TCA Evaluation: Transaction Cost Analysis (VWAP, Implementation Shortfall)
- Long-Horizon Evaluation: FINSABER-inspired 10-20 year backtesting
- Realistic Thresholds: Calibrated to Finance Agent Benchmark (o3: 46.8%)

Version: 2.1 (December 2025) - Added TCA, Long-Horizon, Realistic Thresholds
"""

# Core evaluation framework
# Agent Response Adapters (Phase 1 - December 2025)
from evaluation.adapters import (
    AgentResponseAdapter,
    adapt_response_to_decision,
    adapt_response_to_dict,
    batch_adapt_responses,
    create_test_case_from_response,
)

# Advanced trading metrics (updated with 2025 benchmarks)
from evaluation.advanced_trading_metrics import (
    AdvancedTradingMetrics,
    Trade,
    calculate_advanced_trading_metrics,
    check_overfitting_signals,
    compare_to_2025_benchmarks,
    generate_trading_metrics_report,
    get_trading_metrics_thresholds,
    get_trading_metrics_thresholds_2025,
    get_trading_metrics_thresholds_legacy,
)

# Agent-as-a-Judge evaluation (NEW - December 2025)
from evaluation.agent_as_judge import (
    ALL_RUBRICS,
    HALLUCINATION_CHECK_RUBRIC,
    MARKET_ANALYSIS_RUBRIC,
    REASONING_CHAIN_RUBRIC,
    RISK_ASSESSMENT_RUBRIC,
    TRADING_DECISION_RUBRIC,
    AgentDecision,
    EvaluationCategory,
    EvaluationRubric,
    JudgeEvaluationResult,
    JudgeModel,
    JudgeScore,
    ScoreLevel,
    aggregate_judge_scores,
    calculate_agreement_rate,
    check_judge_thresholds,
    create_judge_prompt,
    create_mock_judge,
    create_real_judge_function,
    evaluate_agent_decision,
    evaluate_with_position_debiasing,
    generate_judge_report,
    parse_judge_response,
)

# Agent Performance Contest (UPGRADE-010 Sprint 2 - December 2025)
from evaluation.agent_contest import (
    AgentRanking,
    AgentRecord,
    ContestManager,
    ELOSystem,
    Prediction,
    PredictionOutcome,
    VoteResult,
    VotingEngine,
    create_contest_manager,
)

# Agent Performance Metrics (UPGRADE-004 - December 2025)
from evaluation.agent_metrics import (
    AgentComparison,
    AgentMetrics,
    AgentMetricsTracker,
    DecisionRecord,
    PerformanceTrend,
    create_metrics_tracker,
    generate_metrics_report,
)

# Sprint 1.7 Alerting Pipeline (UPGRADE-010 Sprint 1.7 - December 2025)
from evaluation.anomaly_alerting_bridge import (
    AlertingAction,
    AlertingPipelineConfig,
    AnomalyAlertingBridge,
    create_alerting_pipeline,
    create_full_sprint1_pipeline,
)

# CLASSic framework (ICLR 2025, enhanced December 2025)
from evaluation.classic_evaluation import (
    AISecurityMetrics,
    CLASSicMetrics,
    ComprehensiveSecurityMetrics,
    DataProtectionMetrics,
    SecurityCategory,
    SecurityIncident,
    SecurityTestCase,
    # Enhanced Security Evaluation (NEW - December 2025)
    SecurityThreatLevel,
    TradingSecurityMetrics,
    calculate_classic_metrics,
    calculate_security_metrics,
    detect_credential_exposure,
    detect_hallucination_trading,
    detect_prompt_injection,
    generate_classic_report,
    generate_security_report,
    get_classic_thresholds,
    get_sample_security_test_cases,
    run_security_test_suite,
)

# Component-level evaluation
from evaluation.component_evaluation import (
    ComponentEvaluationResult,
    ComponentEvaluator,
    ComponentTestCase,
    ComponentType,
    generate_component_report,
    run_component_suite,
)

# Sprint 1.6 Unified Decision Context (UPGRADE-010 Sprint 1.6 - December 2025)
from evaluation.decision_context import (
    ContextCompleteness,
    DecisionContextBuilder,
    DecisionContextManager,
    ExplanationContext,
    MarketContext,
    UnifiedDecisionContext,
    create_context_manager,
    create_decision_context,
)
from evaluation.evaluation_framework import AgentEvaluator, EvaluationResult, TestCase

# SHAP Decision Explainer (UPGRADE-010 Sprint 1 - December 2025)
from evaluation.explainer import (
    LIME_AVAILABLE,
    SHAP_AVAILABLE,
    BaseExplainer,
    ExplainerConfig,
    ExplainerFactory,
    Explanation,
    ExplanationLogger,
    ExplanationType,
    FeatureContribution,
    FeatureImportanceExplainer,
    LIMEExplainer,
    ModelType,
    SHAPExplainer,
    create_explainer,
    create_lime_explainer,
    create_shap_explainer,
)

# Evaluator-Optimizer Feedback Loop (UPGRADE-003 - December 2025)
from evaluation.feedback_loop import (
    ConvergenceReason,
    EvaluatorOptimizerLoop,
    FeedbackCycle,
    FeedbackResult,
    PromptRefinement,
    Weakness,
    WeaknessCategory,
    create_feedback_loop,
    generate_feedback_report,
)

# Real LLM Judge Implementation (Phase 1 - December 2025)
from evaluation.llm_judge import (
    JudgeConfig,
    JudgeCostTracker,
    LLMJudge,
    create_judge_function,
    create_production_judge,
)

# Long-horizon evaluation (NEW - December 2025)
from evaluation.long_horizon_evaluation import (
    LONG_HORIZON_THRESHOLDS,
    DegradationSeverity,
    HorizonResult,
    LongHorizonMetrics,
    LongTermViability,
    check_long_horizon_thresholds,
    generate_long_horizon_report,
    run_long_horizon_backtest,
)
from evaluation.metrics import (
    calculate_agent_metrics,
    calculate_stockbench_metrics,
    calculate_v6_1_metrics,
)

# Overfitting prevention
from evaluation.overfitting_prevention import (
    OverfitRiskLevel,
    OverfittingMetrics,
    calculate_overfitting_metrics,
    generate_overfitting_report,
    validate_quantconnect_best_practices,
)

# PSI drift detection (NEW - December 2025)
from evaluation.psi_drift_detection import (
    DriftLevel,
    PSIResult,
    StrategyHealthMetrics,
    calculate_psi,
    calculate_psi_for_metric,
    calculate_strategy_health,
    check_drift_with_psi,
    generate_psi_report,
)

# QuantConnect integration
from evaluation.quantconnect_integration import (
    BacktestConfig,
    BacktestResult,
    compare_backtests,
    generate_backtest_report,
    run_quantconnect_backtest_cli,
)

# Realistic thresholds calibration (NEW - December 2025)
from evaluation.realistic_thresholds import (
    CONSERVATIVE_THRESHOLDS,
    OPTIMISTIC_THRESHOLDS,
    REALISTIC_THRESHOLDS_2025,
    AccuracyThresholds,
    CostThresholds,
    DecisionComplexity,
    ExpectationLevel,
    RealisticExpectations,
    calculate_profitability_requirements,
    calibrate_expectations,
    generate_realistic_expectations_report,
    get_thresholds,
    validate_against_benchmarks,
)

# Sprint 1.5 Unified Monitoring Report (UPGRADE-010 Sprint 1.5 - December 2025)
# Sprint 1.7: Added DecisionSummary
from evaluation.sprint1_monitoring import (
    AnomalySummary,
    DecisionSummary,
    ExplanationSummary,
    ReasoningChainSummary,
    Sprint1MonitoringReport,
    export_sprint1_report,
    generate_sprint1_report,
    generate_sprint1_text_report,
)

# TCA evaluation (NEW - December 2025)
from evaluation.tca_evaluation import (
    TCA_THRESHOLDS,
    ExecutionQuality,
    ExecutionRecord,
    TCAMetrics,
    TCAReport,
    calculate_arrival_cost,
    calculate_implementation_shortfall,
    calculate_market_impact,
    calculate_tca_metrics,
    calculate_vwap_deviation,
    check_tca_compliance,
    format_tca_report,
    generate_tca_report,
)

# Walk-forward analysis (updated with Monte Carlo - December 2025)
from evaluation.walk_forward_analysis import (
    MonteCarloIteration,
    MonteCarloResult,
    WalkForwardResult,
    WalkForwardWindow,
    create_walk_forward_schedule,
    generate_monte_carlo_report,
    generate_walk_forward_report,
    run_monte_carlo_walk_forward,
    run_walk_forward_analysis,
)


__all__ = [
    # Core framework
    "AgentEvaluator",
    "EvaluationResult",
    "TestCase",
    "calculate_agent_metrics",
    "calculate_stockbench_metrics",
    "calculate_v6_1_metrics",
    # PSI drift detection (NEW - December 2025)
    "PSIResult",
    "DriftLevel",
    "StrategyHealthMetrics",
    "calculate_psi",
    "calculate_psi_for_metric",
    "calculate_strategy_health",
    "generate_psi_report",
    "check_drift_with_psi",
    # CLASSic framework (enhanced December 2025)
    "CLASSicMetrics",
    "calculate_classic_metrics",
    "generate_classic_report",
    "get_classic_thresholds",
    # Enhanced Security Evaluation (NEW - December 2025)
    "SecurityThreatLevel",
    "SecurityCategory",
    "SecurityIncident",
    "SecurityTestCase",
    "AISecurityMetrics",
    "TradingSecurityMetrics",
    "DataProtectionMetrics",
    "ComprehensiveSecurityMetrics",
    "detect_prompt_injection",
    "detect_hallucination_trading",
    "detect_credential_exposure",
    "run_security_test_suite",
    "calculate_security_metrics",
    "generate_security_report",
    "get_sample_security_test_cases",
    # Walk-forward analysis (updated with Monte Carlo - December 2025)
    "WalkForwardResult",
    "WalkForwardWindow",
    "MonteCarloResult",
    "MonteCarloIteration",
    "run_walk_forward_analysis",
    "generate_walk_forward_report",
    "create_walk_forward_schedule",
    "run_monte_carlo_walk_forward",
    "generate_monte_carlo_report",
    # Advanced trading metrics (updated with 2025 benchmarks)
    "AdvancedTradingMetrics",
    "Trade",
    "calculate_advanced_trading_metrics",
    "generate_trading_metrics_report",
    "get_trading_metrics_thresholds",
    "get_trading_metrics_thresholds_2025",
    "get_trading_metrics_thresholds_legacy",
    "check_overfitting_signals",
    "compare_to_2025_benchmarks",
    # Component evaluation
    "ComponentEvaluator",
    "ComponentEvaluationResult",
    "ComponentTestCase",
    "ComponentType",
    "run_component_suite",
    "generate_component_report",
    # Overfitting prevention
    "OverfittingMetrics",
    "OverfitRiskLevel",
    "calculate_overfitting_metrics",
    "generate_overfitting_report",
    "validate_quantconnect_best_practices",
    # QuantConnect integration
    "BacktestConfig",
    "BacktestResult",
    "run_quantconnect_backtest_cli",
    "generate_backtest_report",
    "compare_backtests",
    # TCA evaluation (NEW - December 2025)
    "ExecutionRecord",
    "TCAMetrics",
    "TCAReport",
    "ExecutionQuality",
    "TCA_THRESHOLDS",
    "calculate_vwap_deviation",
    "calculate_implementation_shortfall",
    "calculate_arrival_cost",
    "calculate_market_impact",
    "calculate_tca_metrics",
    "generate_tca_report",
    "format_tca_report",
    "check_tca_compliance",
    # Long-horizon evaluation (NEW - December 2025)
    "HorizonResult",
    "LongHorizonMetrics",
    "DegradationSeverity",
    "LongTermViability",
    "LONG_HORIZON_THRESHOLDS",
    "run_long_horizon_backtest",
    "generate_long_horizon_report",
    "check_long_horizon_thresholds",
    # Realistic thresholds calibration (NEW - December 2025)
    "DecisionComplexity",
    "AccuracyThresholds",
    "CostThresholds",
    "RealisticExpectations",
    "ExpectationLevel",
    "REALISTIC_THRESHOLDS_2025",
    "OPTIMISTIC_THRESHOLDS",
    "CONSERVATIVE_THRESHOLDS",
    "get_thresholds",
    "calibrate_expectations",
    "calculate_profitability_requirements",
    "validate_against_benchmarks",
    "generate_realistic_expectations_report",
    # Agent-as-a-Judge evaluation (NEW - December 2025)
    "JudgeModel",
    "EvaluationCategory",
    "ScoreLevel",
    "JudgeScore",
    "EvaluationRubric",
    "AgentDecision",
    "JudgeEvaluationResult",
    "TRADING_DECISION_RUBRIC",
    "RISK_ASSESSMENT_RUBRIC",
    "REASONING_CHAIN_RUBRIC",
    "HALLUCINATION_CHECK_RUBRIC",
    "MARKET_ANALYSIS_RUBRIC",
    "ALL_RUBRICS",
    "create_judge_prompt",
    "parse_judge_response",
    "evaluate_with_position_debiasing",
    "evaluate_agent_decision",
    "generate_judge_report",
    "check_judge_thresholds",
    "create_mock_judge",
    "create_real_judge_function",
    "calculate_agreement_rate",
    "aggregate_judge_scores",
    # Real LLM Judge Implementation (Phase 1 - December 2025)
    "JudgeConfig",
    "JudgeCostTracker",
    "LLMJudge",
    "create_production_judge",
    "create_judge_function",
    # Agent Response Adapters (Phase 1 - December 2025)
    "AgentResponseAdapter",
    "adapt_response_to_dict",
    "adapt_response_to_decision",
    "create_test_case_from_response",
    "batch_adapt_responses",
    # Evaluator-Optimizer Feedback Loop (UPGRADE-003 - December 2025)
    "EvaluatorOptimizerLoop",
    "FeedbackResult",
    "FeedbackCycle",
    "Weakness",
    "WeaknessCategory",
    "PromptRefinement",
    "ConvergenceReason",
    "create_feedback_loop",
    "generate_feedback_report",
    # Agent Performance Metrics (UPGRADE-004 - December 2025)
    "AgentMetricsTracker",
    "AgentMetrics",
    "DecisionRecord",
    "PerformanceTrend",
    "AgentComparison",
    "create_metrics_tracker",
    "generate_metrics_report",
    # SHAP Decision Explainer (UPGRADE-010 Sprint 1 - December 2025)
    "ExplanationType",
    "ModelType",
    "FeatureContribution",
    "Explanation",
    "ExplainerConfig",
    "BaseExplainer",
    "SHAPExplainer",
    "LIMEExplainer",
    "FeatureImportanceExplainer",
    "ExplainerFactory",
    "ExplanationLogger",
    "create_explainer",
    "create_shap_explainer",
    "create_lime_explainer",
    "SHAP_AVAILABLE",
    "LIME_AVAILABLE",
    # Agent Performance Contest (UPGRADE-010 Sprint 2 - December 2025)
    "PredictionOutcome",
    "Prediction",
    "AgentRecord",
    "AgentRanking",
    "ELOSystem",
    "VoteResult",
    "VotingEngine",
    "ContestManager",
    "create_contest_manager",
    # Sprint 1.5 Unified Monitoring (UPGRADE-010 Sprint 1.5 - December 2025)
    # Sprint 1.7: Added DecisionSummary
    "ReasoningChainSummary",
    "AnomalySummary",
    "ExplanationSummary",
    "DecisionSummary",
    "Sprint1MonitoringReport",
    "generate_sprint1_report",
    "export_sprint1_report",
    "generate_sprint1_text_report",
    # Sprint 1.6 Unified Decision Context (UPGRADE-010 Sprint 1.6 - December 2025)
    "ContextCompleteness",
    "MarketContext",
    "ExplanationContext",
    "UnifiedDecisionContext",
    "DecisionContextBuilder",
    "DecisionContextManager",
    "create_decision_context",
    "create_context_manager",
    # Sprint 1.7 Alerting Pipeline (UPGRADE-010 Sprint 1.7 - December 2025)
    "AlertingAction",
    "AlertingPipelineConfig",
    "AnomalyAlertingBridge",
    "create_alerting_pipeline",
    "create_full_sprint1_pipeline",
]
