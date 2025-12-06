# Analyze Trading Strategy

Perform a code review and analysis of an existing trading strategy.

## Arguments
- `$ARGUMENTS` - Algorithm name or file path to analyze

## Instructions

1. Read the algorithm file and analyze:
   - Strategy logic and entry/exit conditions
   - Risk management implementation
   - Parameter choices and their rationale
   - Code quality and best practices

2. Check for common issues:
   - Look-ahead bias
   - Survivorship bias
   - Overfitting to historical data
   - Missing data validation
   - Inadequate risk controls

3. Review against QuantConnect best practices:
   - Proper indicator warmup
   - Efficient data access patterns
   - Appropriate use of Portfolio/Securities APIs
   - Logging and debugging capabilities

4. Provide recommendations:
   - Potential improvements
   - Additional risk controls to consider
   - Alternative approaches to the strategy
   - Suggested parameter ranges for optimization

Strategy to analyze: $ARGUMENTS
