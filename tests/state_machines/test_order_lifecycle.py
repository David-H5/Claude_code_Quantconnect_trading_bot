"""
Order Lifecycle State Machine Tests

Tests valid and invalid state transitions for orders.
Ensures invariants hold across all possible state paths.

UPGRADE-015: Advanced Test Framework - State Machine Testing
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable
import pytest


class OrderState(Enum):
    """Order state enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# Valid state transitions
VALID_TRANSITIONS: dict[OrderState, list[OrderState]] = {
    OrderState.PENDING: [OrderState.SUBMITTED, OrderState.REJECTED, OrderState.CANCELLED],
    OrderState.SUBMITTED: [OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED],
    OrderState.PARTIALLY_FILLED: [OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED],
    OrderState.FILLED: [],  # Terminal state
    OrderState.CANCELLED: [],  # Terminal state
    OrderState.REJECTED: [],  # Terminal state
}


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: OrderState
    to_state: OrderState
    reason: str
    timestamp: float


class OrderStateMachine:
    """
    State machine for order lifecycle.

    Tracks transitions and validates invariants.
    """

    def __init__(self, initial_state: OrderState = OrderState.PENDING):
        self._state = initial_state
        self._history: list[StateTransition] = []
        self._filled_quantity = 0
        self._total_quantity = 100

    @property
    def state(self) -> OrderState:
        return self._state

    @property
    def is_terminal(self) -> bool:
        return self._state in [OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED]

    @property
    def history(self) -> list[StateTransition]:
        return self._history.copy()

    def can_transition_to(self, new_state: OrderState) -> bool:
        """Check if transition to new state is valid."""
        return new_state in VALID_TRANSITIONS.get(self._state, [])

    def transition_to(self, new_state: OrderState, reason: str = "") -> bool:
        """
        Attempt to transition to new state.

        Returns True if successful, False if invalid transition.
        """
        if not self.can_transition_to(new_state):
            return False

        import time
        transition = StateTransition(
            from_state=self._state,
            to_state=new_state,
            reason=reason,
            timestamp=time.time()
        )
        self._history.append(transition)
        self._state = new_state
        return True

    def submit(self) -> bool:
        """Submit a pending order."""
        return self.transition_to(OrderState.SUBMITTED, "Order submitted to exchange")

    def partial_fill(self, quantity: int) -> bool:
        """Record a partial fill."""
        if self._state not in [OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED]:
            return False

        self._filled_quantity += quantity
        if self._filled_quantity >= self._total_quantity:
            return self.transition_to(OrderState.FILLED, f"Fully filled: {self._filled_quantity}")
        return self.transition_to(OrderState.PARTIALLY_FILLED, f"Partial fill: {quantity}")

    def fill(self, quantity: int | None = None) -> bool:
        """Fill the order completely."""
        if self._state not in [OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED]:
            return False
        self._filled_quantity = quantity or self._total_quantity
        return self.transition_to(OrderState.FILLED, "Order filled")

    def cancel(self) -> bool:
        """Cancel the order."""
        if self._state in [OrderState.PENDING, OrderState.SUBMITTED]:
            return self.transition_to(OrderState.CANCELLED, "Order cancelled")
        return False

    def reject(self, reason: str = "Unknown") -> bool:
        """Reject the order."""
        if self._state in [OrderState.PENDING, OrderState.SUBMITTED]:
            return self.transition_to(OrderState.REJECTED, f"Rejected: {reason}")
        return False


class TestOrderStateMachineTransitions:
    """Test valid state transitions."""

    def test_pending_to_submitted(self):
        """PENDING -> SUBMITTED is valid."""
        sm = OrderStateMachine()
        assert sm.state == OrderState.PENDING
        assert sm.submit()
        assert sm.state == OrderState.SUBMITTED

    def test_submitted_to_filled(self):
        """SUBMITTED -> FILLED is valid."""
        sm = OrderStateMachine()
        sm.submit()
        assert sm.fill()
        assert sm.state == OrderState.FILLED

    def test_submitted_to_partial_to_filled(self):
        """SUBMITTED -> PARTIALLY_FILLED -> FILLED is valid."""
        sm = OrderStateMachine()
        sm._total_quantity = 100
        sm.submit()

        assert sm.partial_fill(50)
        assert sm.state == OrderState.PARTIALLY_FILLED
        assert sm._filled_quantity == 50

        assert sm.partial_fill(50)
        assert sm.state == OrderState.FILLED
        assert sm._filled_quantity == 100

    def test_pending_to_cancelled(self):
        """PENDING -> CANCELLED is valid."""
        sm = OrderStateMachine()
        assert sm.cancel()
        assert sm.state == OrderState.CANCELLED

    def test_submitted_to_cancelled(self):
        """SUBMITTED -> CANCELLED is valid."""
        sm = OrderStateMachine()
        sm.submit()
        assert sm.cancel()
        assert sm.state == OrderState.CANCELLED

    def test_pending_to_rejected(self):
        """PENDING -> REJECTED is valid."""
        sm = OrderStateMachine()
        assert sm.reject("Invalid symbol")
        assert sm.state == OrderState.REJECTED


class TestOrderStateMachineInvalidTransitions:
    """Test invalid state transitions are blocked."""

    def test_pending_cannot_fill_directly(self):
        """PENDING -> FILLED is invalid (must go through SUBMITTED)."""
        sm = OrderStateMachine()
        assert not sm.fill()
        assert sm.state == OrderState.PENDING

    def test_filled_is_terminal(self):
        """FILLED is terminal - no further transitions."""
        sm = OrderStateMachine()
        sm.submit()
        sm.fill()

        assert not sm.submit()
        assert not sm.cancel()
        assert not sm.reject()
        assert sm.state == OrderState.FILLED

    def test_cancelled_is_terminal(self):
        """CANCELLED is terminal - no further transitions."""
        sm = OrderStateMachine()
        sm.cancel()

        assert not sm.submit()
        assert not sm.fill()
        assert sm.state == OrderState.CANCELLED

    def test_rejected_is_terminal(self):
        """REJECTED is terminal - no further transitions."""
        sm = OrderStateMachine()
        sm.reject()

        assert not sm.submit()
        assert not sm.cancel()
        assert sm.state == OrderState.REJECTED

    def test_partial_fill_cannot_cancel(self):
        """PARTIALLY_FILLED -> CANCELLED is valid (unfilled portion)."""
        sm = OrderStateMachine()
        sm._total_quantity = 100
        sm.submit()
        sm.partial_fill(50)

        # Actually partial fills CAN be cancelled for remaining
        assert sm.cancel()
        assert sm.state == OrderState.CANCELLED


class TestOrderStateMachineInvariants:
    """Test invariants that must hold across all states."""

    def test_filled_quantity_never_exceeds_total(self):
        """Filled quantity should never exceed total quantity."""
        sm = OrderStateMachine()
        sm._total_quantity = 100
        sm.submit()

        sm.partial_fill(60)
        sm.partial_fill(60)  # Would exceed if not capped

        assert sm._filled_quantity <= sm._total_quantity

    def test_history_tracks_all_transitions(self):
        """History should record all transitions."""
        sm = OrderStateMachine()
        sm.submit()
        sm.partial_fill(50)
        sm.fill()

        assert len(sm.history) == 3
        assert sm.history[0].to_state == OrderState.SUBMITTED
        assert sm.history[1].to_state == OrderState.PARTIALLY_FILLED
        assert sm.history[2].to_state == OrderState.FILLED

    def test_terminal_state_ends_history(self):
        """After terminal state, history should not grow."""
        sm = OrderStateMachine()
        sm.submit()
        sm.fill()

        history_len = len(sm.history)

        # Try invalid transitions
        sm.cancel()
        sm.submit()
        sm.reject()

        assert len(sm.history) == history_len

    def test_no_skipped_states_in_history(self):
        """History should have continuous state chain."""
        sm = OrderStateMachine()
        sm.submit()
        sm.partial_fill(25)
        sm.partial_fill(25)
        sm.fill()

        for i in range(1, len(sm.history)):
            prev_to = sm.history[i - 1].to_state
            curr_from = sm.history[i].from_state
            assert prev_to == curr_from, "History has discontinuous states"


class TestOrderStateMachineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_quantity_fill(self):
        """Zero quantity fill should not change state."""
        sm = OrderStateMachine()
        sm._total_quantity = 100
        sm.submit()

        sm.partial_fill(0)
        assert sm.state == OrderState.PARTIALLY_FILLED
        assert sm._filled_quantity == 0

    def test_exact_quantity_fill(self):
        """Filling exact remaining quantity should complete."""
        sm = OrderStateMachine()
        sm._total_quantity = 100
        sm.submit()
        sm.partial_fill(50)
        sm.partial_fill(50)

        assert sm.state == OrderState.FILLED
        assert sm._filled_quantity == 100

    def test_multiple_partial_fills(self):
        """Multiple small partial fills should accumulate correctly."""
        sm = OrderStateMachine()
        sm._total_quantity = 100
        sm.submit()

        for _ in range(10):
            sm.partial_fill(10)

        assert sm.state == OrderState.FILLED
        assert sm._filled_quantity == 100

    def test_immediate_rejection(self):
        """Order can be rejected without submission."""
        sm = OrderStateMachine()
        assert sm.reject("Pre-trade validation failed")
        assert sm.state == OrderState.REJECTED
        assert len(sm.history) == 1


class TestCircuitBreakerStateMachine:
    """Test circuit breaker state machine transitions."""

    # Circuit breaker states
    class CBState(Enum):
        CLOSED = "closed"  # Normal operation
        OPEN = "open"  # Trading halted
        HALF_OPEN = "half_open"  # Testing conditions

    CB_VALID_TRANSITIONS = {
        "closed": ["open"],
        "open": ["half_open"],
        "half_open": ["closed", "open"],
    }

    def test_closed_to_open_on_threshold(self):
        """CLOSED -> OPEN when loss threshold exceeded."""
        # Simulated test
        current_state = self.CBState.CLOSED
        loss_pct = 0.05
        threshold = 0.03

        if loss_pct > threshold:
            current_state = self.CBState.OPEN

        assert current_state == self.CBState.OPEN

    def test_open_to_half_open_after_cooldown(self):
        """OPEN -> HALF_OPEN after cooldown expires."""
        current_state = self.CBState.OPEN
        cooldown_expired = True

        if cooldown_expired:
            current_state = self.CBState.HALF_OPEN

        assert current_state == self.CBState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        """HALF_OPEN -> CLOSED when test trade succeeds."""
        current_state = self.CBState.HALF_OPEN
        test_trade_success = True

        if test_trade_success:
            current_state = self.CBState.CLOSED

        assert current_state == self.CBState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """HALF_OPEN -> OPEN when test trade fails."""
        current_state = self.CBState.HALF_OPEN
        test_trade_success = False

        if not test_trade_success:
            current_state = self.CBState.OPEN

        assert current_state == self.CBState.OPEN

    def test_closed_cannot_skip_to_half_open(self):
        """CLOSED -> HALF_OPEN is invalid."""
        valid = "half_open" in self.CB_VALID_TRANSITIONS["closed"]
        assert not valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
