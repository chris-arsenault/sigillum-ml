"""Interfaces implemented by learned whole-score composition components."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from generation.composition.protocol import (
    LearnedCriticResult,
    ProposalRequest,
    ProposalResponse,
    SelectionRequest,
    SelectionResponse,
)


@runtime_checkable
class LearnedProposer(Protocol):
    """Generate explicit source-patch candidates for a Ruby-scheduled action."""

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        """Return zero or more candidates bound to ``request``."""


@runtime_checkable
class LearnedCritic(Protocol):
    """Attach learned features or judgments to Ruby-validated candidates."""

    def evaluate(self, request: SelectionRequest) -> Sequence[LearnedCriticResult]:
        """Evaluate candidates without claiming mechanical validity."""


@runtime_checkable
class LearnedPolicy(Protocol):
    """Select a candidate or the explicit unchanged original."""

    def select(
        self,
        request: SelectionRequest,
        critic_results: Sequence[LearnedCriticResult],
    ) -> SelectionResponse:
        """Return a request-bound decision for Ruby to validate and commit."""


@runtime_checkable
class LearnedCompositionProvider(LearnedProposer, Protocol):
    """Convenience interface for a service providing proposal and selection."""

    def select(self, request: SelectionRequest) -> SelectionResponse:
        """Return a request-bound decision, optionally with learned critics."""
