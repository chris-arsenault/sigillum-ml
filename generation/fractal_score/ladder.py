"""General multi-resolution refinement ladder for fractalized score generation.

This module encodes the *shape* of the fractalization hypothesis without any
music-specific or deterministic composition order. A score attribute (here,
harmony; later, form, orchestration, surface detail) is laid out on a grid of
slots at the finest resolution. A coarse representation reveals only a strided
subset of those slots; a finer representation reveals a denser subset. One
shared refinement operator learns to fill the newly revealed slots at each step
conditioned on the coarser context it already knows.

Nothing in this module chooses *what* to refine next; the schedule only says
"make the grid twice as dense." Which slots carry which content, and whether a
refinement is good, is learned. This keeps the process a learnable refinement
ladder rather than an authored ``form -> harmony -> melody`` pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass


class LadderError(ValueError):
    """Raised when a resolution ladder or schedule is malformed."""


@dataclass(frozen=True)
class RefinementSchedule:
    """A coarse-to-fine sequence of grid strides.

    ``strides`` is strictly decreasing and ends at ``1`` (every slot present at
    the finest resolution). Consecutive pairs ``(parent, child)`` describe one
    refinement step: the parent grid keeps slots at multiples of ``parent`` and
    the child grid additionally reveals slots at multiples of ``child``.
    """

    strides: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.strides) < 2:
            raise LadderError("a refinement schedule needs at least two strides")
        if self.strides[-1] != 1:
            raise LadderError("the finest stride must be 1")
        for coarse, fine in zip(self.strides, self.strides[1:]):
            if fine >= coarse:
                raise LadderError("strides must strictly decrease")
            if coarse % fine != 0:
                raise LadderError("each stride must divide its coarser neighbour")

    @classmethod
    def geometric(cls, coarsest: int) -> "RefinementSchedule":
        """Build ``(coarsest, coarsest/2, ..., 2, 1)`` for a power-of-two stride."""

        if coarsest < 2 or (coarsest & (coarsest - 1)) != 0:
            raise LadderError("coarsest stride must be a power of two >= 2")
        strides: list[int] = []
        value = coarsest
        while value >= 1:
            strides.append(value)
            value //= 2
        return cls(tuple(strides))

    @property
    def coarsest(self) -> int:
        return self.strides[0]

    def steps(self) -> tuple[tuple[int, int], ...]:
        """Return the ``(parent_stride, child_stride)`` refinement steps."""

        return tuple(zip(self.strides, self.strides[1:]))

    def level_index(self, child_stride: int) -> int:
        """Return a stable embedding index for the level that produces ``child_stride``."""

        try:
            return self.strides.index(child_stride)
        except ValueError as error:
            raise LadderError(f"stride {child_stride} is not in the schedule") from error

    @property
    def level_count(self) -> int:
        return len(self.strides)


def revealed_positions(length: int, stride: int) -> list[int]:
    """Grid positions present at ``stride`` for a sequence of ``length`` slots."""

    if length < 0:
        raise LadderError("length must be non-negative")
    if stride < 1:
        raise LadderError("stride must be positive")
    return list(range(0, length, stride))


def refinement_positions(length: int, parent_stride: int, child_stride: int) -> list[int]:
    """Positions newly revealed going from ``parent_stride`` to ``child_stride``.

    These are the slots the refinement operator must fill at this step: present
    at the child grid but absent from the parent grid.
    """

    if parent_stride % child_stride != 0:
        raise LadderError("child stride must divide parent stride")
    child = set(revealed_positions(length, child_stride))
    parent = set(revealed_positions(length, parent_stride))
    return sorted(child - parent)
