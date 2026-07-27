"""ML-only interfaces for Partitura's Ruby-owned composition workflow."""

from generation.composition.benchmark import (
    AblationSpec,
    BenchmarkCase,
    BenchmarkCell,
    BenchmarkError,
    BenchmarkManifest,
    FrozenInput,
    StrategySpec,
)
from generation.composition.interfaces import (
    LearnedCompositionProvider,
    LearnedCritic,
    LearnedPolicy,
    LearnedProposer,
)
from generation.composition.annotation_dataset import (
    AnnotationDataset,
    AnnotationDatasetBuilder,
    AnnotationDatasetError,
    AnnotationDatasetSpec,
    AnnotationObservation,
    AnnotationTargetSpec,
    TrainingExample,
)
from generation.composition.observation_dataset import (
    ObservationDataset,
    ObservationDatasetBuilder,
    ObservationDatasetError,
    ObservationDatasetSpec,
    ScoreObservation,
)
from generation.composition.evidence import (
    CompositionDataset,
    EvidenceError,
    HumanPreferenceRecord,
    PairwiseExample,
    PairwiseReviewRecord,
    TrajectoryRecord,
)
from generation.composition.evaluation import (
    ComparisonCell,
    EvaluationLab,
    EvaluationReport,
    ScorePreferenceRecord,
    ScoreReviewRecord,
)
from generation.composition.evaluation_run import EvaluationRun, RunEffort
from generation.composition.protocol import (
    ORIGINAL_CANDIDATE_ID,
    SCHEMA_VERSION,
    CandidateProposal,
    LearnedCriticResult,
    ProposalRequest,
    ProposalResponse,
    ProtocolError,
    SelectionRequest,
    SelectionResponse,
)
from generation.composition.representation_baselines import (
    RepresentationBaselineRunner,
)

__all__ = [
    "AblationSpec",
    "AnnotationDataset",
    "AnnotationDatasetBuilder",
    "AnnotationDatasetError",
    "AnnotationDatasetSpec",
    "AnnotationObservation",
    "AnnotationTargetSpec",
    "BenchmarkCase",
    "BenchmarkCell",
    "BenchmarkError",
    "BenchmarkManifest",
    "CandidateProposal",
    "CompositionDataset",
    "EvidenceError",
    "EvaluationLab",
    "EvaluationReport",
    "EvaluationRun",
    "FrozenInput",
    "HumanPreferenceRecord",
    "LearnedCompositionProvider",
    "LearnedCritic",
    "LearnedCriticResult",
    "LearnedPolicy",
    "LearnedProposer",
    "ORIGINAL_CANDIDATE_ID",
    "ObservationDataset",
    "ObservationDatasetBuilder",
    "ObservationDatasetError",
    "ObservationDatasetSpec",
    "PairwiseExample",
    "PairwiseReviewRecord",
    "ProposalRequest",
    "ProposalResponse",
    "ProtocolError",
    "RepresentationBaselineRunner",
    "SCHEMA_VERSION",
    "RunEffort",
    "ScorePreferenceRecord",
    "ScoreObservation",
    "ScoreReviewRecord",
    "SelectionRequest",
    "SelectionResponse",
    "TrajectoryRecord",
    "TrainingExample",
    "StrategySpec",
    "ComparisonCell",
]
