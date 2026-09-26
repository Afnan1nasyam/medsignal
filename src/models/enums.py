"""Enumerations shared across MedSignal.

Values are derived directly from the extraction schemas and the knowledge-graph
schema in ``ARCHITECTURE.md``. Source-data enums use lowercase snake-case string
values (matching the raw JSON extractions); graph edge types use the uppercase
relationship names from the KG schema table.
"""

from enum import Enum


class DataSource(str, Enum):
    """The four public data sources ingested by MedSignal."""

    FAERS = "faers"
    PUBMED = "pubmed"
    CLINICAL_TRIALS = "clinical_trials"
    DRUG_LABELS = "drug_labels"


class EvidenceGrade(str, Enum):
    """Overall strength of evidence supporting an answer.

    Grades (see ``ARCHITECTURE.md`` — Evidence Grading System):

    - ``A`` — **Strong**: confirmed in drug label + supported by clinical trial data.
    - ``B`` — **Moderate**: multiple FAERS reports (>50) + supporting literature.
    - ``C`` — **Suggestive**: literature case reports + some FAERS signal.
    - ``D`` — **Weak**: isolated FAERS reports only, or a single case report.
    - ``E`` — **Insufficient**: no evidence found in any source.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

    @property
    def label(self) -> str:
        """Human-readable label for the grade."""
        return {
            "A": "Strong",
            "B": "Moderate",
            "C": "Suggestive",
            "D": "Weak",
            "E": "Insufficient",
        }[self.value]


class DrugRole(str, Enum):
    """Role a drug plays in a FAERS adverse-event report."""

    PRIMARY_SUSPECT = "primary_suspect"
    SECONDARY_SUSPECT = "secondary_suspect"
    CONCOMITANT = "concomitant"
    INTERACTING = "interacting"


class ReactionOutcome(str, Enum):
    """Outcome of an adverse reaction in a FAERS report."""

    HOSPITALIZATION = "hospitalization"
    LIFE_THREATENING = "life_threatening"
    DEATH = "death"
    DISABILITY = "disability"
    CONGENITAL_ANOMALY = "congenital_anomaly"
    OTHER = "other"


class ReporterType(str, Enum):
    """Type of person who filed a FAERS report."""

    PHYSICIAN = "physician"
    PHARMACIST = "pharmacist"
    CONSUMER = "consumer"
    OTHER = "other"


class TrialPhase(str, Enum):
    """Clinical trial phase."""

    PHASE_1 = "1"
    PHASE_2 = "2"
    PHASE_3 = "3"
    PHASE_4 = "4"
    NA = "NA"


class TrialStatus(str, Enum):
    """Recruitment / completion status of a clinical trial."""

    COMPLETED = "completed"
    RECRUITING = "recruiting"
    TERMINATED = "terminated"
    WITHDRAWN = "withdrawn"
    OTHER = "other"


class StudyType(str, Enum):
    """Type of study a PubMed article describes."""

    CASE_REPORT = "case_report"
    CLINICAL_TRIAL = "clinical_trial"
    META_ANALYSIS = "meta_analysis"
    REVIEW = "review"
    OTHER = "other"


class AEFrequency(str, Enum):
    """Frequency of an adverse reaction as stated on a drug label."""

    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    VERY_RARE = "very_rare"
    UNKNOWN = "unknown"


class InteractionSeverity(str, Enum):
    """Severity of a drug-drug interaction."""

    MAJOR = "major"
    MODERATE = "moderate"
    MINOR = "minor"


class QueryIntent(str, Enum):
    """Classified intent of a user query, used by the agent's query planner."""

    SAFETY_PROFILE = "safety_profile"
    DRUG_INTERACTION = "drug_interaction"
    ADVERSE_EVENT_LOOKUP = "adverse_event_lookup"
    SIGNAL_DETECTION = "signal_detection"
    DRUG_COMPARISON = "drug_comparison"
    GENERAL = "general"


class NodeType(str, Enum):
    """Node types in the biomedical knowledge graph."""

    DRUG = "drug"
    ADVERSE_EVENT = "adverse_event"
    CONDITION = "condition"
    CLINICAL_TRIAL = "clinical_trial"
    PUBLICATION = "publication"
    DRUG_CLASS = "drug_class"


class EdgeType(str, Enum):
    """Edge (relationship) types in the biomedical knowledge graph."""

    REPORTED_WITH = "REPORTED_WITH"
    TREATS = "TREATS"
    CONTRAINDICATED_FOR = "CONTRAINDICATED_FOR"
    INTERACTS_WITH = "INTERACTS_WITH"
    SAME_CLASS_AS = "SAME_CLASS_AS"
    BELONGS_TO = "BELONGS_TO"
    STUDIED_IN = "STUDIED_IN"
    TRIAL_REPORTED = "TRIAL_REPORTED"
    DESCRIBES = "DESCRIBES"
    STUDIES_DRUG = "STUDIES_DRUG"
    AE_BELONGS_TO = "AE_BELONGS_TO"
