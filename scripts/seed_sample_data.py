"""Generate realistic MedSignal sample data — fully offline, no API calls.

Writes hardcoded (schema-validated) sample records for all four data sources
into ``data/sample/`` and mirrors them into ``data/raw/`` so the ingestion
pipeline has something to chew on without any network access:

* ``data/sample/faers/``            — 50 FAERS reports across 5 batch files
* ``data/sample/pubmed/``           — 30 PubMed articles (one file)
* ``data/sample/clinical_trials/``  — 20 trials (one file)
* ``data/sample/drug_labels/``      — 15 drug labels (one file)

Every record is constructed through the Pydantic models in ``src.models`` so
the output is guaranteed to round-trip back through them during ingestion. A
fixed RNG seed keeps demographic variety reproducible run-to-run. Deliberately
seeded safety patterns (e.g. metformin -> lactic acidosis) give the knowledge
graph meaningful, queryable structure.

Run:
    python scripts/seed_sample_data.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Make ``src`` importable when run as ``python scripts/seed_sample_data.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from src.config import settings  # noqa: E402
from src.models.enums import (  # noqa: E402
    AEFrequency,
    DrugRole,
    InteractionSeverity,
    ReactionOutcome,
    ReporterType,
    StudyType,
    TrialPhase,
    TrialStatus,
)
from src.models.schemas import (  # noqa: E402
    ClinicalTrialRecord,
    DrugLabel,
    FaersDrug,
    FaersPatient,
    FaersReaction,
    FaersReport,
    LabelAdverseReaction,
    LabelDrugInteraction,
    PubMedArticle,
    TrialAdverseEvent,
    TrialIntervention,
)

# Reproducible demographic variety (does not affect the seeded safety patterns).
RNG = random.Random(20240921)

# Core drug set shared across every source so graph edges converge.
CORE_DRUGS = [
    "metformin",
    "lisinopril",
    "atorvastatin",
    "warfarin",
    "amiodarone",
    "omeprazole",
    "metoprolol",
    "simvastatin",
    "amlodipine",
    "ibuprofen",
    "aspirin",
    "clopidogrel",
    "rosuvastatin",
    "sertraline",
    "gabapentin",
]

# Brand display names used for the FAERS ``name`` field (generic_name = key).
BRAND = {
    "metformin": "Glucophage",
    "lisinopril": "Prinivil",
    "atorvastatin": "Lipitor",
    "warfarin": "Coumadin",
    "amiodarone": "Cordarone",
    "omeprazole": "Prilosec",
    "metoprolol": "Lopressor",
    "simvastatin": "Zocor",
    "amlodipine": "Norvasc",
    "ibuprofen": "Advil",
    "aspirin": "Bayer Aspirin",
    "clopidogrel": "Plavix",
    "rosuvastatin": "Crestor",
    "sertraline": "Zoloft",
    "gabapentin": "Neurontin",
}

_SERIOUS_OUTCOMES = {
    ReactionOutcome.HOSPITALIZATION,
    ReactionOutcome.LIFE_THREATENING,
    ReactionOutcome.DEATH,
    ReactionOutcome.DISABILITY,
}


# --------------------------------------------------------------------------- #
# FAERS — 50 individual case safety reports
# --------------------------------------------------------------------------- #
def _drug(name_key: str, role: DrugRole, indication: str | None = None) -> FaersDrug:
    """Build a :class:`FaersDrug` from a core-drug key."""
    return FaersDrug(
        name=BRAND.get(name_key, name_key.title()),
        generic_name=name_key,
        role=role,
        indication=indication,
    )


def _pick_outcome(reaction: str) -> ReactionOutcome:
    """Pick a plausible outcome for a reaction (serious terms skew serious)."""
    severe_terms = {
        "Lactic acidosis",
        "Rhabdomyolysis",
        "Gastrointestinal haemorrhage",
        "Cerebral haemorrhage",
        "Serotonin syndrome",
        "Pulmonary toxicity",
        "Angioedema",
        "Clostridium difficile colitis",
    }
    if reaction in severe_terms:
        return RNG.choice(
            [
                ReactionOutcome.HOSPITALIZATION,
                ReactionOutcome.LIFE_THREATENING,
                ReactionOutcome.DEATH,
            ]
        )
    return RNG.choice(
        [ReactionOutcome.OTHER, ReactionOutcome.HOSPITALIZATION, ReactionOutcome.OTHER]
    )


# (primary_drug, primary_reaction, [co_drug keys with INTERACTING role])
_FAERS_SPECS: list[tuple[str, str, list[str]]] = []

# metformin + Lactic acidosis (x9)
for i in range(9):
    co = ["ibuprofen"] if i % 3 == 0 else []
    _FAERS_SPECS.append(("metformin", "Lactic acidosis", co))

# warfarin + bleeding (x7), rotating MedDRA terms and interacting drugs
_BLEED_TERMS = [
    "Gastrointestinal haemorrhage",
    "Haemorrhage",
    "Epistaxis",
    "Haematuria",
    "Cerebral haemorrhage",
]
for i in range(7):
    co = ["aspirin"] if i % 2 == 0 else ["ibuprofen"]
    _FAERS_SPECS.append(("warfarin", _BLEED_TERMS[i % len(_BLEED_TERMS)], co))

# statins + Rhabdomyolysis (x6), rotating statin; some with amiodarone interaction
_STATINS = ["atorvastatin", "simvastatin", "rosuvastatin"]
for i in range(6):
    co = ["amiodarone"] if i % 2 == 0 else []
    _FAERS_SPECS.append((_STATINS[i % len(_STATINS)], "Rhabdomyolysis", co))

# amiodarone + thyroid disorder (x5)
_THYROID = ["Hypothyroidism", "Hyperthyroidism", "Thyroid disorder"]
for i in range(5):
    _FAERS_SPECS.append(("amiodarone", _THYROID[i % len(_THYROID)], []))

# ibuprofen + GI bleeding (x6)
for i in range(6):
    co = ["aspirin"] if i % 3 == 0 else []
    _FAERS_SPECS.append(("ibuprofen", "Gastrointestinal haemorrhage", co))

# Remaining varied reports (x17) to round out to 50
_FAERS_FILLER: list[tuple[str, str, list[str]]] = [
    ("lisinopril", "Angioedema", []),
    ("lisinopril", "Cough", []),
    ("metoprolol", "Bradycardia", []),
    ("amlodipine", "Peripheral oedema", []),
    ("omeprazole", "Hypomagnesaemia", []),
    ("omeprazole", "Clostridium difficile colitis", []),
    ("aspirin", "Gastrointestinal haemorrhage", []),
    ("clopidogrel", "Haemorrhage", ["aspirin"]),
    ("sertraline", "Hyponatraemia", []),
    ("sertraline", "Serotonin syndrome", []),
    ("gabapentin", "Somnolence", []),
    ("gabapentin", "Dizziness", []),
    ("metformin", "Diarrhoea", []),
    ("atorvastatin", "Myalgia", []),
    ("simvastatin", "Hepatic function abnormal", ["amiodarone"]),
    ("warfarin", "International normalised ratio increased", ["amiodarone"]),
    ("amiodarone", "Pulmonary toxicity", []),
]
_FAERS_SPECS.extend(_FAERS_FILLER)

_SECONDARY_REACTIONS = ["Nausea", "Headache", "Dizziness", "Fatigue", "Vomiting"]
_INDICATIONS = {
    "metformin": "Type 2 diabetes mellitus",
    "lisinopril": "Hypertension",
    "atorvastatin": "Hypercholesterolaemia",
    "simvastatin": "Hypercholesterolaemia",
    "rosuvastatin": "Hypercholesterolaemia",
    "warfarin": "Atrial fibrillation",
    "amiodarone": "Cardiac arrhythmia",
    "omeprazole": "Gastro-oesophageal reflux disease",
    "metoprolol": "Hypertension",
    "amlodipine": "Hypertension",
    "ibuprofen": "Pain",
    "aspirin": "Secondary prevention",
    "clopidogrel": "Acute coronary syndrome",
    "sertraline": "Major depressive disorder",
    "gabapentin": "Neuropathic pain",
}


def build_faers_reports() -> list[FaersReport]:
    """Construct the 50 FAERS reports from the seeded spec list."""
    reports: list[FaersReport] = []
    for idx, (primary, reaction, co_drugs) in enumerate(_FAERS_SPECS, start=1):
        report_id = f"US-MEDSIGNAL-2024-{idx:05d}"

        drugs = [
            _drug(primary, DrugRole.PRIMARY_SUSPECT, _INDICATIONS.get(primary))
        ]
        for co in co_drugs:
            drugs.append(_drug(co, DrugRole.INTERACTING, _INDICATIONS.get(co)))

        reactions = [FaersReaction(preferred_term=reaction, outcome=_pick_outcome(reaction))]
        if idx % 3 == 0:  # occasionally add a secondary, milder reaction
            reactions.append(
                FaersReaction(
                    preferred_term=RNG.choice(_SECONDARY_REACTIONS),
                    outcome=ReactionOutcome.OTHER,
                )
            )

        serious = any(r.outcome in _SERIOUS_OUTCOMES for r in reactions)
        month = (idx % 12) + 1
        day = (idx % 27) + 1
        report = FaersReport(
            report_id=report_id,
            patient=FaersPatient(
                age=float(RNG.randint(25, 85)),
                age_unit="year",
                sex=RNG.choice(["male", "female"]),
                weight=round(RNG.uniform(52.0, 104.0), 1),
            ),
            drugs=drugs,
            reactions=reactions,
            report_date=f"2024-{month:02d}-{day:02d}",
            reporter_type=RNG.choice(list(ReporterType)),
            serious=serious,
        )
        reports.append(report)
    return reports


# --------------------------------------------------------------------------- #
# PubMed — 30 abstracts
# --------------------------------------------------------------------------- #
def _abstract(bg: str, methods: str, results: str, conclusion: str, impl: str) -> str:
    """Join structured abstract parts into a single paragraph."""
    return " ".join([bg, methods, results, conclusion, impl])


# Study-type-specific methodological paragraphs appended to each abstract so
# that every record reaches the realistic 150-300 word length of a structured
# abstract. Placeholders are filled with the article's own drugs / events, so
# the added prose stays article-specific rather than boilerplate.
_ABSTRACT_TAIL: dict[StudyType, str] = {
    StudyType.CASE_REPORT: (
        "As a single-patient observation, this report cannot establish causality or "
        "incidence and reflects one clinical trajectory rather than a population-level "
        "estimate. Publication bias favours striking presentations, so uncomplicated "
        "courses of {drugs} exposure are likely under-represented in the literature. "
        "The temporal relationship, dechallenge, and absence of a more plausible "
        "alternative explanation nonetheless support a probable drug-related mechanism "
        "for {aes}. Objective investigations and the documented clinical course were "
        "used to exclude common confounders and competing diagnoses. Clinicians should "
        "remain alert to {aes} when prescribing {drugs}, particularly in older adults, "
        "in renal or hepatic impairment, and in the setting of polypharmacy where "
        "interacting agents may amplify risk. Suspected reactions should be submitted "
        "to spontaneous reporting systems so that aggregate signals can be evaluated."
    ),
    StudyType.CLINICAL_TRIAL: (
        "Randomisation and blinding reduced confounding, but the study was powered for "
        "its primary efficacy endpoint rather than for rare safety outcomes, so "
        "uncommon events involving {aes} may be underestimated. Enrolment criteria "
        "excluded some frail, older, and multimorbid patients, which limits "
        "generalisability to the broader population encountered in routine practice. "
        "Adverse events were ascertained through protocol-defined, actively solicited "
        "reporting and adjudicated against predefined definitions, and analyses "
        "followed the intention-to-treat principle. Loss to follow-up and the finite "
        "trial duration constrain inference about long-latency harms, and continued "
        "post-marketing surveillance therefore remains necessary. On balance the "
        "observed safety profile of {drugs} was consistent with prior evidence and "
        "supports continued, structured monitoring for {aes} in clinical use, with "
        "individualised risk assessment for higher-risk subgroups."
    ),
    StudyType.REVIEW: (
        "This narrative synthesis drew on heterogeneous sources and is constrained by "
        "the quality of the underlying reports and by the under-reporting and "
        "notoriety effects inherent to spontaneous surveillance data. Confounding by "
        "indication, channelling, and incomplete denominator information cannot be "
        "excluded, so absolute risks for {aes} could not be estimated with precision. "
        "Included studies varied in design, exposure definition, and follow-up, "
        "limiting the strength of any pooled interpretation. The findings are best "
        "regarded as hypothesis-generating and as a prompt for structured, "
        "prospectively designed surveillance rather than as confirmatory evidence. "
        "Clinicians should weigh the signal linking {drugs} to {aes} against the "
        "established therapeutic benefit, individualise monitoring, and counsel "
        "patients on early warning symptoms. Consistent, transparent evidence grading "
        "would further improve how such safety signals are communicated and acted upon."
    ),
    StudyType.META_ANALYSIS: (
        "Pooling improved statistical precision but introduced between-study "
        "heterogeneity, and the summary estimate should be read in light of differing "
        "populations, exposure definitions, and follow-up durations. Potential "
        "publication bias was assessed and residual confounding cannot be excluded "
        "where observational cohorts contributed to the pooled effect. Absolute risk "
        "for {aes} depends heavily on baseline population characteristics, so relative "
        "measures may overstate clinical impact in low-risk groups. Prespecified "
        "sensitivity and subgroup analyses were broadly consistent with the primary "
        "result, strengthening confidence in the direction of association for {drugs}. "
        "Overall the aggregated data reinforce the relationship between {drugs} and "
        "{aes} while underscoring the need for individualised risk assessment, shared "
        "decision-making, and ongoing pharmacovigilance as prescribing patterns evolve."
    ),
}


def _extend_abstract(abstract: str, study_type: StudyType, drugs: list[str], aes: list[str]) -> str:
    """Append a study-type methodological paragraph, filled with drugs/events."""
    drugs_str = ", ".join(drugs) if drugs else "the agents studied"
    aes_str = ", ".join(aes) if aes else "the adverse events of interest"
    tail = _ABSTRACT_TAIL[study_type].format(drugs=drugs_str, aes=aes_str)
    return f"{abstract} {tail}"


# Each entry: pmid, title, journal, pub_date, study_type, drugs, aes, findings, parts
_PUBMED: list[dict] = [
    # --- metformin safety (5) ---
    {
        "pmid": "34011201",
        "title": "Metformin-Associated Lactic Acidosis: A 10-Year Pharmacovigilance Review",
        "journal": "Drug Safety",
        "pub_date": "2021-04-15",
        "study_type": StudyType.REVIEW,
        "drugs": ["metformin"],
        "aes": ["lactic acidosis", "renal impairment"],
        "findings": "MALA is rare but carries high mortality when renal function is not monitored.",
        "parts": (
            "Metformin remains first-line therapy for type 2 diabetes, yet metformin-associated lactic acidosis (MALA) continues to generate spontaneous adverse-event reports worldwide.",
            "We reviewed a decade of pharmacovigilance submissions and published case series to characterise reporting patterns, patient risk factors, and outcomes.",
            "Across the reviewed reports, MALA clustered strongly in patients with acute kidney injury, dehydration, or concomitant nephrotoxic drugs, and case-fatality estimates ranged from 25 to 45 percent.",
            "The signal supports existing guidance to withhold metformin during acute illness and to respect renal contraindications.",
            "Clinicians should reinforce sick-day rules and periodic renal monitoring to mitigate this uncommon but serious risk.",
        ),
    },
    {
        "pmid": "34011202",
        "title": "Gastrointestinal Tolerability of Extended-Release Metformin: A Randomized Trial",
        "journal": "Diabetes Care",
        "pub_date": "2020-09-01",
        "study_type": StudyType.CLINICAL_TRIAL,
        "drugs": ["metformin"],
        "aes": ["diarrhoea", "nausea"],
        "findings": "Extended-release metformin roughly halved gastrointestinal discontinuations.",
        "parts": (
            "Gastrointestinal intolerance is the leading cause of early metformin discontinuation in type 2 diabetes.",
            "In a randomized, double-blind trial, 480 metformin-naive adults received immediate-release or extended-release metformin titrated over eight weeks.",
            "Diarrhoea and nausea were reported roughly half as often with the extended-release formulation, and treatment discontinuation fell from 14 to 7 percent, with comparable glycaemic control.",
            "Extended-release metformin improved tolerability without sacrificing efficacy.",
            "Formulation choice is a practical lever for improving adherence in newly treated patients.",
        ),
    },
    {
        "pmid": "34011203",
        "title": "Vitamin B12 Deficiency During Long-Term Metformin Therapy",
        "journal": "Journal of Clinical Endocrinology & Metabolism",
        "pub_date": "2022-02-10",
        "study_type": StudyType.META_ANALYSIS,
        "drugs": ["metformin"],
        "aes": ["vitamin B12 deficiency", "peripheral neuropathy"],
        "findings": "Long-term metformin use is associated with a dose-dependent fall in serum B12.",
        "parts": (
            "Chronic metformin exposure has been linked to malabsorption of vitamin B12.",
            "This meta-analysis pooled 21 observational studies and randomized trials comprising more than 15,000 patients.",
            "Metformin use was associated with a dose- and duration-dependent reduction in serum vitamin B12 and a modest increase in reported peripheral neuropathy.",
            "Periodic B12 screening is warranted for patients on prolonged, higher-dose metformin therapy.",
            "Supplementation is inexpensive and may prevent under-recognised neuropathic complications.",
        ),
    },
    {
        "pmid": "34011204",
        "title": "Metformin Use in Patients With Chronic Kidney Disease: A Cohort Study",
        "journal": "Kidney International",
        "pub_date": "2023-06-20",
        "study_type": StudyType.CLINICAL_TRIAL,
        "drugs": ["metformin"],
        "aes": ["lactic acidosis", "renal impairment"],
        "findings": "Metformin appears tolerable down to eGFR 30 with appropriate dose reduction.",
        "parts": (
            "Historical contraindications restricted metformin in chronic kidney disease despite limited outcome data.",
            "We followed a cohort of 6,200 patients with eGFR between 30 and 60 who continued dose-adjusted metformin.",
            "Rates of lactic acidosis remained low and were not significantly elevated compared with matched controls, provided doses were reduced below an eGFR of 45.",
            "Dose-adjusted metformin was reasonably safe in moderate chronic kidney disease.",
            "These findings support contemporary eGFR-based dosing rather than blanket discontinuation.",
        ),
    },
    {
        "pmid": "34011205",
        "title": "A Case Report of Severe Lactic Acidosis Following Metformin and Ibuprofen Co-Use",
        "journal": "BMJ Case Reports",
        "pub_date": "2024-01-30",
        "study_type": StudyType.CASE_REPORT,
        "drugs": ["metformin", "ibuprofen"],
        "aes": ["lactic acidosis", "acute kidney injury"],
        "findings": "NSAID-induced renal hypoperfusion may precipitate metformin lactic acidosis.",
        "parts": (
            "The combination of metformin with nephrotoxic agents can precipitate acute decompensation.",
            "We describe a 74-year-old woman who developed profound lactic acidosis after several days of high-dose ibuprofen for musculoskeletal pain.",
            "She presented with acute kidney injury, a serum lactate of 11 mmol/L, and an arterial pH of 7.10, requiring haemodialysis and full recovery.",
            "NSAID-mediated renal hypoperfusion likely unmasked metformin accumulation.",
            "This case underscores caution when NSAIDs are added to metformin in older adults.",
        ),
    },
    # --- warfarin interactions (4) ---
    {
        "pmid": "34022301",
        "title": "Amiodarone Potentiation of Warfarin: Magnitude and Time Course",
        "journal": "Journal of Thrombosis and Haemostasis",
        "pub_date": "2020-11-05",
        "study_type": StudyType.CLINICAL_TRIAL,
        "drugs": ["warfarin", "amiodarone"],
        "aes": ["haemorrhage", "international normalised ratio increased"],
        "findings": "Amiodarone raised INR for weeks; empiric warfarin dose reduction is advised.",
        "parts": (
            "Amiodarone inhibits CYP2C9 and can markedly enhance warfarin's anticoagulant effect.",
            "We prospectively monitored INR in 140 patients started on amiodarone while stable on warfarin.",
            "Mean INR rose by 40 percent over three weeks and the effect persisted for up to twelve weeks, with several clinically significant bleeding events.",
            "An empiric 30 to 50 percent warfarin dose reduction with intensified INR monitoring is recommended when amiodarone is initiated.",
            "The prolonged half-life of amiodarone demands vigilance long after co-prescription begins.",
        ),
    },
    {
        "pmid": "34022302",
        "title": "Concomitant NSAIDs and Warfarin: Bleeding Risk in a National Cohort",
        "journal": "JAMA Internal Medicine",
        "pub_date": "2021-08-12",
        "study_type": StudyType.META_ANALYSIS,
        "drugs": ["warfarin", "ibuprofen", "aspirin"],
        "aes": ["gastrointestinal haemorrhage", "haemorrhage"],
        "findings": "Adding an NSAID to warfarin roughly doubled major GI bleeding.",
        "parts": (
            "NSAIDs add antiplatelet and mucosal-injury effects to warfarin anticoagulation.",
            "We pooled national cohort data covering more than 90,000 warfarin users with and without NSAID co-exposure.",
            "Concomitant NSAID use approximately doubled the rate of major gastrointestinal haemorrhage, with the highest risk among older adults and aspirin co-users.",
            "Routine NSAID co-prescription with warfarin should be avoided where possible.",
            "Gastroprotection and alternative analgesics should be considered when NSAIDs are unavoidable.",
        ),
    },
    {
        "pmid": "34022303",
        "title": "Warfarin and Aspirin Combination Therapy After Coronary Stenting",
        "journal": "Circulation",
        "pub_date": "2022-05-18",
        "study_type": StudyType.CLINICAL_TRIAL,
        "drugs": ["warfarin", "aspirin", "clopidogrel"],
        "aes": ["haemorrhage", "gastrointestinal haemorrhage"],
        "findings": "Triple therapy increased bleeding without clear ischaemic benefit.",
        "parts": (
            "Combining oral anticoagulation with dual antiplatelet therapy raises bleeding concerns after stenting.",
            "This randomized trial compared warfarin plus aspirin plus clopidogrel against warfarin plus clopidogrel alone in 720 patients.",
            "Triple therapy significantly increased major and minor bleeding without a statistically significant reduction in ischaemic events.",
            "Dropping aspirin from the regimen reduced haemorrhage while preserving efficacy.",
            "Shorter triple-therapy windows are favoured in contemporary practice.",
        ),
    },
    {
        "pmid": "34022304",
        "title": "Drug-Drug Interaction Alerts for Warfarin: Clinical Utility Assessment",
        "journal": "Pharmacoepidemiology and Drug Safety",
        "pub_date": "2023-03-09",
        "study_type": StudyType.REVIEW,
        "drugs": ["warfarin", "amiodarone", "ibuprofen"],
        "aes": ["haemorrhage"],
        "findings": "High alert override rates dilute the value of interaction warnings.",
        "parts": (
            "Electronic warfarin interaction alerts are ubiquitous but frequently overridden.",
            "We reviewed alert-response logs and the supporting evidence base for the most common warfarin interaction pairs.",
            "Clinically important pairs such as warfarin-amiodarone were often buried among low-value alerts, contributing to override rates above 85 percent.",
            "Tiered, evidence-weighted alerting could restore clinician trust and catch high-risk combinations.",
            "Signal-to-noise reduction is central to safe anticoagulation informatics.",
        ),
    },
    # --- statin myopathy (3) ---
    {
        "pmid": "34033401",
        "title": "Statin-Associated Rhabdomyolysis: Risk Factors and Interacting Drugs",
        "journal": "Muscle & Nerve",
        "pub_date": "2020-07-22",
        "study_type": StudyType.REVIEW,
        "drugs": ["simvastatin", "atorvastatin", "amiodarone"],
        "aes": ["rhabdomyolysis", "blood creatine phosphokinase increased"],
        "findings": "Rhabdomyolysis risk rises sharply with CYP3A4-inhibiting co-medications.",
        "parts": (
            "Statin-associated rhabdomyolysis is rare but potentially fatal.",
            "We synthesised registry and case-series data to identify dominant risk factors.",
            "High-dose simvastatin combined with CYP3A4 inhibitors such as amiodarone produced the greatest risk, particularly in older women with renal impairment.",
            "Dose caps and awareness of interacting drugs remain the primary preventive strategy.",
            "Switching to less interaction-prone statins is advisable in high-risk patients.",
        ),
    },
    {
        "pmid": "34033402",
        "title": "Rosuvastatin Versus Atorvastatin: Comparative Myopathy Reporting",
        "journal": "Atherosclerosis",
        "pub_date": "2021-12-01",
        "study_type": StudyType.META_ANALYSIS,
        "drugs": ["rosuvastatin", "atorvastatin"],
        "aes": ["myalgia", "rhabdomyolysis"],
        "findings": "Muscle-symptom reporting was broadly similar between the two statins.",
        "parts": (
            "Perceived differences in muscle safety influence statin selection.",
            "This meta-analysis compared myopathy and myalgia reporting across 34 randomized trials of rosuvastatin and atorvastatin.",
            "Rates of myalgia and serious rhabdomyolysis were low and did not differ significantly between agents after adjusting for dose intensity.",
            "Both statins have comparable, favourable muscle-safety profiles at standard doses.",
            "Interaction burden, not intrinsic statin choice, drives most serious events.",
        ),
    },
    {
        "pmid": "34033403",
        "title": "A Fatal Case of Rhabdomyolysis With Simvastatin and Amiodarone",
        "journal": "BMJ Case Reports",
        "pub_date": "2022-10-14",
        "study_type": StudyType.CASE_REPORT,
        "drugs": ["simvastatin", "amiodarone"],
        "aes": ["rhabdomyolysis", "acute kidney injury"],
        "findings": "High-dose simvastatin with amiodarone precipitated fatal rhabdomyolysis.",
        "parts": (
            "Co-prescription of high-dose simvastatin with amiodarone is discouraged by regulators.",
            "We report an 81-year-old man on simvastatin 80 mg who developed rhabdomyolysis two weeks after amiodarone initiation.",
            "Creatine kinase exceeded 40,000 U/L with anuric acute kidney injury, and the patient died despite renal replacement therapy.",
            "The case illustrates the lethal potential of this well-documented interaction.",
            "Simvastatin doses above 20 mg should be avoided with amiodarone.",
        ),
    },
    # --- amiodarone toxicity (3) ---
    {
        "pmid": "34044501",
        "title": "Amiodarone-Induced Thyroid Dysfunction: Screening and Management",
        "journal": "Thyroid",
        "pub_date": "2020-03-11",
        "study_type": StudyType.REVIEW,
        "drugs": ["amiodarone"],
        "aes": ["hypothyroidism", "hyperthyroidism", "thyroid disorder"],
        "findings": "Up to one in five patients develop thyroid dysfunction on amiodarone.",
        "parts": (
            "Amiodarone's high iodine content disrupts thyroid homeostasis.",
            "We reviewed screening strategies and management of amiodarone-induced thyroid dysfunction.",
            "Between 15 and 20 percent of patients developed hypothyroidism or hyperthyroidism, often months into therapy and sometimes after discontinuation.",
            "Baseline and periodic thyroid function testing is essential throughout treatment.",
            "Type-specific management differs and warrants endocrinology input for thyrotoxic cases.",
        ),
    },
    {
        "pmid": "34044502",
        "title": "Amiodarone Pulmonary Toxicity: Incidence in a Contemporary Cohort",
        "journal": "Chest",
        "pub_date": "2021-09-27",
        "study_type": StudyType.CLINICAL_TRIAL,
        "drugs": ["amiodarone"],
        "aes": ["pulmonary toxicity", "interstitial lung disease"],
        "findings": "Pulmonary toxicity remains a serious, dose-related amiodarone risk.",
        "parts": (
            "Pulmonary toxicity is among the most feared complications of amiodarone.",
            "We prospectively followed 900 patients with serial imaging and pulmonary function testing.",
            "Clinically significant pulmonary toxicity occurred in roughly 5 percent, correlating with cumulative dose and pre-existing lung disease.",
            "Baseline chest imaging and symptom-triggered monitoring support early detection.",
            "Prompt discontinuation and corticosteroids improve outcomes in confirmed cases.",
        ),
    },
    {
        "pmid": "34044503",
        "title": "Hepatotoxicity and Bradycardia With Amiodarone: A Safety Signal Analysis",
        "journal": "Drug Safety",
        "pub_date": "2023-01-19",
        "study_type": StudyType.META_ANALYSIS,
        "drugs": ["amiodarone", "metoprolol"],
        "aes": ["hepatic function abnormal", "bradycardia"],
        "findings": "Combined amiodarone and beta-blockade heightens bradycardia risk.",
        "parts": (
            "Amiodarone affects multiple organ systems beyond the heart.",
            "We performed a disproportionality analysis of spontaneous reports for hepatic and conduction adverse events.",
            "Elevated liver enzymes and symptomatic bradycardia were disproportionately reported, the latter amplified by concomitant beta-blockers such as metoprolol.",
            "Liver function and heart rate should be monitored, especially with rate-limiting co-therapy.",
            "Dose review is prudent when bradycardia or transaminitis emerges.",
        ),
    },
    # --- NSAID GI risk (3) ---
    {
        "pmid": "34055601",
        "title": "NSAID-Associated Upper Gastrointestinal Bleeding: An Updated Meta-Analysis",
        "journal": "Gut",
        "pub_date": "2020-06-08",
        "study_type": StudyType.META_ANALYSIS,
        "drugs": ["ibuprofen", "aspirin"],
        "aes": ["gastrointestinal haemorrhage", "peptic ulcer"],
        "findings": "NSAIDs increase upper GI bleeding roughly four-fold versus non-use.",
        "parts": (
            "Non-steroidal anti-inflammatory drugs are a leading cause of drug-induced GI bleeding.",
            "We meta-analysed 52 studies quantifying upper gastrointestinal haemorrhage risk with common NSAIDs.",
            "Pooled analysis showed an approximately four-fold increase in bleeding risk, rising further with age, prior ulcer, and concurrent antithrombotics.",
            "Risk stratification should guide NSAID prescribing and gastroprotection.",
            "Proton-pump inhibitor co-therapy substantially attenuates the excess risk.",
        ),
    },
    {
        "pmid": "34055602",
        "title": "Proton-Pump Inhibitor Co-Therapy for NSAID Gastroprotection",
        "journal": "American Journal of Gastroenterology",
        "pub_date": "2022-04-25",
        "study_type": StudyType.CLINICAL_TRIAL,
        "drugs": ["ibuprofen", "omeprazole"],
        "aes": ["gastrointestinal haemorrhage", "dyspepsia"],
        "findings": "Omeprazole markedly reduced NSAID-associated ulcers and bleeding.",
        "parts": (
            "Gastroprotection can offset NSAID mucosal injury in at-risk patients.",
            "In a randomized trial, 640 chronic ibuprofen users received omeprazole or placebo for six months.",
            "Endoscopic ulcers and clinically significant bleeding fell by roughly two-thirds with omeprazole compared with placebo.",
            "PPI co-therapy is effective gastroprotection for higher-risk NSAID users.",
            "The strategy is cost-effective in patients with additional bleeding risk factors.",
        ),
    },
    {
        "pmid": "34055603",
        "title": "Ibuprofen and Antiplatelet Interaction: Attenuation of Aspirin Cardioprotection",
        "journal": "Journal of the American College of Cardiology",
        "pub_date": "2021-02-16",
        "study_type": StudyType.REVIEW,
        "drugs": ["ibuprofen", "aspirin"],
        "aes": ["gastrointestinal haemorrhage", "myocardial infarction"],
        "findings": "Ibuprofen may blunt aspirin's irreversible platelet inhibition.",
        "parts": (
            "Timing of ibuprofen relative to aspirin may affect cardioprotection.",
            "We reviewed pharmacodynamic and observational data on the ibuprofen-aspirin interaction.",
            "Ibuprofen taken before aspirin competitively blocked the platelet COX-1 site, potentially attenuating aspirin's antiplatelet benefit while adding bleeding risk.",
            "Patients should separate dosing or consider alternative analgesics.",
            "The interaction is clinically relevant for secondary cardiovascular prevention.",
        ),
    },
    # --- drug interactions in the elderly (3) ---
    {
        "pmid": "34066701",
        "title": "Polypharmacy and Adverse Drug Events in Older Adults: A Prospective Study",
        "journal": "Age and Ageing",
        "pub_date": "2020-10-30",
        "study_type": StudyType.CLINICAL_TRIAL,
        "drugs": ["warfarin", "amiodarone", "metformin", "sertraline"],
        "aes": ["haemorrhage", "hypoglycaemia", "hyponatraemia"],
        "findings": "Each additional medication incrementally raised adverse-event risk.",
        "parts": (
            "Older adults face compounding interaction risk from polypharmacy.",
            "We prospectively followed 2,100 community-dwelling adults over 75 taking five or more medications.",
            "Adverse drug events rose steadily with medication count, driven by anticoagulant bleeding, sulfonylurea and insulin hypoglycaemia, and SSRI-related hyponatraemia.",
            "Structured deprescribing reviews reduced preventable events.",
            "Medication reconciliation should be routine at every transition of care.",
        ),
    },
    {
        "pmid": "34066702",
        "title": "SSRIs and Hyponatraemia in the Elderly: A Case-Control Analysis",
        "journal": "British Journal of Clinical Pharmacology",
        "pub_date": "2022-07-04",
        "study_type": StudyType.META_ANALYSIS,
        "drugs": ["sertraline"],
        "aes": ["hyponatraemia", "syndrome of inappropriate antidiuretic hormone secretion"],
        "findings": "SSRI initiation sharply increased early hyponatraemia in older patients.",
        "parts": (
            "SSRIs can provoke SIADH-mediated hyponatraemia, especially in older adults.",
            "This case-control analysis examined sodium levels after sertraline initiation in patients over 65.",
            "The odds of clinically significant hyponatraemia were roughly tripled in the first month, particularly with concurrent diuretics.",
            "Baseline and early follow-up sodium checks are prudent after SSRI start.",
            "Prompt recognition prevents falls and hospitalisation.",
        ),
    },
    {
        "pmid": "34066703",
        "title": "Anticholinergic Burden and Falls Among Older Patients on CNS Drugs",
        "journal": "Journal of the American Geriatrics Society",
        "pub_date": "2023-08-22",
        "study_type": StudyType.REVIEW,
        "drugs": ["gabapentin", "sertraline"],
        "aes": ["somnolence", "dizziness", "fall"],
        "findings": "CNS-active drug combinations increased sedation and fall risk.",
        "parts": (
            "Sedating medications accumulate in older adults and contribute to falls.",
            "We reviewed the association between CNS-active drug burden and fall-related outcomes.",
            "Gabapentin combined with serotonergic agents was associated with increased somnolence, dizziness, and fall-related injuries.",
            "Cumulative sedative burden should be minimised in frail patients.",
            "Dose reduction and gait assessment are practical mitigations.",
        ),
    },
    # --- GLP-1 agonist safety signals (3) ---
    {
        "pmid": "34077801",
        "title": "GLP-1 Receptor Agonists and Acute Pancreatitis: A Disproportionality Study",
        "journal": "Diabetologia",
        "pub_date": "2021-05-13",
        "study_type": StudyType.META_ANALYSIS,
        "drugs": ["semaglutide", "liraglutide"],
        "aes": ["pancreatitis", "nausea"],
        "findings": "A modest pancreatitis signal persists but absolute risk is low.",
        "parts": (
            "GLP-1 receptor agonists have been scrutinised for pancreatic safety.",
            "We conducted a disproportionality analysis of spontaneous reports for pancreatitis with GLP-1 agonists.",
            "A modest but consistent reporting signal for acute pancreatitis emerged, though the absolute event rate remained low and often confounded by baseline diabetes risk.",
            "Clinicians should counsel patients on pancreatitis symptoms without withholding effective therapy.",
            "Ongoing surveillance is warranted as prescribing expands.",
        ),
    },
    {
        "pmid": "34077802",
        "title": "Gastrointestinal Adverse Events With Semaglutide: Dose-Escalation Matters",
        "journal": "The Lancet Diabetes & Endocrinology",
        "pub_date": "2022-11-08",
        "study_type": StudyType.CLINICAL_TRIAL,
        "drugs": ["semaglutide"],
        "aes": ["nausea", "vomiting", "diarrhoea"],
        "findings": "Slower titration halved discontinuations for nausea.",
        "parts": (
            "Nausea limits tolerability of GLP-1 receptor agonists.",
            "This trial randomized 560 adults to standard or slower semaglutide dose escalation.",
            "Slower titration reduced nausea and vomiting and halved discontinuations, with equivalent weight and glycaemic outcomes at 24 weeks.",
            "Individualised, gradual escalation improves tolerability.",
            "Patient-tailored titration supports long-term adherence.",
        ),
    },
    {
        "pmid": "34077803",
        "title": "GLP-1 Agonists and Gallbladder Disease: A Signal From Pooled Trials",
        "journal": "JAMA",
        "pub_date": "2023-04-02",
        "study_type": StudyType.META_ANALYSIS,
        "drugs": ["liraglutide", "dulaglutide"],
        "aes": ["cholelithiasis", "cholecystitis"],
        "findings": "Higher-dose, longer-duration GLP-1 use raised gallbladder events.",
        "parts": (
            "Rapid weight loss with GLP-1 agonists may promote gallstone formation.",
            "We pooled 76 randomized trials to quantify gallbladder-related adverse events.",
            "GLP-1 receptor agonists were associated with a significant increase in cholelithiasis and cholecystitis, most pronounced at higher doses and longer durations.",
            "Patients should be advised about biliary symptoms during therapy.",
            "The benefit-risk balance remains favourable for most indications.",
        ),
    },
    # --- general pharmacovigilance (6) ---
    {
        "pmid": "34088901",
        "title": "Signal Detection in Spontaneous Reporting Systems: Methods and Pitfalls",
        "journal": "Drug Safety",
        "pub_date": "2020-01-20",
        "study_type": StudyType.REVIEW,
        "drugs": [],
        "aes": [],
        "findings": "Disproportionality methods flag signals but require careful confounding control.",
        "parts": (
            "Spontaneous reporting systems underpin post-marketing drug safety.",
            "We reviewed disproportionality methods including the reporting odds ratio and Bayesian shrinkage estimators.",
            "Each method detected signals effectively but was vulnerable to reporting bias, notoriety effects, and confounding by indication.",
            "Signals should be treated as hypotheses requiring confirmatory study.",
            "Transparent methodology and triangulation across data sources strengthen conclusions.",
        ),
    },
    {
        "pmid": "34088902",
        "title": "The Role of Real-World Evidence in Pharmacovigilance",
        "journal": "Nature Reviews Drug Discovery",
        "pub_date": "2021-10-11",
        "study_type": StudyType.REVIEW,
        "drugs": [],
        "aes": [],
        "findings": "Linked healthcare data complement spontaneous reports for risk quantification.",
        "parts": (
            "Real-world data increasingly supplement traditional safety surveillance.",
            "We examined how electronic health records and claims data are used to quantify drug risks.",
            "Linked datasets enabled incidence estimation and confounding adjustment that spontaneous reports alone cannot provide, at the cost of data-quality challenges.",
            "Combining spontaneous reports with real-world evidence yields more actionable safety insights.",
            "Governance and privacy safeguards are prerequisites for responsible use.",
        ),
    },
    {
        "pmid": "34088903",
        "title": "Under-Reporting of Adverse Drug Reactions: A Systematic Review",
        "journal": "Pharmacoepidemiology and Drug Safety",
        "pub_date": "2022-03-28",
        "study_type": StudyType.META_ANALYSIS,
        "drugs": [],
        "aes": [],
        "findings": "The median under-reporting rate of serious ADRs exceeded 90 percent.",
        "parts": (
            "Under-reporting limits the sensitivity of spontaneous surveillance.",
            "This systematic review pooled 37 studies estimating adverse drug reaction reporting completeness.",
            "The median under-reporting rate exceeded 90 percent and was worse for non-serious and well-known reactions.",
            "Interventions to ease reporting could improve signal detection.",
            "Automated extraction from clinical records is a promising complement.",
        ),
    },
    {
        "pmid": "34088904",
        "title": "Machine Learning for Adverse Event Prediction: A Critical Appraisal",
        "journal": "Journal of the American Medical Informatics Association",
        "pub_date": "2023-02-14",
        "study_type": StudyType.REVIEW,
        "drugs": [],
        "aes": [],
        "findings": "Predictive models show promise but external validation is often lacking.",
        "parts": (
            "Machine learning is increasingly applied to adverse-event prediction.",
            "We critically appraised 48 published models for methodological rigour and external validation.",
            "Many models reported strong internal performance but few were externally validated, and calibration was frequently unreported.",
            "External validation and prospective evaluation are essential before deployment.",
            "Transparent reporting standards would accelerate trustworthy adoption.",
        ),
    },
    {
        "pmid": "34088905",
        "title": "Evidence Grading Frameworks for Drug Safety Communication",
        "journal": "BMJ Evidence-Based Medicine",
        "pub_date": "2021-07-19",
        "study_type": StudyType.REVIEW,
        "drugs": [],
        "aes": [],
        "findings": "Structured grading improves clarity of safety communications.",
        "parts": (
            "Communicating uncertainty in drug safety findings is challenging.",
            "We compared evidence-grading frameworks used in safety advisories and label changes.",
            "Structured grading that weighted source authority and convergence produced clearer, more reproducible communications than narrative summaries.",
            "Adopting explicit grading improves clinician and patient understanding.",
            "Consistent frameworks also support regulatory decision-making.",
        ),
    },
    {
        "pmid": "34088906",
        "title": "Knowledge Graphs for Integrated Drug Safety Intelligence",
        "journal": "Journal of Biomedical Informatics",
        "pub_date": "2024-02-27",
        "study_type": StudyType.REVIEW,
        "drugs": ["metformin", "warfarin", "amiodarone"],
        "aes": ["lactic acidosis", "haemorrhage"],
        "findings": "Graph integration surfaces cross-source safety relationships.",
        "parts": (
            "Fragmented data sources hamper holistic drug safety assessment.",
            "We reviewed knowledge-graph approaches that integrate spontaneous reports, literature, trials, and labels.",
            "Graph representations surfaced non-obvious drug-drug and drug-event relationships, such as amiodarone-mediated warfarin and statin risks, that single sources missed.",
            "Integrated graphs can accelerate signal triage and hypothesis generation.",
            "Provenance and evidence weighting are essential to avoid spurious links.",
        ),
    },
]


def build_pubmed_articles() -> list[PubMedArticle]:
    """Construct the 30 PubMed article models."""
    articles: list[PubMedArticle] = []
    for entry in _PUBMED:
        bg, methods, results, conclusion, impl = entry["parts"]
        base_abstract = _abstract(bg, methods, results, conclusion, impl)
        full_abstract = _extend_abstract(
            base_abstract, entry["study_type"], entry["drugs"], entry["aes"]
        )
        articles.append(
            PubMedArticle(
                pmid=entry["pmid"],
                title=entry["title"],
                abstract=full_abstract,
                authors=[
                    f"{RNG.choice(['Smith','Patel','Nguyen','Garcia','Cohen','Ali','Rossi','Tanaka'])} {chr(RNG.randint(65,90))}",
                    f"{RNG.choice(['Johnson','Kim','Lopez','Muller','Haddad','Silva','Novak','Chen'])} {chr(RNG.randint(65,90))}",
                ],
                journal=entry["journal"],
                pub_date=entry["pub_date"],
                mesh_terms=[d.title() for d in entry["drugs"]] + ["Drug Safety", "Pharmacovigilance"],
                drugs_mentioned=entry["drugs"],
                adverse_events_mentioned=entry["aes"],
                study_type=entry["study_type"],
                key_findings=entry["findings"],
            )
        )
    return articles


# --------------------------------------------------------------------------- #
# ClinicalTrials.gov — 20 trials
# --------------------------------------------------------------------------- #
def _ae(term: str, organ: str, affected: int, at_risk: int) -> TrialAdverseEvent:
    """Build a :class:`TrialAdverseEvent` with a computed frequency percent."""
    freq = round(100.0 * affected / at_risk, 2) if at_risk else 0.0
    return TrialAdverseEvent(
        term=term,
        organ_system=organ,
        affected_count=affected,
        at_risk_count=at_risk,
        frequency_percent=freq,
    )


# nct, title, phase, status, conditions, intervention drugs, [(term, organ, affected, at_risk)], enroll, start, end
_TRIALS: list[dict] = [
    # metformin (3)
    {
        "nct": "NCT03012001", "title": "Metformin for Glycaemic Control in Newly Diagnosed Type 2 Diabetes",
        "phase": TrialPhase.PHASE_3, "status": TrialStatus.COMPLETED,
        "conditions": ["Type 2 Diabetes Mellitus"], "drugs": ["metformin"],
        "aes": [("Diarrhoea", "Gastrointestinal disorders", 42, 300), ("Nausea", "Gastrointestinal disorders", 28, 300), ("Lactic acidosis", "Metabolism and nutrition disorders", 1, 300)],
        "enroll": 300, "start": "2019-03-01", "end": "2021-06-30",
    },
    {
        "nct": "NCT03012002", "title": "Metformin Versus Placebo for Prediabetes Progression",
        "phase": TrialPhase.PHASE_4, "status": TrialStatus.COMPLETED,
        "conditions": ["Prediabetic State"], "drugs": ["metformin"],
        "aes": [("Diarrhoea", "Gastrointestinal disorders", 30, 250), ("Vitamin B12 deficiency", "Metabolism and nutrition disorders", 9, 250)],
        "enroll": 250, "start": "2018-09-15", "end": "2022-01-20",
    },
    {
        "nct": "NCT03012003", "title": "Extended-Release Metformin Tolerability Study",
        "phase": TrialPhase.PHASE_2, "status": TrialStatus.RECRUITING,
        "conditions": ["Type 2 Diabetes Mellitus"], "drugs": ["metformin"],
        "aes": [],
        "enroll": 180, "start": "2023-11-01", "end": None,
    },
    # statins (3)
    {
        "nct": "NCT03022001", "title": "Atorvastatin for Primary Prevention of Cardiovascular Events",
        "phase": TrialPhase.PHASE_3, "status": TrialStatus.COMPLETED,
        "conditions": ["Hypercholesterolaemia", "Cardiovascular Disease"], "drugs": ["atorvastatin"],
        "aes": [("Myalgia", "Musculoskeletal and connective tissue disorders", 55, 500), ("Rhabdomyolysis", "Musculoskeletal and connective tissue disorders", 2, 500), ("Hepatic function abnormal", "Hepatobiliary disorders", 12, 500)],
        "enroll": 500, "start": "2017-05-10", "end": "2021-05-10",
    },
    {
        "nct": "NCT03022002", "title": "Rosuvastatin Intensive Versus Moderate Dosing Trial",
        "phase": TrialPhase.PHASE_3, "status": TrialStatus.COMPLETED,
        "conditions": ["Hypercholesterolaemia"], "drugs": ["rosuvastatin"],
        "aes": [("Myalgia", "Musculoskeletal and connective tissue disorders", 40, 420), ("Blood creatine phosphokinase increased", "Investigations", 18, 420)],
        "enroll": 420, "start": "2019-01-07", "end": "2022-09-30",
    },
    {
        "nct": "NCT03022003", "title": "Simvastatin Safety in Elderly Patients on Amiodarone",
        "phase": TrialPhase.PHASE_4, "status": TrialStatus.TERMINATED,
        "conditions": ["Hypercholesterolaemia", "Cardiac Arrhythmia"], "drugs": ["simvastatin", "amiodarone"],
        "aes": [("Rhabdomyolysis", "Musculoskeletal and connective tissue disorders", 6, 90), ("Myalgia", "Musculoskeletal and connective tissue disorders", 15, 90)],
        "enroll": 90, "start": "2020-02-01", "end": "2021-03-15",
    },
    # antihypertensive (3)
    {
        "nct": "NCT03032001", "title": "Lisinopril Versus Amlodipine for Stage 1 Hypertension",
        "phase": TrialPhase.PHASE_3, "status": TrialStatus.COMPLETED,
        "conditions": ["Hypertension"], "drugs": ["lisinopril", "amlodipine"],
        "aes": [("Cough", "Respiratory, thoracic and mediastinal disorders", 48, 400), ("Angioedema", "Immune system disorders", 3, 400), ("Peripheral oedema", "General disorders", 60, 400)],
        "enroll": 400, "start": "2018-06-01", "end": "2021-12-01",
    },
    {
        "nct": "NCT03032002", "title": "Metoprolol for Rate Control in Atrial Fibrillation",
        "phase": TrialPhase.PHASE_4, "status": TrialStatus.COMPLETED,
        "conditions": ["Atrial Fibrillation", "Hypertension"], "drugs": ["metoprolol"],
        "aes": [("Bradycardia", "Cardiac disorders", 34, 350), ("Fatigue", "General disorders", 41, 350)],
        "enroll": 350, "start": "2019-04-18", "end": "2022-04-18",
    },
    {
        "nct": "NCT03032003", "title": "Amlodipine Add-On Therapy for Resistant Hypertension",
        "phase": TrialPhase.PHASE_2, "status": TrialStatus.RECRUITING,
        "conditions": ["Hypertension"], "drugs": ["amlodipine"],
        "aes": [],
        "enroll": 150, "start": "2024-01-10", "end": None,
    },
    # anticoagulant (2)
    {
        "nct": "NCT03042001", "title": "Warfarin Versus Direct Oral Anticoagulant in Atrial Fibrillation",
        "phase": TrialPhase.PHASE_3, "status": TrialStatus.COMPLETED,
        "conditions": ["Atrial Fibrillation", "Stroke Prevention"], "drugs": ["warfarin"],
        "aes": [("Haemorrhage", "Vascular disorders", 45, 600), ("Gastrointestinal haemorrhage", "Gastrointestinal disorders", 22, 600), ("Cerebral haemorrhage", "Nervous system disorders", 5, 600)],
        "enroll": 600, "start": "2017-10-01", "end": "2021-10-01",
    },
    {
        "nct": "NCT03042002", "title": "Warfarin Plus Aspirin After Mechanical Valve Replacement",
        "phase": TrialPhase.PHASE_4, "status": TrialStatus.RECRUITING,
        "conditions": ["Heart Valve Disease", "Thromboembolism"], "drugs": ["warfarin", "aspirin"],
        "aes": [],
        "enroll": 220, "start": "2023-08-01", "end": None,
    },
    # PPI (2)
    {
        "nct": "NCT03052001", "title": "Omeprazole Gastroprotection During Chronic NSAID Use",
        "phase": TrialPhase.PHASE_3, "status": TrialStatus.COMPLETED,
        "conditions": ["Peptic Ulcer", "Osteoarthritis"], "drugs": ["omeprazole", "ibuprofen"],
        "aes": [("Gastrointestinal haemorrhage", "Gastrointestinal disorders", 8, 320), ("Hypomagnesaemia", "Metabolism and nutrition disorders", 11, 320), ("Diarrhoea", "Gastrointestinal disorders", 19, 320)],
        "enroll": 320, "start": "2019-02-11", "end": "2022-02-11",
    },
    {
        "nct": "NCT03052002", "title": "Long-Term Omeprazole and Risk of Enteric Infection",
        "phase": TrialPhase.PHASE_4, "status": TrialStatus.COMPLETED,
        "conditions": ["Gastro-Oesophageal Reflux Disease"], "drugs": ["omeprazole"],
        "aes": [("Clostridium difficile colitis", "Infections and infestations", 7, 410), ("Hypomagnesaemia", "Metabolism and nutrition disorders", 14, 410)],
        "enroll": 410, "start": "2018-12-03", "end": "2023-01-15",
    },
    # NSAID (2)
    {
        "nct": "NCT03062001", "title": "Ibuprofen Versus Naproxen for Chronic Osteoarthritis Pain",
        "phase": TrialPhase.PHASE_3, "status": TrialStatus.COMPLETED,
        "conditions": ["Osteoarthritis"], "drugs": ["ibuprofen"],
        "aes": [("Gastrointestinal haemorrhage", "Gastrointestinal disorders", 16, 380), ("Dyspepsia", "Gastrointestinal disorders", 52, 380), ("Hypertension", "Vascular disorders", 21, 380)],
        "enroll": 380, "start": "2018-03-22", "end": "2021-03-22",
    },
    {
        "nct": "NCT03062002", "title": "Low-Dose Ibuprofen Cardiovascular Safety Study",
        "phase": TrialPhase.PHASE_4, "status": TrialStatus.RECRUITING,
        "conditions": ["Pain", "Cardiovascular Risk"], "drugs": ["ibuprofen", "aspirin"],
        "aes": [],
        "enroll": 260, "start": "2024-02-05", "end": None,
    },
    # other (5)
    {
        "nct": "NCT03072001", "title": "Clopidogrel Versus Ticagrelor After Acute Coronary Syndrome",
        "phase": TrialPhase.PHASE_3, "status": TrialStatus.COMPLETED,
        "conditions": ["Acute Coronary Syndrome"], "drugs": ["clopidogrel", "aspirin"],
        "aes": [("Haemorrhage", "Vascular disorders", 38, 550), ("Dyspnoea", "Respiratory, thoracic and mediastinal disorders", 12, 550)],
        "enroll": 550, "start": "2017-11-14", "end": "2021-11-14",
    },
    {
        "nct": "NCT03072002", "title": "Sertraline for Major Depressive Disorder in Older Adults",
        "phase": TrialPhase.PHASE_4, "status": TrialStatus.COMPLETED,
        "conditions": ["Major Depressive Disorder"], "drugs": ["sertraline"],
        "aes": [("Hyponatraemia", "Metabolism and nutrition disorders", 17, 300), ("Nausea", "Gastrointestinal disorders", 44, 300)],
        "enroll": 300, "start": "2019-07-09", "end": "2022-07-09",
    },
    {
        "nct": "NCT03072003", "title": "Gabapentin for Diabetic Peripheral Neuropathic Pain",
        "phase": TrialPhase.PHASE_3, "status": TrialStatus.COMPLETED,
        "conditions": ["Diabetic Neuropathy", "Neuropathic Pain"], "drugs": ["gabapentin"],
        "aes": [("Somnolence", "Nervous system disorders", 58, 340), ("Dizziness", "Nervous system disorders", 47, 340), ("Peripheral oedema", "General disorders", 20, 340)],
        "enroll": 340, "start": "2018-08-27", "end": "2021-08-27",
    },
    {
        "nct": "NCT03072004", "title": "Aspirin for Primary Cardiovascular Prevention in Diabetes",
        "phase": TrialPhase.PHASE_3, "status": TrialStatus.COMPLETED,
        "conditions": ["Type 2 Diabetes Mellitus", "Cardiovascular Disease"], "drugs": ["aspirin", "metformin"],
        "aes": [("Gastrointestinal haemorrhage", "Gastrointestinal disorders", 29, 700), ("Epistaxis", "Respiratory, thoracic and mediastinal disorders", 33, 700)],
        "enroll": 700, "start": "2016-04-01", "end": "2021-04-01",
    },
    {
        "nct": "NCT03072005", "title": "Amiodarone Versus Sotalol for Recurrent Atrial Fibrillation",
        "phase": TrialPhase.PHASE_3, "status": TrialStatus.RECRUITING,
        "conditions": ["Atrial Fibrillation"], "drugs": ["amiodarone"],
        "aes": [],
        "enroll": 280, "start": "2023-10-16", "end": None,
    },
]


def build_clinical_trials() -> list[ClinicalTrialRecord]:
    """Construct the 20 clinical-trial models."""
    trials: list[ClinicalTrialRecord] = []
    for t in _TRIALS:
        trials.append(
            ClinicalTrialRecord(
                nct_id=t["nct"],
                title=t["title"],
                phase=t["phase"],
                status=t["status"],
                conditions=t["conditions"],
                interventions=[
                    TrialIntervention(name=d, type="drug") for d in t["drugs"]
                ],
                adverse_events=[_ae(*a) for a in t["aes"]],
                enrollment=t["enroll"],
                start_date=t["start"],
                completion_date=t["end"],
            )
        )
    return trials


# --------------------------------------------------------------------------- #
# Drug labels — 15 (one per core drug)
# --------------------------------------------------------------------------- #
def _rx(reaction: str, freq: AEFrequency, desc: str = "") -> LabelAdverseReaction:
    return LabelAdverseReaction(reaction=reaction, frequency=freq, description=desc)


def _ix(drug: str, sev: InteractionSeverity, desc: str = "") -> LabelDrugInteraction:
    return LabelDrugInteraction(interacting_drug=drug, severity=sev, description=desc)


_LABELS: list[dict] = [
    {
        "drug_name": "Glucophage", "generic": "metformin", "ingredient": "metformin hydrochloride",
        "manufacturer": "Bristol-Myers Squibb",
        "indications": ["Type 2 diabetes mellitus"],
        "contraindications": ["Severe renal impairment (eGFR < 30)", "Metabolic acidosis", "Diabetic ketoacidosis"],
        "warnings": ["Risk of lactic acidosis", "Vitamin B12 deficiency with long-term use"],
        "rx": [_rx("Diarrhoea", AEFrequency.COMMON), _rx("Nausea", AEFrequency.COMMON), _rx("Lactic acidosis", AEFrequency.VERY_RARE, "Serious; discontinue immediately")],
        "ix": [_ix("ibuprofen", InteractionSeverity.MODERATE, "NSAID may reduce renal clearance, raising lactic acidosis risk"), _ix("contrast media", InteractionSeverity.MAJOR, "Withhold around iodinated contrast administration")],
        "boxed": None,
    },
    {
        "drug_name": "Prinivil", "generic": "lisinopril", "ingredient": "lisinopril",
        "manufacturer": "Merck",
        "indications": ["Hypertension", "Heart failure", "Post-myocardial infarction"],
        "contraindications": ["History of angioedema", "Pregnancy", "Bilateral renal artery stenosis"],
        "warnings": ["Angioedema", "Hyperkalaemia", "Renal impairment"],
        "rx": [_rx("Cough", AEFrequency.COMMON), _rx("Dizziness", AEFrequency.COMMON), _rx("Angioedema", AEFrequency.RARE, "May be life-threatening")],
        "ix": [_ix("ibuprofen", InteractionSeverity.MODERATE, "NSAIDs blunt antihypertensive effect and worsen renal function"), _ix("potassium supplements", InteractionSeverity.MODERATE, "Risk of hyperkalaemia")],
        "boxed": None,
    },
    {
        "drug_name": "Lipitor", "generic": "atorvastatin", "ingredient": "atorvastatin calcium",
        "manufacturer": "Pfizer",
        "indications": ["Hypercholesterolaemia", "Prevention of cardiovascular disease"],
        "contraindications": ["Active liver disease", "Pregnancy", "Breastfeeding"],
        "warnings": ["Myopathy and rhabdomyolysis", "Hepatic enzyme elevation"],
        "rx": [_rx("Myalgia", AEFrequency.COMMON), _rx("Hepatic function abnormal", AEFrequency.UNCOMMON), _rx("Rhabdomyolysis", AEFrequency.RARE, "Serious; monitor CK")],
        "ix": [_ix("amiodarone", InteractionSeverity.MAJOR, "CYP3A4 inhibition increases statin exposure and myopathy risk"), _ix("clarithromycin", InteractionSeverity.MAJOR)],
        "boxed": None,
    },
    {
        "drug_name": "Coumadin", "generic": "warfarin", "ingredient": "warfarin sodium",
        "manufacturer": "Bristol-Myers Squibb",
        "indications": ["Atrial fibrillation", "Venous thromboembolism", "Mechanical heart valves"],
        "contraindications": ["Active bleeding", "Pregnancy", "Recent major surgery"],
        "warnings": ["Major and fatal bleeding", "Requires regular INR monitoring", "Numerous drug and food interactions"],
        "rx": [_rx("Haemorrhage", AEFrequency.COMMON, "Bleeding is the principal risk"), _rx("Gastrointestinal haemorrhage", AEFrequency.UNCOMMON), _rx("Skin necrosis", AEFrequency.RARE)],
        "ix": [_ix("aspirin", InteractionSeverity.MAJOR, "Additive bleeding risk"), _ix("ibuprofen", InteractionSeverity.MAJOR, "Additive bleeding and GI injury"), _ix("amiodarone", InteractionSeverity.MAJOR, "CYP2C9 inhibition raises INR; reduce warfarin dose")],
        "boxed": "WARNING: BLEEDING RISK. Warfarin can cause major or fatal bleeding. Regular INR monitoring is required.",
    },
    {
        "drug_name": "Cordarone", "generic": "amiodarone", "ingredient": "amiodarone hydrochloride",
        "manufacturer": "Wyeth",
        "indications": ["Life-threatening ventricular arrhythmias", "Atrial fibrillation"],
        "contraindications": ["Severe sinus-node dysfunction", "Second- or third-degree heart block", "Iodine hypersensitivity"],
        "warnings": ["Pulmonary toxicity", "Hepatotoxicity", "Thyroid dysfunction", "Proarrhythmia"],
        "rx": [_rx("Hypothyroidism", AEFrequency.COMMON), _rx("Hyperthyroidism", AEFrequency.UNCOMMON), _rx("Pulmonary toxicity", AEFrequency.UNCOMMON, "May be fatal"), _rx("Bradycardia", AEFrequency.COMMON)],
        "ix": [_ix("warfarin", InteractionSeverity.MAJOR, "Potentiates anticoagulation"), _ix("simvastatin", InteractionSeverity.MAJOR, "Increases rhabdomyolysis risk; limit simvastatin dose"), _ix("metoprolol", InteractionSeverity.MODERATE, "Additive bradycardia")],
        "boxed": "WARNING: PULMONARY, HEPATIC, AND PROARRHYTHMIC TOXICITY. Reserve for life-threatening arrhythmias due to potentially fatal toxicities.",
    },
    {
        "drug_name": "Prilosec", "generic": "omeprazole", "ingredient": "omeprazole",
        "manufacturer": "AstraZeneca",
        "indications": ["Gastro-oesophageal reflux disease", "Peptic ulcer disease", "NSAID gastroprotection"],
        "contraindications": ["Hypersensitivity to proton-pump inhibitors"],
        "warnings": ["Hypomagnesaemia with long-term use", "Increased risk of enteric infection", "Possible B12 deficiency"],
        "rx": [_rx("Headache", AEFrequency.COMMON), _rx("Diarrhoea", AEFrequency.COMMON), _rx("Hypomagnesaemia", AEFrequency.UNCOMMON), _rx("Clostridium difficile colitis", AEFrequency.RARE)],
        "ix": [_ix("clopidogrel", InteractionSeverity.MODERATE, "May reduce clopidogrel activation via CYP2C19")],
        "boxed": None,
    },
    {
        "drug_name": "Lopressor", "generic": "metoprolol", "ingredient": "metoprolol tartrate",
        "manufacturer": "Novartis",
        "indications": ["Hypertension", "Angina", "Heart failure", "Rate control in atrial fibrillation"],
        "contraindications": ["Severe bradycardia", "Second- or third-degree heart block", "Decompensated heart failure"],
        "warnings": ["Do not withdraw abruptly", "Bradycardia and hypotension", "May mask hypoglycaemia"],
        "rx": [_rx("Fatigue", AEFrequency.COMMON), _rx("Bradycardia", AEFrequency.COMMON), _rx("Dizziness", AEFrequency.COMMON)],
        "ix": [_ix("amiodarone", InteractionSeverity.MODERATE, "Additive bradycardia and conduction slowing"), _ix("verapamil", InteractionSeverity.MAJOR)],
        "boxed": None,
    },
    {
        "drug_name": "Zocor", "generic": "simvastatin", "ingredient": "simvastatin",
        "manufacturer": "Merck",
        "indications": ["Hypercholesterolaemia", "Prevention of cardiovascular disease"],
        "contraindications": ["Active liver disease", "Pregnancy", "Concomitant strong CYP3A4 inhibitors"],
        "warnings": ["Dose-dependent myopathy and rhabdomyolysis", "Avoid high doses with interacting drugs"],
        "rx": [_rx("Myalgia", AEFrequency.COMMON), _rx("Rhabdomyolysis", AEFrequency.RARE, "Higher risk at 80 mg dose"), _rx("Hepatic function abnormal", AEFrequency.UNCOMMON)],
        "ix": [_ix("amiodarone", InteractionSeverity.MAJOR, "Limit simvastatin to 20 mg daily"), _ix("warfarin", InteractionSeverity.MODERATE)],
        "boxed": None,
    },
    {
        "drug_name": "Norvasc", "generic": "amlodipine", "ingredient": "amlodipine besylate",
        "manufacturer": "Pfizer",
        "indications": ["Hypertension", "Chronic stable angina"],
        "contraindications": ["Severe hypotension", "Hypersensitivity to dihydropyridines"],
        "warnings": ["Peripheral oedema", "Symptomatic hypotension"],
        "rx": [_rx("Peripheral oedema", AEFrequency.COMMON), _rx("Flushing", AEFrequency.COMMON), _rx("Dizziness", AEFrequency.UNCOMMON)],
        "ix": [_ix("simvastatin", InteractionSeverity.MODERATE, "Limit simvastatin dose with amlodipine")],
        "boxed": None,
    },
    {
        "drug_name": "Advil", "generic": "ibuprofen", "ingredient": "ibuprofen",
        "manufacturer": "Pfizer Consumer Healthcare",
        "indications": ["Pain", "Inflammation", "Fever"],
        "contraindications": ["Active peptic ulcer", "Severe heart failure", "Third trimester of pregnancy"],
        "warnings": ["Gastrointestinal bleeding and ulceration", "Cardiovascular thrombotic events", "Renal impairment"],
        "rx": [_rx("Dyspepsia", AEFrequency.COMMON), _rx("Gastrointestinal haemorrhage", AEFrequency.UNCOMMON, "Risk increases with age and antithrombotics"), _rx("Hypertension", AEFrequency.UNCOMMON)],
        "ix": [_ix("warfarin", InteractionSeverity.MAJOR, "Additive bleeding risk"), _ix("aspirin", InteractionSeverity.MODERATE, "May attenuate cardioprotection; additive GI injury"), _ix("lisinopril", InteractionSeverity.MODERATE), _ix("metformin", InteractionSeverity.MODERATE, "Reduced renal clearance may raise lactic acidosis risk")],
        "boxed": None,
    },
    {
        "drug_name": "Bayer Aspirin", "generic": "aspirin", "ingredient": "acetylsalicylic acid",
        "manufacturer": "Bayer",
        "indications": ["Secondary cardiovascular prevention", "Pain", "Fever"],
        "contraindications": ["Active bleeding", "Children with viral illness (Reye syndrome)", "Aspirin hypersensitivity"],
        "warnings": ["Gastrointestinal bleeding", "Bleeding risk with anticoagulants"],
        "rx": [_rx("Dyspepsia", AEFrequency.COMMON), _rx("Gastrointestinal haemorrhage", AEFrequency.UNCOMMON), _rx("Epistaxis", AEFrequency.UNCOMMON)],
        "ix": [_ix("warfarin", InteractionSeverity.MAJOR, "Additive bleeding risk"), _ix("ibuprofen", InteractionSeverity.MODERATE, "Ibuprofen may block aspirin antiplatelet effect"), _ix("clopidogrel", InteractionSeverity.MODERATE)],
        "boxed": None,
    },
    {
        "drug_name": "Plavix", "generic": "clopidogrel", "ingredient": "clopidogrel bisulfate",
        "manufacturer": "Sanofi",
        "indications": ["Acute coronary syndrome", "Recent myocardial infarction or stroke", "Peripheral arterial disease"],
        "contraindications": ["Active pathological bleeding", "Severe hepatic impairment"],
        "warnings": ["Bleeding risk", "Reduced effect in CYP2C19 poor metabolisers"],
        "rx": [_rx("Haemorrhage", AEFrequency.COMMON), _rx("Bruising", AEFrequency.COMMON), _rx("Gastrointestinal haemorrhage", AEFrequency.UNCOMMON)],
        "ix": [_ix("omeprazole", InteractionSeverity.MODERATE, "May reduce clopidogrel activation"), _ix("aspirin", InteractionSeverity.MODERATE), _ix("warfarin", InteractionSeverity.MAJOR)],
        "boxed": "WARNING: DIMINISHED EFFECTIVENESS IN POOR METABOLIZERS. Effectiveness depends on CYP2C19 activation to an active metabolite.",
    },
    {
        "drug_name": "Crestor", "generic": "rosuvastatin", "ingredient": "rosuvastatin calcium",
        "manufacturer": "AstraZeneca",
        "indications": ["Hypercholesterolaemia", "Prevention of cardiovascular disease"],
        "contraindications": ["Active liver disease", "Pregnancy", "Breastfeeding"],
        "warnings": ["Myopathy and rhabdomyolysis", "Proteinuria at high doses"],
        "rx": [_rx("Myalgia", AEFrequency.COMMON), _rx("Blood creatine phosphokinase increased", AEFrequency.UNCOMMON), _rx("Rhabdomyolysis", AEFrequency.RARE)],
        "ix": [_ix("warfarin", InteractionSeverity.MODERATE, "May increase INR"), _ix("ciclosporin", InteractionSeverity.MAJOR)],
        "boxed": None,
    },
    {
        "drug_name": "Zoloft", "generic": "sertraline", "ingredient": "sertraline hydrochloride",
        "manufacturer": "Pfizer",
        "indications": ["Major depressive disorder", "Anxiety disorders", "Obsessive-compulsive disorder"],
        "contraindications": ["Concomitant monoamine oxidase inhibitors", "Concomitant pimozide"],
        "warnings": ["Serotonin syndrome", "Hyponatraemia", "Increased bleeding risk with antithrombotics"],
        "rx": [_rx("Nausea", AEFrequency.COMMON), _rx("Hyponatraemia", AEFrequency.UNCOMMON, "More common in the elderly"), _rx("Serotonin syndrome", AEFrequency.RARE)],
        "ix": [_ix("warfarin", InteractionSeverity.MODERATE, "May increase bleeding risk"), _ix("ibuprofen", InteractionSeverity.MODERATE, "Additive bleeding risk")],
        "boxed": "WARNING: SUICIDAL THOUGHTS AND BEHAVIORS. Increased risk in children, adolescents, and young adults.",
    },
    {
        "drug_name": "Neurontin", "generic": "gabapentin", "ingredient": "gabapentin",
        "manufacturer": "Pfizer",
        "indications": ["Neuropathic pain", "Partial seizures", "Postherpetic neuralgia"],
        "contraindications": ["Hypersensitivity to gabapentin"],
        "warnings": ["CNS depression and sedation", "Respiratory depression with opioids", "Withdrawal on abrupt discontinuation"],
        "rx": [_rx("Somnolence", AEFrequency.COMMON), _rx("Dizziness", AEFrequency.COMMON), _rx("Peripheral oedema", AEFrequency.UNCOMMON)],
        "ix": [_ix("opioids", InteractionSeverity.MAJOR, "Additive respiratory depression"), _ix("sertraline", InteractionSeverity.MINOR)],
        "boxed": None,
    },
]


def build_drug_labels() -> list[DrugLabel]:
    """Construct the 15 drug-label models (one per core drug)."""
    labels: list[DrugLabel] = []
    for lb in _LABELS:
        labels.append(
            DrugLabel(
                drug_name=lb["drug_name"],
                generic_name=lb["generic"],
                active_ingredient=lb["ingredient"],
                manufacturer=lb["manufacturer"],
                indications=lb["indications"],
                contraindications=lb["contraindications"],
                warnings=lb["warnings"],
                adverse_reactions=lb["rx"],
                drug_interactions=lb["ix"],
                boxed_warning=lb["boxed"],
            )
        )
    return labels


# --------------------------------------------------------------------------- #
# Writing & orchestration
# --------------------------------------------------------------------------- #
def _dump(models: list) -> list[dict]:
    """Serialize a list of Pydantic models to JSON-ready dicts."""
    return [m.model_dump(mode="json") for m in models]


def _write_json(path: Path, payload) -> None:
    """Write ``payload`` as pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _raw_dir(sample_subdir: Path) -> Path:
    """Map a ``data/sample/<x>`` directory to its ``data/raw/<x>`` mirror."""
    return settings.data_path / "raw" / sample_subdir.name


def seed() -> dict[str, int]:
    """Generate every dataset, write it to sample + raw, and return counts."""
    faers = build_faers_reports()
    pubmed = build_pubmed_articles()
    trials = build_clinical_trials()
    labels = build_drug_labels()

    # FAERS: 5 batches of 10 -> faers_batch_01.json .. faers_batch_05.json
    faers_dicts = _dump(faers)
    faers_batches: list[tuple[str, list[dict]]] = []
    for batch_no in range(5):
        start = batch_no * 10
        faers_batches.append(
            (f"faers_batch_{batch_no + 1:02d}.json", faers_dicts[start : start + 10])
        )

    # Write to sample/ then mirror the whole set into raw/.
    targets = {
        settings.sample_faers_dir: faers_batches,
        settings.sample_pubmed_dir: [("pubmed_articles.json", _dump(pubmed))],
        settings.sample_trials_dir: [("clinical_trials.json", _dump(trials))],
        settings.sample_labels_dir: [("drug_labels.json", _dump(labels))],
    }

    for sample_dir, files in targets.items():
        raw_dir = _raw_dir(sample_dir)
        for filename, payload in files:
            _write_json(sample_dir / filename, payload)
            _write_json(raw_dir / filename, payload)

    return {
        "faers": len(faers),
        "pubmed": len(pubmed),
        "trials": len(trials),
        "labels": len(labels),
    }


def main() -> None:
    """Entry point: seed all sample data and print a summary."""
    logger.info("Seeding MedSignal sample data (offline, hardcoded)...")
    counts = seed()
    summary = (
        f"Created {counts['faers']} FAERS reports, "
        f"{counts['pubmed']} PubMed articles, "
        f"{counts['trials']} trials, "
        f"{counts['labels']} drug labels"
    )
    logger.info(summary)
    print(summary)
    print(f"Sample data written to: {settings.sample_dir}")
    print(f"Mirrored to:            {settings.data_path / 'raw'}")


if __name__ == "__main__":
    main()
