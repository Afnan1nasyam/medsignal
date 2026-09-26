"""Offline helpers for normalizing and matching biomedical terms.

Pure-Python utilities (no network, no model download) for drug-name
normalization, brand→generic mapping, adverse-event normalization, and simple
substring extraction of drugs / adverse events from free text. Also exposes
reference constants (common adverse events, drug-class groupings) used across
ingestion, retrieval, and the knowledge graph.
"""

import re

# Salt / hydrate suffixes stripped from drug names during normalization.
_SALT_SUFFIXES: frozenset[str] = frozenset(
    {
        "hydrochloride",
        "hcl",
        "sodium",
        "potassium",
        "calcium",
        "magnesium",
        "tartrate",
        "bitartrate",
        "sulfate",
        "sulphate",
        "maleate",
        "mesylate",
        "succinate",
        "besylate",
        "citrate",
        "phosphate",
        "acetate",
        "fumarate",
        "hydrobromide",
        "monohydrate",
        "dihydrate",
        "hemihydrate",
    }
)

# Brand → generic mapping for the drugs in our sample data (and common others).
BRAND_TO_GENERIC: dict[str, str] = {
    "glucophage": "metformin",
    "zestril": "lisinopril",
    "prinivil": "lisinopril",
    "coumadin": "warfarin",
    "cordarone": "amiodarone",
    "lipitor": "atorvastatin",
    "prilosec": "omeprazole",
    "lopressor": "metoprolol",
    "toprol": "metoprolol",
    "zocor": "simvastatin",
    "crestor": "rosuvastatin",
    "norvasc": "amlodipine",
    "synthroid": "levothyroxine",
    "lasix": "furosemide",
    "zoloft": "sertraline",
    "prozac": "fluoxetine",
    "nexium": "esomeprazole",
    "plavix": "clopidogrel",
    "ventolin": "albuterol",
    "tenormin": "atenolol",
    "vasotec": "enalapril",
    "cozaar": "losartan",
    "diovan": "valsartan",
}

# ~50 common adverse-event preferred terms (lowercase).
COMMON_ADVERSE_EVENTS: list[str] = [
    "nausea",
    "vomiting",
    "diarrhea",
    "constipation",
    "headache",
    "dizziness",
    "fatigue",
    "rash",
    "pruritus",
    "urticaria",
    "dyspepsia",
    "abdominal pain",
    "insomnia",
    "somnolence",
    "anxiety",
    "depression",
    "tremor",
    "palpitations",
    "tachycardia",
    "bradycardia",
    "hypotension",
    "hypertension",
    "dyspnea",
    "cough",
    "edema",
    "peripheral edema",
    "myalgia",
    "arthralgia",
    "back pain",
    "weakness",
    "fever",
    "chills",
    "sweating",
    "dry mouth",
    "blurred vision",
    "tinnitus",
    "weight gain",
    "weight loss",
    "decreased appetite",
    "anorexia",
    "hyperglycemia",
    "hypoglycemia",
    "hyperkalemia",
    "anemia",
    "thrombocytopenia",
    "neutropenia",
    "elevated liver enzymes",
    "renal impairment",
    "angioedema",
    "syncope",
    # MedDRA / pharmacovigilance preferred terms seen across the sources, plus
    # common lay synonyms — shared by ingestion and the query planner.
    "lactic acidosis",
    "rhabdomyolysis",
    "muscle pain",
    "gastrointestinal haemorrhage",
    "gastrointestinal hemorrhage",
    "haemorrhage",
    "hemorrhage",
    "bleeding",
    "cerebral haemorrhage",
    "epistaxis",
    "haematuria",
    "hypothyroidism",
    "hyperthyroidism",
    "thyroid disorder",
    "serotonin syndrome",
    "hyponatraemia",
    "hyponatremia",
    "pulmonary toxicity",
    "hepatic function abnormal",
    "liver damage",
    "international normalised ratio increased",
    "qt prolongation",
    "pancreatitis",
    "cholelithiasis",
    "diarrhoea",
    "peripheral oedema",
]

# Drug-class groupings (generic names, lowercase).
DRUG_CLASSES: dict[str, list[str]] = {
    "statins": [
        "atorvastatin",
        "simvastatin",
        "rosuvastatin",
        "pravastatin",
        "lovastatin",
        "fluvastatin",
        "pitavastatin",
    ],
    "ace_inhibitors": [
        "lisinopril",
        "enalapril",
        "ramipril",
        "captopril",
        "benazepril",
        "quinapril",
        "perindopril",
    ],
    "beta_blockers": [
        "metoprolol",
        "atenolol",
        "propranolol",
        "carvedilol",
        "bisoprolol",
        "nebivolol",
        "labetalol",
    ],
    "arbs": [
        "losartan",
        "valsartan",
        "irbesartan",
        "candesartan",
        "olmesartan",
        "telmisartan",
    ],
    "proton_pump_inhibitors": [
        "omeprazole",
        "esomeprazole",
        "pantoprazole",
        "lansoprazole",
        "rabeprazole",
        "dexlansoprazole",
    ],
    "ssris": [
        "sertraline",
        "fluoxetine",
        "paroxetine",
        "citalopram",
        "escitalopram",
        "fluvoxamine",
    ],
    "anticoagulants": [
        "warfarin",
        "apixaban",
        "rivaroxaban",
        "dabigatran",
        "edoxaban",
    ],
    "biguanides": ["metformin"],
    "nsaids": [
        "ibuprofen",
        "naproxen",
        "diclofenac",
        "celecoxib",
        "ketorolac",
        "meloxicam",
    ],
    "antiplatelets": [
        "aspirin",
        "clopidogrel",
        "ticagrelor",
        "prasugrel",
        "dipyridamole",
    ],
    "gabapentinoids": ["gabapentin", "pregabalin"],
}


def normalize_drug_name(name: str) -> str:
    """Normalize a drug name to a canonical generic form.

    Lowercases, collapses whitespace, strips salt/hydrate suffixes
    (e.g. "hydrochloride", "sodium", "calcium"), and resolves known brand
    names to their generic equivalent.

    Args:
        name: Raw drug name, brand or generic, possibly with a salt suffix.

    Returns:
        The normalized generic name (empty string if ``name`` is blank).
    """
    if not name:
        return ""
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    # Drop salt/hydrate words while preserving the base drug name.
    tokens = [tok for tok in cleaned.split(" ") if tok not in _SALT_SUFFIXES]
    base = " ".join(tokens).strip() or cleaned
    return BRAND_TO_GENERIC.get(base, base)


def normalize_adverse_event(term: str) -> str:
    """Normalize an adverse-event term.

    Lowercases, collapses whitespace, and strips a trailing "nos"
    (not otherwise specified) qualifier.

    Args:
        term: Raw adverse-event term.

    Returns:
        The normalized term (empty string if ``term`` is blank).
    """
    if not term:
        return ""
    cleaned = re.sub(r"\s+", " ", term.strip().lower())
    cleaned = re.sub(r"[\s,]+nos$", "", cleaned).strip()
    return cleaned


def are_same_drug(name1: str, name2: str) -> bool:
    """Return whether two drug names refer to the same drug after normalization."""
    n1 = normalize_drug_name(name1)
    n2 = normalize_drug_name(name2)
    return bool(n1) and n1 == n2


def extract_drug_names_from_text(text: str, known_drugs: list[str]) -> list[str]:
    """Extract known drug names mentioned in free text.

    Performs case-insensitive substring matching against ``known_drugs`` and
    also matches known brand names, returning the corresponding generic when
    that generic is in ``known_drugs``.

    Args:
        text: Free text to scan.
        known_drugs: Candidate drug names to look for.

    Returns:
        Sorted list of unique normalized generic names found in ``text``.
    """
    if not text:
        return []
    text_lower = text.lower()
    known_normalized = {normalize_drug_name(d) for d in known_drugs}
    found: set[str] = set()

    for drug in known_drugs:
        if drug and drug.lower() in text_lower:
            found.add(normalize_drug_name(drug))

    for brand, generic in BRAND_TO_GENERIC.items():
        if brand in text_lower and generic in known_normalized:
            found.add(generic)

    return sorted(found)


def extract_ae_terms_from_text(text: str, known_terms: list[str]) -> list[str]:
    """Extract known adverse-event terms mentioned in free text.

    Case-insensitive substring matching against ``known_terms``.

    Args:
        text: Free text to scan.
        known_terms: Candidate adverse-event terms to look for.

    Returns:
        Sorted list of unique normalized terms found in ``text``.
    """
    if not text:
        return []
    text_lower = text.lower()
    found: set[str] = set()
    for term in known_terms:
        if term and term.lower() in text_lower:
            found.add(normalize_adverse_event(term))
    return sorted(found)
