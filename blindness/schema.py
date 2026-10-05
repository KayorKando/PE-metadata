"""bioRxiv metadata schema (MAPLE, Listing 1) and label cleaning.

The Gemini annotations in MAPLE's CSV contain off-schema values
(e.g. "Psychology" for primary_research_area). We map each value to the
schema: exact match, then prefix match, then the attribute's fallback.
"""
import numpy as np
import pandas as pd

BIORXIV_SCHEMA = {
    "primary_research_area": [
        "Biochemistry", "Bioinformatics", "Biophysics", "Cancer Biology", "Cell Biology",
        "Clinical Trials", "Developmental Biology", "Ecology", "Epidemiology",
        "Evolutionary Biology", "Genetics", "Genomics", "Immunology", "Microbiology",
        "Molecular Biology", "Neuroscience", "Paleontology", "Pathology",
        "Pharmacology and Toxicology", "Physiology", "Plant Biology", "Public Health",
        "Scientific Communication and Education", "Structural Biology", "Synthetic Biology",
        "Systems Biology", "Zoology", "Other",
    ],
    "model_organism": [
        "Human", "Mouse/Rat", "Zebrafish", "Drosophila melanogaster", "Caenorhabditis elegans",
        "Saccharomyces cerevisiae", "Escherichia coli", "Arabidopsis thaliana", "Plant",
        "Cell Culture", "In Silico / Computational", "Other Mammal", "Other Vertebrate",
        "Other Invertebrate", "Other Microbe", "Not Applicable / Review", "Other",
    ],
    "experimental_approach": [
        "Wet Lab Experimentation", "Computational / In Silico Analysis", "Clinical Study",
        "Field Study / Observation", "Case Study / Case Review", "Review / Meta-analysis",
        "New Method Development", "Theoretical Modeling", "Other",
    ],
    "dominant_data_type": [
        "Genomic", "Transcriptomic", "Proteomic", "Metabolomic", "Imaging", "Structural",
        "Phenotypic / Behavioral", "Ecological / Environmental", "Clinical / Patient Data",
        "Simulation / Model Output", "Multi-omics", "Other",
    ],
    "research_focus_scale": [
        "Molecular", "Cellular", "Circuit / Network", "Tissue / Organ", "Organismal",
        "Population", "Ecosystem", "Multi-scale", "Other",
    ],
    "disease_mention": [
        "Cancer", "Neurodegenerative Disease", "Infectious Disease", "Metabolic Disease",
        "Cardiovascular Disease", "Autoimmune / Inflammatory Disease",
        "Psychiatric / Neurological Disorder", "Genetic Disorder",
        "No Specific Disease Mentioned", "Other",
    ],
    "sample_size": [
        "Single Subject / Case Study", "Small Cohort (<50 subjects)",
        "Medium Cohort (50-1000 subjects)", "Large Cohort / Population-scale (>1000 subjects)",
        "Relies on Cell/Animal Replicates", "Not Specified / Not Applicable",
    ],
    "research_goal": [
        "Investigating a mechanism", "Characterizing a system/molecule", "Developing a method/tool",
        "Identifying novel elements", "Testing a hypothesis", "Quantifying a parameter",
        "Evaluating/Comparing approaches", "Other",
    ],
}
FALLBACK = {k: "Other" for k in BIORXIV_SCHEMA}
FALLBACK["sample_size"] = "Not Specified / Not Applicable"

# The 9 attributes MAPLE uses (PE.py: METADATA).
BIORXIV_ATTRIBUTES = list(BIORXIV_SCHEMA) + ["word_count"]


def clean_value(value, options, fallback):
    if not isinstance(value, str):
        return fallback
    v = value.strip()
    if v in options:
        return v
    for opt in options:  # e.g. "Large Cohort / Population-scale" -> "... (>1000 subjects)"
        if opt.startswith(v) or v.startswith(opt):
            return opt
    return fallback


def word_count_bin(texts):
    """Same binning as MAPLE's compute_mauve_jsd.py: round(words / 50) * 50."""
    wc = pd.Series(texts).astype(str).str.split().str.len()
    return ((wc / 50).round() * 50).astype(int).astype(str).to_numpy()


def clean_labels(df, attributes):
    out = {}
    for a in attributes:
        if a == "word_count":
            out[a] = word_count_bin(df["text"])
        elif a in BIORXIV_SCHEMA:
            out[a] = np.array([clean_value(v, BIORXIV_SCHEMA[a], FALLBACK[a]) for v in df[a]])
        else:  # non-schema attribute: use as-is
            out[a] = df[a].astype(str).to_numpy()
    return out
