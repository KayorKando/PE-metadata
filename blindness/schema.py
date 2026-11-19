"""Metadata schemas (MAPLE: bioRxiv Listing 1, OpenReview Listing 2) and label cleaning.

The Gemini annotations in MAPLE's CSV contain off-schema values
(e.g. "Psychology" for primary_research_area). We map each value to the
schema: exact match, then prefix match, then the attribute's fallback.

Which dataset a process works on is set by the environment variable PE_DATASET
(biorxiv, the default, or openreview). SCHEMA / FALLBACK / ATTRIBUTES and clean_labels
follow it; BIORXIV_SCHEMA and OPENREVIEW_SCHEMA are always the fixed dicts.
"""
import os

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
BIORXIV_FALLBACK = {k: "Other" for k in BIORXIV_SCHEMA}
BIORXIV_FALLBACK["sample_size"] = "Not Specified / Not Applicable"

# The 9 attributes MAPLE uses (PE.py: METADATA).
BIORXIV_ATTRIBUTES = list(BIORXIV_SCHEMA) + ["word_count"]

# OpenReview (ICLR 2023 reviews). area ... keywords_group_3: MAPLE Listing 2 (the AIM attributes), with "Other"
# added to primary_strength / primary_weakness (Gemini uses it). The other five attributes are in MAPLE's CSVs
# but not in Listing 2; their options are the values of MAPLE's Gemini labels (train set), one-off values dropped.
OPENREVIEW_SCHEMA = {
    "area": [
        "Deep Learning and representational learning",
        "Applications (eg, speech processing, computer vision, NLP)",
        "Reinforcement Learning (eg, decision and control, planning, hierarchical RL, robotics)",
        "Neuroscience and Cognitive Science (e.g., neural coding, brain-computer interfaces)",
        "Probabilistic Methods (eg, variational inference, causal inference, Gaussian processes)",
        "Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)",
        "Unsupervised and Self-supervised learning",
        "Machine Learning for Sciences (eg biology, physics, health sciences, social sciences, climate/sustainability )",
        "General Machine Learning",
        "Theory (eg, control theory, learning theory, algorithmic game theory)",
        "Generative models",
        "Optimization (eg, convex and non-convex optimization)",
    ],
    "recommendation": [
        "6: marginally above the acceptance threshold", "3: reject, not good enough",
        "5: marginally below the acceptance threshold", "8: accept, good paper", "1: strong reject",
    ],
    "primary_contribution_type": [
        "New Algorithm/Method", "Theoretical Analysis", "Empirical Study/Ablation", "New Benchmark/Dataset",
        "Position/Survey Paper", "Other",
    ],
    "primary_strength": [
        "Novelty/Originality of Idea", "Strong Empirical Performance", "Significant Problem or Application",
        "High-Quality Presentation/Clarity", "Solid Theoretical Guarantees",
        "Good Reproducibility (Code/Data Provided)", "Other",
    ],
    "primary_weakness": [
        "Insufficient/Flawed Experiments", "Weak or Missing Baselines", "Lack of Novelty",
        "Unclear/Poor Presentation", "Flawed Theoretical Claims", "Limited Scope or Impact", "Ethical Concerns",
        "Other",
    ],
    "novelty_assessment": [
        "Groundbreaking", "Significant Advance", "Incremental Improvement", "Well-Known Concept",
        "Unclear/Not Discussed",
    ],
    "reviewer_tone": [
        "Enthusiastic/Supportive", "Constructive/Balanced", "Critical/Dismissive", "Formal/Detached",
    ],
    "clarity_and_presentation_quality": [
        "Excellent (Easy to follow)", "Good (Mostly clear)", "Needs Improvement (Key parts are confusing)",
        "Poor (Very difficult to understand)",
    ],
    "actionability_of_feedback": [
        "Highly Actionable (Specific experiments to run)",
        "Moderately Actionable (Clarifications or rephrasing needed)",
        "Low Actionability (Vague or fundamental conceptual issues)",
    ],
    "keywords_group_1": [
        "Large Language Models (LLMs)", "Diffusion Models", "Foundation Models", "Prompting",
        "Text-to-Image Synthesis", "Transformers", "Self-Supervised Learning (SSL)", "Datasets & Benchmarks",
        "Data Augmentation", "Other",
    ],
    "keywords_group_2": [
        "Few-Shot & Zero-Shot Learning", "Meta-Learning", "Continual Learning", "Transfer Learning",
        "Federated Learning", "Graph Neural Networks (GNNs)", "Knowledge Graphs", "Weak Supervision",
        "Causal Inference", "Other",
    ],
    "keywords_group_3": [
        "Efficiency", "Stochastic Gradient Descent (SGD)", "Model Compression", "Robustness",
        "Out-of-Distribution (OOD) Generalization", "Fairness & Bias", "Privacy",
        "Interpretability & Explainability", "Reinforcement Learning (RL)", "Decision Making", "Other",
    ],
}
# "Other" is the fallback everywhere (also where it is not an option: only unparsed annotations land there),
# except novelty, which has its own "not discussed" option.
OPENREVIEW_FALLBACK = {k: "Other" for k in OPENREVIEW_SCHEMA}
OPENREVIEW_FALLBACK["novelty_assessment"] = "Unclear/Not Discussed"

SCHEMAS = {"biorxiv": (BIORXIV_SCHEMA, BIORXIV_FALLBACK), "openreview": (OPENREVIEW_SCHEMA, OPENREVIEW_FALLBACK)}
DATASET = os.environ.get("PE_DATASET", "biorxiv")
if DATASET not in SCHEMAS:
    raise ValueError(f"PE_DATASET={DATASET!r}: expected one of {sorted(SCHEMAS)}")
SCHEMA, FALLBACK = SCHEMAS[DATASET]
ATTRIBUTES = list(SCHEMA) + ["word_count"]     # the content attributes of the active dataset + word_count


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
        elif a in SCHEMA:
            out[a] = np.array([clean_value(v, SCHEMA[a], FALLBACK[a]) for v in df[a]])
        else:  # non-schema attribute: use as-is
            out[a] = df[a].astype(str).to_numpy()
    return out
