"""Build the two released task prompts without changing their scientific wording."""
from importlib.resources import files
import json
import math
import re


def _targets(targets):
    if not isinstance(targets, (list, tuple)) or not targets:
        raise ValueError("Supply a nonempty list of age-domain names.")
    if any(not isinstance(t, str) or not t.strip() or any(c in t for c in "\n\r{}") for t in targets):
        raise ValueError("Target names must be nonempty single-line strings without braces.")
    values = [t.strip() for t in targets]
    if len(set(values)) != len(values):
        raise ValueError("Target age-domain names must be unique.")
    return values


def load_template(name):
    if name not in ("age_estimation", "recommendation"):
        raise ValueError("Unknown prompt template.")
    return files("smart").joinpath("templates", name + ".txt").read_text(encoding="utf-8")


def build_age_prompt(targets, clinical_narrative):
    targets = _targets(targets)
    if not isinstance(clinical_narrative, str) or not clinical_narrative.strip():
        raise ValueError("A clinical narrative is required.")
    schema = {}
    for target in targets:
        for suffix in (" aging causal chain", " youthful causal chain", " chain of thought"):
            schema[target + suffix] = "string"
        schema[target] = "float"
    slots = {"target_age_types": json.dumps(targets, ensure_ascii=False),
             "output_schema": json.dumps(schema, ensure_ascii=False, indent=2),
             "input": clinical_narrative}
    # A single substitution pass leaves any braces in the input narrative untouched.
    return re.sub(r"\{(target_age_types|output_schema|input)\}",
                  lambda match: slots[match.group(1)], load_template("age_estimation"))


def build_recommendation_prompt(clinical_narrative, biological_age_table):
    if not isinstance(clinical_narrative, str) or not clinical_narrative.strip():
        raise ValueError("A clinical narrative is required.")
    if not isinstance(biological_age_table, str) or not biological_age_table.strip():
        raise ValueError("Supply a labelled baseline/counterfactual age table.")
    return (load_template("recommendation").rstrip()
            + "\n\nINPUT_1 (CLINICAL_NARRATIVE):\n" + clinical_narrative.strip()
            + "\n\nINPUT_2 (BIOLOGICAL_AGE_TABLE):\n" + biological_age_table.strip() + "\n")


def parse_age_response(text, targets):
    """Require every requested age and rationale field; never invent missing values."""
    targets = _targets(targets)
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)[:-3].strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("The response must be a JSON object.")
    for domain in targets:
        value = result.get(domain)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing or invalid numerical age for {domain}.")
        for suffix in (" aging causal chain", " youthful causal chain", " chain of thought"):
            rationale = result.get(domain + suffix)
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError(f"Missing explanation field for {domain}{suffix}.")
    return result
