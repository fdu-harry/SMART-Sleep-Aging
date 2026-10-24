"""Render a supplied, ordered field dictionary into the manuscript's SCN format.

This is a format renderer. Original cohort extraction, variable recoding and
clinical reference selection must be performed before this step.
"""
from collections.abc import Mapping

SECTIONS = (
    "Basic information", "Lifestyle", "Physical examination",
    "Sleep study measurements", "Family history", "Medical history",
)


def render_scn(record):
    """Keep recorded negatives separate from missing values (rendered 'unknown')."""
    if not isinstance(record, Mapping) or not record:
        raise ValueError("SCN input must be a nonempty mapping of section headings to fields.")
    unknown = set(record) - set(SECTIONS)
    if unknown:
        raise ValueError(f"Unrecognized sections: {sorted(unknown)}")
    parts = []
    for section in SECTIONS:
        if section not in record:
            continue
        fields = record[section]
        if not isinstance(fields, Mapping):
            raise ValueError(f"{section} must contain a field mapping.")
        lines = [f"[{section}]"]
        for key, value in fields.items():
            if not isinstance(key, str) or not key.strip() or "\n" in key:
                raise ValueError("Field names must be nonempty, single-line strings.")
            if value is None:
                value = "unknown"
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Supply field values as nonempty strings, including their units.")
            lines.append(f"{key}: {value}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"
