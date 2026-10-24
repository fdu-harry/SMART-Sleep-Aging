"""Explicit field perturbation and paired age-shift primitives.

Clinical targets and admissible fields must be specified by the investigator.
"""
from copy import deepcopy
import numpy as np


def perturb_field(record, section, field, new_value):
    """Copy a structured record and alter exactly one named field.

    Use atomic input fields when a single measurement is to be modified. This
    function does not search/replace numbers in prose or choose a clinical target.
    """
    if section not in record or field not in record[section]:
        raise KeyError("The requested field must already exist in the input record.")
    if not isinstance(new_value, str) or not new_value.strip():
        raise ValueError("Supply the replacement field text, including its unit.")
    previous=record[section][field]
    if previous == new_value:
        raise ValueError("A counterfactual perturbation must change the selected field.")
    modified=deepcopy(record)
    modified[section][field]=new_value
    return modified, {'section':section,'field':field,'baseline':previous,'counterfactual':new_value}


def age_shift(baseline, counterfactual):
    """Counterfactual minus baseline ages in matching participant/domain order."""
    baseline=np.asarray(baseline,dtype=float)
    counterfactual=np.asarray(counterfactual,dtype=float)
    if baseline.shape != counterfactual.shape or not np.isfinite(baseline).all() or not np.isfinite(counterfactual).all():
        raise ValueError("Use finite, identically shaped, aligned baseline and counterfactual estimates.")
    return counterfactual-baseline


def age_reduction(baseline, counterfactual):
    """Baseline minus counterfactual; positive values denote estimated reductions."""
    return -age_shift(baseline, counterfactual)


class _MeasurementUpdater:
    """Numerical adapter for the report generator's CascadeUpdater methods."""

    def __init__(self, measurements):
        self.values = deepcopy(measurements)

    def get_val(self, key):
        return self.values.get(key)

    def set_val(self, key, value, is_int=False):
        if key not in self.values or self.values[key] is None or value is None:
            return
        if is_int:
            value = int(round(value))
        else:
            value = float(format(value, '.2f' if key == 'BMI' else '.1f'))
        self.values[key] = value

    def update_bmi(self, new_bmi):
        self.set_val('BMI', new_bmi)
        height_cm = self.get_val('Height')
        if height_cm:
            height_m = height_cm / 100.0
            new_weight = new_bmi * height_m ** 2
            self.set_val('Weight', new_weight)

    def update_tst(self, new_tst):
        old_tst = self.get_val('TST')
        if not old_tst or old_tst <= 0:
            return
        self.set_val('TST', new_tst)
        trt = self.get_val('TRT')
        if trt and trt > 0:
            new_se = new_tst / trt * 100
            self.set_val('SE', min(new_se, 100.0))
        for stage in ['N1', 'N2', 'N3', 'REM']:
            pct = self.get_val(f'{stage}_pct')
            if pct is not None:
                self.set_val(f'{stage}_time', new_tst * (pct / 100.0))
        new_hours = new_tst / 60.0
        for prefix, idx_name in [('O', 'OAI'), ('C', 'CAI'), ('M', 'MAI'), ('H', 'HI'), ('Total_Apnea', 'Total_Apnea_Index')]:
            idx_val = self.get_val(idx_name)
            if idx_val is not None:
                self.set_val(f'{prefix}_events', idx_val * new_hours, is_int=True)
        arousal_idx = self.get_val('Arousal_index')
        if arousal_idx is not None:
            self.set_val('Total_arousals', arousal_idx * new_hours, is_int=True)

    def update_se(self, new_se):
        self.set_val('SE', new_se)
        trt = self.get_val('TRT')
        if trt and trt > 0:
            new_tst = trt * (new_se / 100.0)
            self.update_tst(new_tst)

    def update_sleep_stage_pct(self, target_stage, new_pct):
        old_pct = self.get_val(f'{target_stage}_pct')
        if old_pct is None:
            return
        diff = old_pct - new_pct
        self.set_val(f'{target_stage}_pct', new_pct)
        compensate_stage = 'N1' if target_stage == 'N2' else 'N2'
        comp_old_pct = self.get_val(f'{compensate_stage}_pct')
        if comp_old_pct is not None:
            new_comp_pct = max(0.0, comp_old_pct + diff)
            self.set_val(f'{compensate_stage}_pct', new_comp_pct)
        tst = self.get_val('TST')
        if tst:
            for stage in ['N1', 'N2', 'N3', 'REM']:
                pct = self.get_val(f'{stage}_pct')
                if pct is not None:
                    self.set_val(f'{stage}_time', tst * (pct / 100.0))

    def update_ahi(self, new_ahi):
        old_ahi = self.get_val('AHI')
        if not old_ahi or old_ahi <= 0:
            self.set_val('AHI', new_ahi)
            return
        ratio = new_ahi / old_ahi
        self.set_val('AHI', new_ahi)
        for idx in ['OAI', 'CAI', 'MAI', 'HI']:
            val = self.get_val(idx)
            if val is not None:
                self.set_val(idx, val * ratio)
        new_oai = self.get_val('OAI') or 0
        new_cai = self.get_val('CAI') or 0
        new_mai = self.get_val('MAI') or 0
        self.set_val('Total_Apnea_Index', new_oai + new_cai + new_mai)
        tst = self.get_val('TST')
        if tst:
            hours = tst / 60.0
            for prefix, idx_name in [('O', 'OAI'), ('C', 'CAI'), ('M', 'MAI'), ('H', 'HI'), ('Total_Apnea', 'Total_Apnea_Index')]:
                idx_val = self.get_val(idx_name)
                if idx_val is not None:
                    self.set_val(f'{prefix}_events', idx_val * hours, is_int=True)


def perturb_measurements(measurements, target, value):
    """Apply the original report-generator cascade to numerical fields.

    TST and TRT are in minutes, indices in events/h and stage fractions in %.
    Updates preserve the source formatting: one decimal, two for BMI, and
    integer event counts. Only recorded fields are updated.
    """
    from math import isfinite
    if target not in measurements or measurements[target] is None:
        raise ValueError("The target must be recorded.")
    if isinstance(value, bool) or not isfinite(value) or value < 0:
        raise ValueError("The target must be finite and nonnegative.")
    if value == measurements[target]:
        raise ValueError("The scenario must change the target.")
    for key, number in measurements.items():
        if number is not None and (isinstance(number, bool) or not isfinite(number) or number < 0):
            raise ValueError(f"Invalid measurement: {key}")
    updater = _MeasurementUpdater(measurements)
    if target == 'TST':
        if value <= 0 or (measurements.get('TRT') is not None and value > measurements['TRT']):
            raise ValueError("TST must be positive and cannot exceed TRT.")
        updater.update_tst(value)
    elif target == 'SE':
        if not 0 < value <= 100 or not measurements.get('TRT') or not measurements.get('TST'):
            raise ValueError("SE requires TST, TRT and a target in (0, 100].")
        updater.update_se(value)
    elif target == 'AHI':
        if measurements['AHI'] <= 0:
            raise ValueError("Proportional AHI changes require a positive baseline.")
        updater.update_ahi(value)
    elif target == 'BMI':
        if value <= 0:
            raise ValueError("BMI must be positive.")
        updater.update_bmi(value)
    elif target in ['N1_pct', 'N2_pct', 'N3_pct', 'REM_pct']:
        compensate = 'N1_pct' if target == 'N2_pct' else 'N2_pct'
        if measurements.get(compensate) is None:
            raise ValueError("The compensating stage must be recorded.")
        remainder = measurements[compensate] + measurements[target] - value
        if value > 100 or not 0 <= remainder <= 100:
            raise ValueError("The target would produce invalid stage percentages.")
        updater.update_sleep_stage_pct(target.removesuffix('_pct'), value)
    else:
        raise ValueError("Unsupported target.")
    out = updater.values
    changes = [dict(field=key, baseline=measurements[key], counterfactual=number)
               for key, number in out.items() if number != measurements[key]]
    return out, changes
