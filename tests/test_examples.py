from pathlib import Path
import copy,json,re
import numpy as np
import pytest
from smart.counterfactual import perturb_measurements,age_shift,age_reduction
from smart.entropy import homeostasis_entropy,fit_entropy_cutpoints

def test_reference_boundaries_enter_higher_category():
    # np.digitize(right=False) on [0,0.5,1,1.5,2] -> [0,1,2,2,3].
    p=np.array([1,1,2,1])/5
    expected=-np.sum(p*np.log2(p))
    assert homeostasis_entropy([[0,.5,1,1.5,2]],[1e-9,1,2])[0]==pytest.approx(expected)
    np.testing.assert_array_equal(fit_entropy_cutpoints([1,2,3]),[.5,1.5,3.])

def full_measurements():
    return dict(TST=360,TRT=450,SE=80,N1_pct=12,N2_pct=54,N3_pct=14,REM_pct=20,
                N1_time=43.2,N2_time=194.4,N3_time=50.4,REM_time=72,
                AHI=36,OAI=12,CAI=1,MAI=1,HI=22,Total_Apnea_Index=14,
                O_events=72,C_events=6,M_events=6,H_events=132,Total_Apnea_events=84,
                Arousal_index=30,Total_arousals=180,ODI=32,T90=10)

def test_tst_linked_fields_and_unrelated_fields():
    baseline=full_measurements();original=copy.deepcopy(baseline)
    out,_=perturb_measurements(baseline,'TST',420)
    assert baseline==original
    assert out['SE']==93.3
    assert out['REM_time']==84 and out['O_events']==84 and out['Total_arousals']==210
    assert out['ODI']==32 and out['T90']==10 and out['AHI']==36

def test_ahi_scaling_and_counts():
    out,_=perturb_measurements(full_measurements(),'AHI',6)
    assert out['OAI']==2 and out['HI']==3.7
    assert out['AHI']==6
    assert [out[k] for k in ['OAI','CAI','MAI','HI']]==[2,.2,.2,3.7]
    assert out['Total_Apnea_Index']==2.4
    assert out['Total_Apnea_events']==14 and out['O_events']==12
    assert out['TST']==360 and out['Total_arousals']==180

def test_stage_compensation_and_invalid_targets():
    out,_=perturb_measurements(full_measurements(),'N3_pct',20)
    assert out['N2_pct']==48 and out['N3_time']==72
    assert sum(out[s+'_pct'] for s in ['N1','N2','N3','REM'])==100
    with pytest.raises(ValueError):perturb_measurements(full_measurements(),'TST',600)
    with pytest.raises(ValueError):perturb_measurements(full_measurements(),'N3_pct',99)
    np.testing.assert_array_equal(age_shift([60,62],[57,63]),[-3,1])
    np.testing.assert_array_equal(age_reduction([60,62],[57,63]),[3,-1])

def test_psg_examples_are_internally_consistent_and_limited_input_has_no_staging():
    root=Path(__file__).resolve().parents[1]/'examples'
    for name in ['01_respiratory_context','02_sleep_continuity']:
        d=json.loads((root/(name+'.json')).read_text(encoding='utf-8'))['Sleep study measurements']
        def val(k):return float(re.match(r'^[0-9.]+',d[k])[0])
        assert val('Total recording time')==val('Total sleep time')+val('Sleep onset latency')+val('Wake after sleep onset')
        assert val('Sleep efficiency')==pytest.approx(100*val('Total sleep time')/val('Total recording time'),abs=.01)
        assert sum(val(s+' sleep') for s in ['N1','N2','N3','REM'])==100
        if 'Obstructive apnea index' in d:
            assert val('AHI')==sum(val(k) for k in ['Obstructive apnea index','Central apnea index','Mixed apnea index','Hypopnea index'])
    d=json.loads((root/'03_limited_measurements.json').read_text(encoding='utf-8'))['Sleep study measurements']
    assert 'AHI' not in d and d['Sleep architecture'] is None and d['Total sleep time'] is None
