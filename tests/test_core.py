import json
from pathlib import Path
import numpy as np
import pytest

from smart.metrics import age_gap, organ_asynchrony, summarize_predictions, concordance_correlation
from smart.entropy import reference_deviation, fit_entropy_cutpoints, homeostasis_entropy
from smart.prompts import build_age_prompt, build_recommendation_prompt, parse_age_response, load_template
from smart.scn import render_scn


def test_age_gap_preserves_sign():
    np.testing.assert_array_equal(age_gap([48,50,55],50),[-2,0,5])


def test_asynchrony_is_sample_sd_and_offset_invariant():
    gaps=np.array([[-2,0,2],[1,7,10.]])
    np.testing.assert_allclose(organ_asynchrony(gaps), [2,np.sqrt(21)])
    np.testing.assert_allclose(organ_asynchrony(gaps+60),organ_asynchrony(gaps))


def test_summary_excludes_overall_sleep_and_distinguishes_variation_axes():
    ages=np.array([[[90,50,54],[110,52,56.]]])
    out=summarize_predictions(ages,[50],['overall sleep biological age','cardiovascular age','renal age'])
    assert out['asynchrony'][0] == pytest.approx(np.sqrt(8))
    assert out['instability'][0] == pytest.approx(np.sqrt(2))
    np.testing.assert_allclose(out['mcse'][0],[10,1,1])
    np.testing.assert_allclose(out['age_gap'][0],[50,1,5])


def test_single_decoding_does_not_imply_zero_uncertainty():
    out=summarize_predictions([[[50,55]]],[49],['cardiovascular age','renal age'])
    assert np.isnan(out['decoding_sd']).all()
    assert np.isnan(out['instability']).all()
    assert out['asynchrony'][0] == pytest.approx(5/np.sqrt(2))


def test_incomplete_or_duplicated_domain_predictions_rejected():
    with pytest.raises(ValueError):
        summarize_predictions([[[50,np.nan]]],[50],['a','b'])
    with pytest.raises(ValueError):
        summarize_predictions([[[50,55]]],[50],['a','a'])


def test_ccc_identity_and_offset():
    assert concordance_correlation([1,2,3],[1,2,3]) == pytest.approx(1)
    assert concordance_correlation([1,2,3],[2,3,4]) == pytest.approx(4/7)
    assert np.isnan(concordance_correlation([1,1],[1,1]))


def test_reference_thresholds():
    np.testing.assert_allclose(reference_deviation([0,3,5,10],kind='upper',threshold=5,endpoint=0),[0,0,0,2])
    np.testing.assert_allclose(reference_deviation([100,85,70],kind='lower',threshold=85,endpoint=100),[0,0,-2])
    np.testing.assert_allclose(reference_deviation([270,360,450,540,630],kind='interval',lower=360,upper=540),[-1,0,0,0,1])
    assert np.isnan(reference_deviation([np.nan],kind='upper',threshold=5,endpoint=0)[0])
    with pytest.raises(ValueError):
        reference_deviation([5],kind='upper',threshold=5,endpoint=10)


def test_entropy_counts_normal_values_and_enforces_observations():
    x=np.array([[0,0,1,1,2,2,3,3],[0,0,0,0,0,0,0,0],[3,3,3,3,3,3,3,3],[0,1,2,3,np.nan,np.nan,np.nan,np.nan]])
    h=homeostasis_entropy(x,[1e-9,1.5,2.5])
    np.testing.assert_allclose(h[:3],[2,0,0])
    assert np.isnan(h[3])
    np.testing.assert_allclose(fit_entropy_cutpoints(np.arange(1,101)),
                               np.r_[1e-9,np.round(np.percentile(np.arange(1,101),[100/3,200/3]),6)])


def test_prompt_keeps_braces_in_input_and_validates_targets():
    text=build_age_prompt(['renal age','cardiovascular age'],'literal {input} and {target_age_types}')
    assert 'literal {input} and {target_age_types}' in text
    assert '"renal age": "float"' in text
    with pytest.raises(ValueError):
        build_age_prompt(['renal age','renal age'],'text')
    assert 'INPUT_2 (BIOLOGICAL_AGE_TABLE):\nBASELINE' in build_recommendation_prompt('SCN','BASELINE')


def test_response_requires_numeric_ages_and_all_fields():
    obj={'renal age':61.5}
    for suffix in [' aging causal chain',' youthful causal chain',' chain of thought']:
        obj['renal age'+suffix]='Example rationale for software testing.'
    assert parse_age_response('```json\n'+json.dumps(obj)+'\n```',['renal age'])['renal age']==61.5
    obj['renal age']=True
    with pytest.raises(ValueError):
        parse_age_response(json.dumps(obj),['renal age'])


def test_scn_preserves_unknown_and_recorded_negative():
    text=render_scn({'Medical history':{'Diabetes':'No reported history','Hypertension':None}})
    assert 'Diabetes: No reported history' in text
    assert 'Hypertension: unknown' in text


def test_public_and_installed_prompt_copies_are_identical():
    root=Path(__file__).resolve().parents[1]
    for name in ['age_estimation','recommendation']:
        assert (root/'smart/templates'/f'{name}.txt').read_text(encoding='utf-8')==load_template(name)


def test_all_synthetic_inputs_render_without_inventing_fields():
    root=Path(__file__).resolve().parents[1]/'examples'
    for path in root.glob('0*.json'):
        text=render_scn(json.loads(path.read_text(encoding='utf-8')))
        assert '[Sleep study measurements]' in text
        assert 'unknown' in text
