import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from smart.bootstrap import paired_decoding_bootstrap


def test_three_auc_estimands_and_paired_resamples_match_direct_calculation():
    y=np.repeat([0,1],8)
    rng=np.random.default_rng(7)
    g=rng.integers(0,5,16).astype(float)
    s=rng.integers(0,5,(16,3)).astype(float)
    result=paired_decoding_bootstrap(y,g,s,participant_ids=np.arange(16),n_resamples=20,seed=11)
    expected=[roc_auc_score(y,g),np.mean([roc_auc_score(y,s[:,k]) for k in range(3)]),roc_auc_score(y,s.mean(axis=1))]
    np.testing.assert_allclose(result.estimates,expected)
    manual=[]
    rng=np.random.default_rng(11)
    while len(manual)<20:
        idx=rng.integers(0,16,16)
        if np.unique(y[idx]).size < 2:
            continue
        manual.append([roc_auc_score(y[idx],g[idx]),np.mean([roc_auc_score(y[idx],s[idx,k]) for k in range(3)]),roc_auc_score(y[idx],s.mean(axis=1)[idx])])
    np.testing.assert_allclose(result.samples,manual)
    np.testing.assert_allclose([result.lower,result.upper],np.quantile(manual,[.025,.975],axis=0))


def test_cindex_sign_ties_censoring_and_external_ensemble_score():
    t=np.array([1,2,2,3,4,5,6,7.])
    event=np.array([1,1,0,1,0,1,1,0])
    g=np.array([8,7,6,4,4,2,2,1.])
    s=np.column_stack([g,g+1,g-1])
    ensemble=g**2
    result=paired_decoding_bootstrap(t,g,s,participant_ids=np.arange(8),metric='cindex',event=event,
                                     ensemble_score=ensemble,n_resamples=20,seed=6)
    assert result.estimates[0]==pytest.approx(concordance_index(t,-g,event))
    assert result.estimates[2]==pytest.approx(concordance_index(t,-ensemble,event))
    np.testing.assert_allclose(result.samples[:,0],result.samples[:,1])


def test_duplicate_ids_or_missing_scores_rejected():
    with pytest.raises(ValueError):
        paired_decoding_bootstrap([0,0,1,1],[1,2,3,4],np.ones((4,2)),participant_ids=['a','a','b','c'],n_resamples=5)
    with pytest.raises(ValueError):
        paired_decoding_bootstrap([0,0,1,1],[1,2,np.nan,4],np.ones((4,2)),participant_ids=np.arange(4),n_resamples=5)


def test_single_class_bootstraps_rejected_for_all_strategies():
    r=paired_decoding_bootstrap([0,1],[0,1],np.array([[0,0],[1,1.]]),participant_ids=['a','b'],n_resamples=20,seed=4)
    assert r.rejected_draws > 0
    np.testing.assert_array_equal(r.samples,np.ones((20,3)))
