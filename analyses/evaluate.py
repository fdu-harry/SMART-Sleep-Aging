"""Evaluate aligned participant scores using 2,000 paired bootstrap resamples.

CSV columns: participant_id, outcome, greedy, stochastic_01 ... stochastic_10.
For C-index add event. Optional ensemble_score supplies the risk score fitted
from mean ages, which must be provided when comparing fitted Cox models.
For repeated visits, first define one index observation per participant.
"""
from pathlib import Path
import argparse,sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smart.bootstrap import paired_decoding_bootstrap

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--metric',choices=['auc','cindex'],required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/bootstrap.csv')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    df=pd.read_csv(args.input,dtype={'participant_id':str})
    cols=[f'stochastic_{k:02d}' for k in range(1,11)]
    required=['participant_id','outcome','greedy']+cols+(['event'] if args.metric=='cindex' else [])
    missing=set(required)-set(df)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    if df['participant_id'].isna().any():raise ValueError('Participant identifiers cannot be missing.')
    result=paired_decoding_bootstrap(df.outcome.to_numpy(),df.greedy.to_numpy(),df[cols].to_numpy(),
                                    participant_ids=df.participant_id.to_numpy(),metric=args.metric,
                                    event=df.event.to_numpy() if args.metric=='cindex' else None,
                                    ensemble_score=df.ensemble_score.to_numpy() if 'ensemble_score' in df else None,
                                    n_resamples=2000,seed=args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(result.rows()).to_csv(args.output,index=False)
    np.save(args.output.with_suffix('.bootstrap.npy'),result.samples)
    print(f'Saved {len(result.samples):,} paired resamples for {result.n_participants:,} participants.')

if __name__=='__main__':main()
