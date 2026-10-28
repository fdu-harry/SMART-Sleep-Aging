"""Offline input walkthrough; optional summarization of genuine saved predictions."""
from pathlib import Path
import argparse, json, sys, re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smart.scn import render_scn
from smart.prompts import build_age_prompt, parse_age_response
from smart.metrics import summarize_predictions
from smart.counterfactual import perturb_measurements

FIELD_MAP = {'AHI':'AHI','Total sleep time':'TST','Total recording time':'TRT',
             'Sleep efficiency':'SE','Obstructive apnea index':'OAI',
             'Central apnea index':'CAI','Mixed apnea index':'MAI','Hypopnea index':'HI',
             'N1 sleep':'N1_pct','N2 sleep':'N2_pct','N3 sleep':'N3_pct',
             'REM sleep':'REM_pct','Arousal index':'Arousal_index'}

def scenario(record,target,value):
    """Apply linked changes to the atomic fields of the illustrated PSG inputs."""
    source=record['Sleep study measurements']
    numeric={alias:float(re.match(r'^[0-9.]+',source[field])[0])
             for field,alias in FIELD_MAP.items() if source.get(field) is not None}
    modified,audit=perturb_measurements(numeric,target,value)
    result=json.loads(json.dumps(record))
    for field,alias in FIELD_MAP.items():
        if alias in modified and modified[alias] != numeric[alias]:
            result['Sleep study measurements'][field]=re.sub(r'^[0-9.]+',f'{modified[alias]:.6g}',source[field],count=1)
    if target in {'TST', 'SE'}:
        timing = ('Total recording time', 'Total sleep time',
                  'Sleep onset latency', 'Wake after sleep onset')
        if all(source.get(field) is not None for field in timing):
            baseline = {field: float(re.match(r'^[0-9.]+', source[field])[0])
                        for field in timing}
            terminal_wake = (baseline['Total recording time']
                             - baseline['Total sleep time']
                             - baseline['Sleep onset latency']
                             - baseline['Wake after sleep onset'])
            if terminal_wake < -1e-6:
                raise ValueError('The constructed example has inconsistent sleep timing.')
            # In these constructed inputs, recording time spans lights off to lights on.
            # Hold latency and terminal wake fixed and let WASO absorb the TST change.
            new_waso = (baseline['Wake after sleep onset']
                        + baseline['Total sleep time'] - modified['TST'])
            if new_waso < -1e-6:
                raise ValueError('The TST scenario leaves insufficient time for the fixed sleep latency and terminal wake.')
            new_waso = max(0.0, new_waso)
            if new_waso != baseline['Wake after sleep onset']:
                field = 'Wake after sleep onset'
                result['Sleep study measurements'][field] = re.sub(
                    r'^[0-9.]+', f'{new_waso:.6g}', source[field], count=1)
                audit.append(dict(field='WASO', baseline=baseline[field],
                                  counterfactual=new_waso))
    return result,audit

def run(output_dir=None,predictions=None,case='01_respiratory_context'):
    dest=Path(output_dir) if output_dir else ROOT/'outputs/walkthrough'
    dest.mkdir(parents=True,exist_ok=True)
    domains=json.loads((ROOT/'examples/age_domains.json').read_text(encoding='utf-8-sig'))
    for path in sorted((ROOT/'examples').glob('0*.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        scn=render_scn(record)
        (dest/(path.stem+'.txt')).write_text(scn,encoding='utf-8')
        (dest/(path.stem+'_age_prompt.txt')).write_text(build_age_prompt(domains,scn),encoding='utf-8')
    specifications=[('01_respiratory_context','AHI',5),('02_sleep_continuity','TST',360)]
    audits=[]
    for case_name,target,value in specifications:
        record=json.loads((ROOT/'examples'/f'{case_name}.json').read_text(encoding='utf-8'))
        changed,audit=scenario(record,target,value)
        scn=render_scn(changed)
        (dest/(case_name+'_counterfactual.txt')).write_text(scn,encoding='utf-8')
        (dest/(case_name+'_counterfactual_prompt.txt')).write_text(build_age_prompt(domains,scn),encoding='utf-8')
        audits.extend([dict(example=case_name,target=target,**row) for row in audit])
    pd.DataFrame(audits).to_csv(dest/'input_change_audit.csv',index=False)
    if predictions:
        # Load responses in the domain order specified by age_domains.json.
        records=json.loads(Path(predictions).read_text(encoding='utf-8'))
        validated=[parse_age_response(json.dumps(x),domains) for x in records]
        age=float(json.loads((ROOT/'examples'/f'{case}.json').read_text(encoding='utf-8'))['Basic information']['Age'].split()[0])
        tensor=np.array([[r[d] for d in domains] for r in validated])[None,:,:]
        summary=summarize_predictions(tensor,[age],domains)
        pd.DataFrame({'domain':domains,'mean_age':summary['mean_age'][0],
                      'age_gap':summary['age_gap'][0],'decoding_sd':summary['decoding_sd'][0],
                      'mcse':summary['mcse'][0]}).to_csv(dest/'model_age_summary.csv',index=False)
        with (dest/'model_state.json').open('w',encoding='utf-8') as f:
            json.dump({'case':case,'n_decodings':len(validated),
                       'asynchrony_years':float(summary['asynchrony'][0]),
                       'instability_years':float(summary['instability'][0]) if len(validated)>1 else None},f,indent=2,allow_nan=False)
    print(f'Prepared three constructed SCNs, age prompts and two linked-input scenarios in {dest}')
    if not predictions:
        print('Use --predictions to load saved model responses and calculate aging-state summaries.')
    return dest

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--case',default='01_respiratory_context',choices=['01_respiratory_context','02_sleep_continuity','03_limited_measurements'])
    args=parser.parse_args()
    run(args.output_dir,args.predictions,args.case)
