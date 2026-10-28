# Reproducibility

## Release coverage

SMART v0.3.0 provides the two task prompts, structured clinical narrative (SCN) examples, core aging-state calculations, counterfactual field updates and paired-bootstrap evaluation of saved predictions. Prediction generation, cohort extraction, full cohort analysis pipelines and mobile application code are outside this release.

## Installation

Use Python 3.11 or later. From the repository root:

```bash
python -m pip install -e ".[analysis,dev]"
python examples/run_demo.py
python -m pytest -q
```

For Jupyter, install `python -m pip install -e ".[analysis,dev,notebook]"` and open `examples/SMART_walkthrough.ipynb`. The walkthrough prepares three SCNs, task prompts and two counterfactual input scenarios in `outputs/walkthrough/`. All included examples and analysis commands run locally. Tested direct dependencies are listed in `requirements-tested.txt`.

## Using saved predictions

`examples/run_demo.py` accepts a JSON list of saved age-estimation responses. Each record follows the output schema in `smart/templates/age_estimation.txt`, with domains listed in `examples/age_domains.json`. The selected clinical example supplies the chronological age.

```bash
python examples/run_demo.py --predictions private_data/saved_predictions.json --case 01_respiratory_context
```

This writes mean ages, age gaps, decoding SDs, Monte Carlo standard errors, organ asynchrony and decoding instability. To prepare a recommendation prompt, supply an SCN and saved age estimates for the same individual: one baseline and at least nine distinct valid counterfactual scenarios. The prompt assigns three recommendations to each priority tier. The two walkthrough perturbations belong to different constructed individuals and are not a complete recommendation input. The builder accepts table text without checking participant identity or scenario count, so these requirements must be checked before use:

```bash
smart recommendation-prompt --input SCN.txt --table AGE_TABLE.csv --output recommendation_prompt.txt
```

The principal study backbone was Llama-3.3-70B-Instruct. Stochastic decoding used temperature 0.7 and top-p 0.9. Baseline and counterfactual comparisons used the same greedy settings. This repository consumes the resulting estimates.

## Clinical input examples

The three constructed examples use respiratory-event frequency, oxygenation, sleep duration, efficiency and routine sleep stages, together with demographics, BMI, blood pressure and brief clinical history. They contain no laboratory panels, questionnaire scores, spectral measures or participant records. The home-monitoring example reports REI per hour of recording time and leaves unmeasured sleep duration and architecture unknown.

`smart/scn.py` renders prepared fields under six headings. It preserves field values, units and the distinction between a recorded negative and an unknown history. It is a shared output formatter, not a replacement for the original cohort-specific extraction programs. The notebook's separate arithmetic exercise uses constructed numbers to illustrate the formulas.

## Sources of the released calculations

The analysis functions are portable adaptations of the following study code. File loading, plotting and cohort-specific column names are handled separately.

| Released component | Study source | Preserved calculation |
| --- | --- | --- |
| `smart/metrics.py` | Figure 5 signed-gap revision, `hi_source_copy.py`; SHHS/MrOS `load_bio_age`; `technical_plot_source.py` and `common.py` | Mean ages, age gaps, sample SD across organs, within-organ decoding SD, MCSE and Lin's CCC |
| `smart/entropy.py` | `patient_disorder_entropy_SHHS.ipynb` and `patient_disorder_entropy_MrOS.ipynb`, analysis cell 3 | Reference scaling, adaptive bin boundaries, fixed-bin fallback and Shannon entropy |
| `smart/counterfactual.py` | `counterfactual_report.ipynb`, `CascadeUpdater` | Linked TST, efficiency, stage, AHI and BMI updates, including intermediate rounding |
| `smart/bootstrap.py` | `Decoding_paired_bootstrap_20261013/code/paired_metrics.py`, unchanged in the 27 October SHHS revision | Paired participant resampling, tied-score rules and the three decoding summaries |
| ED9c summary table | `SHHS_followup_unified_20261027/code/analyse_ED9c.py` and `shared_survival.py` | Individual censoring times, common samples across strategies and fixed-score C-index evaluation |

The SCN renderer, command-line interface and input validation were added to package these calculations for reuse. The numerical comparison checks operate on the original computational functions without executing notebook file-loading or plotting cells.

## Aging-state definitions

Mean age is the arithmetic mean across repeated decodings. Age gap is mean predicted age minus chronological age. Organ asynchrony is the sample SD across organ-specific age gaps, with denominator M−1. Decoding instability is the mean of the within-organ sample SDs across decodings, with denominator K−1. Overall sleep age is excluded from both organ summaries. MCSE is the within-domain decoding SD divided by √K. All these quantities are in years.

The summary function accepts a complete participant × decoding × domain array. Resolve missing estimates before summarizing. A single decoding has no estimable decoding SD, instability or MCSE.

## Sleep homeostasis entropy

Reference departures are zero inside a reference interval or on the normal side of a single threshold. Interval departures are scaled by half the interval width. For an upper or lower threshold, the scale is half the distance to its fixed normal-side endpoint. Examples are AHI with threshold 5 and endpoint 0, and sleep efficiency with threshold 85 and endpoint 100.

`fit_entropy_cutpoints` returns the **complete bin boundaries**. With at least 80 nonzero departures, the original notebooks use a zero boundary of 1e-9 followed by nonzero-departure tertiles, rounded to six decimals. Repeated quantiles are collapsed. Smaller reference sets use the original fixed boundaries `[0.5, 1.5, 3.0]`. Exact ties enter the higher bin. The boundary array is passed directly to `homeostasis_entropy`.

Normal-range indicators remain in the calculation, missing indicators are omitted and the default five-indicator minimum matches the SHHS/MrOS merged-record filter. Entropy is expressed in bits. Cohort-specific indicator sets and reference settings must be supplied separately.

## Counterfactual field updates

The numerical adapter preserves the original `CascadeUpdater` operations:

| Modified field | Linked fields |
| --- | --- |
| TST | Efficiency at fixed recording time, stage durations, respiratory-event counts and arousal count |
| Sleep efficiency | TST at fixed recording time, followed by its linked fields |
| AHI | Apnea/hypopnea component indices, total apnea index and event counts |
| Stage percentage | Compensation in N2, or N1 when N2 changes, followed by stage durations |
| BMI | Weight at recorded height |

The source rounds updated measurements to one decimal place, BMI to two decimals and event counts to integers. The adapter retains that rounding at each update, so rounded component indices may differ slightly from their stated total. Only recorded fields are changed. The wrapper rejects invalid input ranges and stage combinations before applying the source operations.

The constructed TST example applies an additional timing constraint in `examples/run_demo.py`. It fixes recording time, sleep onset latency and terminal wake time, updating WASO from 90 to 30 min as TST increases from 300 to 360 min. This example-specific constraint is separate from the original report-generator cascade in `smart/counterfactual.py`.

`age_shift` calculates counterfactual minus baseline age. `age_reduction` calculates baseline minus counterfactual age. These are model-estimate comparisons, not measured treatment responses.

## Decoding comparisons

`analyses/data/fig4h_decoding_auc.csv` contains ten organ groups × three strategies, and `ed9c_decoding_cindex.csv` contains eight cardiovascular outcomes × three strategies. The 54 summary rows match the submitted comparisons. The ED9c table uses the revised 27 October 2026 analysis shown in the 28 October submission.

ED9c reports apparent Harrell C-indices from univariable Cox models using cardiovascular age gaps. Observed events use their recorded event times. Participants without an event use their own `censdate` from the SHHS cardiovascular summary. Invalid outcome status, observed events without an event date, and missing or nonpositive follow-up are excluded before defining a common complete-prediction sample for the three strategies within each outcome. The supplied counts are outcome-specific. Model fitting precedes resampling, and the bootstrap retains the fitted risk scores.

```bash
python analyses/replot.py
```

The strategies are greedy decoding, mean performance across ten individual stochastic decodings, and performance of participant-level mean predictions. `analyses/evaluate.py` recomputes 95% CIs using 2,000 paired participant bootstrap resamples. Each draw carries all predictions for the same participants. The middle strategy contributes one mean performance per draw. Confidence limits are the 2.5th and 97.5th percentiles.

Input columns are `participant_id`, `outcome`, `greedy` and `stochastic_01` through `stochastic_10`. For C-index, `outcome` is follow-up time and `event` gives the event indicator. Higher scores indicate greater risk.

```bash
python analyses/evaluate.py --input private_data/aligned_scores.csv --metric auc --output outputs/auc_bootstrap.csv
```

For fitted Cox models, also supply `ensemble_score` from the model fitted using mean ages. These functions resample fixed scores. Covariate adjustment, risk-set selection and model fitting precede this step.

## Prompts and artwork

The two prompt files retain the selected manuscript templates. The recommendation template uses an SCN and a labelled baseline/counterfactual age table. The age template supports a configurable list of output domains.

`assets/overview.png` is Figure 1a without its panel letter. `assets/mobile_interface.png` shows the mobile prototype. The README and optional HTML page share the approved SMART artwork. The MIT license covers code, documentation, prompt text and constructed examples. Manuscript artwork, interface images and third-party marks retain their respective rights.

## Data access

Public-cohort data remain subject to their original access conditions. Hospital-cohort data require institutional approval. Participant records and individual prediction tables are not redistributed in this package.

## Validation

The release is checked against the original entropy and cascade functions, the signed-gap and decoding-variability calculations, and the paired-bootstrap metric implementation. Tests cover normal-range observations, threshold ties, small reference samples, intermediate rounding, tied survival times and aligned participant resampling. The notebook and aggregate-plotting commands are executed before packaging.
