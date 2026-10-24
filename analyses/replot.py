"""Replot released aggregate estimates without refitting or recomputing their CIs.

Source: Fig. 4h and Extended Data Fig. 9c paired-bootstrap result tables.
This produces an analysis preview, not the manually assembled manuscript artwork.
All source rows are retained. Error bars are the stored participant-bootstrap 95% CIs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#5679A6', '#C8A24A', '#8C5C9E']
DISPLAY = ['Greedy decoding', 'Single stochastic decoding (mean performance)',
           'Mean stochastic prediction']


def draw(path, destination):
    data = pd.read_csv(path)
    outcomes = data.endpoint.drop_duplicates().tolist()
    strategies = data.strategy.drop_duplicates().tolist()
    if len(strategies) != 3 or data.duplicated(['endpoint', 'strategy']).any():
        raise ValueError('Expected one row per outcome and decoding strategy.')
    if len(data) != 3 * len(outcomes) or not data.bootstrap_resamples.eq(2000).all():
        raise ValueError('Expected complete three-strategy data with 2,000 resamples.')
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10, 'svg.fonttype':'none', 'pdf.fonttype':42,
                         'axes.edgecolor':'black', 'axes.labelcolor':'black',
                         'xtick.color':'black','ytick.color':'black','text.color':'black',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(12, 4.0))
    x = np.arange(len(outcomes))
    for j, name in enumerate(strategies):
        part = data[data.strategy.eq(name)].set_index('endpoint').loc[outcomes]
        position = x + (j-1)*0.23
        ax.bar(position, part.estimate, width=0.21, color=COLORS[j],
               edgecolor='black', linewidth=0.5, label=DISPLAY[j], zorder=2)
        ax.errorbar(position, part.estimate,
                    yerr=np.vstack([part.estimate-part.lower, part.upper-part.estimate]),
                    fmt='none', ecolor='black', elinewidth=0.8, capsize=0, zorder=3)
    ax.set_xticks(x, outcomes, rotation=28, ha='right')
    ax.set_ylabel(str(data.metric.iloc[0]))
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.55, len(outcomes)-0.45)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,1.01), ncol=3,
              frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.8)
    fig.subplots_adjust(left=0.065,right=0.995,bottom=0.27,top=0.83)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(destination.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(destination.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    for stem in ['fig4h_decoding_auc','ed9c_decoding_cindex']:
        draw(ROOT/'analyses/data'/f'{stem}.csv', ROOT/'outputs/summary_figures'/stem)
    print('Replotted all released aggregate rows. Stored estimates and CIs are unchanged.')
