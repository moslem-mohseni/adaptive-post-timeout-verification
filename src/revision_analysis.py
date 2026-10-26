"""Secondary risk intervals and fresh-stream sensitivity summaries."""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stats_utils import mean_ci,paired_diff_ci,wilson

ROOT=Path(__file__).resolve().parents[1]

def main():
    rows=[]
    for experiment,group in [('e1_main','policy'),('e5_retry_loss','retry_loss'),('e6_heavy_tail','sigma_factor')]:
        frame=pd.read_csv(ROOT/'results'/f'{experiment}.csv')
        keys=['policy'] if group=='policy' else [group,'policy']
        for values,g in frame.groupby(keys):
            values=values if isinstance(values,tuple) else (values,)
            row=dict(experiment=experiment,**dict(zip(keys,values)),n=len(g))
            row['cost'],row['cost_lo'],row['cost_hi']=mean_ci(g.cost,n_boot=2000,seed=852)
            for metric in ['dup','false_success','retry_missing','premature_success','escalated']:
                row[metric],row[metric+'_lo'],row[metric+'_hi']=wilson(int(g[metric].sum()),len(g))
            rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'results/revision_risks.csv',index=False)
    contrasts=[]
    for experiment,factor in [('e5_retry_loss','retry_loss'),('e6_heavy_tail','sigma_factor')]:
        frame=pd.read_csv(ROOT/'results'/f'{experiment}.csv')
        for value,g in frame.groupby(factor):
            ad=g[g.policy=='adaptive'].set_index('seed').cost
            for policy in sorted(set(g.policy)-{'adaptive','oracle'}):
                other=g[g.policy==policy].set_index('seed').cost.loc[ad.index]
                effect,lo,hi,dz=paired_diff_ci(ad,other,n_boot=2000,seed=852)
                contrasts.append(dict(experiment=experiment,factor=factor,value=value,comparator=policy,diff=effect,lo=lo,hi=hi,dz=dz))
    pd.DataFrame(contrasts).to_csv(ROOT/'results/revision_contrasts.csv',index=False)
    main_frame=pd.read_csv(ROOT/'results/e1_main.csv')
    pricing=[]
    for penalty in (0.,25.,100.):
        for policy,g in main_frame.groupby('policy'):
            cost=g.cost+penalty*g.premature_success
            mean,lo,hi=mean_ci(cost,n_boot=2000,seed=852)
            pricing.append(dict(penalty=penalty,policy=policy,cost=mean,lo=lo,hi=hi))
    pd.DataFrame(pricing).to_csv(ROOT/'results/premature_pricing.csv',index=False)
    print('Secondary risk intervals and paired sensitivity contrasts saved')

if __name__=='__main__':main()
