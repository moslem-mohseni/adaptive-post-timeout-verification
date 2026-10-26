"""Separate model-driven post-timeout decisions with exact simulator scoring."""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from world import sample_service,true_episode
from policies import Ep,Adaptive,FixedWait,ContractGuard,ImmediateRetry,make_ctx,COSTS
from client import LLMClient,discover
from stats_utils import paired_diff_ci,sign_flip_p,holm,wilson

def parse_action(text):
    value=json.loads(text)
    action=value['action']
    if action not in ('wait','probe','evidence','success','retry','escalate'):
        raise ValueError('unknown action')
    seconds=value.get('seconds',0.)
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<=seconds<=24:
        raise ValueError('wait must be finite and within 0 to 24 seconds')
    return action,float(seconds)

def model_decision(ep,client,seed):
    svc=ep.svc
    contract=dict(status_lag_mean=svc.lag_mean,idempotency_documented=svc.idem_doc,inflight_bound=svc.bound_doc)
    prompt='A synthetic write timed out at time 2 after dispatch at 0. It may have been lost, committed, or still be in flight. Choose JSON {"action":"wait|probe|evidence|success|retry|escalate","seconds":0}. Status reads never falsely report present but can lag; independent evidence sees commits so far. Documentation can be wrong. A success claim before commit is premature even if the write later completes. Costs are abstract, not dollars: '+json.dumps(COSTS,sort_keys=True)+'. Contract: '+json.dumps(contract,sort_keys=True)
    messages=[dict(role='user',content=prompt)]
    for step in range(8):
        text=client.complete(messages,seed,temperature=0)
        messages.append(dict(role='assistant',content=text))
        try:action,seconds=parse_action(text)
        except (ValueError,KeyError,TypeError):return 'escalate',step+1,True
        if action in ('success','retry','escalate'):return action,step+1,False
        if action=='wait':ep.wait(seconds);observation='waited'
        elif action=='probe':observation='present' if ep.probe() else 'absent'
        else:observation='present' if ep.evidence() else 'absent'
        messages.append(dict(role='user',content=json.dumps(dict(time=ep.t,observation=observation))))
    return 'escalate',8,False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',default='mock');ap.add_argument('--model',default='mock')
    ap.add_argument('--episodes',type=int,default=200)
    ap.add_argument('--input-price-per-million',type=float);ap.add_argument('--output-price-per-million',type=float)
    args=ap.parse_args()
    if not 100<=args.episodes<=300:raise ValueError('Study 2 protocol requires 100 to 300 episodes')
    out=HERE.parents[1]/'results/llm';out.mkdir(parents=True,exist_ok=True)
    client=LLMClient(args.backend,args.model,out/'cache',budget_usd=5,max_tokens=256,
                     price_input_per_million=args.input_price_per_million,price_output_per_million=args.output_price_per_million)
    best_wait=float((HERE.parents[1]/'results/best_wait.txt').read_text())
    rows=[]
    baselines=[Adaptive(),FixedWait(best_wait),ContractGuard(),ImmediateRetry()]
    for seed in range(70000,70000+args.episodes):
        rng=np.random.default_rng(seed);svc=sample_service(rng);truth=true_episode(rng,svc)
        ep=Ep(svc,truth);decision,calls,invalid=model_decision(ep,client,seed)
        row=ep.finish(decision);row.update(seed=seed,policy='model',calls=calls,invalid=int(invalid),mock=args.backend=='mock');rows.append(row)
        for policy in baselines:
            ep=Ep(svc,truth)
            decision=policy.run(ep,make_ctx(np.random.default_rng(seed*7+1),svc))
            row=ep.finish(decision);row.update(seed=seed,policy=policy.name,calls=0,invalid=0,mock=args.backend=='mock');rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/('dry_run.csv' if args.backend=='mock' else 'episodes.csv'),index=False)
    status=dict(real_llm_run=args.backend!='mock',backend=args.backend,model=args.model,temperature=0,
                episodes=args.episodes,availability=discover(),local_runtime_attempt=json.loads((HERE/'availability_attempt.json').read_text()))
    (out/'status.json').write_text(json.dumps(status,indent=2))
    if args.backend!='mock':
        model=frame[frame.policy=='model'].set_index('seed');contrasts=[]
        for policy in baselines:
            other=frame[frame.policy==policy.name].set_index('seed').loc[model.index]
            effect,lo,hi,dz=paired_diff_ci(model.cost,other.cost)
            contrasts.append(dict(comparator=policy.name,cost_diff=effect,lo=lo,hi=hi,dz=dz,p=sign_flip_p(model.cost,other.cost)))
        result=pd.DataFrame(contrasts);result['p_holm']=holm(result.p)
        result.to_csv(out/'paired_costs.csv',index=False)
        risks=[]
        for metric in ('dup','false_success','premature_success'):
            mean,lo,hi=wilson(int(model[metric].sum()),len(model));risks.append(dict(metric=metric,mean=mean,lo=lo,hi=hi))
        pd.DataFrame(risks).to_csv(out/'model_risks.csv',index=False)
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
