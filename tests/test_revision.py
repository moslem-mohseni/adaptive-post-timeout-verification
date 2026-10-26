"""Effect-ledger accounting and fresh-stream sensitivity invariants."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from policies import Ep,Oracle,Adaptive
from world import Service,true_episode,T_TIMEOUT

def service():
    return Service('slow_backend',0.0,False,False,None,False)

def test_retry_loss_does_not_create_duplicate_and_penalizes_missing_effect():
    absent=dict(lost=True,c=5.,lag=0.,retry_lost=True)
    row=Ep(service(),absent,retry_loss=1.).finish('retry')
    assert row['retry_missing']==1 and row['dup']==0 and row['cost']==101 and row['eventual_effects']==0
    assert Oracle().run(Ep(service(),absent),{})=='escalate'
    late=dict(lost=False,c=5.,lag=0.,retry_lost=True)
    assert Ep(service(),late,retry_loss=1.).finish('retry')['dup']==0
    late['retry_lost']=False
    assert Ep(service(),late).finish('retry')['eventual_effects']==2

def test_premature_success_is_distinct_from_permanently_false_claim():
    row=Ep(service(),dict(lost=False,c=5.,lag=0.)).finish('success')
    assert row['premature_success']==1 and row['false_success']==0
    assert Ep(service(),dict(lost=True,c=5.,lag=0.)).finish('success')['false_success']==1

def test_oracle_does_not_pay_for_unnecessary_idempotent_retry():
    svc=Service('ack_loss',0.,True,True,None,False)
    ep=Ep(svc,dict(lost=False,c=.1,lag=0.))
    assert Oracle().run(ep,{})=='success'
    assert ep.finish('success')['cost']==0

def test_timeout_condition_retains_real_acknowledgement_evidence():
    for seed in range(100):
        tr=true_episode(np.random.default_rng(seed),service(),sigma_factor=2.)
        assert tr['lost'] or tr['ack_lost'] or tr['c']+tr['ack_delay']>T_TIMEOUT

def test_retry_loss_changes_expected_terminal_risk():
    controller=Adaptive()
    from policies import COSTS
    assert controller._terminal(np.ones(100,dtype=bool),COSTS,0.,0.)==(1.,'retry')
    assert controller._terminal(np.ones(100,dtype=bool),COSTS,0.,1.)==(25.,'escalate')
