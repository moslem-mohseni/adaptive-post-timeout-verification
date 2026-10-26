from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/llm_study'))
from run import parse_action,model_decision
from client import LLMClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from policies import Ep
from world import Service

@pytest.mark.parametrize('text',['{"action":"wait","seconds":-1}','{"action":"wait","seconds":true}','{"action":"destroy"}','{"action":"wait","seconds":1e100}'])
def test_invalid_actions_cannot_mutate_clock(text):
    with pytest.raises((ValueError,KeyError)):parse_action(text)

def test_mock_invalid_response_escalates_without_effect(tmp_path):
    ep=Ep(Service('ack_loss',0.,False,False,None,False),dict(lost=True,c=1.,lag=0.))
    assert model_decision(ep,LLMClient(cache_dir=tmp_path/'cache'),1)==('escalate',1,True)
    assert ep.t==2 and ep.finish('escalate')['eventual_effects']==0
