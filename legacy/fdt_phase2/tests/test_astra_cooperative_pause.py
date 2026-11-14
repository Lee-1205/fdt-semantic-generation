import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_pause_request_survives_optimizer_recovery_without_deleting_evidence(tmp_path):
    from run_astra_bounded_pilot import cooperative_pause,PilotStop
    from astra_bounded_pilot import write_recovery,read_recovery
    folder=tmp_path/'A';folder.mkdir()
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters())
    model(torch.ones(1,2)).sum().backward();opt.step()
    state={'step':1,'cursor':8,'supervised':365}
    cooperative_pause(folder,state)
    request=tmp_path/'STOP_REQUESTED.json';request.write_text('{"reason":"save now"}')
    with pytest.raises(PilotStop,match='COOPERATIVE_PAUSE'):cooperative_pause(folder,state)
    write_recovery(folder/'recovery.pt',model,opt,state,'same')
    restored=read_recovery(folder/'recovery.pt',model,opt,'same')
    assert restored['step']==1 and request.exists()
    cooperative_pause(folder,restored)
    request.write_text('{"reason":"a new stop request"}')
    with pytest.raises(PilotStop,match='COOPERATIVE_PAUSE'):cooperative_pause(folder,restored)


def test_training_pause_saves_latest_update_and_resume_does_not_replay_it(tmp_path,monkeypatch):
    import run_astra_bounded_pilot as p
    from astra_bounded_pilot import read_recovery
    monkeypatch.setattr(p,'load_model',lambda _:torch.nn.Linear(2,1))
    monkeypatch.setattr(p,'configure_training',lambda m,_:torch.optim.AdamW(m.parameters()))
    monkeypatch.setattr(p,'pending_gate',lambda *a:None)
    monkeypatch.setattr(p,'save_model',lambda path,*args:path.write_bytes(b'test model'))
    updates=[]
    def update(model,opt,entries,*args):
        opt.zero_grad();model(torch.ones(1,2)).sum().backward();opt.step()
        updates.append(len(entries))
        request=tmp_path/'STOP_REQUESTED.json'
        if not request.exists():request.write_text('{"reason":"test cooperative stop"}')
        return {},None,0
    monkeypatch.setattr(p,'update',update)
    rows=[{'supervised':1,'active':2} for _ in range(16)]
    plan={'seed':47,'parent':'unused','parent_sha256':'unchanged','rows':rows,'supervised':16}
    folder=tmp_path/'A'
    result=p.train_arm('A','row_mean',plan,folder,'identity',1e15,{}, {},[],[])
    assert result['paused'] and result['supervised']==8 and updates==[8]
    assert not (folder/'final_status.json').exists() and not (folder/'final.pt').exists()
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters())
    state=read_recovery(folder/'recovery.pt',model,opt,'identity')
    assert state['step']==1 and state['cursor']==8
    result=p.train_arm('A','row_mean',plan,folder,'identity',1e15,{}, {},[],[])
    assert result['stop']=='PLAN_COMPLETE' and result['supervised']==16
    assert updates==[8,8] and (tmp_path/'STOP_REQUESTED.json').exists()
