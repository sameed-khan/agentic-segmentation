"""Offline: recompute the candidates shown in each finish review of a run (deterministic, GT-free generator)
and score them against GT; compare with what the agent accepted. Usage: edit RUN, then
uv run --project harness/server python eval/v2/candidate_judgement.py"""
import json, sys, numpy as np
sys.path.insert(0,'eval/v2')
import score as S
from segd.data import get_case, load_image, load_mask
from segd import candidates as C
from segd.marks import body_mask
RUN='results/phase1-10-high-cand'
tot=dict(shown=0,good=0,acc=0,acc_good=0); gains=[]
for row in [get_case(c) for c in sorted(__import__('os').listdir(RUN)) if c[:1] in 'FLMNV' and len(c)==3]:
    t=[json.loads(l) for l in open(f'{RUN}/{row.case}/trajectory.jsonl')]
    revs=[x for x in t if x['op']=='finish' and x.get('candidates')]
    if not revs: continue
    img,aff,sp=load_image(row); gt=load_mask(row)>0; z=np.load(f'{RUN}/{row.case}/versions.npz')
    accepted=set(sum([x['args'].get('ids',[]) for x in t if x['op']=='accept' and not x.get('error')],[]))
    body=body_mask(img)
    for r in revs:
        vid=int(str(r['parent']).lstrip('v')); m=S.unpack_version(z,vid,img.shape)
        if m is None: continue
        cands=C.propose(img,m,sp,method='intensity',max_n=5,cache={'body':body})
        start=S.dice(m,gt) if hasattr(S,'dice') else None
        good=[c for c in cands if (c.region&gt).sum()/max(c.region.sum(),1)>0.5]
        newm=m.copy()
        for c in good: newm|=c.region
        d0=2*(m&gt).sum()/(m.sum()+gt.sum()); d1=2*(newm&gt).sum()/(newm.sum()+gt.sum())
        info=[(c.id, round(c.volume_ml,1), round((c.region&gt).sum()/max(c.region.sum(),1),2)) for c in cands]
        print(row.case, 'v%d'%vid, 'dice %.3f oracle-accept %.3f'%(d0,d1), info, 'agent accepted', sorted(accepted))
        gains.append(d1-d0); tot['shown']+=len(cands); tot['good']+=len(good)
print(tot, 'mean oracle gain over reviews %.3f'%np.mean(gains), 'n reviews', len(gains))
