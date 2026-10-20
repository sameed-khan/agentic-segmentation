import csv, json, glob, statistics as st, numpy as np
from scipy.stats import wilcoxon
from concurrent.futures import ProcessPoolExecutor
from segd.data import get_case, load_mask
a={r['case']:r for r in csv.DictReader(open('results/p6-sammed3d-high/scores.csv'))}
b={r['case']:r for r in csv.DictReader(open('results/phase3-high/scores.csv'))}
def ext(c):
    m=load_mask(get_case(c))>0; i=np.argwhere(m); return c,int((i.max(0)-i.min(0)+1).max())
with ProcessPoolExecutor(12) as ex: E=dict(ex.map(ext,sorted(a)))
big=[c for c in a if E[c]>192]; small=[c for c in a if E[c]<=192]
for nm,cs in [('>192mm',big),('<=192mm',small)]:
    x=[float(a[c]['dice']) for c in cs]; y=[float(b[c]['dice']) for c in cs]
    print(nm,len(cs),'sammed3d %.3f nnint %.3f diff %+.3f p=%.2g'%(st.mean(x),st.mean(y),st.mean(x)-st.mean(y),wilcoxon(x,y).pvalue))
from collections import Counter
ops=Counter()
for f in glob.glob('results/p6-sammed3d-high/*/trajectory.jsonl'):
    for l in open(f):
        t=json.loads(l)
        if t['op'] in ('point','box','lasso','scribble','refine') and not t.get('error'): ops[t['op']]+=1
print('model-prompt ops',dict(ops), 'emulated share %.2f'%((ops['box']+ops['lasso']+ops['scribble'])/sum(ops.values())))
ww=[json.load(open(f))['wall_time_s'] for f in glob.glob('results/p6-sammed3d-high/*/meta.json')]
print('wall median %.0f'%st.median(ww))
