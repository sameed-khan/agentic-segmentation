import csv, numpy as np, nibabel as nib, sys
from scipy import ndimage as ndi
from segd.data import get_case, load_mask
from concurrent.futures import ProcessPoolExecutor
RUNS=['phase1-07-xhigh','phase1-09-xhigh-ex','phase1-10-high-cand','phase1-06-high']
R=[0,1,2,3,4,5]
def one(arg):
    run,case=arg
    gt=load_mask(get_case(case))>0
    p=np.asarray(nib.as_closest_canonical(nib.load(f'results/{run}/{case}/final_mask.nii.gz')).dataobj)>0
    out=[]
    ball=ndi.generate_binary_structure(3,1)
    for r in R:
        q=ndi.binary_dilation(p,ball,iterations=r) if r else p
        s=q.sum()+gt.sum(); out.append(2*(q&gt).sum()/s if s else 0)
    return run,case,out
args=[(r,c['case']) for r in RUNS for c in csv.DictReader(open(f'results/{r}/scores.csv'))]
with ProcessPoolExecutor(16) as ex: res=list(ex.map(one,args))
for run in RUNS:
    rr=[x[2] for x in res if x[0]==run]
    print(run, ' '.join(f'r{r}:{np.mean([v[i] for v in rr]):.3f}' for i,r in enumerate(R)))
