import os, numpy as np, pandas as pd, torch, json
from scipy.stats import wilcoxon
SRC='/mnt/data/d2_confirmatory/run_d2_confirmatory.py'
text=open(SRC).read().replace('@njit(cache=True)','@njit(cache=False)'); prefix=text.split("print('Collecting training data...'",1)[0]
ns={}; exec(prefix,ns)
# aliases
SCENARIOS=ns['SCENARIOS']; TRAIN_SCENARIOS=ns['TRAIN_SCENARIOS']; TRAIN_SEEDS=ns['TRAIN_SEEDS']; TEST_SEEDS=ns['TEST_SEEDS']; STEPS=ns['STEPS']; WARMUP=ns['WARMUP']; DT=ns['DT']; UMAX=ns['UMAX']; RES_MAX=ns['RES_MAX']; AUTH=ns['AUTH']; SEED=ns['SEED']
collect=ns['collect']; train_ae=ns['train_ae']; enc_t=ns['enc_t']; train_pred=ns['train_pred']; CeNN=ns['CeNN']; latt=ns['latt']; scenario=ns['scenario']; FeatureState=ns['FeatureState']; geo=ns['geo']; PREF=ns['PREF']; VREF=ns['VREF']
# training data exactly same seed/scenario split; current-only samples use same even-k post-warm-up ordering
print('collecting current-only capacity training data...',flush=True)
TT=[];YY=[]
for sc in TRAIN_SCENARIOS:
    t,y=collect(sc,TRAIN_SEEDS); TT.append(t); YY.append(y)
TT=np.concatenate(TT,1); YY=np.concatenate(YY,1); tm=TT.reshape(-1,3).mean(0); ts=TT.reshape(-1,3).std(0)+1e-6
Cs=[];Ys=[]
for ei in range(TT.shape[1]):
    for k in range(STEPS):
        if k%2==0 and k>=WARMUP:
            Cs.append((TT[k,ei]-tm)/ts); Ys.append(YY[k,ei])
C=np.asarray(Cs);Y=np.asarray(Ys); rng=np.random.default_rng(SEED); idx=rng.choice(len(C),min(32000,len(C)),replace=False); C=C[idx];Y=Y[idx]
ccells=C.reshape(-1,3);ccells=ccells[rng.choice(len(ccells),min(70000,len(ccells)),replace=False)]
# Active inference encoder params: 3->64 (256) + 64->3 (195) =451, plus CeNN43 =494, exactly D2 active path.
ae=train_ae(ccells,3,64,3,71,6); z=enc_t(ae,C.reshape(-1,3)).reshape(len(C),9,3); zm=z.reshape(-1,3).mean(0); zs=z.reshape(-1,3).std(0)+1e-6; ce=train_pred(CeNN(),latt((z-zm)/zs),Y,72,7)
# numpy export
linpar=lambda l:(l.weight.detach().numpy(),l.bias.detach().numpy())
def ex_ae(m):return [linpar(m.e1),linpar(m.e2),linpar(m.d1),linpar(m.d2)]
def ae_np(par,x,encode=False):
    (w1,b1),(w2,b2),(w3,b3),(w4,b4)=par;h=np.tanh(x@w1.T+b1);zz=np.tanh(h@w2.T+b2)
    if encode:return zz
    return np.tanh(zz@w3.T+b3)@w4.T+b4
def ex_c(m):return [m.A.detach().numpy(),m.B.detach().numpy(),float(m.b.detach()),m.g.detach().numpy(),m.ob.detach().numpy()]
def conv_np(x,k):
    xp=np.pad(x,((0,0),(1,1),(1,1)));out=np.zeros_like(x)
    for i in range(3):
        for j in range(3):out+=k[i,j]*xp[:,i:i+3,j:j+3]
    return out
def cenn_np(par,inp):
    A,B,b,g,ob=par;x=np.zeros((len(inp),3,3));ff=np.zeros_like(x)
    for c in range(3):ff+=conv_np(inp[:,c],B[c])
    for _ in range(4):x+=.32*(-x+conv_np(np.tanh(x),A)+ff+b)
    return np.tanh(np.tanh(x)[:,2,:]*g+ob)*RES_MAX
AE=ex_ae(ae);CE=ex_c(ce)
def pred(temp):
    tn=(temp-tm)/ts;zz=ae_np(AE,tn.reshape(-1,3),True).reshape(len(tn),9,3);zz=(zz-zm)/zs;return AUTH*np.clip(cenn_np(CE,latt(zz)),-RES_MAX,RES_MAX)

def run(name,seeds):
    sc=scenario(name,seeds);n=len(seeds);p=np.repeat(PREF[0][None],n,0);v=np.repeat(VREF[0][None],n,0)
    for j,s in enumerate(seeds):
        rr=np.random.default_rng(700000+int(s)*19+sum(map(ord,name))*5);p[j]+=rr.normal(0,.012,3);v[j]+=rr.normal(0,.018,3)
    fs=FeatureState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];errs=np.zeros((STEPS,n));eff=np.zeros((STEPS,n));sat=np.zeros((STEPS,n));res=np.zeros((STEPS,n))
    for k in range(STEPS):
        te=fs.feature(p,v,k,sc['pn'][k],sc['vn'][k]);base=geo(p,v,k);r=pred(te);raw=base+r;u=np.clip(raw,-UMAX,UMAX);fs.cmd(u);sat[k]=np.any(np.abs(raw)>=UMAX-1e-12,axis=1);q.append(u.copy());q.pop(0);ud=np.array([q[-1-int(d)][j] for j,d in enumerate(sc['delay'])]);act+=(DT/.055)*(ud-act);a=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]+sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);v+=DT*a;p+=DT*v;errs[k]=np.linalg.norm(p-PREF[k],axis=1);eff[k]=np.sum(u*u,1);res[k]=np.linalg.norm(r,axis=1)
    rows=[]
    for j,s in enumerate(seeds):
        e=errs[WARMUP:,j]; rows.append([name,'ActiveCapacityMatched-NoHistory-AE-CeNN',int(s),np.sqrt(np.mean(e*e)),np.quantile(e,.95),np.max(e),float(np.max(e)>1),np.mean(eff[WARMUP:,j]),100*np.mean(sat[WARMUP:,j]),np.sqrt(np.mean(res[WARMUP:,j]**2))])
    return rows
rows=[]
for sc in SCENARIOS:
    print('eval',sc,flush=True);rows+=run(sc,TEST_SEEDS)
cols=['Scenario','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','excursion_gt_1m','control_effort','saturation_pct','residual_rms_ms2']
df=pd.DataFrame(rows,columns=cols);out='/mnt/data/d2_confirmatory';df.to_csv(out+'/active_capacity_trials_200seeds.csv',index=False)
sm=df.groupby(['Scenario','Controller']).agg(n=('SeedIndex','size'),rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),p95_error_mean=('p95_error_m','mean'),max_error_mean=('max_error_m','mean'),excursion_pct=('excursion_gt_1m',lambda x:100*x.mean()),effort_mean=('control_effort','mean'),saturation_pct_mean=('saturation_pct','mean')).reset_index();sm.to_csv(out+'/active_capacity_summary_200seeds.csv',index=False)
# Compare against previously frozen D2 exact-history outcomes.
main=pd.read_csv(out+'/main_trials_200seeds.csv');d2=main[main.Controller=='D2-History-AE-CeNN'];stats=[]
for sc in SCENARIOS:
    a=df[df.Scenario==sc].sort_values('SeedIndex');b=d2[d2.Scenario==sc].sort_values('SeedIndex');d=a.rmse_m.to_numpy()-b.rmse_m.to_numpy();rg=np.random.default_rng(8844+sum(map(ord,sc)));ii=rg.integers(0,len(d),(10000,len(d)));bs=d[ii].mean(1);p=wilcoxon(d).pvalue;stats.append([sc,a.rmse_m.mean(),b.rmse_m.mean(),d.mean(),100*d.mean()/a.rmse_m.mean(),np.quantile(bs,.025),np.quantile(bs,.975),d.mean()/(d.std(ddof=1)+1e-12),p])
st=pd.DataFrame(stats,columns=['Scenario','ActiveCapacity_RMSE_m','D2_RMSE_m','D2_improvement_m','D2_improvement_pct','CI95_low_m','CI95_high_m','Paired_effect_dz','Wilcoxon_p']);pv=st.Wilcoxon_p.to_numpy();order=np.argsort(pv);adj=np.empty(len(pv));running=0
for rank,i in enumerate(order):running=max(running,(len(pv)-rank)*pv[i]);adj[i]=min(1,running)
st['Holm_adjusted_p']=adj;st.to_csv(out+'/active_capacity_vs_D2_paired_statistics.csv',index=False)
# counts
active_d2=494; active_cap=sum(p.numel() for p in [ae.e1.weight,ae.e1.bias,ae.e2.weight,ae.e2.bias])+sum(p.numel() for p in ce.parameters())
json.dump({'active_inference_parameters_D2':active_d2,'active_inference_parameters_capacity':int(active_cap),'training_total_parameters_capacity':sum(p.numel() for p in ae.parameters())+sum(p.numel() for p in ce.parameters()),'note':'Capacity controller uses only encoder bottleneck and CeNN at inference; decoder is training-only, matching D2 inference semantics.'},open(out+'/active_capacity_model_info.json','w'),indent=2)
print(sm.to_string(index=False));print(st.to_string(index=False));print('active cap',active_cap)
