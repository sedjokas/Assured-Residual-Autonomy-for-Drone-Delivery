"""
Illustrative AP-QI-CeNN residual-control feasibility experiment (v2)
=================================================================
Purpose
-------
This script provides a small, reproducible synthetic experiment for the role
assigned to the CeNN layer in the AP-QI-CeNN architecture: a resource-bounded,
bounded-authority residual around a stable nominal controller and underneath an
independent runtime-assurance (RTA) governor.

Scope boundary
--------------
This is NOT a 6-DoF flight model, not hardware-in-the-loop evidence, and not a
claim that CeNN is superior to MPC/robust/DNN flight-control alternatives.
A compact MLP residual of comparable parameter count is included specifically
to prevent a CeNN-specific superiority claim from being inferred from a weak
baseline.
"""
from __future__ import annotations

from pathlib import Path
import json, math, platform, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon, binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/mnt/data/cenn_experiment_v2")
OUT.mkdir(parents=True, exist_ok=True)

SEED_TRAIN = 7
SEED_SCENARIO = 17
DT = 0.02                    # 50 Hz
TSEC = 28.0
N_MC = 120                   # paired Monte-Carlo missions
CORRIDOR = 1.60              # route-contract half-width [m]
RESIDUAL_LIMIT = 2.40        # residual authority [m/s^2]
PRIMARY_TUBE = 0.50          # tighter tracking tube used as secondary duration KPI

np.random.seed(SEED_TRAIN)
torch.manual_seed(SEED_TRAIN)
torch.set_num_threads(1)


def sat(x, lo=-1.0, hi=1.0):
    return np.clip(x, lo, hi)


class TinyCeNN(nn.Module):
    """3x3 Chua-Yang-style locally coupled CeNN with shared local templates."""
    def __init__(self, relaxation_steps: int = 5, cenn_dt: float = 0.30):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(1, 1, 3, 3))
        self.B = nn.Parameter(torch.randn(1, 1, 3, 3) * 0.10)
        self.bias = nn.Parameter(torch.zeros(1))
        self.relaxation_steps = relaxation_steps
        self.cenn_dt = cenn_dt

    def forward(self, xi):
        x = torch.zeros_like(xi)
        input_term = F.conv2d(xi, self.B, padding=1) + self.bias.view(1, 1, 1, 1)
        for _ in range(self.relaxation_steps):
            y = torch.clamp(x, -1.0, 1.0)
            feedback = F.conv2d(y, self.A, padding=1)
            x = x + self.cenn_dt * (-x + feedback + input_term)
        return torch.clamp(x, -1.0, 1.0)[:, :, 1, 1].squeeze(1)


class TinyMLP(nn.Module):
    """Parameter-comparable compact DNN residual baseline (5 -> 3 -> 1)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, z):
        return torch.tanh(self.fc2(torch.tanh(self.fc1(z)))).squeeze(1)


def wind_disturbance(w, dw):
    """Synthetic nonlinear crosswind-to-lateral-acceleration mapping."""
    return 0.55*w + 0.095*w*np.abs(w) + 0.10*np.tanh(1.2*dw)


def make_training_data(n=12000, seed=1):
    rng = np.random.default_rng(seed)
    w = rng.uniform(-3.0, 3.0, n)
    w_prev = 0.85*w + 0.15*rng.uniform(-3.0, 3.0, n)
    e = np.clip(rng.normal(0.0, 0.45, n), -1.2, 1.2)
    v = np.clip(rng.normal(0.0, 0.50, n), -1.5, 1.5)
    dw = w - w_prev
    w_hat = w + rng.normal(0.0, 0.18, n) + 0.08*np.sin(1.7*w)
    w_prev_hat = w_prev + rng.normal(0.0, 0.18, n)
    target = np.clip(-wind_disturbance(w, dw) / RESIDUAL_LIMIT, -1.0, 1.0).astype(np.float32)

    xi = np.zeros((n, 1, 3, 3), dtype=np.float32)
    xi[:,0,0,1] = w_hat / 4.0
    xi[:,0,1,0] = e / 1.5
    xi[:,0,1,1] = v / 2.0
    xi[:,0,1,2] = dw / 2.0
    xi[:,0,2,1] = w_prev_hat / 4.0
    xi[:,0,2,2] = 1.0

    z = np.column_stack([e/1.5, v/2.0, w_hat/4.0, dw/2.0, w_prev_hat/4.0]).astype(np.float32)
    return torch.tensor(xi), torch.tensor(z), torch.tensor(target)


def train_models():
    Xc, Xm, y = make_training_data()
    models = {"CeNN": TinyCeNN(), "MLP": TinyMLP()}
    losses = {}
    for name, model in models.items():
        torch.manual_seed(SEED_TRAIN)
        opt = torch.optim.Adam(model.parameters(), lr=0.03)
        for _ in range(300):
            idx = torch.randint(0, len(y), (1024,))
            pred = model(Xc[idx] if name == "CeNN" else Xm[idx])
            loss = F.mse_loss(pred, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xc[:3000] if name == "CeNN" else Xm[:3000])
            losses[name] = float(torch.sqrt(F.mse_loss(pred, y[:3000])))
    return models, losses


def generate_test_scenarios(n=N_MC, seed=SEED_SCENARIO):
    rng = np.random.default_rng(seed)
    steps = int(TSEC/DT)
    t = np.arange(steps)*DT

    w = np.zeros((steps,n), dtype=np.float32)
    for k in range(1,steps):
        w[k] = 0.995*w[k-1] + 0.12*rng.normal(size=n)

    for i in range(n):
        gust_specs = [
            ((7.5,10.5),(2.0,3.2),(0.50,0.90)),
            ((18.0,22.0),(2.0,3.8),(0.40,0.80)),
        ]
        for center_rng, amp_rng, width_rng in gust_specs:
            amp = rng.uniform(*amp_rng)*rng.choice([-1.0,1.0])
            center = rng.uniform(*center_rng)
            width = rng.uniform(*width_rng)
            w[:,i] += amp*np.exp(-0.5*((t-center)/width)**2)
    w = np.clip(w,-4.8,4.8)

    sensor_bias = rng.normal(0.0,0.05,n)
    w_hat = w*(1.0+sensor_bias[None,:]) + rng.normal(0.0,0.20,(steps,n))
    aero_scale = rng.uniform(0.88,1.15,n)
    actuator_gain = rng.uniform(0.88,1.05,n)
    drag = rng.uniform(0.08,0.13,n)
    w_prev = np.vstack([np.zeros((1,n),dtype=np.float32),w[:-1]])
    dw = w-w_prev
    disturbance = wind_disturbance(w,dw)*aero_scale[None,:]

    dropout = np.zeros((steps,n),dtype=bool)
    dropout_windows=[]
    for i in range(n):
        center = rng.uniform(18.0,21.0)
        duration = rng.uniform(1.5,2.5)
        dropout[:,i]=(t>=center)&(t<center+duration)
        dropout_windows.append((float(center),float(center+duration)))

    return dict(t=t,w=w,w_hat=w_hat,disturbance=disturbance,
                actuator_gain=actuator_gain,drag=drag,dropout=dropout,
                dropout_windows=dropout_windows)


def residual_prediction(model_name, model, e, v, w_hat, w_hat_prev, dw_hat):
    if model_name == "CeNN":
        n=len(e)
        xi=np.zeros((n,1,3,3),dtype=np.float32)
        xi[:,0,0,1]=w_hat/4.0
        xi[:,0,1,0]=e/1.5
        xi[:,0,1,1]=v/2.0
        xi[:,0,1,2]=dw_hat/2.0
        xi[:,0,2,1]=w_hat_prev/4.0
        xi[:,0,2,2]=1.0
        inp=torch.from_numpy(xi)
    else:
        z=np.column_stack([e/1.5,v/2.0,w_hat/4.0,dw_hat/2.0,w_hat_prev/4.0]).astype(np.float32)
        inp=torch.from_numpy(z)
    with torch.no_grad():
        y=model(inp).numpy()
    return RESIDUAL_LIMIT*y


def simulate(profile, models, variant):
    t=profile['t']; w_hat=profile['w_hat']; d=profile['disturbance']
    act=profile['actuator_gain']; drag=profile['drag']; dropout=profile['dropout']
    steps,n=w_hat.shape
    e=np.zeros(n); v=np.zeros(n); prev_w_hat=w_hat[0].copy()
    e_hist=np.zeros((steps,n),dtype=np.float32)
    u_hist=np.zeros((steps,n),dtype=np.float32)
    rta_hist=np.zeros((steps,n),dtype=bool)
    residual_hist=np.zeros((steps,n),dtype=np.float32)

    use_rta='rta' in variant
    residual_kind = 'CeNN' if 'cenn' in variant else ('MLP' if 'mlp' in variant else None)
    use_dropout='dropout' in variant

    for k in range(steps):
        u0=np.clip(-1.9*e-2.3*v,-4.5,4.5)
        u_prop=u0.copy()
        if residual_kind:
            res=residual_prediction(residual_kind,models[residual_kind],e.astype(np.float32),v.astype(np.float32),
                                    w_hat[k].astype(np.float32),prev_w_hat.astype(np.float32),
                                    (w_hat[k]-prev_w_hat).astype(np.float32))
            if use_dropout:
                res=np.where(dropout[k],0.0,res)
            res=np.clip(res,-RESIDUAL_LIMIT,RESIDUAL_LIMIT)
            u_prop=np.clip(u0+res,-5.0,5.0)
            residual_hist[k]=res

        if use_rta:
            # Independent conservative model, not trained with either residual.
            d_hat_rta=0.55*w_hat[k]+0.08*w_hat[k]*np.abs(w_hat[k])
            horizon=0.50
            e_pred=e+horizon*v+0.5*horizon**2*(act*u_prop+d_hat_rta)
            danger=(np.abs(e_pred)>1.25)|((np.abs(e)>1.05)&(e*v>0))
            u_safe=np.clip(-4.2*e-3.5*v-d_hat_rta/np.maximum(act,0.85),-5.0,5.0)
            u=np.where(danger,u_safe,u_prop)
            rta_hist[k]=danger
        else:
            u=u_prop

        a=act*u+d[k]-drag*v*np.abs(v)
        v=v+DT*a
        e=e+DT*v
        e_hist[k]=e; u_hist[k]=u
        prev_w_hat=w_hat[k].copy()

    rmse=np.sqrt(np.mean(e_hist**2,axis=0))
    maxerr=np.max(np.abs(e_hist),axis=0)
    time05=np.sum(np.abs(e_hist)>PRIMARY_TUBE,axis=0)*DT
    effort=np.mean(u_hist**2,axis=0)
    violation=np.any(np.abs(e_hist)>CORRIDOR,axis=0).astype(int)
    rta_rate=np.mean(rta_hist,axis=0)
    max_resid=np.max(np.abs(residual_hist),axis=0)
    return dict(e=e_hist,u=u_hist,rta=rta_hist,residual=residual_hist,
                metrics=dict(rmse_m=rmse,max_abs_error_m=maxerr,seconds_outside_0p5m=time05,
                             control_effort_u2=effort,corridor_violation=violation,
                             rta_intervention_rate=rta_rate,max_abs_residual=max_resid))


def bootstrap_mean_diff(a,b,n_boot=5000,seed=123):
    """Paired mean difference a-b with percentile CI."""
    rng=np.random.default_rng(seed); diff=np.asarray(a)-np.asarray(b); n=len(diff)
    idx=rng.integers(0,n,(n_boot,n)); boot=diff[idx].mean(axis=1)
    return float(diff.mean()), tuple(np.quantile(boot,[0.025,0.975]))


def exact_mcnemar(a,b):
    a=np.asarray(a).astype(int); b=np.asarray(b).astype(int)
    n10=int(np.sum((a==1)&(b==0))); n01=int(np.sum((a==0)&(b==1)))
    n=n10+n01
    p=1.0 if n==0 else float(binomtest(min(n10,n01),n,p=0.5,alternative='two-sided').pvalue)
    return n10,n01,p


def param_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def render_figures(profile,runs,summary):
    # Fig 1: architecture schematic - single axes, publication-friendly.
    fig=plt.figure(figsize=(8.6,4.6)); ax=fig.add_subplot(111); ax.axis('off')
    boxes=[
        (0.03,0.58,0.19,0.22,'Route contract\ncenterline / corridor'),
        (0.28,0.58,0.18,0.22,'Stable nominal\nPD controller'),
        (0.28,0.17,0.18,0.22,'Bounded residual\nCeNN or compact MLP'),
        (0.54,0.46,0.17,0.22,'Proposed command\n$u_0+\\Delta u$'),
        (0.78,0.46,0.19,0.22,'Independent RTA\nsafety governor'),
    ]
    for x,y,w,h,txt in boxes:
        rect=plt.Rectangle((x,y),w,h,fill=False,linewidth=1.6)
        ax.add_patch(rect); ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=9.4)
    arrows=[((0.22,0.69),(0.28,0.69)),((0.46,0.69),(0.54,0.60)),((0.46,0.28),(0.54,0.52)),((0.71,0.57),(0.78,0.57))]
    for a,b in arrows: ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',linewidth=1.6))
    ax.annotate('local state / wind features',xy=(0.37,0.39),xytext=(0.37,0.52),ha='center',fontsize=8,arrowprops=dict(arrowstyle='->'))
    ax.text(0.80,0.22,'Deadline miss / low confidence:\nset $\\Delta u=0$; nominal + RTA remain',fontsize=8.7,ha='center',va='center')
    ax.annotate('',xy=(0.86,0.46),xytext=(0.82,0.31),arrowprops=dict(arrowstyle='->',linestyle='--'))
    ax.text(0.50,0.93,'Illustrative residual-control experiment aligned with the AP-QI-CeNN authority split',ha='center',fontsize=12,fontweight='bold')
    fig.tight_layout();
    p=OUT/'fig7_experiment_schematic.png'; fig.savefig(p,dpi=450,bbox_inches='tight'); fig.savefig(OUT/'fig7_experiment_schematic.pdf',bbox_inches='tight'); plt.close(fig)

    # Fig 2: transparent stress-case selection: mission closest to the 90th percentile of nominal RMSE.
    nominal_rmse=runs['Nominal']['metrics']['rmse_m']
    target=np.quantile(nominal_rmse,0.90)
    idx=int(np.argmin(np.abs(nominal_rmse-target)))
    fig=plt.figure(figsize=(8.6,4.8)); ax=fig.add_subplot(111)
    for label,ls in [('Nominal','-'),('Nominal + RTA','--'),('Compact MLP + RTA','-.'),('CeNN + RTA',':'),('CeNN dropout -> nominal + RTA',(0,(5,1,1,1)))]:
        ax.plot(profile['t'],runs[label]['e'][:,idx],label=label,linestyle=ls,linewidth=1.5)
    ax.axhline(CORRIDOR,linestyle='--',linewidth=1.0); ax.axhline(-CORRIDOR,linestyle='--',linewidth=1.0)
    # Mark dropout interval for selected mission.
    s,e=profile['dropout_windows'][idx]; ax.axvspan(s,e,alpha=0.10)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Cross-track error $e$ [m]')
    ax.set_title('Pre-specified 90th-percentile nominal-RMSE mission under nonlinear gusts')
    ax.legend(fontsize=7.5,ncol=2,loc='upper right'); ax.grid(True,alpha=0.25); fig.tight_layout()
    p=OUT/'fig8_representative_tracking.png'; fig.savefig(p,dpi=450); fig.savefig(OUT/'fig8_representative_tracking.pdf'); plt.close(fig)

    # Fig 3: RMSE empirical distributions as box plot.
    labels=['Nominal','Nominal + RTA','Compact MLP + RTA','CeNN + RTA','CeNN dropout -> nominal + RTA']
    data=[runs[l]['metrics']['rmse_m'] for l in labels]
    fig=plt.figure(figsize=(8.6,4.9)); ax=fig.add_subplot(111)
    ax.boxplot(data,labels=labels,showmeans=True,meanline=True)
    ax.set_ylabel('Mission cross-track RMSE [m]'); ax.set_title('Paired Monte-Carlo distribution (120 identical disturbance realizations)')
    ax.tick_params(axis='x',rotation=18,labelsize=8); ax.grid(True,axis='y',alpha=0.25); fig.tight_layout()
    p=OUT/'fig9_rmse_distribution.png'; fig.savefig(p,dpi=450); fig.savefig(OUT/'fig9_rmse_distribution.pdf'); plt.close(fig)


def main():
    t0=time.perf_counter(); models,train_rmse=train_models(); train_time=time.perf_counter()-t0
    profile=generate_test_scenarios()
    variants={
        'Nominal':'nominal',
        'Nominal + RTA':'nominal_rta',
        'Compact MLP + RTA':'mlp_rta',
        'CeNN only':'cenn',
        'CeNN + RTA':'cenn_rta',
        'CeNN dropout -> nominal + RTA':'cenn_dropout_rta',
    }
    runs={label:simulate(profile,models,var) for label,var in variants.items()}

    # Summary and trial-level data.
    rows=[]; trial_rows=[]
    for label,r in runs.items():
        m=r['metrics']
        rows.append({
            'Controller':label,
            'RMSE mean [m]':float(np.mean(m['rmse_m'])),
            'RMSE SD [m]':float(np.std(m['rmse_m'],ddof=1)),
            'Max |error| mean [m]':float(np.mean(m['max_abs_error_m'])),
            'Time |e|>0.5 m mean [s]':float(np.mean(m['seconds_outside_0p5m'])),
            'Control effort mean(u^2)':float(np.mean(m['control_effort_u2'])),
            'Corridor violation [% missions]':float(100*np.mean(m['corridor_violation'])),
            'RTA interventions [% cycles]':float(100*np.mean(m['rta_intervention_rate'])),
            'Max residual mean [m/s^2]':float(np.mean(m['max_abs_residual'])),
        })
        for i in range(N_MC):
            trial_rows.append({
                'mission_id':i,'controller':label,
                'rmse_m':float(m['rmse_m'][i]),'max_abs_error_m':float(m['max_abs_error_m'][i]),
                'seconds_outside_0p5m':float(m['seconds_outside_0p5m'][i]),
                'control_effort_u2':float(m['control_effort_u2'][i]),
                'corridor_violation':int(m['corridor_violation'][i]),
                'rta_intervention_rate':float(m['rta_intervention_rate'][i]),
                'max_abs_residual':float(m['max_abs_residual'][i]),
            })
    summary=pd.DataFrame(rows); trials=pd.DataFrame(trial_rows)
    base=np.mean(runs['Nominal']['metrics']['rmse_m'])
    summary['RMSE reduction vs nominal [%]']=100*(base-summary['RMSE mean [m]'])/base

    comparisons=[]
    comp_pairs=[
        ('CeNN + RTA','Nominal'),('CeNN + RTA','Nominal + RTA'),('CeNN + RTA','Compact MLP + RTA'),
        ('CeNN dropout -> nominal + RTA','Nominal'),('Compact MLP + RTA','Nominal + RTA')]
    for j,(cand,ref) in enumerate(comp_pairs):
        a=runs[ref]['metrics']['rmse_m']; b=runs[cand]['metrics']['rmse_m']
        diff,ci=bootstrap_mean_diff(a,b,seed=100+j)
        try:
            wp=float(wilcoxon(a,b,alternative='two-sided',zero_method='wilcox').pvalue)
        except Exception:
            wp=float('nan')
        n10,n01,mp=exact_mcnemar(runs[ref]['metrics']['corridor_violation'],runs[cand]['metrics']['corridor_violation'])
        comparisons.append({
            'Candidate':cand,'Reference':ref,
            'RMSE mean improvement [m]':diff,'RMSE improvement 95% CI low [m]':ci[0],
            'RMSE improvement 95% CI high [m]':ci[1],
            'RMSE relative improvement [%]':100*diff/float(np.mean(a)),
            'Wilcoxon paired p':wp,
            'McNemar ref-only violations':n10,'McNemar cand-only violations':n01,'McNemar exact p':mp,
        })
    comp=pd.DataFrame(comparisons)

    summary.to_csv(OUT/'cenn_feasibility_summary.csv',index=False)
    trials.to_csv(OUT/'cenn_feasibility_trials.csv',index=False)
    comp.to_csv(OUT/'cenn_feasibility_paired_tests.csv',index=False)

    # Parameter/template exports.
    templates={
        'CeNN_A':models['CeNN'].A.detach().cpu().numpy().reshape(3,3).tolist(),
        'CeNN_B':models['CeNN'].B.detach().cpu().numpy().reshape(3,3).tolist(),
        'CeNN_bias':float(models['CeNN'].bias.detach().cpu().item()),
        'CeNN_parameter_count':param_count(models['CeNN']),
        'MLP_parameter_count':param_count(models['MLP']),
        'training_target_rmse_normalized':train_rmse,
    }
    (OUT/'learned_templates_and_model_info.json').write_text(json.dumps(templates,indent=2))

    env={
        'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,
        'torch':torch.__version__,'matplotlib':matplotlib.__version__,
        'seeds':{'train':SEED_TRAIN,'scenario':SEED_SCENARIO},'dt_s':DT,'duration_s':TSEC,'n_missions':N_MC,
        'corridor_half_width_m':CORRIDOR,'residual_limit_mps2':RESIDUAL_LIMIT,
        'training_wall_clock_s_on_this_host':train_time,
        'scope_warning':'Host timing is not an embedded-hardware benchmark and is not used as H3 evidence.'
    }
    (OUT/'experiment_manifest.json').write_text(json.dumps(env,indent=2))

    readme=f"""# AP-QI-CeNN illustrative residual-control feasibility experiment\n\nThis archive is a reproducible **synthetic** demonstration of the role assigned to a bounded CeNN residual in the AP-QI-CeNN paper. It is not a 6-DoF flight validation, not HIL evidence, and not a CeNN superiority claim.\n\n## Reproduce\n\nRun `python cenn_feasibility_demo_v2.py` from an environment with NumPy, pandas, SciPy, PyTorch and Matplotlib. Fixed seeds are recorded in `experiment_manifest.json`.\n\n## Primary design\n- 50 Hz lateral cross-track model for 28 s.\n- 120 paired Monte-Carlo missions.\n- Stable nominal PD controller.\n- 3x3 locally coupled CeNN residual, bounded to +/- {RESIDUAL_LIMIT:.2f} m/s^2.\n- Parameter-comparable compact MLP (5-3-1) residual baseline.\n- Independent 0.5 s look-ahead RTA envelope.\n- Nonlinear stochastic gusts, sensor noise/bias, aerodynamic and actuator uncertainty.\n- Forced 1.5-2.5 s CeNN compute outage with residual set to zero.\n\n## Claim boundary\nThe experiment demonstrates architectural feasibility: a residual module can improve tracking while being bounded, supervised, and safely disposable. Full H3 validation still requires stronger model-based baselines, embedded timing/power, SIL/HIL, and flight tests.\n"""
    (OUT/'README.md').write_text(readme)
    (OUT/'requirements.txt').write_text('numpy\npandas\nscipy\ntorch\nmatplotlib\n')

    render_figures(profile,runs,summary)

    print('\nMODEL INFO')
    print('CeNN params:',param_count(models['CeNN']),'MLP params:',param_count(models['MLP']))
    print('Calibration RMSE normalized:',train_rmse)
    print('\nSUMMARY')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nPAIRED TESTS')
    print(comp.to_string(index=False,float_format=lambda x:f'{x:.5g}'))
    return summary,comp,templates

if __name__=='__main__':
    main()
