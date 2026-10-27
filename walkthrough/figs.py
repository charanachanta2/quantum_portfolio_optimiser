import sys, os, json
HERE=os.path.dirname(os.path.abspath(__file__))
P=os.path.join(HERE,"..","qpo_project")
RES=os.environ.get("QPO_RESULTS","results")          # e.g. QPO_RESULTS=results_real
sys.path.insert(0,P); os.chdir(P)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from qpo.config import Config
from qpo.data import synthetic_market, CRISES
from qpo.backtest import make_instance
from qpo.qaoa import QAOA, train_qaoa, apply_flip_channel, estimate_energy
from qpo.noise import NoiseModel
from qpo import theory
O=os.path.join(HERE,"figs")+os.sep
FACTS={}
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
NAVY="#1B2A49"; TEAL="#139A8C"; COR="#E4572E"; GOLD="#E9A820"; GRY="#8A94A6"

# 1 architecture
fig,ax=plt.subplots(figsize=(10,4.6)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,4.6)
def box(x,y,w,h,t,c):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.1",fc=c,ec=NAVY,lw=1.2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=9.5,color=NAVY)
def arr(x1,y1,x2,y2): ax.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle="->",color=NAVY,lw=1.5))
box(.2,3.3,2.6,1,"1 DATA\nprices · VIX · ESG\n(data.py)","#E8EEF7")
box(3.7,3.3,2.6,1,"2 ESTIMATION\nmu, shrunk Sigma, VIX z\n(backtest.make_instance)","#E8EEF7")
box(7.2,3.3,2.6,1,"3 QUBO\ntime-dependent, ESG, costs\n(qubo.py)","#DDF3EF")
box(.2,1.7,2.6,1,"4 QAOA SOLVER\nadaptive depth, COBYLA\n(qaoa.py)","#FCE1D8")
box(3.7,1.7,2.6,1,"5 NOISE + MITIGATION\nDD · ZNE · PEC · logical\n(noise.py, qec.py)","#FFF1CC")
box(7.2,1.7,2.6,1,"6 CLASSICAL BASELINES\nexact · SA · CVaR · Gurobi · RL\n(classical.py)","#E6E1F5")
box(.2,.2,4.4,.9,"7 EVALUATION  walk-forward backtest\nSharpe · drawdown · stats tests (stats.py)","#E8EEF7")
box(5.4,.2,4.4,.9,"8 DASHBOARD  live re-optimisation\n(live.py, dashboard/app.py)","#E8EEF7")
arr(2.8,3.8,3.7,3.8); arr(6.3,3.8,7.2,3.8); arr(8.5,3.3,1.5,2.7); arr(8.5,3.3,8.5,2.7); arr(1.5,1.7,1.5,1.1); arr(8.5,1.7,6,1.1); arr(2.8,2.2,3.7,2.2); arr(5,1.7,7,1.1)
fig.savefig(O+"01_architecture.png",bbox_inches="tight"); plt.close()

# 2 market
m=synthetic_market(n_assets=16)
fig,ax=plt.subplots(2,1,figsize=(9,5),sharex=True,gridspec_kw={"height_ratios":[2,1]})
ax[0].plot(m.prices.index,m.prices.iloc[:,:5].values,lw=.8); ax[0].set_yscale("log"); ax[0].set_ylabel("price (log)")
ax[1].plot(m.macro.index,m.macro["VIX"],color=COR,lw=.8); ax[1].set_ylabel("VIX")
for a in ax:
    for k,(s,e) in CRISES.items(): a.axvspan(pd.Timestamp(s),pd.Timestamp(e),color=GOLD,alpha=.3)
ax[0].set_title("Market data (SYNTHETIC simulation): prices and VIX, crisis windows shaded"); 
ax[1].text(pd.Timestamp("2008-09-01"),30,"2008",fontsize=9); ax[1].text(pd.Timestamp("2020-02-20"),30,"2020",fontsize=9)
fig.tight_layout(); fig.savefig(O+"02_market.png"); plt.close()

# instance
cfg=Config(n_assets=3,n_bits=2,n_periods=2)
i=int(np.searchsorted(m.returns.index,pd.Timestamp("2008-10-15")))
N=m.returns.shape[1]; inst=make_instance(m,i,np.full(N,1/N),cfg)
qb=inst.qubo; E=qb.energies()
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
im=ax[0].imshow(qb.Q,cmap="RdBu_r",vmin=-abs(qb.Q).max(),vmax=abs(qb.Q).max())
ax[0].set_title(f"QUBO matrix Q ({qb.n}x{qb.n} = {qb.n} qubits)"); plt.colorbar(im,ax=ax[0],fraction=.046)
lab=[f"t{t}A{a}b{k}" for t in range(qb.T) for a in range(qb.A) for k in range(qb.K)]
ax[0].set_xticks(range(qb.n)); ax[0].set_xticklabels(lab,rotation=90,fontsize=7); ax[0].set_yticks(range(qb.n)); ax[0].set_yticklabels(lab,fontsize=7)
ax[1].axis("off")
txt=("Encoding\n\nweight w = level / 6\nlevel = 1·b0 + 2·b1  (0..3)\n\n3 assets × 2 bits × 2 periods\n= 12 qubits  →  4096 bitstrings\n\n"
 f"Instance date: {inst.date.date()} (GFC)\nVIX-driven caps per period: {inst.caps}\nrisk aversion λ_t: {[round(x,2) for x in inst.lam]}\nESG-eligible assets: {len(inst.elig)}/16\ncouplings: {qb.n_couplings()}  linear: {qb.n_linear()}")
ax[1].text(0,.95,txt,va="top",fontsize=11,family="monospace",color=NAVY)
fig.tight_layout(); fig.savefig(O+"03_qubo.png"); plt.close()

# QAOA
q=QAOA(E,qb.n,qb.n_couplings(),qb.n_linear())
tr=train_qaoa(q,None,p_max=3,maxiter=60)
pr=q.probs(tr.params); zopt=int(np.argmin(E))
print("p",tr.p,"popt",pr[zopt],"E expect",q.energy_exact(tr.params),"Emin",E.min(),"Emean",E.mean())
FACTS.update(p=int(tr.p),popt=float(pr[zopt]),E_qaoa=float(q.energy_exact(tr.params)),Emin=float(E.min()),Emean=float(E.mean()),date=str(inst.date.date()),n=int(qb.n),caps=[int(c) for c in inst.caps],lam=[float(x) for x in inst.lam],n_elig=int(len(inst.elig)),n_coup=int(qb.n_couplings()),n_lin=int(qb.n_linear()))
fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].hist(E,bins=60,color=GRY); ax[0].axvline(E.min(),color=TEAL,lw=2,label="optimum"); ax[0].axvline(q.energy_exact(tr.params),color=COR,lw=2,label="QAOA mean energy"); ax[0].axvline(E.mean(),color=NAVY,ls="--",label="random mean")
ax[0].set_xlabel("QUBO energy"); ax[0].set_ylabel("# bitstrings"); ax[0].set_title("Energy landscape (4096 bitstrings)"); ax[0].legend(fontsize=8)
o=np.argsort(E)[:60]
ax[1].bar(range(60),pr[o],color=[TEAL if k==0 else COR for k in range(60)]); ax[1].axhline(1/len(E),color=NAVY,ls="--",label="uniform 1/2^n"); ax[1].legend(fontsize=8)
ax[1].set_xlabel("60 lowest-energy bitstrings (rank)"); ax[1].set_ylabel("QAOA probability"); ax[1].set_title(f"QAOA (p={tr.p}) boosts low-energy states")
fig.tight_layout(); fig.savefig(O+"04_qaoa.png"); plt.close()

# noise
nm_raw=NoiseModel(); nm_dd=NoiseModel(dd_suppression=.9); nm_log=NoiseModel(logical_q_per_layer=1e-6)
fig,ax=plt.subplots(figsize=(7,4))
labs=["ideal","physical raw","physical + DD","logical d=7"]; vals=[q.energy_exact(tr.params)]
for nm in (nm_raw,nm_dd,nm_log): vals.append(float(q.noisy_probs(tr.params,nm)@E))
ax.bar(labs,vals,color=[TEAL,COR,GOLD,NAVY]); ax.axhline(E.min(),color="k",ls=":",label="true optimum"); ax.axhline(E.mean(),color=GRY,ls="--",label="random")
ax.set_ylabel("expected QUBO energy (lower = better)"); ax.set_title("Noise degrades QAOA; DD and logical qubits recover it"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(O+"05_noise.png"); plt.close()
print("noise vals",vals, [q.q_eff(tr.p,n) for n in (nm_raw,nm_dd,nm_log)])
FACTS.update(noise_vals=[float(v) for v in vals],q_eff=[float(q.q_eff(tr.p,n)) for n in (nm_raw,nm_dd,nm_log)])

# ZNE
scales=np.array([1,2,3.]); rng=np.random.default_rng(0)
ev=[estimate_energy(q,tr.params,nm_raw,shots=None,rng=rng,method="raw")[0]]
ev=[float(q.noisy_probs(tr.params,nm_raw,s)@E) for s in scales]
z=estimate_energy(q,tr.params,nm_raw,shots=None,rng=rng,method="zne")[0]
pe=estimate_energy(q,tr.params,nm_raw,shots=None,rng=rng,method="pec")[0]
xs=np.linspace(0,3.2,50); co=np.polyfit(scales,ev,2)
fig,ax=plt.subplots(figsize=(7,4))
ax.plot(xs,np.polyval(co,xs),color=GRY,ls="--",label="Richardson fit"); ax.scatter(scales,ev,color=COR,zorder=3,label="noisy runs at 1x,2x,3x noise")
ax.scatter([0],[z],color=TEAL,s=90,zorder=3,label=f"ZNE estimate ({z:.4f})"); ax.scatter([0],[q.energy_exact(tr.params)],marker="*",color=NAVY,s=150,zorder=3,label=f"ideal ({q.energy_exact(tr.params):.4f})")
ax.set_xlabel("noise scale factor"); ax.set_ylabel("energy"); ax.set_title("Zero-noise extrapolation"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(O+"06_zne.png"); plt.close()
print("zne",z,"pec",pe)
FACTS.update(zne=float(z),pec=float(pe),raw=float(ev[0]),zne_pts=[float(v) for v in ev])

# PEC gamma^2
fig,ax=plt.subplots(figsize=(7,4))
ns=np.arange(4,101)
for qq,c in ((0.001,TEAL),(0.01,GOLD),(0.03,COR)): ax.plot(ns,[theory.pec_gamma(n,qq)**2 for n in ns],color=c,label=f"q={qq}")
ax.set_yscale("log"); ax.set_xlabel("qubits n"); ax.set_ylabel("PEC shot overhead γ²"); ax.set_title("PEC cost explodes with n·q → need logical qubits"); ax.legend()
fig.tight_layout(); fig.savefig(O+"07_pec.png"); plt.close()

# resources
r=pd.read_csv(f"{RES}/resources.csv"); r=r[r.p_phys==1e-3]; r=r[r.p_layers==3]
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
ax[0].bar(r.n_logical.astype(str),r.physical_qubits,color=TEAL); ax[0].set_xlabel("logical qubits"); ax[0].set_ylabel("physical qubits"); ax[0].set_title("Physical qubits needed (p=1e-3, p_layers=3)")
for x,(a,d) in enumerate(zip(r.physical_qubits,r.distance)): ax[0].text(x,a,f"d={d}",ha="center",va="bottom",fontsize=8)
ax[1].bar(r.n_logical.astype(str),r.T_count,color=COR); ax[1].set_xlabel("logical qubits"); ax[1].set_ylabel("T-count"); ax[1].set_title("T gates per circuit")
fig.tight_layout(); fig.savefig(O+"08_resources.png"); plt.close()

# ablation
a=pd.read_csv(f"{RES}/ablation_summary.csv").rename(columns={"Unnamed: 0":"config"}) if "config" not in pd.read_csv(f"{RES}/ablation_summary.csv",nrows=1) else pd.read_csv(f"{RES}/ablation_summary.csv")
fig,ax=plt.subplots(figsize=(8,4)); a2=a[a.config!="raw (noise-aware training)"]
ax.barh(a2.config[::-1],a2.est_err[::-1],xerr=a2.est_err_sd[::-1],color=TEAL,ecolor=GRY)
ax.set_xlabel("energy-estimate error (lower = better)"); ax.set_title("Error-mitigation ablation")
fig.tight_layout(); fig.savefig(O+"09_ablation.png"); plt.close()

# scaling
s=pd.read_csv(f"{RES}/scaling.csv")
if "n" not in s: s=s.rename(columns={s.columns[0]:"n"})
fig,ax=plt.subplots(figsize=(7,4))
for c,l,col in (("N99_qaoa","QAOA (ideal)",TEAL),("N99_qaoa_noisy_LB","QAOA (noisy bound)",COR),("N99_random","random",GRY),("SA_evals_median","simulated annealing",NAVY),("grover_queries","Grover",GOLD)):
    ax.plot(s.n,s[c],"o-",label=l,color=col)
ax.set_yscale("log"); ax.set_xlabel("qubits n"); ax.set_ylabel("samples to find optimum (99%)"); ax.set_title("Scaling: no quantum advantage over SA at n≤16"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(O+"10_scaling.png"); plt.close()

# backtest combined
fig,ax=plt.subplots(1,2,figsize=(11,4))
for a_,k in zip(ax,("GFC_2008","COVID_2020")):
    d=pd.read_csv(f"{RES}/backtest_{k}_returns.csv",index_col=0,parse_dates=True)
    for c,col,ls in (("equal_weight_full",GRY,"-"),("cvar_sub",NAVY,"-"),("miqp_sub","#7B5EA7",":"),("exact_qubo",GOLD,"--"),("qaoa_ideal",TEAL,"-"),("qaoa_logical",COR,"-")):
        if c not in d: continue
        a_.plot(d.index,(1+d[c]).cumprod(),label=c,color=col,ls=ls,lw=1.2)
    a_.set_title(k); a_.legend(fontsize=7)
fig.tight_layout(); fig.savefig(O+"11_backtest.png"); plt.close()
import shutil; shutil.copy(f"{RES}/qec_threshold.png",O+"12_qec.png")

FACTS["pec_gamma2_n12"]=float(theory.pec_gamma(12,FACTS["q_eff"][0])**2); FACTS["pec_gamma2_n100_q03"]=float(theory.pec_gamma(100,0.03)**2)
FACTS["results_dir"]=RES
json.dump(FACTS,open(os.path.join(HERE,"figs","facts.json"),"w"),indent=1)
