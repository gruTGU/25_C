# -*- coding: utf-8 -*-
"""
Q4（第三问思想版，Dual）：
- 男胎：两层模型（μ(GA,BMI,...) + σ(GA,QC)），P_hit = 1 - Φ((thr - μ)/σ)
- 女胎：在无 Y 浓度阈值的前提下，用多项式回归近似 P_pass(hit) 随 GA/BMI 的变化（默认标签：抽血次数==1）
- 在 P_hit 基础上优化一次检测 t1 与两次检测 (t1,Δ) 策略
"""
import os, re, math, argparse, warnings, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 安静 & 字体
import logging, matplotlib
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
class NullWriter:
    def write(self, arg): pass
    def flush(self): pass
_stderr = sys.stderr
sys.stderr = NullWriter()
plt.rcParams['font.family'] = [
    'Microsoft YaHei','SimHei','Arial Unicode MS','STHeiti',
    'PingFang SC','WenQuanYi Zen Hei','Source Han Sans SC','sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False
sys.stderr = _stderr

# ---------- 工具 ----------
def parse_ga_to_weeks(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float,np.integer,np.floating)):
        try: return float(x)
        except: return np.nan
    s=str(x).strip().replace("W","w").replace("D","d").replace("周","w").replace("天","d")
    s=s.replace("．",".").replace("＋","+")
    m=re.match(r"^\s*(\d{1,2})\s*(?:w)?\s*(?:\+)?\s*(\d{1,2})?\s*(?:d)?\s*$", s)
    if m:
        w=int(m.group(1)); d=int(m.group(2)) if m.group(2) else 0
        return w + d/7.0
    try: return float(s)
    except: return np.nan

def guess_columns(df):
    aliases={
        "id":["孕妇代码","孕妇ID","孕妇编号","受检者ID","样本ID","样本编号","ID","code","patient_id"],
        "age":["年龄","Age"], "ga":["检测孕周","孕周","孕周(周+天)","孕周（周+天）","GA"],
        "bmi":["孕妇BMI","BMI"], "y_frac":["Y染色体浓度","Y浓度","Y浓度(%)","fetal_fraction_Y","Y_fetal_fraction"],
        "y_z":["Y染色体的Z值","Y染色体Z值","Y_Z"], "draws":["检测抽血次数","抽血次数","draw_count"],
        "date":["检测日期","检测时间","日期","sample_time","采血时间"],
        "gc":["整体GC","GC含量","GC"], "filt_ratio":["被过滤读段比例","filtered_ratio","AA"],
        "unique_mapped":["唯一比对读段数","unique_mapped_reads","O"],
        "sex":["胎儿性别","性别","FetalSex"]
    }
    m={}; low={str(c).strip().lower(): c for c in df.columns}
    for k,cands in aliases.items():
        for c in cands:
            key=str(c).strip().lower()
            if key in low: m[k]=low[key]; break
    return m

def to_ratio(y):
    y=pd.to_numeric(y, errors="coerce")
    if (y>1).mean(skipna=True)>0.5: return y/100.0
    return y

def normal_cdf(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))

def risk_level(t):
    if t <= 12: return 1.0
    elif t < 28: return 2.0
    else: return 3.0

def make_bmi_group(bmi):
    bins=[20,28,32,36,40,100]; labels=["[20,28)","[28,32)","[32,36)","[36,40)","[40,+)"]
    return pd.cut(bmi, bins=bins, labels=labels, right=False)

def ensure_outdir(p): os.makedirs(p, exist_ok=True)

# ---------- 男胎：μ/σ ----------
def build_design_matrix(df, deg=3, include_interactions=True):
    GA=df["GA_weeks"].values.astype(float)
    BMI=df["BMI"].values.astype(float)
    cols=[np.ones_like(GA)]
    for d in range(1,deg+1): cols.append(GA**d)
    cols.append(BMI)
    for k in ["Age","Height","Weight"]:
        if k in df.columns:
            cols.append(pd.to_numeric(df[k], errors="coerce").values.astype(float))
    if include_interactions:
        for d in range(1,deg+1): cols.append((GA**d)*BMI)
    return np.column_stack(cols)

def fit_linear_mu(df, deg=3, include_interactions=True):
    X=build_design_matrix(df, deg, include_interactions)
    y=df["Y_frac"].values.astype(float)
    ok=(~np.isnan(X).any(axis=1)) & (~np.isnan(y))
    X=X[ok]; y=y[ok]
    if X.size==0:
        c=float(np.nanmedian(df["Y_frac"]))
        return lambda d: np.clip(np.full(len(d), c), 1e-4, 0.499)
    beta, *_=np.linalg.lstsq(X,y,rcond=None)
    def mu_fn(dn):
        Xn=build_design_matrix(dn, deg, include_interactions)
        yhat=Xn @ beta
        return np.clip(yhat, 1e-4, 0.499)
    return mu_fn

def estimate_sigma(df, mu_fn, mode="ga_local", bin_width=0.5, qc_cols=None):
    y=df["Y_frac"].values.astype(float)
    mu=mu_fn(df); resid=y-mu; resid=resid[~np.isnan(resid)]
    if len(resid)==0: return lambda d: np.full(len(d), 0.02)
    if mode=="global":
        s=float(max(np.nanstd(resid),0.01)); return lambda d: np.full(len(d), s)
    if mode=="ga_local":
        ga=df["GA_weeks"].values.astype(float)
        ok=(~np.isnan(ga)) & (~np.isnan(y))
        ga=ga[ok]; rr=(y-mu)[ok]
        if len(ga)==0:
            s=float(np.clip(np.nanstd(resid),0.01,0.05)); return lambda d: np.full(len(d), s)
        start, stop = max(9.0,float(np.nanmin(ga))), min(30.0,float(np.nanmax(ga)))
        if not np.isfinite(start) or not np.isfinite(stop) or stop-start<1e-6:
            bins=np.array([start-0.5, start+0.5], dtype=float)
        else:
            bins=np.arange(start, stop+bin_width, bin_width, dtype=float)
            if bins.size<2: bins=np.array([start-0.5, start+0.5], dtype=float)
        mids=(bins[:-1]+bins[1:])/2.0
        gsd=float(np.nanstd(rr)); gsd=0.02 if (not np.isfinite(gsd) or gsd==0) else gsd
        if mids.size==0:
            s=float(np.clip(gsd,0.005,0.06)); return lambda d: np.full(len(d), s)
        sds=np.empty_like(mids); sds.fill(np.nan)
        for i in range(mids.size):
            m=(ga>=bins[i])&(ga<bins[i+1])
            if np.any(m): sds[i]=np.nanstd(rr[m])
        sds=np.where(np.isnan(sds), gsd, sds)
        sds=np.clip(sds,0.005,0.06)
        def sigma_fn(dn):
            GA=dn["GA_weeks"].values.astype(float)
            if sds.size==0: return np.full(len(GA), float(np.clip(gsd,0.005,0.06)))
            idx=np.digitize(GA, bins)-1; idx=np.clip(idx,0,sds.size-1)
            return sds[idx]
        return sigma_fn
    s=float(np.clip(np.nanstd(resid),0.01,0.05)); return lambda d: np.full(len(d), s)

# ---------- 女胎：多项式概率近似 ----------
def build_poly_prob(df, deg=3, include_interactions=True, label_col="hit"):
    GA=df["GA_weeks"].values.astype(float)
    BMI=df["BMI"].values.astype(float)
    y=pd.to_numeric(df[label_col], errors="coerce").values.astype(float)
    cols=[np.ones_like(GA)]
    for d in range(1,deg+1): cols.append(GA**d)
    cols.append(BMI)
    for k in ["Age","Height","Weight"]:
        if k in df.columns: cols.append(pd.to_numeric(df[k], errors="coerce").values.astype(float))
    if include_interactions:
        for d in range(1,deg+1): cols.append((GA**d)*BMI)
    X=np.column_stack(cols)
    ok=(~np.isnan(X).any(axis=1)) & (~np.isnan(y))
    X=X[ok]; y=y[ok]
    if X.size==0:
        p=float(np.clip(np.nanmean(y),0.05,0.95))
        return lambda d: np.full(len(d), p)
    beta,*_=np.linalg.lstsq(X,y,rcond=None)
    def p_fn(dn):
        GA=dn["GA_weeks"].values.astype(float); BMI=dn["BMI"].values.astype(float)
        cols=[np.ones_like(GA)]
        for dd in range(1,deg+1): cols.append(GA**dd)
        cols.append(BMI)
        for k in ["Age","Height","Weight"]:
            if k in dn.columns: cols.append(pd.to_numeric(dn[k], errors="coerce").values.astype(float))
        if include_interactions:
            for dd in range(1,deg+1): cols.append((GA**dd)*BMI)
        Xn=np.column_stack(cols)
        phat=Xn @ beta
        return np.clip(phat, 0.0, 1.0)
    return p_fn

# ---------- 读数 & 标签 ----------
def load_data(excel="附件.xlsx", clean_csv=None, sheet=None):
    if clean_csv and os.path.exists(clean_csv):
        df=pd.read_csv(clean_csv)
    else:
        df=pd.read_excel(excel, sheet_name=sheet) if sheet else pd.read_excel(excel)
    mp=guess_columns(df)
    if "ga" in mp: df["GA_weeks"]=df[mp["ga"]].apply(parse_ga_to_weeks)
    if "bmi" in mp: df["BMI"]=pd.to_numeric(df[mp["bmi"]], errors="coerce")
    if "y_frac" in mp: df["Y_frac"]=to_ratio(df[mp["y_frac"]])
    if "age" in mp: df["Age"]=pd.to_numeric(df[mp["age"]], errors="coerce")
    if "draws" in mp: df["Draws"]=pd.to_numeric(df[mp["draws"]], errors="coerce")
    if "sex" in mp: df["Sex_raw"]=df[mp["sex"]].astype(str)
    for k,std in [("身高","Height"),("体重","Weight")]:
        if k in df.columns and std not in df.columns:
            df[std]=pd.to_numeric(df[k], errors="coerce")
    df=df[(~df["GA_weeks"].isna()) & (~df["BMI"].isna())]
    df=df[(df["GA_weeks"]>=9)&(df["GA_weeks"]<=30)&(df["BMI"]>10)&(df["BMI"]<60)]
    if "Sex_raw" in df.columns:
        df["is_male"]=df["Sex_raw"].str.contains("男").fillna(False)
    else:
        if "Y_frac" in df.columns: df["is_male"]=df["Y_frac"]>0.005
        else: df["is_male"]=np.nan
    return df

def define_hit(df, sex="auto", y_threshold=0.04, female_rule="draw1", pass_col=None, pass_pos="是"):
    if sex=="auto":
        if df["is_male"].isna().all(): sex="male"
    if sex=="male":
        if "Y_frac" not in df.columns:
            raise ValueError("男胎标签需要 Y 浓度列。")
        df["hit"]=(df["Y_frac"]>=y_threshold).astype(int)
        return df[df["is_male"]==True].copy(), "male"
    else:
        dff=df[df["is_male"]==False].copy() if df["is_male"].notna().any() else df.copy()
        if pass_col and pass_col in dff.columns:
            val=pass_pos
            if isinstance(val,str):
                dff["hit"]=dff[pass_col].astype(str).str.contains(str(val)).fillna(False).astype(int)
            else:
                dff["hit"]=(pd.to_numeric(dff[pass_col], errors="coerce")==float(val)).astype(int)
            rule=f"pass_col={pass_col}::{pass_pos}"
        elif "Draws" in dff.columns:
            dff["hit"]=(pd.to_numeric(dff["Draws"], errors="coerce")<=1).astype(int)
            rule="draws<=1"
        else:
            dff["hit"]=1; rule="fallback_all_pass(1)"
        return dff.copy(), f"female:{rule}"

# ---------- 策略优化 ----------
def optimize_strategy(phit_fn, scen, outdir, prefix,
                      ga_min=10.0, ga_max=29.0, ga_step=0.25,
                      c1=1.0, cr=1.0, lam=1.0, kappa=0.05, alpha=1.0,
                      delta_choices=(1,1.5,2,3)):
    t_grid=np.arange(ga_min, ga_max+1e-9, ga_step).astype(float)
    def phit_t(t):
        tmp=pd.DataFrame({"GA_weeks":[t], "BMI":[scen["BMI"]]})
        for k in ["Age","Height","Weight"]:
            tmp[k]=[scen[k]] if scen.get(k) is not None else None
        return float(phit_fn(tmp)[0])
    P=np.array([phit_t(t) for t in t_grid])

    J1=np.array([risk_level(t)+lam*(1-P[i])+c1+kappa*(t-ga_min) for i,t in enumerate(t_grid)])
    t1_single=float(t_grid[np.argmin(J1)]); J1_min=float(J1.min())

    best=None; heat=np.full((len(delta_choices), len(t_grid)), np.nan)
    for di,d in enumerate(delta_choices):
        valid=t_grid[t_grid+d<=ga_max+1e-9]
        for i,t1 in enumerate(valid):
            t2=t1+d
            P1=phit_t(t1); P2=phit_t(t2); P2p=alpha*P2+(1-alpha)*P1
            Psucc=P1+(1-P1)*P2p
            E_Tres=P1*t1+(1-P1)*t2
            E_risk=P1*risk_level(t1)+(1-P1)*risk_level(t2)
            J2=c1+(1-P1)*cr+E_risk+lam*(1-Psucc)+kappa*(E_Tres-ga_min)
            heat[di,i]=J2
            if (best is None) or (J2<best["J2"]):
                best=dict(t1=t1,d=d,t2=t2,J2=J2,Psucc=Psucc,E_Tres=E_Tres)

    plt.figure()
    extent=[t_grid.min(), t_grid.max(), 0, len(delta_choices)]
    plt.imshow(heat, aspect='auto', origin='lower', extent=extent)
    plt.colorbar(label="目标值 J₂")
    yticks=np.arange(len(delta_choices))+0.5
    plt.yticks(yticks,[f"Δ={d}" for d in delta_choices])
    plt.axvline(best["t1"], linestyle="--", linewidth=1)
    plt.xlabel("首检孕周 t₁（周）"); plt.ylabel("复检间隔 Δ（周）")
    plt.title(f"Q4：J₂(t₁,Δ) 热力图（{prefix}）")
    ensure_outdir(outdir); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"q4_dual_heatmap_{prefix}.png"), dpi=150); plt.close()

    plt.figure()
    plt.plot(t_grid, P, linewidth=2, label="P(hit)")
    plt.axvline(best["t1"], linestyle="--", linewidth=1, label="t₁*")
    plt.axvline(best["t2"], linestyle="--", linewidth=1, label="t₂*")
    plt.xlabel("孕周（周）"); plt.ylabel("P(hit)")
    plt.title(f"P(hit) 曲线与推荐时点（{prefix}）"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"q4_dual_phit_{prefix}.png"), dpi=150); plt.close()

    return {"t1_single":t1_single,"J1":J1_min}, best

# ---------- 主流程 ----------
def run(excel="附件.xlsx", clean_csv=None, sheet=None, outdir="outputs_q4_dual",
        sex="auto", y_threshold=0.04, female_rule="draw1", pass_col=None, pass_pos="是",
        deg=3, include_interactions=True, sigma_mode="ga_local",
        ga_min=10.0, ga_max=29.0, ga_step=0.25,
        c1=1.0, cr=1.0, lam=1.0, kappa=0.05, alpha=1.0,
        delta_choices="1,1.5,2,3"):
    ensure_outdir(outdir)
    df=load_data(excel, clean_csv, sheet)
    df, used_rule = define_hit(df, sex=sex, y_threshold=y_threshold,
                               female_rule=female_rule, pass_col=pass_col, pass_pos=pass_pos)
    print(f"[INFO] 样本量: {len(df)} | 标签规则: {used_rule}")
    df["BMI_group"]=make_bmi_group(df["BMI"])

    recs=[]
    for g in (df["BMI_group"].cat.categories if hasattr(df["BMI_group"],"cat")
              else sorted(df["BMI_group"].dropna().unique())):
        sub=df[df["BMI_group"]==g].copy()
        if len(sub)==0: continue
        scen={"BMI": float(np.nanmedian(sub["BMI"])),
              "Age": float(np.nanmedian(sub["Age"])) if "Age" in sub.columns else None,
              "Height": float(np.nanmedian(sub["Height"])) if "Height" in sub.columns else None,
              "Weight": float(np.nanmedian(sub["Weight"])) if "Weight" in sub.columns else None}

        if used_rule.startswith("male"):
            mu_fn=fit_linear_mu(sub, deg=deg, include_interactions=include_interactions)
            sigma_fn=estimate_sigma(sub, mu_fn, mode=sigma_mode, bin_width=0.5,
                                    qc_cols=[c for c in ["unique_mapped","filt_ratio","GC"] if c in sub.columns])
            def phit_fn(dn):
                mu=mu_fn(dn); sigma=sigma_fn(dn)
                z=(y_threshold - mu)/np.maximum(sigma,1e-4)
                return 1.0 - np.array([normal_cdf(zz) for zz in z])
        else:
            p_fn=build_poly_prob(sub, deg=deg, include_interactions=include_interactions, label_col="hit")
            def phit_fn(dn): return p_fn(dn)

        single, two = optimize_strategy(phit_fn, scen, outdir,
                                        prefix=f"{g}_{'M' if used_rule.startswith('male') else 'F'}",
                                        ga_min=ga_min, ga_max=ga_max, ga_step=ga_step,
                                        c1=c1, cr=cr, lam=lam, kappa=kappa, alpha=alpha,
                                        delta_choices=tuple(float(x) for x in str(delta_choices).split(",") if x))

        recs.append({
            "BMI_group": str(g), "n": len(sub), "label_rule": used_rule,
            "t1_single": round(single["t1_single"],2), "J1": round(single["J1"],4),
            "t1_two": round(two["t1"],2), "delta": round(two["d"],2), "t2_two": round(two["t2"],2),
            "Psucc_two": round(two["Psucc"],4), "E_Tres_two": round(two["E_Tres"],3), "J2": round(two["J2"],4)
        })
    out_csv=os.path.join(outdir, f"q4_dual_policy_table_{'M' if used_rule.startswith('male') else 'F'}.csv")
    pd.DataFrame(recs).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[OK] 已保存：", out_csv)

def main():
    ap=argparse.ArgumentParser(description="Q4（第三问思想版，Dual）：男胎(μ/σ)/女胎(多项式) + 策略优化")
    ap.add_argument("--excel", type=str, default="附件.xlsx")
    ap.add_argument("--clean_csv", type=str, default=None)
    ap.add_argument("--sheet", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="outputs_q4_dual")
    ap.add_argument("--sex", type=str, default="auto", choices=["auto","male","female"])
    ap.add_argument("--y_threshold", type=float, default=0.04)
    ap.add_argument("--female_rule", type=str, default="draw1", choices=["draw1","pass_col"])
    ap.add_argument("--pass_col", type=str, default=None)
    ap.add_argument("--pass_pos", type=str, default="是")
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--no_inter", action="store_true")
    ap.add_argument("--sigma_mode", type=str, default="ga_local", choices=["global","ga_local"])
    ap.add_argument("--ga_min", type=float, default=10.0)
    ap.add_argument("--ga_max", type=float, default=29.0)
    ap.add_argument("--ga_step", type=float, default=0.25)
    ap.add_argument("--delta_choices", type=str, default="1,1.5,2,3")
    ap.add_argument("--c1", type=float, default=1.0)
    ap.add_argument("--cr", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=1.0)
    args=ap.parse_args(); warnings.filterwarnings("ignore")
    run(excel=args.excel, clean_csv=args.clean_csv, sheet=args.sheet, outdir=args.outdir,
        sex=args.sex, y_threshold=args.y_threshold, female_rule=args.female_rule, pass_col=args.pass_col, pass_pos=args.pass_pos,
        deg=args.deg, include_interactions=not args.no_inter, sigma_mode=args.sigma_mode,
        ga_min=args.ga_min, ga_max=args.ga_max, ga_step=args.ga_step,
        c1=args.c1, cr=args.cr, lam=args.lam, kappa=args.kappa, alpha=args.alpha,
        delta_choices=args.delta_choices)
if __name__=="__main__": main()
