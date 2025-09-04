# -*- coding: utf-8 -*-
"""
NIPT Q4 (ML版)：可解释概率模型 + 校准 + 策略优化（按 BMI 组）
优先 LightGBM(对 GA 单调↑) -> EBM -> 样条GLM(GAM风格)，并做 Isotonic/Platt 校准。
输出：策略表与图。
"""
import os, re, math, argparse, warnings, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 静音与中文字体
import logging, matplotlib
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams['font.family'] = [
    'Microsoft YaHei','SimHei','Arial Unicode MS','STHeiti',
    'PingFang SC','WenQuanYi Zen Hei','Source Han Sans SC','sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# ----------------- 数据 & 工具 -----------------
def parse_ga_to_weeks(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int,float,np.integer,np.floating)):
        try: return float(x)
        except: return np.nan
    s = str(x).strip().replace("W","w").replace("D","d").replace("周","w").replace("天","d")
    s = s.replace("．",".").replace("＋","+")
    m = re.match(r"^\s*(\d{1,2})\s*(?:w)?\s*(?:\+)?\s*(\d{1,2})?\s*(?:d)?\s*$", s)
    if m:
        w = int(m.group(1)); d = int(m.group(2)) if m.group(2) else 0
        return w + d/7.0
    try: return float(s)
    except: return np.nan

def guess_columns(df):
    aliases = {
        "ga": ["检测孕周","孕周","孕周(周+天)","孕周（周+天）","GA"],
        "bmi": ["孕妇BMI","BMI"],
        "y_frac": ["Y染色体浓度","Y浓度","Y浓度(%)","fetal_fraction_Y","Y_fetal_fraction"],
        "age": ["年龄","Age"],
        "height": ["身高","Height","身高(cm)","身高（cm）"],
        "weight": ["体重","Weight","体重(kg)","体重（kg）"],
        "gc": ["整体GC","GC含量","GC"],
    }
    m = {}
    low = {str(c).strip().lower(): c for c in df.columns}
    for k, cands in aliases.items():
        for c in cands:
            key = str(c).strip().lower()
            if key in low:
                m[k] = low[key]; break
    return m

def to_ratio(y):
    y = pd.to_numeric(y, errors="coerce")
    if (y > 1).mean(skipna=True) > 0.5: return y/100.0
    return y

def make_bmi_group(bmi):
    bins_bmi = [20, 28, 32, 36, 40, 100]
    labels_bmi = ["[20,28)","[28,32)","[32,36)","[36,40)","[40,+)"]
    return pd.cut(bmi, bins=bins_bmi, labels=labels_bmi, right=False)

def load_clean(excel_path, clean_csv=None, sheet=None, y_thr=0.04):
    if clean_csv and os.path.exists(clean_csv):
        df = pd.read_csv(clean_csv)
        if "GA_weeks" not in df.columns and "检测孕周" in df.columns:
            df["GA_weeks"] = df["检测孕周"].apply(parse_ga_to_weeks)
        if "BMI" not in df.columns and "孕妇BMI" in df.columns:
            df["BMI"] = pd.to_numeric(df["孕妇BMI"], errors="coerce")
        if "Y_frac" not in df.columns and "Y染色体浓度" in df.columns:
            df["Y_frac"] = to_ratio(df["Y染色体浓度"])
        if "Age" not in df.columns and "年龄" in df.columns:
            df["Age"] = pd.to_numeric(df["年龄"], errors="coerce")
        for k,std in [("身高","Height"),("体重","Weight")]:
            if k in df.columns and std not in df.columns:
                df[std] = pd.to_numeric(df[k], errors="coerce")
    else:
        df_raw = pd.read_excel(excel_path, sheet_name=sheet) if sheet else pd.read_excel(excel_path)
        mp = guess_columns(df_raw); df = df_raw.copy()
        df["GA_weeks"] = df[mp["ga"]].apply(parse_ga_to_weeks)
        df["BMI"] = pd.to_numeric(df[mp["bmi"]], errors="coerce")
        df["Y_frac"] = to_ratio(df[mp["y_frac"]])
        if "age" in mp: df["Age"] = pd.to_numeric(df[mp["age"]], errors="coerce")
        if "height" in mp: df["Height"] = pd.to_numeric(df[mp["height"]], errors="coerce")
        if "weight" in mp: df["Weight"] = pd.to_numeric(df[mp["weight"]], errors="coerce")
        if "gc" in mp: df["GC"] = pd.to_numeric(df[mp["gc"]], errors="coerce")

    # 过滤范围
    df = df[(~df["GA_weeks"].isna()) & (~df["BMI"].isna()) & (~df["Y_frac"].isna())]
    df = df[(df["GA_weeks"]>=9)&(df["GA_weeks"]<=30)&(df["BMI"]>10)&(df["BMI"]<60)&(df["Y_frac"]>0)&(df["Y_frac"]<0.5)]
    # 标签
    df["hit"] = (df["Y_frac"] >= y_thr).astype(int)
    return df

# ----------------- 模型 & 校准 -----------------
def fit_model(X, y, model_pref="auto", monotone=True, ga_col=0, random_state=42):
    # 1) LightGBM（单调）
    if model_pref in ("auto","lgbm"):
        try:
            import lightgbm as lgb
            mono = [0]*X.shape[1]
            if monotone and 0 <= ga_col < X.shape[1]:
                mono[ga_col] = 1  # GA 单调↑
            clf = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                random_state=random_state,
                monotone_constraints=mono if monotone else None
            )
            clf.fit(X, y)
            return (lambda Xn: clf.predict_proba(Xn)[:,1], "lgbm")
        except Exception:
            pass
    # 2) EBM
    if model_pref in ("auto","ebm"):
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
            ebm = ExplainableBoostingClassifier(random_state=random_state)
            ebm.fit(X, y)
            return (lambda Xn: ebm.predict_proba(Xn)[:,1], "ebm")
        except Exception:
            pass
    # 3) 朴素 GLM（无依赖）
    try:
        import statsmodels.api as sm
        ga = X[:,0]
        Phi = np.column_stack([np.ones_like(ga), ga, ga**2, ga**3, X[:,1:]])
        model = sm.GLM(y, Phi, family=sm.families.Binomial())
        res = model.fit()
        def proba(Xn):
            ga = Xn[:,0]
            Phi = np.column_stack([np.ones_like(ga), ga, ga**2, ga**3, Xn[:,1:]])
            eta = res.predict(Phi)
            return np.clip(eta, 1e-6, 1-1e-6)
        return (proba, "glm_poly3")
    except Exception:
        pass
    # 4) 兜底常数
    pbar = float(np.clip(np.mean(y), 1e-6, 1-1e-6))
    return (lambda Xn: np.full(Xn.shape[0], pbar), "constant")

def fit_calibrator(p_raw, y, method="isotonic"):
    if method=="isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p_raw, y)
            return lambda z: np.clip(ir.predict(z), 0, 1), "isotonic"
        except Exception:
            pass
    if method=="platt":
        try:
            from sklearn.linear_model import LogisticRegression
            z = np.clip(p_raw, 1e-6, 1-1e-6)
            x = np.log(z/(1-z)).reshape(-1,1)
            lr = LogisticRegression(max_iter=1000).fit(x, y)
            return lambda s: lr.predict_proba(np.log(np.clip(s,1e-6,1-1e-6)/(1-np.clip(s,1e-6,1-1e-6))).reshape(-1,1))[:,1], "platt"
        except Exception:
            pass
    return (lambda z: np.clip(z, 0, 1), "none")

# ----------------- 策略优化 -----------------
def risk_level(t):
    if t <= 12: return 1.0
    elif t < 28: return 2.0
    else: return 3.0

def optimize_for_group(df, proba_fn, cal_fn, outdir,
                       ga_min=10.0, ga_max=29.0, ga_step=0.25,
                       c1=1.0, cr=1.0, lam=1.0, kappa=0.05, alpha=1.0,
                       delta_choices=(1,1.5,2,3)):
    scen = {"BMI": float(np.nanmedian(df["BMI"])),
            "Age": float(np.nanmedian(df["Age"])) if "Age" in df.columns else None,
            "Height": float(np.nanmedian(df["Height"])) if "Height" in df.columns else None,
            "Weight": float(np.nanmedian(df["Weight"])) if "Weight" in df.columns else None}
    def X_from_t(t):
        cols = [float(t), scen["BMI"]]
        for k in ["Age","Height","Weight"]:
            if scen[k] is not None: cols.append(scen[k])
        return np.array(cols, dtype=float).reshape(1,-1)
    def phit_t(t):
        p = proba_fn(X_from_t(t))
        return float(cal_fn(np.array([p]))[0])

    t_grid = np.arange(ga_min, ga_max+1e-9, ga_step).astype(float)
    P_grid = np.array([phit_t(t) for t in t_grid])

    # 单次
    J1 = np.array([risk_level(t) + lam*(1-P_grid[i]) + c1 + kappa*(t - ga_min) for i,t in enumerate(t_grid)])
    t1_single = float(t_grid[np.argmin(J1)])
    J1_min = float(J1.min())

    # 两次
    best = None
    heat = np.full((len(delta_choices), len(t_grid)), np.nan)
    for di, d in enumerate(delta_choices):
        valid = t_grid[t_grid + d <= ga_max + 1e-9]
        for i, t1 in enumerate(valid):
            t2 = t1 + d
            P1 = phit_t(t1); P2 = phit_t(t2)
            P2p = alpha*P2 + (1-alpha)*P1
            Psucc = P1 + (1-P1)*P2p
            E_Tres = P1*t1 + (1-P1)*t2
            E_risk = P1*risk_level(t1) + (1-P1)*risk_level(t2)
            J2 = c1 + (1-P1)*cr + E_risk + lam*(1-Psucc) + kappa*(E_Tres - ga_min)
            heat[di, i] = J2
            if (best is None) or (J2 < best["J2"]):
                best = dict(t1=t1, d=d, t2=t2, J2=J2, Psucc=Psucc, E_Tres=E_Tres)

    # 图
    plt.figure()
    extent = [t_grid.min(), t_grid.max(), 0, len(delta_choices)]
    plt.imshow(heat, aspect='auto', origin='lower', extent=extent)
    plt.colorbar(label="J₂")
    yticks = np.arange(len(delta_choices)) + 0.5
    plt.yticks(yticks, [f"Δ={d}" for d in delta_choices])
    plt.axvline(best["t1"], linestyle="--", linewidth=1)
    plt.xlabel("首检孕周 t₁（周）"); plt.ylabel("复检间隔 Δ（周）"); plt.title("J₂(t₁,Δ) 热力图")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "q4_ml_heatmap.png"), dpi=150); plt.close()

    plt.figure()
    plt.plot(t_grid, P_grid, linewidth=2, label="P(hit)")
    plt.axvline(best["t1"], linestyle="--", linewidth=1, label="t₁*")
    plt.axvline(best["t2"], linestyle="--", linewidth=1, label="t₂*")
    plt.xlabel("孕周（周）"); plt.ylabel("P(hit)"); plt.title("P(hit) 曲线与推荐时点"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "q4_ml_phit.png"), dpi=150); plt.close()

    return {"t1_single": t1_single, "J1": J1_min}, best

def run(excel="附件.xlsx", clean_csv="outputs_q1/q1_clean_male.csv", sheet=None,
        outdir="outputs_q4_ml",
        y_thr=0.04, model="auto", cal="isotonic", monotone=True,
        ga_min=10.0, ga_max=29.0, ga_step=0.25,
        c1=1.0, cr=1.0, lam=1.0, kappa=0.05, alpha=1.0,
        delta_choices="1,1.5,2,3", seed=42):
    os.makedirs(outdir, exist_ok=True)
    df = load_clean(excel, clean_csv=clean_csv, sheet=sheet, y_thr=y_thr)
    print("[INFO] 样本量：", len(df))

    cols = ["GA_weeks","BMI"] + [c for c in ["Age","Height","Weight"] if c in df.columns]
    X = df[cols].values.astype(float)
    y = df["hit"].values.astype(int)

    proba_fn, used_model = fit_model(X, y, model_pref=model, monotone=monotone, ga_col=0, random_state=seed)
    p_raw = proba_fn(X)

    cal_fn, used_cal = fit_calibrator(p_raw, y, method=cal)

    df["BMI_group"] = make_bmi_group(df["BMI"])
    recs = []
    for g in (df["BMI_group"].cat.categories if hasattr(df["BMI_group"],"cat") else sorted(df["BMI_group"].dropna().unique())):
        sub = df[df["BMI_group"]==g]
        if len(sub)==0: continue
        single, two = optimize_for_group(sub, proba_fn, cal_fn, outdir,
                                         ga_min=ga_min, ga_max=ga_max, ga_step=ga_step,
                                         c1=c1, cr=cr, lam=lam, kappa=kappa, alpha=alpha,
                                         delta_choices=tuple(float(x) for x in str(delta_choices).split(",") if x))
        recs.append({
            "BMI_group": str(g), "n": len(sub),
            "t1_single": round(single["t1_single"],2), "J1": round(single["J1"],4),
            "t1_two": round(two["t1"],2), "delta": round(two["d"],2), "t2_two": round(two["t2"],2),
            "Psucc_two": round(two["Psucc"],4), "E_Tres_two": round(two["E_Tres"],3), "J2": round(two["J2"],4),
            "model": used_model, "calibration": used_cal
        })

    res = pd.DataFrame(recs)
    res.to_csv(os.path.join(outdir, "q4_ml_policy_table.csv"), index=False, encoding="utf-8-sig")
    print("[OK] 已保存：", os.path.join(outdir, "q4_ml_policy_table.csv"))
    return used_model, used_cal

def main():
    ap = argparse.ArgumentParser(description="NIPT Q4 (ML)：可解释模型+校准+策略优化")
    ap.add_argument("--excel", type=str, default="附件.xlsx")
    ap.add_argument("--clean_csv", type=str, default="outputs_q1/q1_clean_male.csv")
    ap.add_argument("--sheet", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="outputs_q4_ml")
    ap.add_argument("--y_thr", type=float, default=0.04)
    ap.add_argument("--model", type=str, default="auto", choices=["auto","lgbm","ebm","glm"])
    ap.add_argument("--cal", type=str, default="isotonic", choices=["isotonic","platt","none"])
    ap.add_argument("--no_monotone", action="store_true")
    ap.add_argument("--ga_min", type=float, default=10.0)
    ap.add_argument("--ga_max", type=float, default=29.0)
    ap.add_argument("--ga_step", type=float, default=0.25)
    ap.add_argument("--delta_choices", type=str, default="1,1.5,2,3")
    ap.add_argument("--c1", type=float, default=1.0)
    ap.add_argument("--cr", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    used_model, used_cal = run(excel=args.excel, clean_csv=args.clean_csv, sheet=args.sheet,
               outdir=args.outdir, y_thr=args.y_thr, model=args.model, cal=args.cal,
               monotone=not args.no_monotone,
               ga_min=args.ga_min, ga_max=args.ga_max, ga_step=args.ga_step,
               c1=args.c1, cr=args.cr, lam=args.lam, kappa=args.kappa, alpha=args.alpha,
               delta_choices=args.delta_choices, seed=args.seed)
    print("[INFO] 使用模型/校准：", used_model, "/", used_cal)

if __name__ == "__main__":
    main()
