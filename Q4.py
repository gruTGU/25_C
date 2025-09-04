# -*- coding: utf-8 -*-
import os
import re
import math
import argparse
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- 安静字体/告警设置（与前面风格保持一致） --------------------
import logging
import matplotlib
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

# -----------------------------
# 工具函数（与Q3一致/精简）
# -----------------------------

def parse_ga_to_weeks(x):
    """将孕周字符串或数字统一为周（float）。如: '11w+6' -> 11 + 6/7"""
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        try: return float(x)
        except Exception: return np.nan
    s = str(x).strip()
    s = s.replace("W","w").replace("D","d").replace("周","w").replace("天","d")
    s = s.replace("．",".").replace("＋","+")
    m = re.match(r"^\s*(\d{1,2})\s*(?:w)?\s*(?:\+)?\s*(\d{1,2})?\s*(?:d)?\s*$", s)
    if m:
        w = int(m.group(1)); d = int(m.group(2)) if m.group(2) else 0
        return w + d/7.0
    try: return float(s)
    except Exception: return np.nan


def guess_columns(df):
    """
    依据常见别名猜测列名映射，返回 {标准名: 实际列名}
    标准名: id, age, ga, bmi, y_frac, date, gc, unique_mapped, filt_ratio, height, weight
    """
    aliases = {
        "id": ["孕妇代码","孕妇ID","孕妇编号","受检者ID","样本ID","样本编号","ID","code","patient_id"],
        "age": ["年龄","Age","C"],
        "ga": ["检测孕周","孕周","孕周(周+天)","孕周（周+天）","J","GA","gestational_age"],
        "bmi": ["孕妇BMI","BMI","K"],
        "y_frac": ["Y染色体浓度","Y浓度","Y浓度(%)","V","fetal_fraction_Y","Y_fetal_fraction"],
        "date": ["检测日期","检测时间","日期","H","sample_time","采血时间"],
        "unique_mapped": ["唯一比对读段数","唯一比对reads","O","unique_mapped_reads"],
        "gc": ["整体GC","GC含量","P","GC"],
        "filt_ratio": ["被过滤读段比例","AA","filtered_ratio"],
        "height": ["身高","Height","身高(cm)","身高（cm）"],
        "weight": ["体重","Weight","体重(kg)","体重（kg）"],
    }
    mapping = {}
    low = {str(c).strip().lower(): c for c in df.columns}
    for std, cands in aliases.items():
        for c in cands:
            key = str(c).strip().lower()
            if key in low:
                mapping[std] = low[key]
                break
    return mapping


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def to_ratio(y):
    """将可能为百分比的 Y 浓度转换为比例 0-1"""
    y = pd.to_numeric(y, errors="coerce")
    if (y > 1).mean(skipna=True) > 0.5:
        return y / 100.0
    return y


def normal_cdf(x):
    """标准正态分布 CDF"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def make_bmi_group(bmi):
    bins_bmi = [20, 28, 32, 36, 40, 100]
    labels_bmi = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "[40,+)"]
    return pd.cut(bmi, bins=bins_bmi, labels=labels_bmi, right=False)


# ---- 均值 μ 与方差 σ 的拟合（与Q3思路一致，轻依赖版） ----

def build_design_matrix(df, deg=3, include_interactions=True):
    GA = df["GA_weeks"].values.astype(float)
    BMI = df["BMI"].values.astype(float)
    cols = [np.ones_like(GA)]
    # GA 多项式
    for d in range(1, deg+1):
        cols.append(GA**d)
    # 线性项
    cols.append(BMI)
    if "Age" in df: cols.append(df["Age"].values.astype(float))
    if "Height" in df: cols.append(df["Height"].values.astype(float))
    if "Weight" in df: cols.append(df["Weight"].values.astype(float))
    # 交互：GA^d * BMI
    if include_interactions:
        for d in range(1, deg+1):
            cols.append((GA**d) * BMI)
    X = np.column_stack(cols)
    return X


def fit_linear_mu(df, deg=3, include_interactions=True):
    X = build_design_matrix(df, deg=deg, include_interactions=include_interactions)
    y = df["Y_frac"].values.astype(float)
    ok = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
    X = X[ok]; y = y[ok]
    if X.size == 0:
        # 极端情况，返回常数函数
        c = float(np.nanmedian(df["Y_frac"]))
        return (lambda d: np.full(len(d), max(min(c,0.49),1e-4)))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    def mu_fn(df_new):
        Xn = build_design_matrix(df_new, deg=deg, include_interactions=include_interactions)
        yhat = Xn @ beta
        return np.clip(yhat, 1e-4, 0.499)
    return mu_fn


def estimate_sigma(df, mu_fn, mode="ga_local", bin_width=0.5, qc_cols=None):
    """
    健壮的 σ 估计，避免小样本/单点导致的索引错误。
    """
    y = df["Y_frac"].values.astype(float)
    mu = mu_fn(df)
    resid = y - mu
    resid = resid[~np.isnan(resid)]
    if len(resid) == 0:
        return lambda d: np.full(len(d), 0.02)

    if mode == "global":
        s = float(np.nanstd(resid))
        s = max(s, 0.01)
        return lambda d: np.full(len(d), s)

    if mode == "ga_local":
        ga = df["GA_weeks"].values.astype(float)
        ok = (~np.isnan(ga)) & (~np.isnan(y))
        ga = ga[ok]; rr = (y - mu)[ok]

        if len(ga) == 0:
            s = float(np.clip(np.nanstd(resid), 0.01, 0.05))
            return lambda d: np.full(len(d), s)

        start = max(9.0, float(np.nanmin(ga)))
        stop  = min(30.0, float(np.nanmax(ga)))
        if not np.isfinite(start) or not np.isfinite(stop):
            s = float(np.clip(np.nanstd(resid), 0.01, 0.05))
            return lambda d: np.full(len(d), s)
        if stop - start < 1e-6:
            bins = np.array([start - bin_width, start + bin_width], dtype=float)
        else:
            bins = np.arange(start, stop + bin_width, bin_width, dtype=float)
            if bins.size < 2:
                bins = np.array([start - bin_width, start + bin_width], dtype=float)

        mids = (bins[:-1] + bins[1:]) / 2.0
        global_sd = float(np.nanstd(rr))
        if not np.isfinite(global_sd) or global_sd == 0:
            global_sd = 0.02

        if mids.size == 0:
            s = float(np.clip(global_sd, 0.005, 0.06))
            return lambda d: np.full(len(d), s)

        sds = np.empty_like(mids)
        sds.fill(np.nan)
        for i in range(mids.size):
            m = (ga >= bins[i]) & (ga < bins[i+1])
            if np.any(m):
                sds[i] = np.nanstd(rr[m])

        sds = np.where(np.isnan(sds), global_sd, sds)
        sds = np.clip(sds, 0.005, 0.06)

        def sigma_fn(dnew):
            GA = dnew["GA_weeks"].values.astype(float)
            if sds.size == 0:
                return np.full(len(GA), float(np.clip(global_sd, 0.005, 0.06)))
            idx = np.digitize(GA, bins) - 1
            idx = np.clip(idx, 0, sds.size - 1)
            return sds[idx]
        return sigma_fn

    if mode == "by_qc" and qc_cols:
        ga = df["GA_weeks"].values.astype(float)
        ok = (~np.isnan(ga)) & (~np.isnan(y))
        if not np.any(ok):
            s = float(np.clip(np.nanstd(resid), 0.01, 0.05))
            return lambda d: np.full(len(d), s)
        bins = np.arange(max(9, np.nanmin(ga[ok])), min(30, np.nanmax(ga[ok])) + 0.5, 0.5)
        if bins.size < 2:
            s = float(np.clip(np.nanstd(resid), 0.01, 0.05))
            return lambda d: np.full(len(d), s)

        idx = np.digitize(ga[ok], bins) - 1
        local_std = np.zeros(np.sum(ok), dtype=float)
        for i in range(np.min(idx), np.max(idx)+1):
            m = (idx == i)
            if np.any(m):
                local_std[m] = np.nanstd((y - mu)[ok][m])
        if np.all(local_std == 0):
            local_std[:] = np.nanmedian(local_std) if np.isfinite(np.nanmedian(local_std)) else np.nanstd(resid)

        Zs = []
        for c in qc_cols:
            if c in df.columns:
                z = pd.to_numeric(df.loc[ok, c], errors="coerce").values.astype(float)
                m, s = np.nanmean(z), np.nanstd(z)
                s = s if (s and np.isfinite(s)) else 1.0
                Zs.append((z - m) / s)
            else:
                Zs.append(np.zeros(np.sum(ok)))
        Z = np.column_stack(Zs + [np.ones(np.sum(ok))])
        coefs, *_ = np.linalg.lstsq(Z, local_std, rcond=None)
        def sigma_fn(dnew):
            Feats = []
            for c in qc_cols:
                if c in dnew.columns:
                    z = pd.to_numeric(dnew[c], errors="coerce").values.astype(float)
                    m, s = np.nanmean(z), np.nanstd(z)
                    s = s if (s and np.isfinite(s)) else 1.0
                    Feats.append((z - m) / s)
                else:
                    Feats.append(np.zeros(len(dnew)))
            Zn = np.column_stack(Feats + [np.ones(len(dnew))])
            s_hat = Zn @ coefs
            s_hat = np.clip(s_hat, 0.005, 0.06)
            return s_hat
        return sigma_fn

    s = float(np.nanstd(resid))
    s = np.clip(s, 0.01, 0.05)
    return lambda d: np.full(len(d), s)


def load_clean_male(excel_path, sheet_name=None, clean_csv=None,
                    min_ga=9.0, max_ga=30.0, gc_low=0.3, gc_high=0.7):
    """
    优先读 Q1 清洗结果；否则从 Excel 清洗。
    """
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
        for k, std in [("身高","Height"), ("体重","Weight")]:
            if k in df.columns and std not in df.columns:
                df[std] = pd.to_numeric(df[k], errors="coerce")
        if "is_male" not in df.columns:
            df["is_male"] = df["Y_frac"] > 0.005

        df = df[(~df["GA_weeks"].isna()) & (~df["BMI"].isna()) & (~df["Y_frac"].isna())]
        df = df[(df["GA_weeks"] >= min_ga) & (df["GA_weeks"] <= max_ga)]
        df = df[(df["BMI"] > 10) & (df["BMI"] < 60)]
        df = df[(df["Y_frac"] > 0) & (df["Y_frac"] < 0.5)]
        df = df[df["is_male"]].copy()
        return df

    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(excel_path)
    mapping = guess_columns(df_raw)
    need = ["ga","bmi","y_frac"]
    for k in need:
        if k not in mapping:
            raise ValueError("Excel 缺少必要列，至少需要 孕周/孕妇BMI/Y浓度。")
    df = df_raw.copy()
    df["GA_weeks"] = df[mapping["ga"]].apply(parse_ga_to_weeks)
    df["BMI"] = pd.to_numeric(df[mapping["bmi"]], errors="coerce")
    df["Y_frac"] = to_ratio(df[mapping["y_frac"]])
    if "age" in mapping:
        df["Age"] = pd.to_numeric(df[mapping["age"]], errors="coerce")
    if "height" in mapping:
        df["Height"] = pd.to_numeric(df[mapping["height"]], errors="coerce")
    if "weight" in mapping:
        df["Weight"] = pd.to_numeric(df[mapping["weight"]], errors="coerce")
    if "gc" in mapping:
        df["GC"] = pd.to_numeric(df[mapping["gc"]], errors="coerce")
    if "filt_ratio" in mapping:
        df["filt_ratio"] = pd.to_numeric(df[mapping["filt_ratio"]], errors="coerce")
    if "unique_mapped" in mapping:
        df["unique_mapped"] = pd.to_numeric(df[mapping["unique_mapped"]], errors="coerce")

    df = df[(~df["GA_weeks"].isna()) & (~df["BMI"].isna()) & (~df["Y_frac"].isna())]
    df = df[(df["GA_weeks"] >= min_ga) & (df["GA_weeks"] <= max_ga)]
    df = df[(df["BMI"] > 10) & (df["BMI"] < 60)]
    df = df[(df["Y_frac"] > 0) & (df["Y_frac"] < 0.5)]
    df["is_male"] = df["Y_frac"] > 0.005
    df = df[df["is_male"]].copy()
    if "GC" in df.columns:
        df = df[(df["GC"] >= gc_low) & (df["GC"] <= gc_high)]
    return df


# -----------------------------
# P_hit 构建（复用 Q3 思路）
# -----------------------------

def build_phit_model(df, deg=3, include_interactions=True, sigma_mode="ga_local", y_threshold=0.04):
    mu_fn = fit_linear_mu(df, deg=deg, include_interactions=include_interactions)
    qc_cols = [c for c in ["unique_mapped","filt_ratio","GC"] if c in df.columns]
    sigma_fn = estimate_sigma(df, mu_fn, mode=sigma_mode, bin_width=0.5, qc_cols=qc_cols)
    def phit_fn(df_new):
        mu = mu_fn(df_new)
        sigma = sigma_fn(df_new)
        z = (y_threshold - mu) / np.maximum(sigma, 1e-4)
        phit = 1.0 - np.array([normal_cdf(zz) for zz in z])
        return np.clip(phit, 0.0, 1.0)
    return phit_fn


# -----------------------------
# 策略评价
# -----------------------------

def risk_level(t):
    if t <= 12: return 1.0
    elif t < 28: return 2.0
    else: return 3.0


def eval_single_test(t, phit, c1=1.0, lam=1.0, kappa=0.05, tmin=10.0):
    P1 = float(phit(t))
    J = risk_level(t) + lam*(1 - P1) + c1 + kappa*(t - tmin)
    return dict(J=J, P_success=P1, E_draws=1.0, E_Tres=t)


def eval_two_step(t1, delta, phit, c1=1.0, cr=1.0, lam=1.0, kappa=0.05, tmin=10.0, alpha=1.0):
    t2 = t1 + delta
    P1 = float(phit(t1))
    P2 = float(phit(t2))
    # 相关性调整：alpha=1 独立，alpha=0 完全相关（P2'=P1）
    P2p = alpha * P2 + (1 - alpha) * P1
    Psucc = P1 + (1 - P1) * P2p
    E_draws = 2 - P1  # 失败才二抽
    E_Tres = P1 * t1 + (1 - P1) * t2
    E_risk = P1 * risk_level(t1) + (1 - P1) * risk_level(t2)
    J = c1 + (1 - P1) * cr + E_risk + lam * (1 - Psucc) + kappa * (E_Tres - tmin)
    return dict(J=J, P_success=Psucc, E_draws=E_draws, E_Tres=E_Tres, t2=t2)


# -----------------------------
# 主流程：分组优化（按 BMI 组）
# -----------------------------

def run_q4(excel_path="附件.xlsx",
           clean_csv="outputs_q1/q1_clean_male.csv",
           sheet_name=None,
           outdir="outputs_q4",
           deg=3,
           include_interactions=True,
           sigma_mode="ga_local",
           y_threshold=0.04,
           ga_min=10.0, ga_max=29.0, ga_step=0.25,
           c1=1.0, cr=1.0, lam=1.0, kappa=0.05,
           alpha=1.0,              # 相关性：1表示独立，0表示完全相关
           delta_choices="1,1.5,2,3"):
    """
    Q4：在P_hit模型上做策略优化，给出一次与两次检测的最优时点。
    """
    ensure_outdir(outdir)
    print(f"[INFO] 读取数据：{clean_csv if (clean_csv and os.path.exists(clean_csv)) else excel_path}")
    df = load_clean_male(excel_path, sheet_name=sheet_name, clean_csv=clean_csv)
    print("[INFO] 样本量（男胎/清洗后）:", len(df))
    print("孕周范围: [{:.2f}, {:.2f}] 周".format(df["GA_weeks"].min(), df["GA_weeks"].max()))
    print("BMI 中位数: {:.2f}".format(df["BMI"].median()))
    print("Y 浓度中位数: {:.3f}".format(df["Y_frac"].median()))

    # 构建 P_hit
    phit_fn = build_phit_model(df, deg=deg, include_interactions=include_interactions,
                               sigma_mode=sigma_mode, y_threshold=y_threshold)

    # 分组中位情景
    df["BMI_group"] = make_bmi_group(df["BMI"])
    groups = df["BMI_group"].cat.categories if hasattr(df["BMI_group"], "cat") else sorted(df["BMI_group"].dropna().unique().tolist())

    t_grid = np.arange(ga_min, ga_max + 1e-9, ga_step).astype(float)
    deltas = [float(x) for x in str(delta_choices).split(",") if str(x).strip()!=""]
    recs = []

    def phit_scalar(t, scen):
        tmp = pd.DataFrame({"GA_weeks":[t],
                            "BMI":[scen["BMI"]],
                            "Age":[scen["Age"]] if scen["Age"] is not None else None,
                            "Height":[scen["Height"]] if scen["Height"] is not None else None,
                            "Weight":[scen["Weight"]] if scen["Weight"] is not None else None})
        return float(phit_fn(tmp)[0])

    for g in groups:
        sub = df[df["BMI_group"] == g].copy()
        if len(sub) == 0:
            continue
        scen = {
            "BMI": float(np.nanmedian(sub["BMI"])),
            "Age": float(np.nanmedian(df["Age"])) if "Age" in df.columns else None,
            "Height": float(np.nanmedian(df["Height"])) if "Height" in df.columns else None,
            "Weight": float(np.nanmedian(df["Weight"])) if "Weight" in df.columns else None,
        }

        # 预先向量化 phit
        def _phit_vec(ts):
            tmp = pd.DataFrame({"GA_weeks": ts,
                                "BMI": scen["BMI"],
                                "Age": scen["Age"] if scen["Age"] is not None else None,
                                "Height": scen["Height"] if scen["Height"] is not None else None,
                                "Weight": scen["Weight"] if scen["Weight"] is not None else None})
            return phit_fn(tmp)

        P_vec = _phit_vec(t_grid)

        # 单次检测基线
        J1 = np.array([eval_single_test(t, lambda x: _phit_vec(np.array([x]))[0], c1=c1, lam=lam, kappa=kappa)["J"] for t in t_grid])
        t1_star_single = float(t_grid[np.argmin(J1)])
        rec_single = eval_single_test(t1_star_single, lambda x: _phit_vec(np.array([x]))[0], c1=c1, lam=lam, kappa=kappa)

        # 两次检测搜索
        J_mat = []
        Psucc_mat = []
        for d in deltas:
            # 合法 t1 区间需保证 t1+d <= ga_max
            valid = t_grid[t_grid + d <= ga_max + 1e-9]
            if valid.size == 0:
                J_mat.append(np.array([])); Psucc_mat.append(np.array([]))
                continue
            # 计算每个 t1 的目标
            Js = []
            Ps = []
            for t1 in valid:
                res = eval_two_step(t1, d, lambda tt: phit_scalar(tt, scen),
                                    c1=c1, cr=cr, lam=lam, kappa=kappa, alpha=alpha)
                Js.append(res["J"]); Ps.append(res["P_success"])
            J_mat.append(np.array(Js))
            Psucc_mat.append(np.array(Ps))

        # 找最优 (t1, delta)
        best_J = np.inf; best_t1 = None; best_d = None; best_res = None
        for di, d in enumerate(deltas):
            if J_mat[di].size == 0: continue
            idx = int(np.argmin(J_mat[di]))
            t1 = float(t_grid[idx])
            # 检查 t1+d 是否越界（对齐 valid 的构造）
            if t1 + d > ga_max + 1e-9: 
                continue
            res = eval_two_step(t1, d, lambda tt: phit_scalar(tt, scen),
                                c1=c1, cr=cr, lam=lam, kappa=kappa, alpha=alpha)
            if res["J"] < best_J:
                best_J = res["J"]; best_t1 = t1; best_d = d; best_res = res

        if best_res is None:
            continue

        recs.append({
            "BMI_group": str(g),
            "n": len(sub),
            "BMI_median": scen["BMI"],
            "best_t1": round(best_t1, 2),
            "best_delta": round(best_d, 2),
            "best_t2": round(best_res["t2"], 2),
            "P_success_two": round(best_res["P_success"], 4),
            "E_draws_two": round(best_res["E_draws"], 3),
            "E_Tres_two": round(best_res["E_Tres"], 3),
            "J_two": round(best_res["J"], 4),
            "t1_single": round(t1_star_single, 2),
            "P_success_single": round(rec_single["P_success"], 4),
            "E_draws_single": round(rec_single["E_draws"], 3),
            "E_Tres_single": round(rec_single["E_Tres"], 3),
            "J_single": round(rec_single["J"], 4),
        })

        # 可视化：该组的 J2 热力图
        # 统一矩阵大小：对每个 delta，用与 t_grid 等长的数组填充（越界处设 NaN），便于 imshow
        J_full = np.full((len(deltas), len(t_grid)), np.nan)
        for di, d in enumerate(deltas):
            valid_len = len(t_grid[t_grid + d <= ga_max + 1e-9])
            if valid_len > 0:
                J_full[di, :valid_len] = J_mat[di]

        plt.figure()
        extent = [t_grid.min(), t_grid.max(), 0, len(deltas)]
        plt.imshow(J_full, aspect='auto', origin='lower', extent=extent)
        plt.colorbar(label="目标值 J₂")
        yticks = np.arange(len(deltas)) + 0.5
        plt.yticks(yticks, [f"Δ={d}" for d in deltas])
        plt.axvline(best_t1, linestyle="--", linewidth=1)
        plt.xlabel("首检孕周 t₁（周）")
        plt.ylabel("复检间隔 Δ（周）")
        plt.title(f"Q4：J₂(t₁,Δ) 热力图（{g}，α={alpha}）")
        fig_heat = os.path.join(outdir, f"q4_heatmap_{str(g)}.png")
        plt.tight_layout(); plt.savefig(fig_heat, dpi=150); plt.close()

        # 可视化：该组的 P_hit(t) 与推荐 t1,t2
        plt.figure()
        plt.plot(t_grid, P_vec, linewidth=2, label="P(hit)")
        plt.axvline(best_t1, linestyle="--", linewidth=1, label="t₁*")
        plt.axvline(best_res["t2"], linestyle="--", linewidth=1, label="t₂*")
        plt.xlabel("孕周（周）"); plt.ylabel("P(hit)")
        plt.title(f"Q4：P(hit) 曲线与推荐时点（{g}）")
        plt.legend()
        fig_curve = os.path.join(outdir, f"q4_phit_{str(g)}.png")
        plt.tight_layout(); plt.savefig(fig_curve, dpi=150); plt.close()

    # 导出策略表
    rec_df = pd.DataFrame(recs)
    csv_path = os.path.join(outdir, "q4_policy_table.csv")
    rec_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 策略表已保存：{csv_path}")

    # 控制台总结
    print("\n===== Q4 最优策略（按 BMI组）=====")
    if not rec_df.empty:
        for _, r in rec_df.iterrows():
            print(f"{r['BMI_group']:>8} n={int(r['n'])} | "
                  f"两次：t1={r['best_t1']}, Δ={r['best_delta']} → t2={r['best_t2']}, "
                  f"Psucc={r['P_success_two']}, E_draws={r['E_draws_two']}, E_Tres={r['E_Tres_two']}, J2={r['J_two']} "
                  f"|| 单次：t1={r['t1_single']}, Psucc={r['P_success_single']}, J1={r['J_single']}")
    else:
        print("无可用结果，请检查数据与分组。")


def main():
    parser = argparse.ArgumentParser(description="NIPT Q4：基于 P(hit) 的检测与复检策略优化（按 BMI 组）")
    parser.add_argument("--excel", type=str, default="附件.xlsx", help="原始 Excel（若 clean_csv 不存在则使用）")
    parser.add_argument("--clean_csv", type=str, default="outputs_q1/q1_clean_male.csv", help="Q1清洗输出（优先使用）")
    parser.add_argument("--sheet", type=str, default=None, help="工作表名称")
    parser.add_argument("--outdir", type=str, default="outputs_q4", help="输出目录")

    parser.add_argument("--deg", type=int, default=3, help="GA 多项式次数")
    parser.add_argument("--no_inter", action="store_true", help="不使用 GA×BMI 交互项")
    parser.add_argument("--sigma_mode", type=str, default="ga_local", choices=["global","ga_local","by_qc"], help="σ 估计方式")
    parser.add_argument("--y_threshold", type=float, default=0.04, help="达标阈值（比例）")

    parser.add_argument("--ga_min", type=float, default=10.0, help="t1 网格最小（周）")
    parser.add_argument("--ga_max", type=float, default=29.0, help="t1 网格最大（周）")
    parser.add_argument("--ga_step", type=float, default=0.25, help="t1 网格步长（周）")
    parser.add_argument("--delta_choices", type=str, default="1,1.5,2,3", help="备选复检间隔，逗号分隔（周）")

    parser.add_argument("--c1", type=float, default=1.0, help="一次抽血成本/惩罚")
    parser.add_argument("--cr", type=float, default=1.0, help="复检额外成本")
    parser.add_argument("--lam", type=float, default=1.0, help="未达标惩罚权重 λ")
    parser.add_argument("--kappa", type=float, default=0.05, help="延迟惩罚权重 κ（每周）")
    parser.add_argument("--alpha", type=float, default=1.0, help="两次命中概率独立性参数 α（1独立，0完全相关）")

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    run_q4(excel_path=args.excel,
           clean_csv=args.clean_csv,
           sheet_name=args.sheet,
           outdir=args.outdir,
           deg=args.deg,
           include_interactions=not args.no_inter,
           sigma_mode=args.sigma_mode,
           y_threshold=args.y_threshold,
           ga_min=args.ga_min, ga_max=args.ga_max, ga_step=args.ga_step,
           c1=args.c1, cr=args.cr, lam=args.lam, kappa=args.kappa,
           alpha=args.alpha,
           delta_choices=args.delta_choices)


if __name__ == "__main__":
    main()
