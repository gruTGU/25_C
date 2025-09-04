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

# -------------------- 安静字体/告警设置（与 Q2 一致风格） --------------------
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
# 工具函数
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
    标准名: id, age, ga, bmi, y_z, y_frac, draws, date, 
            total_reads, unique_mapped, map_rate, dup_rate, gc, filt_ratio,
            height, weight
    """
    aliases = {
        "id": ["孕妇代码","孕妇ID","孕妇编号","受检者ID","样本ID","样本编号","ID","code","patient_id"],
        "age": ["年龄","Age","C"],
        "ga": ["检测孕周","孕周","孕周(周+天)","孕周（周+天）","J","GA","gestational_age"],
        "bmi": ["孕妇BMI","BMI","K"],
        "y_z": ["Y染色体的Z值","Y染色体Z值","Y_Z","U"],
        "y_frac": ["Y染色体浓度","Y浓度","Y浓度(%)","V","fetal_fraction_Y","Y_fetal_fraction"],
        "draws": ["检测抽血次数","抽血次数","I","draw_count"],
        "date": ["检测日期","检测时间","日期","H","sample_time","采血时间"],
        "total_reads": ["总读段数","总reads","L","total_reads"],
        "unique_mapped": ["唯一比对读段数","唯一比对reads","O","unique_mapped_reads"],
        "map_rate": ["比对比例","比对率","M","map_rate"],
        "dup_rate": ["重复读段比例","重复率","N","dup_rate"],
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


def build_design_matrix(df, deg=3, include_interactions=True, mapping=None):
    """
    设计矩阵 X（用于拟合 μ）：列含常数、GA 的多项式、BMI、Age、可选 Height/Weight，
    以及 GA 多项式×BMI 的交互项。
    """
    GA = df["GA_weeks"].values.astype(float)
    BMI = df["BMI"].values.astype(float)
    cols = [np.ones_like(GA)]
    names = ["const"]

    # GA 多项式
    for d in range(1, deg+1):
        cols.append(GA**d)
        names.append(f"GA^{d}")

    # BMI、Age、Height、Weight（若存在）
    cols.append(BMI); names.append("BMI")
    if "Age" in df:
        cols.append(df["Age"].values.astype(float)); names.append("Age")
    if "Height" in df:
        cols.append(df["Height"].values.astype(float)); names.append("Height")
    if "Weight" in df:
        cols.append(df["Weight"].values.astype(float)); names.append("Weight")

    # 交互：GA^d * BMI
    if include_interactions:
        for d in range(1, deg+1):
            cols.append((GA**d) * BMI)
            names.append(f"GA^{d}:BMI")

    X = np.column_stack(cols)
    return X, names


def fit_linear_mu(df, deg=3, include_interactions=True):
    """
    在 Y（原始 0-1 比例）尺度上拟合 μ(x)：最小二乘
    返回：beta, names, 预测函数 mu_fn
    """
    X, names = build_design_matrix(df, deg=deg, include_interactions=include_interactions)
    y = df["Y_frac"].values.astype(float)
    # 去除 NaN
    ok = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
    X = X[ok]; y = y[ok]
    # 最小二乘
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    def mu_fn(df_new):
        Xn, _ = build_design_matrix(df_new, deg=deg, include_interactions=include_interactions)
        yhat = Xn @ beta
        return np.clip(yhat, 1e-4, 0.499)  # 合理范围，避免0与>0.5
    return beta, names, mu_fn


def estimate_sigma(df, mu_fn, mode="ga_local", bin_width=0.5, qc_cols=None):
    """
    估计观测误差 σ：
      - global: 全局残差标准差
      - ga_local: 按孕周分箱估计局部残差 std，并沿 GA 插值
      - by_qc: 若提供 qc_cols（例如 ['unique_mapped','filt_ratio','gc']），
               用一个简单线性模型 std = a + b1*z1 + b2*z2 + ... 来拟合，然后按 df_new 预测
    返回：sigma_fn(df_new) -> 预测 σ（下限1e-4）
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
        # 按 GA 分箱
        ga = df["GA_weeks"].values.astype(float)
        ok = (~np.isnan(ga)) & (~np.isnan(y))
        ga = ga[ok]; rr = (y - mu)[ok]
        bins = np.arange(max(9, np.nanmin(ga)), min(30, np.nanmax(ga)) + bin_width, bin_width)
        mids = (bins[:-1] + bins[1:]) / 2
        sds = []
        for i in range(len(mids)):
            mask = (ga >= bins[i]) & (ga < bins[i+1])
            if np.any(mask):
                sds.append(np.nanstd(rr[mask]))
            else:
                sds.append(np.nan)
        sds = np.array(sds, dtype=float)
        # 缺失用邻近/全局填
        global_sd = float(np.nanstd(rr))
        sds = np.where(np.isnan(sds), global_sd, sds)
        sds = np.clip(sds, 0.005, 0.06)  # 合理范围
        def sigma_fn(dnew):
            GA = dnew["GA_weeks"].values.astype(float)
            # 最近邻插值
            idx = np.digitize(GA, bins) - 1
            idx = np.clip(idx, 0, len(mids)-1)
            return sds[idx]
        return sigma_fn

    if mode == "by_qc" and qc_cols:
        # 简单线性回归：std ~ qc 标准化后线性组合
        # 先为每个点构造一个局部 std（用 KNN/分箱近似）
        ga = df["GA_weeks"].values.astype(float)
        ok = (~np.isnan(ga)) & (~np.isnan(y))
        bins = np.arange(max(9, np.nanmin(ga)), min(30, np.nanmax(ga)) + 0.5, 0.5)
        idx = np.digitize(ga[ok], bins) - 1
        local_std = np.zeros(np.sum(ok))
        for i in range(np.min(idx), np.max(idx)+1):
            m = (idx == i)
            if np.any(m):
                local_std[m] = np.nanstd((y - mu)[ok][m])
        local_std = np.where(local_std==0, np.nanmedian(local_std[local_std>0]), local_std)
        # 构造 QC 特征
        Z = []
        for c in qc_cols:
            if c in df.columns:
                z = pd.to_numeric(df.loc[ok, c], errors="coerce").values.astype(float)
                # 标准化
                m, s = np.nanmean(z), np.nanstd(z)
                if s == 0 or np.isnan(s): s = 1.0
                Z.append((z - m) / s)
            else:
                Z.append(np.zeros(np.sum(ok)))
        Z = np.column_stack(Z + [np.ones(np.sum(ok))])  # 最后一列常数
        # 拟合最小二乘 local_std ~ Z
        coefs, *_ = np.linalg.lstsq(Z, local_std, rcond=None)
        def sigma_fn(dnew):
            Feats = []
            for c in qc_cols:
                if c in dnew.columns:
                    z = pd.to_numeric(dnew[c], errors="coerce").values.astype(float)
                    m, s = np.nanmean(z), np.nanstd(z)
                    s = s if (s and not np.isnan(s)) else 1.0
                    Feats.append((z - m) / s)
                else:
                    Feats.append(np.zeros(len(dnew)))
            Zn = np.column_stack(Feats + [np.ones(len(dnew))])
            s_hat = Zn @ coefs
            s_hat = np.clip(s_hat, 0.005, 0.06)
            return s_hat
        return sigma_fn

    # 兜底：全局
    s = float(np.nanstd(resid))
    s = np.clip(s, 0.01, 0.05)
    return lambda d: np.full(len(d), s)


def earliest_t_by_target(ga_grid, phit, target=0.9):
    """在 phit(ga) 曲线上找到最早达到 target 的孕周；未达则取最大值。"""
    phit = np.asarray(phit, dtype=float)
    meets = phit >= target
    if not np.any(meets):
        return float(ga_grid.max())
    idx = int(np.argmax(meets))  # 第一次为 True 的索引
    return float(ga_grid[idx])


def risk_level(t):
    """题面风险分档：≤12 ->1；13–27 ->2；≥28 ->3"""
    if t <= 12: return 1.0
    elif t < 28: return 2.0
    else: return 3.0


def argmin_risk(ga_grid, phit, lam=1.0):
    """最小化 Risk(t)=risk_level(t)+lam*(1-phit(t))，多处平坦取最早。"""
    vals = np.array([risk_level(t) + lam*(1 - p) for t, p in zip(ga_grid, phit)], dtype=float)
    idx = int(np.argmin(vals))
    return float(ga_grid[idx])


def bootstrap_ci(df_group, build_and_predict_fn, target=0.9, lam=1.0, B=200, seed=42):
    """
    对某 BMI 组做 bootstrap：
      build_and_predict_fn() -> (ga_grid, phit)  用于一次重建并预测曲线
    返回：(t_target_mean, lo, hi, t_risk_mean, lo, hi)
    """
    if len(df_group) == 0:
        return [np.nan]*6
    rng = np.random.default_rng(seed)
    t1s, t2s = [], []
    n = len(df_group)
    arr = df_group.index.values
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        dfb = df_group.iloc[idx].reset_index(drop=True)
        ga_grid, phit = build_and_predict_fn(dfb)
        t1 = earliest_t_by_target(ga_grid, phit, target=target)
        t2 = argmin_risk(ga_grid, phit, lam=lam)
        t1s.append(t1); t2s.append(t2)
    def _ci(a):
        a = np.array(a, dtype=float)
        return float(np.nanmean(a)), float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))
    m1, lo1, hi1 = _ci(t1s)
    m2, lo2, hi2 = _ci(t2s)
    return m1, lo1, hi1, m2, lo2, hi2


def make_bmi_group(bmi):
    bins_bmi = [20, 28, 32, 36, 40, 100]
    labels_bmi = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "[40,+)"]
    return pd.cut(bmi, bins=bins_bmi, labels=labels_bmi, right=False)

# -----------------------------
# 数据准备
# -----------------------------

def load_clean_male(excel_path, sheet_name=None, clean_csv=None,
                    min_ga=9.0, max_ga=30.0, gc_low=0.3, gc_high=0.7):
    """
    优先读 Q1 清洗结果；否则从 Excel 清洗，匹配第一问示例：仅按 GC 做轻质控也可。
    """
    if clean_csv and os.path.exists(clean_csv):
        df = pd.read_csv(clean_csv)
        # 兼容列
        if "GA_weeks" not in df.columns and "检测孕周" in df.columns:
            df["GA_weeks"] = df["检测孕周"].apply(parse_ga_to_weeks)
        if "BMI" not in df.columns and "孕妇BMI" in df.columns:
            df["BMI"] = pd.to_numeric(df["孕妇BMI"], errors="coerce")
        if "Y_frac" not in df.columns and "Y染色体浓度" in df.columns:
            df["Y_frac"] = to_ratio(df["Y染色体浓度"])
        if "Age" not in df.columns and "年龄" in df.columns:
            df["Age"] = pd.to_numeric(df["年龄"], errors="coerce")
        # 可选身高体重
        for k, std in [("身高","Height"), ("体重","Weight")]:
            if k in df.columns and std not in df.columns:
                df[std] = pd.to_numeric(df[k], errors="coerce")

        # 男胎推断
        if "is_male" not in df.columns:
            df["is_male"] = df["Y_frac"] > 0.005

        df = df[(~df["GA_weeks"].isna()) & (~df["BMI"].isna()) & (~df["Y_frac"].isna())]
        df = df[(df["GA_weeks"] >= min_ga) & (df["GA_weeks"] <= max_ga)]
        df = df[(df["BMI"] > 10) & (df["BMI"] < 60)]
        df = df[(df["Y_frac"] > 0) & (df["Y_frac"] < 0.5)]
        df = df[df["is_male"]].copy()
        return df

    # 否则从 Excel 清洗
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
    # QC
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
    # 男胎筛选
    df["is_male"] = df["Y_frac"] > 0.005
    df = df[df["is_male"]].copy()
    # 轻质控：GC 区间
    if "GC" in df.columns:
        df = df[(df["GC"] >= gc_low) & (df["GC"] <= gc_high)]

    return df

# -----------------------------
# Q3 主流程
# -----------------------------

def run_q3(excel_path="附件.xlsx",
           clean_csv="outputs_q1/q1_clean_male.csv",
           sheet_name=None,
           outdir="outputs_q3",
           deg=3,
           include_interactions=True,
           sigma_mode="ga_local",  # global / ga_local / by_qc
           y_threshold=0.04,
           target_hit=0.90,
           lam=1.0,
           ga_min=9.0, ga_max=30.0, ga_step=0.25,
           bootstrap_B=200,
           seed=42):
    """
    Q3：多因素 + 测量误差的两层建模，输出按 BMI 组的最早达标时点与风险-命中权衡时点（含CI），并做σ敏感性分析。
    """
    ensure_outdir(outdir)
    print(f"[INFO] 读取数据（优先 clean_csv）：{clean_csv if (clean_csv and os.path.exists(clean_csv)) else excel_path}")
    df = load_clean_male(excel_path, sheet_name=sheet_name, clean_csv=clean_csv)
    print("[INFO] 样本量（男胎/清洗后）:", len(df))
    print("孕周范围: [{:.2f}, {:.2f}] 周".format(df["GA_weeks"].min(), df["GA_weeks"].max()))
    print("BMI 中位数: {:.2f}".format(df["BMI"].median()))
    print("Y 浓度中位数: {:.3f}".format(df["Y_frac"].median()))
    print("≥{:.0%} 的比例: {:.1%}".format(y_threshold, (df["Y_frac"] >= y_threshold).mean()))

    # 拟合 μ(x)
    beta, names, mu_fn = fit_linear_mu(df, deg=deg, include_interactions=include_interactions)
    coef_df = pd.DataFrame({"term": names, "coef": beta})
    coef_path = os.path.join(outdir, "q3_mu_coefs.csv")
    coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")
    print(f"[OK] μ 拟合系数已保存：{coef_path}")

    # 估计 σ(x)
    qc_cols = [c for c in ["unique_mapped","filt_ratio","GC"] if c in df.columns]
    sigma_fn = estimate_sigma(df, mu_fn, mode=sigma_mode, bin_width=0.5, qc_cols=qc_cols)

    # 生成 GA 网格与情景（按 BMI 组中位数、Age/Height/Weight 中位数）
    df["BMI_group"] = make_bmi_group(df["BMI"])
    groups = df["BMI_group"].cat.categories if hasattr(df["BMI_group"], "cat") else sorted(df["BMI_group"].dropna().unique().tolist())

    ga_grid = np.arange(max(ga_min, df["GA_weeks"].min()), min(ga_max, df["GA_weeks"].max()), ga_step)
    recs = []
    fig1 = plt.figure()
    for g in groups:
        sub = df[df["BMI_group"] == g].copy()
        if len(sub) == 0:
            continue
        # 组内中位情景
        scen = {
            "BMI": float(np.nanmedian(sub["BMI"])),
            "Age": float(np.nanmedian(df["Age"])) if "Age" in df.columns else np.nan,
            "Height": float(np.nanmedian(df["Height"])) if "Height" in df.columns else np.nan,
            "Weight": float(np.nanmedian(df["Weight"])) if "Weight" in df.columns else np.nan,
        }
        df_grid = pd.DataFrame({
            "GA_weeks": ga_grid,
            "BMI": scen["BMI"],
            "Age": scen["Age"] if not np.isnan(scen["Age"]) else None,
            "Height": scen["Height"] if not np.isnan(scen["Height"]) else None,
            "Weight": scen["Weight"] if not np.isnan(scen["Weight"]) else None,
        })

        mu = mu_fn(df_grid)
        sigma = sigma_fn(df_grid)
        # P_hit(t) = 1 - Φ((thr - μ)/σ)
        z = (y_threshold - mu) / np.maximum(sigma, 1e-4)
        phit = 1.0 - np.array([normal_cdf(zz) for zz in z])

        # 两类时点
        t_target = earliest_t_by_target(ga_grid, phit, target=target_hit)
        t_risk = argmin_risk(ga_grid, phit, lam=lam)

        # bootstrap：定义一个闭包用于重建与预测
        def build_and_predict(dfb):
            # 重拟合 μ 与 σ（按当前 bootstrap 子样本）
            _, _, mu_fn_b = fit_linear_mu(dfb, deg=deg, include_interactions=include_interactions)
            sigma_fn_b = estimate_sigma(dfb, mu_fn_b, mode=sigma_mode, bin_width=0.5, qc_cols=qc_cols)
            tmp = pd.DataFrame({
                "GA_weeks": ga_grid,
                "BMI": scen["BMI"],
                "Age": scen["Age"] if "Age" in dfb.columns else None,
                "Height": scen["Height"] if "Height" in dfb.columns else None,
                "Weight": scen["Weight"] if "Weight" in dfb.columns else None,
            })
            mu_b = mu_fn_b(tmp); sigma_b = sigma_fn_b(tmp)
            z_b = (y_threshold - mu_b) / np.maximum(sigma_b, 1e-4)
            phit_b = 1.0 - np.array([normal_cdf(zz) for zz in z_b])
            return ga_grid, phit_b

        m1, lo1, hi1, m2, lo2, hi2 = bootstrap_ci(sub.reset_index(drop=True), build_and_predict,
                                                   target=target_hit, lam=lam, B=bootstrap_B, seed=seed)

        # 记录
        def round05(x): return float(np.round(x * 2) / 2.0) if not np.isnan(x) else np.nan
        recs.append({
            "BMI_group": str(g),
            "n": len(sub),
            "BMI_median": scen["BMI"],
            "t_target_hit": round05(t_target),
            "t_target_hit_CI95": f"[{round05(lo1)}, {round05(hi1)}]",
            "t_risk_min": round05(t_risk),
            "t_risk_min_CI95": f"[{round05(lo2)}, {round05(hi2)}]",
            "P_hit_at_t_target": float(phit[np.argmin(np.abs(ga_grid - t_target))]),
        })

        # 画 P_hit 曲线
        plt.plot(ga_grid, phit, linewidth=2, label=f"{g} (n={len(sub)})")
        plt.axvline(t_target, linestyle="--", linewidth=1)

    plt.axhline(target_hit, linestyle="--", linewidth=1)
    plt.xlabel("孕周（周）"); plt.ylabel("P(hit): Y 浓度 ≥ {:.0%}".format(y_threshold))
    plt.title("Q3：P(hit) 曲线与最早达标时点（按 BMI 组）")
    plt.legend()
    fig1_path = os.path.join(outdir, "q3_phit_curves.png")
    plt.tight_layout(); plt.savefig(fig1_path, dpi=150); plt.close()
    print(f"[OK] 已保存图：{fig1_path}")

    # 风险-命中率权衡曲线
    plt.figure()
    for g in [r["BMI_group"] for r in recs]:
        sub = df[df["BMI_group"] == g].copy()
        if len(sub) == 0: continue
        scen_BMI = float(np.nanmedian(sub["BMI"]))
        tmp = pd.DataFrame({"GA_weeks": ga_grid, "BMI": scen_BMI,
                            "Age": float(np.nanmedian(df["Age"])) if "Age" in df.columns else None,
                            "Height": float(np.nanmedian(df["Height"])) if "Height" in df.columns else None,
                            "Weight": float(np.nanmedian(df["Weight"])) if "Weight" in df.columns else None})
        mu = mu_fn(tmp); sigma = sigma_fn(tmp)
        z = (y_threshold - mu) / np.maximum(sigma, 1e-4)
        phit = 1.0 - np.array([normal_cdf(zz) for zz in z])
        obj = np.array([risk_level(t) + lam*(1-p) for t,p in zip(ga_grid, phit)])
        plt.plot(ga_grid, obj, linewidth=2, label=g)
    plt.xlabel("孕周（周）"); plt.ylabel("目标：风险 + λ×(1-P_hit)")
    plt.title(f"Q3：风险-命中率权衡曲线（λ={lam}）")
    plt.legend()
    fig2_path = os.path.join(outdir, "q3_risk_tradeoff_curves.png")
    plt.tight_layout(); plt.savefig(fig2_path, dpi=150); plt.close()
    print(f"[OK] 已保存图：{fig2_path}")

    # σ 敏感性分析（±20%）
    plt.figure()
    g = recs[0]["BMI_group"] if len(recs)>0 else None
    if g is not None:
        sub0 = df[df["BMI_group"] == g].copy()
        scen_BMI = float(np.nanmedian(sub0["BMI"]))
        tmp = pd.DataFrame({"GA_weeks": ga_grid, "BMI": scen_BMI,
                            "Age": float(np.nanmedian(df["Age"])) if "Age" in df.columns else None,
                            "Height": float(np.nanmedian(df["Height"])) if "Height" in df.columns else None,
                            "Weight": float(np.nanmedian(df["Weight"])) if "Weight" in df.columns else None})
        mu = mu_fn(tmp); sigma = sigma_fn(tmp)
        for scale in [0.8, 1.0, 1.2]:
            z = (y_threshold - mu) / np.maximum(sigma*scale, 1e-4)
            phit = 1.0 - np.array([normal_cdf(zz) for zz in z])
            plt.plot(ga_grid, phit, linewidth=2, label=f"σ×{scale:.1f}")
        plt.axhline(target_hit, linestyle="--", linewidth=1)
        plt.xlabel("孕周（周）"); plt.ylabel("P(hit)")
        plt.title("Q3：σ 敏感性分析（示例：首个 BMI 组的中位 BMI 情景）")
        plt.legend()
        fig3_path = os.path.join(outdir, "q3_sigma_sensitivity.png")
        plt.tight_layout(); plt.savefig(fig3_path, dpi=150); plt.close()
        print(f"[OK] 已保存图：{fig3_path}")

    # 导出推荐表
    rec_df = pd.DataFrame(recs)
    rec_csv = os.path.join(outdir, "q3_recommendations.csv")
    rec_df.to_csv(rec_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 推荐时点表已保存：{rec_csv}")

    # 输出 σ 按孕周的估计（若为 ga_local）
    if sigma_mode == "ga_local":
        # 用全体样本估计一次 sigma vs GA（便于附录）
        bins = np.arange(max(ga_min, df["GA_weeks"].min()), min(ga_max, df["GA_weeks"].max()) + 0.5, 0.5)
        mids = (bins[:-1] + bins[1:]) / 2
        tmp = []
        for m in mids:
            df_mid = pd.DataFrame({"GA_weeks":[m],
                                   "BMI":[df["BMI"].median()],
                                   "Age":[df["Age"].median() if "Age" in df.columns else None]})
            tmp.append(float(sigma_fn(df_mid)[0]))
        sigma_by_ga = pd.DataFrame({"GA_mid": mids, "sigma": tmp})
        sigma_csv = os.path.join(outdir, "q3_sigma_by_ga.csv")
        sigma_by_ga.to_csv(sigma_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] σ(孕周) 表已保存：{sigma_csv}")

    # 控制台汇总
    print("\n===== Q3 建议时点（按 BMI 组）=====")
    if not rec_df.empty:
        for _, r in rec_df.iterrows():
            print(f"{r['BMI_group']:>8}  n={int(r['n'])}  "
                  f"最早达标(≥{int(target_hit*100)}%命中): {r['t_target_hit']} 周  CI95 {r['t_target_hit_CI95']}  |  "
                  f"风险-命中权衡最优: {r['t_risk_min']} 周  CI95 {r['t_risk_min_CI95']}")
    else:
        print("无可用分组结果，请检查数据。")


def main():
    parser = argparse.ArgumentParser(description="NIPT Q3：多因素+测量误差建模下的最佳检测时点（男胎，按BMI分组）")
    parser.add_argument("--excel", type=str, default="附件.xlsx", help="原始 Excel（若 clean_csv 不存在则使用）")
    parser.add_argument("--clean_csv", type=str, default="outputs_q1/q1_clean_male.csv", help="Q1清洗输出（优先使用）")
    parser.add_argument("--sheet", type=str, default=None, help="工作表名称")
    parser.add_argument("--outdir", type=str, default="outputs_q3", help="输出目录")
    parser.add_argument("--deg", type=int, default=3, help="GA 多项式次数")
    parser.add_argument("--no_inter", action="store_true", help="不使用 GA×BMI 交互项")
    parser.add_argument("--sigma_mode", type=str, default="ga_local", choices=["global","ga_local","by_qc"], help="σ 估计方式")
    parser.add_argument("--y_threshold", type=float, default=0.04, help="达标阈值（比例）")
    parser.add_argument("--target_hit", type=float, default=0.90, help="达标命中率阈值")
    parser.add_argument("--lam", type=float, default=1.0, help="风险-命中率权衡中的 λ")
    parser.add_argument("--ga_min", type=float, default=9.0, help="GA 网格最小")
    parser.add_argument("--ga_max", type=float, default=30.0, help="GA 网格最大")
    parser.add_argument("--ga_step", type=float, default=0.25, help="GA 网格步长（周）")
    parser.add_argument("--bootstrap_B", type=int, default=200, help="bootstrap 次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    run_q3(excel_path=args.excel,
           clean_csv=args.clean_csv,
           sheet_name=args.sheet,
           outdir=args.outdir,
           deg=args.deg,
           include_interactions=not args.no_inter,
           sigma_mode=args.sigma_mode,
           y_threshold=args.y_threshold,
           target_hit=args.target_hit,
           lam=args.lam,
           ga_min=args.ga_min, ga_max=args.ga_max, ga_step=args.ga_step,
           bootstrap_B=args.bootstrap_B,
           seed=args.seed)


if __name__ == "__main__":
    main()
