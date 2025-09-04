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

import logging
import matplotlib
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

class NullWriter:
    def write(self, arg):
        pass
    def flush(self):
        pass

stderr_fileno = sys.stderr
sys.stderr = NullWriter()

plt.rcParams['font.family'] = [
    'Microsoft YaHei','SimHei','Arial Unicode MS','STHeiti',
    'PingFang SC','WenQuanYi Zen Hei','Source Han Sans SC','sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

sys.stderr = stderr_fileno

# -----------------------------
# 工具函数
# -----------------------------

def parse_ga_to_weeks(x):
    """
    将孕周字符串（如 '11w+6', '11周+6天', '11+6', '11w6d'）或数字统一转换为连续周（float）。
    若无法解析，返回 np.nan。
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return float(x)
        except Exception:
            return np.nan
    s = str(x).strip()
    s = s.replace("W", "w").replace("D", "d").replace("周", "w").replace("天", "d")
    s = s.replace("．", ".").replace("＋", "+")
    m = re.match(r"^\s*(\d{1,2})\s*(?:w)?\s*(?:\+)?\s*(\d{1,2})?\s*(?:d)?\s*$", s)
    if m:
        w = int(m.group(1)); d = int(m.group(2)) if m.group(2) is not None else 0
        return w + d / 7.0
    try:
        return float(s)
    except Exception:
        return np.nan


def guess_columns(df):
    """
    依据常见别名猜测列名映射，返回 {标准名: 实际列名}
    标准名: id, age, ga, bmi, y_z, y_frac, draws, date, gc
    """
    aliases = {
        "id": ["孕妇代码", "孕妇ID", "孕妇编号", "受检者ID", "样本ID", "样本编号", "ID", "code", "patient_id"],
        "age": ["年龄", "Age", "C"],
        "ga": ["检测孕周", "孕周", "孕周(周+天)", "孕周（周+天）", "J", "GA", "gestational_age"],
        "bmi": ["孕妇BMI", "BMI", "K"],
        "y_z": ["Y染色体的Z值", "Y染色体Z值", "Y_Z", "U"],
        "y_frac": ["Y染色体浓度", "Y浓度", "Y浓度(%)", "V", "fetal_fraction_Y", "Y_fetal_fraction"],
        "draws": ["检测抽血次数", "抽血次数", "I", "draw_count"],
        "date": ["检测日期", "检测时间", "日期", "H", "sample_time", "采血时间"],
        "gc": ["整体GC", "GC含量", "P", "GC"],
    }
    mapping = {}
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    for std, cand_list in aliases.items():
        for cand in cand_list:
            key = str(cand).strip().lower()
            if key in lower_cols:
                mapping[std] = lower_cols[key]
                break
    return mapping


def ensure_outdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def to_ratio(y):
    """将可能为百分比的 Y 浓度转换为比例 0-1"""
    y = pd.to_numeric(y, errors="coerce")
    if (y > 1).mean(skipna=True) > 0.5:
        return y / 100.0
    return y


def make_bmi_group(bmi):
    bins_bmi = [20, 28, 32, 36, 40, 100]
    labels_bmi = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "[40,+)"]
    return pd.cut(bmi, bins=bins_bmi, labels=labels_bmi, right=False)


def smooth_hit_rate(ga, hit, bin_width=0.5, window=3, ga_min=None, ga_max=None):
    """
    计算平滑后的命中率曲线：
    1) 将数据按孕周以 bin_width 分箱，计算每箱命中率；
    2) 对命中率做滑动窗口平均（窗口=window 个箱）。
    返回 DataFrame: [ga_mid, hit_rate, n]
    """
    ga = np.asarray(ga, dtype=float); hit = np.asarray(hit, dtype=float)
    ok = (~np.isnan(ga)) & (~np.isnan(hit))
    ga = ga[ok]; hit = hit[ok]

    if ga_min is None: ga_min = np.nanmin(ga) if len(ga) else 9.0
    if ga_max is None: ga_max = np.nanmax(ga) if len(ga) else 30.0

    bins = np.arange(ga_min, ga_max + bin_width, bin_width)
    idx = np.digitize(ga, bins) - 1
    mids = (bins[:-1] + bins[1:]) / 2
    rate = np.full(len(mids), np.nan)
    counts = np.zeros(len(mids), dtype=int)
    for i in range(len(mids)):
        mask = idx == i
        if np.any(mask):
            counts[i] = int(np.sum(mask))
            rate[i] = float(np.nanmean(hit[mask]))

    # 滑动平均（简单等权）
    sm_rate = rate.copy()
    if window > 1 and len(rate) >= window:
        half = window // 2
        for i in range(len(rate)):
            lo = max(0, i - half); hi = min(len(rate), i + half + 1)
            vals = rate[lo:hi]
            sm_rate[i] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan

    return pd.DataFrame({"ga_mid": mids, "hit_rate": sm_rate, "n": counts})


def earliest_t_by_target(hit_curve, target=0.9):
    """
    在平滑命中率曲线上找到最早达到 target 的孕周。
    若永远未达，返回最大孕周。
    """
    df = hit_curve.dropna(subset=["hit_rate"]).copy()
    if df.empty:
        return np.nan
    meets = df["hit_rate"] >= target
    if not meets.any():
        return float(df["ga_mid"].max())
    return float(df.loc[meets, "ga_mid"].iloc[0])


def risk_level(t):
    """
    题面风险分档：≤12 低风险=1；13–27 高风险=2；≥28 极高风险=3
    """
    if t <= 12:
        return 1.0
    elif t < 28:
        return 2.0
    else:
        return 3.0


def argmin_risk(hit_curve, lam=1.0):
    """
    选择 t 以最小化 期望风险 = risk_level(t) + lam * (1 - hit_rate(t))
    """
    df = hit_curve.dropna(subset=["hit_rate"]).copy()
    if df.empty:
        return np.nan
    df["obj"] = df["ga_mid"].apply(risk_level) + lam * (1 - df["hit_rate"])
    # 如果多处平坦相同，取最早的 t
    idx = int(df["obj"].values.argmin())
    return float(df["ga_mid"].iloc[idx])


def bootstrap_ci(group_df, target=0.9, lam=1.0, bin_width=0.5, window=3, B=200, seed=42):
    """
    自助法（组内重采样）估计两种策略 t 的置信区间。
    返回：t_target_mean, t_target_lo, t_target_hi, t_risk_mean, t_risk_lo, t_risk_hi
    """
    if len(group_df) == 0:
        return [np.nan]*6
    rng = np.random.default_rng(seed)
    t_targets, t_risks = [], []
    arr = group_df[["GA_weeks", "hit"]].values
    n = len(arr)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sub = pd.DataFrame(arr[idx], columns=["GA_weeks","hit"])
        hc = smooth_hit_rate(sub["GA_weeks"].values, sub["hit"].values, bin_width=bin_width, window=window)
        t1 = earliest_t_by_target(hc, target=target)
        t2 = argmin_risk(hc, lam=lam)
        t_targets.append(t1)
        t_risks.append(t2)
    def ci(a):
        a = np.array(a, dtype=float)
        return float(np.nanmean(a)), float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))
    m1, lo1, hi1 = ci(t_targets)
    m2, lo2, hi2 = ci(t_risks)
    return m1, lo1, hi1, m2, lo2, hi2


def load_data_for_q2(excel_path, clean_csv, sheet_name=None, min_ga=9.0, max_ga=30.0, gc_low=0.3, gc_high=0.7):
    """
    优先读取 Q1 产出的清洗文件；若不存在则从 Excel 自行清洗（与Q1一致的最小规则）。
    """
    if clean_csv and os.path.exists(clean_csv):
        df = pd.read_csv(clean_csv)
        # 兼容列名
        if "GA_weeks" not in df.columns and "检测孕周" in df.columns:
            df["GA_weeks"] = df["检测孕周"].apply(parse_ga_to_weeks)
        if "BMI" not in df.columns and "孕妇BMI" in df.columns:
            df["BMI"] = pd.to_numeric(df["孕妇BMI"], errors="coerce")
        if "Y_frac" not in df.columns and "Y染色体浓度" in df.columns:
            df["Y_frac"] = to_ratio(df["Y染色体浓度"])
        # 推断 is_male
        if "is_male" not in df.columns:
            df["is_male"] = df["Y_frac"] > 0.005
        df = df[(~df["GA_weeks"].isna()) & (~df["BMI"].isna()) & (~df["Y_frac"].isna())]
        df = df[(df["GA_weeks"] >= min_ga) & (df["GA_weeks"] <= max_ga)]
        df = df[(df["BMI"] > 10) & (df["BMI"] < 60)]
        df = df[(df["Y_frac"] > 0) & (df["Y_frac"] < 0.5)]
        df_male = df[df["is_male"]].copy()
        return df_male

    # 否则，从 excel 清洗
    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(excel_path)
    mapping = guess_columns(df_raw)
    for k in ["ga","bmi","y_frac"]:
        if k not in mapping:
            raise ValueError("Excel 缺少必要列，至少需要 孕周/孕妇BMI/Y浓度。")
    df = df_raw.copy()
    df["GA_weeks"] = df[mapping["ga"]].apply(parse_ga_to_weeks)
    df["BMI"] = pd.to_numeric(df[mapping["bmi"]], errors="coerce")
    df["Y_frac"] = to_ratio(df[mapping["y_frac"]])
    df["ID"] = df[mapping["id"]] if "id" in mapping else np.arange(len(df))
    # 日期用于首检筛选
    df["Date"] = pd.to_datetime(df[mapping["date"]], errors="coerce") if "date" in mapping else pd.NaT
    # 仅按 GC 做轻质控（与示例一致）
    df["GC"] = pd.to_numeric(df[mapping["gc"]], errors="coerce") if "gc" in mapping else np.nan

    # 基础过滤
    df = df[(~df["GA_weeks"].isna()) & (~df["BMI"].isna()) & (~df["Y_frac"].isna())]
    df = df[(df["GA_weeks"] >= min_ga) & (df["GA_weeks"] <= max_ga)]
    df = df[(df["BMI"] > 10) & (df["BMI"] < 60)]
    df = df[(df["Y_frac"] > 0) & (df["Y_frac"] < 0.5)]

    # 男胎筛选
    df["is_male"] = df["Y_frac"] > 0.005
    df = df[df["is_male"]].copy()

    # 质控：仅 GC 区间
    if not df["GC"].isna().all():
        df = df[(df["GC"] >= gc_low) & (df["GC"] <= gc_high)]

    # 重复测量：仅保留首检
    if df["Date"].notna().any():
        df = df.sort_values(["ID","Date"]).groupby("ID", as_index=False).first()
    else:
        df = df.sort_values(["ID","GA_weeks"]).groupby("ID", as_index=False).first()

    return df


# -----------------------------
# Q2 主流程
# -----------------------------

def run_q2(excel_path="附件.xlsx",
           clean_csv="outputs_q1/q1_clean_male.csv",
           sheet_name=None,
           outdir="outputs_q2",
           y_threshold=0.04,
           bin_width=0.5,
           smooth_window=3,
           target_hit=0.90,
           lam=1.0,
           bootstrap_B=200,
           seed=42):
    """
    Q2：在仅按 BMI 分组的前提下，给出各组 NIPT 的“最早达标时点”（命中率达到 target_hit 的最早孕周）
        与“风险-命中率权衡”的最优时点（最小化 risk_level + lam*(1-hit)）。
    """

    ensure_outdir(outdir)
    print(f"[INFO] 读取数据：{excel_path if clean_csv is None else (clean_csv if os.path.exists(clean_csv) else excel_path)}")
    df = load_data_for_q2(excel_path, clean_csv, sheet_name=sheet_name)
    print("[INFO] 清洗后样本量（男胎）:", len(df))
    print("孕周范围: [{:.2f}, {:.2f}] 周".format(df["GA_weeks"].min(), df["GA_weeks"].max()))
    print("BMI 中位数: {:.2f}".format(df["BMI"].median()))
    print("Y 浓度≥{:.0%} 的比例: {:.1%}".format(y_threshold, (df["Y_frac"] >= y_threshold).mean()))

    # 命中标记
    df["hit"] = df["Y_frac"] >= y_threshold
    # BMI 分组
    df["BMI_group"] = make_bmi_group(df["BMI"])

    # 结果收集
    recs = []

    # 画图：命中率曲线（各 BMI 组）
    plt.figure()
    groups = df["BMI_group"].cat.categories if hasattr(df["BMI_group"], "cat") else sorted(df["BMI_group"].dropna().unique().tolist())
    for g in groups:
        sub = df[df["BMI_group"] == g].copy()
        if len(sub) == 0:
            continue
        hc = smooth_hit_rate(sub["GA_weeks"].values, sub["hit"].values, bin_width=bin_width, window=smooth_window)
        t_target = earliest_t_by_target(hc, target=target_hit)
        t_risk = argmin_risk(hc, lam=lam)
        # bootstrap 置信区间
        m1, lo1, hi1, m2, lo2, hi2 = bootstrap_ci(sub, target=target_hit, lam=lam,
                                                   bin_width=bin_width, window=smooth_window,
                                                   B=bootstrap_B, seed=seed)

        # 保存记录（四舍五入到 0.5 周便于呈现）
        def round05(x): return float(np.round(x * 2) / 2.0) if not np.isnan(x) else np.nan
        recs.append({
            "BMI_group": str(g),
            "n": len(sub),
            "t_target_hit": round05(t_target),
            "t_target_hit_CI95": f"[{round05(lo1)}, {round05(hi1)}]",
            "t_risk_min": round05(t_risk),
            "t_risk_min_CI95": f"[{round05(lo2)}, {round05(hi2)}]",
            "hit_at_t_target": float(hc.loc[(hc["ga_mid"] - t_target).abs().idxmin(), "hit_rate"]) if not np.isnan(t_target) and not hc.empty else np.nan
        })

        # 曲线与参考线
        plt.plot(hc["ga_mid"], hc["hit_rate"], linewidth=2, label=f"{g} (n={len(sub)})")
        plt.axvline(t_target, linestyle="--", linewidth=1)
    plt.axhline(target_hit, linestyle="--", linewidth=1)
    plt.xlabel("孕周（周）")
    plt.ylabel("命中率：Y 浓度 ≥ {:.0%}".format(y_threshold))
    plt.title("Q2：不同 BMI 组的命中率曲线与推荐时点")
    plt.legend()
    fig1 = os.path.join(outdir, "q2_hit_curves.png")
    plt.tight_layout(); plt.savefig(fig1, dpi=150); plt.close()
    print(f"[OK] 已保存图：{fig1}")

    # 将结果保存为 CSV
    rec_df = pd.DataFrame(recs)
    csv_path = os.path.join(outdir, "q2_recommendations.csv")
    rec_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 推荐时点表已保存：{csv_path}")

    # 同时输出每组的风险目标函数曲线图（可选，合并在一张图）
    plt.figure()
    for g in groups:
        sub = df[df["BMI_group"] == g].copy()
        if len(sub) == 0:
            continue
        hc = smooth_hit_rate(sub["GA_weeks"].values, sub["hit"].values, bin_width=bin_width, window=smooth_window)
        if hc.empty: 
            continue
        hc["obj"] = hc["ga_mid"].apply(risk_level) + lam * (1 - hc["hit_rate"])
        plt.plot(hc["ga_mid"], hc["obj"], linewidth=2, label=f"{g}")
    plt.xlabel("孕周（周）"); plt.ylabel("目标值：风险 + λ×(1-命中)")
    plt.title(f"Q2：风险-命中率权衡曲线（λ={lam}）")
    plt.legend()
    fig2 = os.path.join(outdir, "q2_risk_tradeoff_curves.png")
    plt.tight_layout(); plt.savefig(fig2, dpi=150); plt.close()
    print(f"[OK] 已保存图：{fig2}")

    # 控制台汇总
    print("\n===== Q2 建议时点（按 BMI 组）=====")
    if not rec_df.empty:
        for _, r in rec_df.iterrows():
            print(f"{r['BMI_group']:>8}  n={int(r['n'])}  "
                  f"最早达标(≥{int(target_hit*100)}%命中): {r['t_target_hit']} 周  CI95 {r['t_target_hit_CI95']}  |  "
                  f"风险-命中权衡最优: {r['t_risk_min']} 周  CI95 {r['t_risk_min_CI95']}")
    else:
        print("无可用分组结果，请检查数据。")


def main():
    parser = argparse.ArgumentParser(description="NIPT Q2：按 BMI 分组的最佳检测时点（最早达标 & 风险权衡）")
    parser.add_argument("--excel", type=str, default="附件.xlsx", help="原始 Excel（若找不到清洗CSV则使用）")
    parser.add_argument("--clean_csv", type=str, default="outputs_q1/q1_clean_male.csv", help="Q1清洗输出（优先使用）")
    parser.add_argument("--sheet", type=str, default=None, help="工作表名称")
    parser.add_argument("--outdir", type=str, default="outputs_q2", help="输出目录")
    parser.add_argument("--y_threshold", type=float, default=0.04, help="达标阈值（比例）")
    parser.add_argument("--bin_width", type=float, default=0.5, help="孕周分箱宽度（周）")
    parser.add_argument("--smooth_window", type=int, default=3, help="滑动平均窗口（箱数）")
    parser.add_argument("--target_hit", type=float, default=0.90, help="达标命中率阈值（如 0.90）")
    parser.add_argument("--lam", type=float, default=1.0, help="风险-命中率权衡的 λ 系数")
    parser.add_argument("--bootstrap_B", type=int, default=200, help="bootstrap 次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    run_q2(excel_path=args.excel,
           clean_csv=args.clean_csv,
           sheet_name=args.sheet,
           outdir=args.outdir,
           y_threshold=args.y_threshold,
           bin_width=args.bin_width,
           smooth_window=args.smooth_window,
           target_hit=args.target_hit,
           lam=args.lam,
           bootstrap_B=args.bootstrap_B,
           seed=args.seed)


if __name__ == "__main__":
    main()
