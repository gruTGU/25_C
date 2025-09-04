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
        # 假设已经是以周为单位的数
        try:
            return float(x)
        except Exception:
            return np.nan
    s = str(x).strip()
    # 常见替换
    s = s.replace("W", "w").replace("D", "d").replace("周", "w").replace("天", "d")
    s = s.replace("．", ".").replace("＋", "+")
    # 形如 "11w+6", "11w6d", "11+6"
    m = re.match(r"^\s*(\d{1,2})\s*(?:w)?\s*(?:\+)?\s*(\d{1,2})?\s*(?:d)?\s*$", s)
    if m:
        w = int(m.group(1))
        d = int(m.group(2)) if m.group(2) is not None else 0
        return w + d / 7.0
    # 形如 "11.86" 直接按周数
    try:
        return float(s)
    except Exception:
        return np.nan


def guess_columns(df):
    """
    尝试根据多种可能的列名映射，返回标准列名映射字典。
    标准列名：
      id, age, ga, bmi, y_z, y_frac, draws, date,
      total_reads, unique_mapped, map_rate, dup_rate, gc, filt_ratio
    """
    # 所有候选别名（按优先顺序）
    aliases = {
        "id": ["孕妇代码", "孕妇ID", "孕妇编号", "受检者ID", "样本ID", "样本编号", "ID", "code", "patient_id"],
        "age": ["年龄", "Age", "C"],
        "ga": ["检测孕周", "孕周", "孕周(周+天)", "孕周（周+天）", "J", "GA", "gestational_age"],
        "bmi": ["孕妇BMI", "BMI", "K"],
        "y_z": ["Y染色体的Z值", "Y染色体Z值", "Y_Z", "U"],
        "y_frac": ["Y染色体浓度", "Y浓度", "Y浓度(%)", "V", "fetal_fraction_Y", "Y_fetal_fraction"],
        "draws": ["检测抽血次数", "抽血次数", "I", "draw_count"],
        "date": ["检测时间", "检测日期", "日期", "H", "sample_time", "采血时间"],
        "total_reads": ["总读段数", "总reads", "L", "total_reads"],
        "unique_mapped": ["唯一比对读段数", "唯一比对reads", "O", "unique_mapped_reads"],
        "map_rate": ["比对比例", "比对率", "M", "map_rate"],
        "dup_rate": ["重复读段比例", "重复率", "N", "dup_rate"],
        "gc": ["整体GC", "GC含量", "P", "GC"],
        "filt_ratio": ["被过滤读段比例", "AA", "filtered_ratio"],
    }

    mapping = {}
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    for std, cand_list in aliases.items():
        found = None
        for cand in cand_list:
            key = str(cand).strip().lower()
            if key in lower_cols:
                found = lower_cols[key]
                break
        if found is not None:
            mapping[std] = found
    return mapping


def clip01(x, eps=1e-6):
    return np.clip(x, eps, 1 - eps)


def compute_rolling_mean(x, y, bin_width=0.5, x_min=None, x_max=None):
    """
    在 x 轴上做等宽分箱（默认 0.5 周），计算 y 的均值与样本数。
    返回 DataFrame: [x_mid, y_mean, n]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = (~np.isnan(x)) & (~np.isnan(y))
    x = x[valid]
    y = y[valid]
    if x_min is None:
        x_min = np.nanmin(x) if len(x) else 0.0
    if x_max is None:
        x_max = np.nanmax(x) if len(x) else 1.0
    if x_max <= x_min:
        x_max = x_min + 1.0
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    idx = np.digitize(x, bins) - 1
    mids = (bins[:-1] + bins[1:]) / 2
    y_mean = []
    n_count = []
    x_mid = []
    for i in range(len(mids)):
        mask = idx == i
        if np.any(mask):
            y_mean.append(np.nanmean(y[mask]))
            n_count.append(int(np.sum(mask)))
            x_mid.append(mids[i])
    return pd.DataFrame({"x_mid": x_mid, "y_mean": y_mean, "n": n_count})


def ensure_outdir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)


# -----------------------------
# 主流程
# -----------------------------

def run_q1(excel_path, sheet_name=None, outdir="outputs_q1",
           min_ga=9.0, max_ga=30.0,
           y_threshold=0.04,  # 4%
           apply_qc=True,
           min_unique_reads=2_000_000,
           min_map_rate=0.95,
           max_filt_ratio=0.6,
           gc_low=0.3, gc_high=0.7,
           keep_first_draw_only=True):
    """
    Q1：建模 Y 浓度 ~ 孕周 + BMI（仅男胎），并输出若干可视化与汇总。
    """

    ensure_outdir(outdir)
    print(f"[INFO] 读取数据：{excel_path}")
    if sheet_name is not None:
        df_raw = pd.read_excel(excel_path, sheet_name=sheet_name)
    else:
        df_raw = pd.read_excel(excel_path)

    mapping = guess_columns(df_raw)
    print("[INFO] 列名映射（猜测）:", mapping)

    # 必要列检查
    required = ["ga", "bmi", "y_frac"]
    for col in required:
        if col not in mapping:
            raise ValueError(f"缺少关键列: {col}，请检查 Excel 列名。")

    df = df_raw.copy()

    # 标准化核心列
    df["GA_weeks"] = df[mapping["ga"]].apply(parse_ga_to_weeks)
    df["BMI"] = pd.to_numeric(df[mapping["bmi"]], errors="coerce")
    # Y 浓度，可能是百分比或比例
    y_raw = pd.to_numeric(df[mapping["y_frac"]], errors="coerce")
    # 若大多数值 > 1，则认为单位是百分比
    if (y_raw > 1).mean(skipna=True) > 0.5:
        df["Y_frac"] = y_raw / 100.0
    else:
        df["Y_frac"] = y_raw

    # 可选列
    df["ID"] = df[mapping["id"]] if "id" in mapping else np.arange(len(df))
    df["Age"] = pd.to_numeric(df[mapping["age"]], errors="coerce") if "age" in mapping else np.nan
    df["Draws"] = pd.to_numeric(df[mapping["draws"]], errors="coerce") if "draws" in mapping else np.nan
    if "date" in mapping:
        # 尝试解析日期时间
        df["Date"] = pd.to_datetime(df[mapping["date"]], errors="coerce")
    else:
        df["Date"] = pd.NaT

    # 质控相关（若存在）
    df["total_reads"] = pd.to_numeric(df[mapping["total_reads"]], errors="coerce") if "total_reads" in mapping else np.nan
    df["unique_mapped"] = pd.to_numeric(df[mapping["unique_mapped"]], errors="coerce") if "unique_mapped" in mapping else np.nan
    df["map_rate"] = pd.to_numeric(df[mapping["map_rate"]], errors="coerce") if "map_rate" in mapping else np.nan
    df["dup_rate"] = pd.to_numeric(df[mapping["dup_rate"]], errors="coerce") if "dup_rate" in mapping else np.nan
    df["GC"] = pd.to_numeric(df[mapping["gc"]], errors="coerce") if "gc" in mapping else np.nan
    df["filt_ratio"] = pd.to_numeric(df[mapping["filt_ratio"]], errors="coerce") if "filt_ratio" in mapping else np.nan

    # ---------- 基础清洗 ----------
    # 解析失败 / 缺失
    df_clean = df.copy()
    df_clean = df_clean[(~df_clean["GA_weeks"].isna()) & (~df_clean["BMI"].isna()) & (~df_clean["Y_frac"].isna())]
    # 限制孕周范围
    df_clean = df_clean[(df_clean["GA_weeks"] >= min_ga) & (df_clean["GA_weeks"] <= max_ga)]
    # 合理范围
    df_clean = df_clean[(df_clean["BMI"] > 10) & (df_clean["BMI"] < 60)]
    df_clean = df_clean[(df_clean["Y_frac"] > 0) & (df_clean["Y_frac"] < 0.5)]  # 经验范围：<50%

    # 推断胎儿性别：Y_frac>0 视为男胎（女胎通常为空/0）；保守一点，>0.005
    df_clean["is_male"] = df_clean["Y_frac"] > 0.005
    df_male = df_clean[df_clean["is_male"]].copy()

    # ---------- 可选：质控过滤 ----------
    qc_notes = []
    if apply_qc:
        before = len(df_male)
        mask = np.ones(len(df_male), dtype=bool)
        if not df_male["unique_mapped"].isna().all():
            mask = mask & (df_male["unique_mapped"] >= min_unique_reads)
            qc_notes.append(f"unique_mapped >= {min_unique_reads}")
        if not df_male["map_rate"].isna().all():
            mask = mask & (df_male["map_rate"] >= min_map_rate)
            qc_notes.append(f"map_rate >= {min_map_rate}")
        if not df_male["filt_ratio"].isna().all():
            mask = mask & (df_male["filt_ratio"] <= max_filt_ratio)
            qc_notes.append(f"filt_ratio <= {max_filt_ratio}")
        if not df_male["GC"].isna().all():
            mask = mask & (df_male["GC"] >= gc_low) & (df_male["GC"] <= gc_high)
            qc_notes.append(f"{gc_low} <= GC <= {gc_high}")
        df_male = df_male[mask].copy()
        print(f"[INFO] 质控过滤：{before} -> {len(df_male)} 条；条件: {', '.join(qc_notes) if qc_notes else '无'}")

    # ---------- 重复测量处理：仅保留首检 ----------
    if keep_first_draw_only:
        if df_male["Date"].notna().any():
            df_male = df_male.sort_values(["ID", "Date"]).groupby("ID", as_index=False).first()
        else:
            # 若无日期，则按最小孕周视为首检
            df_male = df_male.sort_values(["ID", "GA_weeks"]).groupby("ID", as_index=False).first()

    # 保存清洗结果
    clean_path = os.path.join(outdir, "q1_clean_male.csv")
    df_male.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 清洗后的男胎数据已保存：{clean_path}")
    print("[INFO] 清洗后样本量：", len(df_male))

    # ---------- 可视化 1：散点 + 滚动均值 ----------
    plt.figure()
    plt.scatter(df_male["GA_weeks"], df_male["Y_frac"], s=8, alpha=0.6)
    # 按孕周滚动均值（0.5周分箱）
    rm = compute_rolling_mean(df_male["GA_weeks"].values, df_male["Y_frac"].values, bin_width=0.5)
    if len(rm) > 0:
        plt.plot(rm["x_mid"], rm["y_mean"], linewidth=2)
    plt.axhline(y_threshold, linestyle="--", linewidth=1)
    plt.xlabel("孕周（周）")
    plt.ylabel("Y 染色体浓度（比例）")
    plt.title("Q1：Y 浓度 vs 孕周（散点 + 分箱均值）")
    fig1_path = os.path.join(outdir, "fig_q1_scatter_ga_vs_yfrac.png")
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=150)
    plt.close()
    print(f"[OK] 已保存图：{fig1_path}")

    # ---------- 可视化 2：不同 BMI 组的均值曲线 ----------
    # BMI 组：按题意的常见分组
    bins_bmi = [20, 28, 32, 36, 40, 100]
    labels_bmi = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "[40,+)"]
    df_male["BMI_group"] = pd.cut(df_male["BMI"], bins=bins_bmi, labels=labels_bmi, right=False)

    plt.figure()
    for label in labels_bmi:
        sub = df_male[df_male["BMI_group"] == label]
        if len(sub) == 0:
            continue
        rm = compute_rolling_mean(sub["GA_weeks"].values, sub["Y_frac"].values, bin_width=0.7)
        if len(rm) > 0:
            plt.plot(rm["x_mid"], rm["y_mean"], linewidth=2, label=label)
    plt.axhline(y_threshold, linestyle="--", linewidth=1)
    plt.xlabel("孕周（周）")
    plt.ylabel("Y 染色体浓度（比例）")
    plt.title("Q1：不同 BMI 组的 Y 浓度均值曲线（分箱）")
    plt.legend()
    fig2_path = os.path.join(outdir, "fig_q1_bmi_groups_curves.png")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"[OK] 已保存图：{fig2_path}")

    # ---------- 可选：使用 statsmodels 进行（加权）回归 ----------
    model_txt_path = os.path.join(outdir, "q1_model_summary.txt")
    preds_csv_path = os.path.join(outdir, "q1_model_predictions.csv")
    try:
        import statsmodels.api as sm
        from patsy import dmatrix

        # 准备数据
        dat = df_male[["GA_weeks", "BMI", "Y_frac", "unique_mapped"]].dropna().copy()
        # 逻辑变换，避免 0/1
        dat["Y_logit"] = np.log(clip01(dat["Y_frac"]) / (1 - clip01(dat["Y_frac"])))
        # 样条基，用 5 自由度
        X_spline = dmatrix("bs(GA_weeks, df=5, include_intercept=False)", data=dat, return_type="dataframe")
        # 交互：样条 * BMI
        for c in X_spline.columns:
            dat[f"{c}:BMI"] = X_spline[c].values * dat["BMI"].values
        X = pd.concat([X_spline, dat.filter(like=":BMI"), dat[["BMI"]]], axis=1)
        # 加权最小二乘（用 log(唯一比对读段数+1) 作为权重）
        if "unique_mapped" in dat and not dat["unique_mapped"].isna().all():
            w = np.log1p(dat["unique_mapped"].values)
        else:
            w = None
        X = sm.add_constant(X, has_constant="add")
        model = sm.WLS(dat["Y_logit"].values, X.values, weights=w) if w is not None else sm.OLS(dat["Y_logit"].values, X.values)
        res = model.fit()
        with open(model_txt_path, "w", encoding="utf-8") as f:
            f.write(res.summary().as_text())
        print(f"[OK] 统计模型摘要已保存：{model_txt_path}")

        # 画预测曲线（选取若干 BMI 值）
        ga_grid = np.linspace(max(9, dat["GA_weeks"].min()), min(30, dat["GA_weeks"].max()), 100)
        bmi_grid = [22, 28, 32, 36, 40]
        plt.figure()
        for bmi in bmi_grid:
            # 生成设计矩阵
            df_pred = pd.DataFrame({"GA_weeks": ga_grid, "BMI": bmi})
            Xg = dmatrix("bs(GA_weeks, df=5, include_intercept=False)", data=df_pred, return_type="dataframe")
            # 交互
            for c in Xg.columns:
                df_pred[f"{c}:BMI"] = Xg[c].values * df_pred["BMI"].values
            Xp = pd.concat([Xg, df_pred.filter(like=':BMI'), df_pred[["BMI"]]], axis=1)
            Xp = sm.add_constant(Xp, has_constant="add")
            y_logit_hat = res.predict(Xp.values)
            y_hat = 1 / (1 + np.exp(-y_logit_hat))
            plt.plot(ga_grid, y_hat, linewidth=2, label=f"BMI={bmi}")
        plt.axhline(y_threshold, linestyle="--", linewidth=1)
        plt.xlabel("孕周（周）")
        plt.ylabel("预测的 Y 染色体浓度（比例）")
        plt.title("Q1：样条+交互回归的预测曲线（若 statsmodels 可用）")
        plt.legend()
        fig3_path = os.path.join(outdir, "fig_q1_model_pred_curves.png")
        plt.tight_layout()
        plt.savefig(fig3_path, dpi=150)
        plt.close()
        print(f"[OK] 已保存图：{fig3_path}")

        # 保存预测细表
        pred_recs = []
        for bmi in bmi_grid:
            for ga in ga_grid:
                df_pred = pd.DataFrame({"GA_weeks": [ga], "BMI": [bmi]})
                Xg = dmatrix("bs(GA_weeks, df=5, include_intercept=False)", data=df_pred, return_type="dataframe")
                for c in Xg.columns:
                    df_pred[f"{c}:BMI"] = Xg[c].values * df_pred["BMI"].values
                Xp = pd.concat([Xg, df_pred.filter(like=':BMI'), df_pred[["BMI"]]], axis=1)
                Xp = sm.add_constant(Xp, has_constant="add")
                y_logit_hat = res.predict(Xp.values)[0]
                y_hat = 1 / (1 + np.exp(-y_logit_hat))
                pred_recs.append({"GA_weeks": ga, "BMI": bmi, "Y_frac_hat": y_hat})
        pd.DataFrame(pred_recs).to_csv(preds_csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 预测结果明细已保存：{preds_csv_path}")

    except Exception as e:
        with open(model_txt_path, "w", encoding="utf-8") as f:
            f.write("未能运行 statsmodels 模型。原因：{}\n".format(str(e)))
            f.write("建议：pip install statsmodels patsy\n")
            f.write("已生成基础可视化（散点与分组均值），可用于答题与报告。\n")
        print(f"[WARN] statsmodels 不可用或建模失败，已写入说明：{model_txt_path}")

    # ---------- 输出一些关键表，用于Q2预览 ----------
    # 各孕周（按0.5周分箱）达到阈值的比例（命中率），分 BMI 组
    ga_bins = np.arange(9, 30.5, 0.5)
    df_male["GA_bin"] = pd.cut(df_male["GA_weeks"], bins=ga_bins, right=False)
    df_male["hit"] = df_male["Y_frac"] >= y_threshold
    hit_table = (
        df_male.groupby(["BMI_group", "GA_bin"])["hit"]
        .mean()
        .reset_index()
        .rename(columns={"hit": "hit_rate"})
    )
    hit_csv = os.path.join(outdir, "q1_hit_rate_by_ga_bmi.csv")
    hit_table.to_csv(hit_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 命中率表已保存：{hit_csv}")

    # 简要汇总打印
    print("\n===== Q1 简要汇总 =====")
    print("样本量（男胎/清洗后）:", len(df_male))
    if len(df_male):
        print("孕周范围: [{:.2f}, {:.2f}] 周".format(df_male["GA_weeks"].min(), df_male["GA_weeks"].max()))
        print("BMI 中位数: {:.2f}".format(df_male["BMI"].median()))
        print("Y 浓度中位数: {:.3f}".format(df_male["Y_frac"].median()))
        print("在阈值 {:.0%} 以上的比例: {:.1%}".format(y_threshold, (df_male["Y_frac"] >= y_threshold).mean()))
    else:
        print("清洗后数据为空，请检查过滤条件与列名映射。")


def main():
    parser = argparse.ArgumentParser(description="NIPT Q1：Y 浓度 ~ 孕周 + BMI 分析（仅男胎）")
    parser.add_argument("--excel", type=str, default="附件.xlsx", help="输入 Excel 文件路径（默认：附件.xlsx）")
    parser.add_argument("--sheet", type=str, default=None, help="工作表名称（默认读取第一个）")
    parser.add_argument("--outdir", type=str, default="outputs_q1", help="输出目录")
    parser.add_argument("--min_ga", type=float, default=9.0, help="孕周下限（周）")
    parser.add_argument("--max_ga", type=float, default=30.0, help="孕周上限（周）")
    parser.add_argument("--y_threshold", type=float, default=0.04, help="达标阈值（比例），默认 0.04 即 4%")
    parser.add_argument("--no_qc", action="store_true", help="不进行质控过滤（默认进行）")
    parser.add_argument("--min_unique_reads", type=float, default=2000000, help="唯一比对读段数下限（若存在该列）")
    parser.add_argument("--min_map_rate", type=float, default=0.95, help="比对率下限（若存在该列）")
    parser.add_argument("--max_filt_ratio", type=float, default=0.6, help="被过滤读段比例上限（若存在该列）")
    parser.add_argument("--gc_low", type=float, default=0.30, help="整体 GC 下限（若存在该列）")
    parser.add_argument("--gc_high", type=float, default=0.70, help="整体 GC 上限（若存在该列）")
    parser.add_argument("--keep_all_draws", action="store_true", help="保留重复检测（默认仅保留首检）")

    args = parser.parse_args()

    run_q1(
        excel_path=args.excel,
        sheet_name=args.sheet,
        outdir=args.outdir,
        min_ga=args.min_ga,
        max_ga=args.max_ga,
        y_threshold=args.y_threshold,
        apply_qc=not args.no_qc,
        min_unique_reads=args.min_unique_reads,
        min_map_rate=args.min_map_rate,
        max_filt_ratio=args.max_filt_ratio,
        gc_low=args.gc_low,
        gc_high=args.gc_high,
        keep_first_draw_only=not args.keep_all_draws,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
