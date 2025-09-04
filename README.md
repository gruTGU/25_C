————Q1————

  ## 基本用法 命令行（同目录放 附件.xlsx）
    python nipt_q1_pipeline.py --excel 附件.xlsx
  
  指定工作表、关闭质控、保留所有重复检测
    python nipt_q1_pipeline.py --excel 附件.xlsx --sheet Sheet1 --no_qc --keep_all_draws
  
    脚本会输出：
      outputs_q1/q1_clean_male.csv 清洗后的男胎样本
      outputs_q1/fig_q1_scatter_ga_vs_yfrac.png 散点 + 分箱均值
      outputs_q1/fig_q1_bmi_groups_curves.png 不同 BMI 组均值曲线
      outputs_q1/q1_hit_rate_by_ga_bmi.csv 不同孕周×BMI 组的达标命中率表（给后续 Q2 用）
      若本机装有 statsmodels+patsy：
      outputs_q1/q1_model_summary.txt 样条+交互回归摘要
      outputs_q1/fig_q1_model_pred_curves.png 预测曲线（若安装成功）
      outputs_q1/q1_model_predictions.csv 预测细表


————Q2————
  ## 可直接用 Q1 的清洗结果
    python nipt_q2_pipeline.py --clean_csv outputs_q1/q1_clean_male.csv
  
  或者让脚本自行从 Excel 清洗（与 Q1 示例一致，仅按 GC 做轻质控）：  python nipt_q2_pipeline.py --excel 附件.xlsx --sheet Sheet1
  
  改命中率阈值/风险权衡参数： python nipt_q2_pipeline.py --target_hit 0.9 --lam 1.0
  
  调整分箱与平滑：python nipt_q2_pipeline.py --bin_width 0.5 --smooth_window 3
  
  提高自助法次数以获得更稳定的CI：  python nipt_q2_pipeline.py --bootstrap_B 500
