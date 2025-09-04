- **Q1.py**
- **Q2.py**
- Q3.py
- Q4.py
- **文献**



# Q1

  ## 基本用法 最好用命令行（同目录放 附件.xlsx）
    python nipt_q1_pipeline.py --excel 附件.xlsx
  
  ### 指定工作表、关闭质控、保留所有重复检测
    python nipt_q1_pipeline.py --excel 附件.xlsx --sheet Sheet1 --no_qc --keep_all_draws
  
  #### 脚本会输出：
      outputs_q1/q1_clean_male.csv 清洗后的男胎样本
      outputs_q1/fig_q1_scatter_ga_vs_yfrac.png 散点 + 分箱均值
      outputs_q1/fig_q1_bmi_groups_curves.png 不同 BMI 组均值曲线
      outputs_q1/q1_hit_rate_by_ga_bmi.csv 不同孕周×BMI 组的达标命中率表（给后续 Q2 用）
      若本机装有 statsmodels+patsy：
      outputs_q1/q1_model_summary.txt 样条+交互回归摘要
      outputs_q1/fig_q1_model_pred_curves.png 预测曲线（若安装成功）
      outputs_q1/q1_model_predictions.csv 预测细表


# Q2
  ## 可直接用 Q1 的清洗结果
    python nipt_q2_pipeline.py --clean_csv outputs_q1/q1_clean_male.csv
  
  ### 或者让脚本自行从 Excel 清洗（与 Q1 示例一致，仅按 GC 做轻质控）
      python nipt_q2_pipeline.py --excel 附件.xlsx --sheet Sheet1
  
  ### 改命中率阈值/风险权衡参数
     python nipt_q2_pipeline.py --target_hit 0.9 --lam 1.0
  
  ### 调整分箱与平滑
     python nipt_q2_pipeline.py --bin_width 0.5 --smooth_window 3
  
  ### 提高自助法次数以获得更稳定的CI
     python nipt_q2_pipeline.py --bootstrap_B 500
  # Q3
  ---
    第三问总体目标
    在男胎样本里，综合考虑孕周、BMI、年龄、身高、体重等对“Y 浓度≥4%（达标）”的影响，并把“检测误差/测序质量”纳入，给出按 BMI 分组的最佳 NIPT 时点，使潜在风险最小，同时评估误差的影响。
    建模思路
    变量与标签
    标签：hit = 1{Y_frac ≥ 0.04}（达标）
    自变量：孕周（GA，周）、BMI（及分组）、年龄、身高、体重（必要时可加入读段/GC/过滤比例等质控指标作为权重或误差因子）。
    两层概率模型（把“检测误差”明确建模）
    层A：先对Y 浓度本身做回归，得到条件均值 μ 与方差结构 σ：
    用 logit 变换的 Y（或直接 Y）做回归：
    g( E[Y | GA,BMI,Age,Height,Weight] ) = f(GA,BMI,交互) + Age + Height + Weight。
    其中 f(GA,BMI) 用多项式/样条逼近；σ 可设为随测序质量(QC)变化（如 GC、过滤比例）或分箱估计得到的测量方差。
    层B：把观测 Y 的误差当作高斯噪声：Y_obs ~ Normal(μ, σ^2)，于是
    P(hit | x) = P(Y_obs ≥ 0.04) = 1 - Φ((0.04 - μ)/σ)，得到达标概率曲线随孕周的变化。
    （没有 statsmodels 也能用纯 numpy 拟合：对 μ 用多项式回归，对 σ 用局部残差方差或按 QC 指标分层估计。）
    按 BMI 分组给“最佳时点”
    目标1：最早达标时点 t* = min{ t : P_hit(t | 组, 其余取中位) ≥ τ }，τ 默认 0.90（可敏感性分析 0.85–0.95）。
    目标2：风险-命中权衡：最小化
    Risk(t) = risk_level(t) + λ · (1 - P_hit(t))，其中
    risk_level(t)=1(≤12周), 2(13–27周), 3(≥28周)（题面风险分档）；λ 做灵敏度分析（如 0.5/1/2）。
    不确定性与“检测误差影响”
    Bootstrap（按孕妇ID重采样）→ 给 t* 的 95% CI。
    误差敏感性：把 σ 按 ±20% 扰动或按不同 QC 分层，重算 P_hit 与 t*，比较差异；或用 Monte Carlo 对 Y 添加噪声后重算 hit 曲线，量化“检测误差对结果的影响”（题面要求）。
    交付物
    表：各 BMI 组的 最早达标时点 与 风险-命中最优时点（含 95% CI、对 τ/λ 的敏感性）。
    图：① P_hit–GA 曲线（分 BMI 组，误差带）；② 风险目标函数曲线；③ 误差敏感性对比图。
    说明：哪些协变量（年龄/身高/体重）显著、会让曲线整体平移多少（可用偏依赖/分位数情景展示）。
  ---
  ## 优先使用 Q1 的清洗结果（推荐）
    python nipt_q3_pipeline.py --clean_csv outputs_q1/q1_clean_male.csv

## 或者直接从 Excel 清洗（与前文要求一致，含 GC 轻质控）
    python nipt_q3_pipeline.py --excel 附件.xlsx --sheet Sheet1

## 可调参数（示例）
    python nipt_q3_pipeline.py --target_hit 0.9 --lam 1.0 --sigma_mode ga_local --deg 3 --bootstrap_B 300
 ##
    输出：
    outputs_q3/q3_phit_curves.png：各 BMI 组的 P(hit) 曲线 + 90% 阈值与竖线
    outputs_q3/q3_risk_tradeoff_curves.png：风险-命中率目标函数曲线（λ 可调）
    outputs_q3/q3_sigma_sensitivity.png：σ±20% 敏感性示例图
    outputs_q3/q3_recommendations.csv：各 BMI 组样本量、两类推荐孕周及 95% CI
    outputs_q3/q3_mu_coefs.csv：均值模型系数（便于写报告）
    （若 sigma_mode=ga_local）outputs_q3/q3_sigma_by_ga.csv：σ(孕周) 估计表
  >> 脚本会自动兼容你给的列名映射（如“孕妇代码/年龄/检测孕周/孕妇BMI/Y染色体浓度/GC含量”等），若身高/体重不存在会自动略过，不影响运行。
  # Q4
  ......
  ---
