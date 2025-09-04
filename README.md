
--Q1
  ## 基本用法 命令行（同目录放 附件.xlsx）
  python nipt_q1_pipeline.py --excel 附件.xlsx
  ## 指定工作表、关闭质控、保留所有重复检测
  python nipt_q1_pipeline.py --excel 附件.xlsx --sheet Sheet1 --no_qc --keep_all_draws

