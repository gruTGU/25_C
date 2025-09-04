# 25_C
C 
--1
  ## 基本用法（同目录放 附件.xlsx）
  python nipt_q1_pipeline.py --excel 附件.xlsx

  ## 指定工作表、关闭质控、保留所有重复检测
  python nipt_q1_pipeline.py --excel 附件.xlsx --sheet Sheet1 --no_qc --keep_all_draws
  
  ## 修改一些阈值（例）
  python nipt_q1_pipeline.py --y_threshold 0.04 --min_unique_reads 3000000 --min_ga 10 --max_ga 28
