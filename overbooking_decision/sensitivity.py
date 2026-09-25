"""
ขั้นที่ 5: Sensitivity analysis ของค่า k
รันทั้ง 4 นโยบายซ้ำสำหรับทุกค่า k ใน config เพื่อตรวจว่า "นโยบายตามโมเดล" ยังชนะทุกค่า k หรือไม่

v2: รับ backtest_fn เป็นพารามิเตอร์ได้ (default = backtest_policies เดิม เพื่อไม่ให้ของเก่าพัง)
ใช้กับ backtest_sequential.backtest_policies_sequential ได้โดยส่ง backtest_fn เข้ามา
"""
from __future__ import annotations

import pandas as pd

from backtest import aggregate_results, backtest_policies


def run_sensitivity(
    df: pd.DataFrame,
    cost_config,
    avg_cancel_rate_lookup: dict | None = None,
    backtest_fn=backtest_policies,
) -> pd.DataFrame:
    all_summaries = []
    for k in cost_config.k_values:
        bt = backtest_fn(df, cost_config, k=k, avg_cancel_rate_lookup=avg_cancel_rate_lookup)
        summary = aggregate_results(bt)
        summary["k"] = k
        all_summaries.append(summary)
    return pd.concat(all_summaries, ignore_index=True)


def plot_sensitivity(sensitivity_df: pd.DataFrame, save_path: str = "sensitivity_plot.png"):
    import matplotlib.pyplot as plt

    # หมายเหตุ: ใช้ label ภาษาอังกฤษ เพราะ font เริ่มต้นของ matplotlib (DejaVu Sans)
    # ไม่มี glyph ภาษาไทย ถ้าต้องการ label ไทยในกราฟ ต้องติดตั้ง font ไทย
    # (เช่น TH Sarabun) แล้วตั้ง plt.rcParams["font.family"] เอง
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy, grp in sensitivity_df.groupby("policy"):
        grp = grp.sort_values("k")
        ax.plot(grp["k"], grp["total_profit"], marker="o", label=policy)
    ax.set_xlabel("k (compensation multiplier = cost_overbook / adr)")
    ax.set_ylabel("Total profit")
    ax.set_title("Profit sensitivity to k, by policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return save_path
