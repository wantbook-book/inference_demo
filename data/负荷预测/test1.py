# -*- coding: utf-8 -*-
"""
功能：读取 CSV 文件并绘制折线图
支持日期格式：2020-3-4、2020/3/4、2020.3.4、2020年3月4日 等
只需修改文件路径即可使用
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from pathlib import Path


def plot_csv_line(
    csv_path,
    ts_col="timestamp",
    save_path=None,
    title=None,
    figsize=(12, 6),
    encoding="utf-8-sig"
):
    """读取 CSV 并绘制折线图。"""
    # 读取文件
    df = pd.read_csv(csv_path, encoding=encoding)

    if ts_col not in df.columns:
        raise ValueError(f"未找到时间列 '{ts_col}'，现有列：{list(df.columns)}")

    # 自动识别多种日期格式
    ts = df[ts_col].astype(str).str.strip()
    ts = (
        ts.str.replace("年", "-", regex=False)
          .str.replace("月", "-", regex=False)
          .str.replace("日", "", regex=False)
    )
    df[ts_col] = pd.to_datetime(ts, errors="coerce", infer_datetime_format=True)

    # 丢弃无效时间并排序
    before = len(df)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    if df.empty:
        raise ValueError("时间列无法解析为有效日期。")
    dropped = before - len(df)
    if dropped > 0:
        print(f"提示：有 {dropped} 行时间解析失败，已忽略。")

    # 选择数值列
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        for c in df.columns:
            if c != ts_col:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        raise ValueError("未找到可绘制的数值列。")

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    for col in numeric_cols:
        ax.plot(df[ts_col], df[col], label=col)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title or Path(csv_path).stem)

    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

    ax.legend()
    ax.grid(True, alpha=0.5)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"✅ 图像已保存：{save_path}")
    else:
        plt.show()



if __name__ == "__main__":
    csv_path = r"./data/sample1/test_1024.csv"        
    save_path = r"result.png"     

    plot_csv_line(
        csv_path=csv_path,
        ts_col="date",        # 时间列列名
        save_path=save_path,       # 保存路径（或 None）
        title="timeseries",
        figsize=(12, 6)
    )
