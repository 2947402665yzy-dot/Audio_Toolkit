"""
数据集统计模块

扫描音频文件夹，统计总文件数、总时长、采样率分布、格式分布等信息，
并导出详细 CSV 报告。
依赖：pandas（数据统计与 CSV 导出）、librosa（音频信息读取）
"""

import os
import sys

# 导入共享工具模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scan_audio_files, ProgressTracker, format_time


def compute_dataset_stats(input_dir, progress_callback=None):
    """
    扫描文件夹中所有音频文件，计算数据集统计信息并导出 CSV。

    统计内容包括：总文件数、总时长、平均时长、最短/最长文件、
    采样率分布、格式分布、总文件大小。每个文件的详细信息存入
    DataFrame 并导出为 CSV。

    参数:
        input_dir (str): 音频文件夹路径
        progress_callback (callable, optional): 进度回调函数，
            签名为 callback(current, total, message)

    返回:
        dict: 包含以下键的统计字典:
            - total_files: 总文件数
            - total_duration: 总时长（格式化字符串）
            - avg_duration: 平均时长（格式化字符串）
            - shortest: 最短文件信息 {"filename":..., "duration":...}
            - longest: 最长文件信息 {"filename":..., "duration":...}
            - sr_distribution: 采样率分布 {采样率: 文件数}
            - format_distribution: 格式分布 {格式: 文件数}
            - total_size_mb: 总文件大小（MB）
            - output_csv: 输出 CSV 路径
    """
    import pandas as pd
    import librosa

    # 扫描音频文件
    files = scan_audio_files(input_dir)
    tracker = ProgressTracker(len(files), progress_callback)

    records = []

    for file_path in files:
        try:
            # 用 librosa 读取音频信息（mono=False 保留原始声道）
            y, sr = librosa.load(file_path, sr=None, mono=False)
            duration = librosa.get_duration(y=y, sr=sr)
            file_size = os.path.getsize(file_path)
            file_format = os.path.splitext(file_path)[1].lower().lstrip(".")
            channels = 1 if y.ndim == 1 else int(y.shape[0])

            records.append({
                "filename": os.path.basename(file_path),
                "path": file_path,
                "duration_sec": round(duration, 2),
                "sample_rate": int(sr),
                "channels": channels,
                "format": file_format,
                "file_size_mb": round(file_size / (1024 * 1024), 3),
            })
        except Exception:
            # 跳过无法读取的文件
            pass
        tracker.update(os.path.basename(file_path))

    # 导出详细 CSV
    df = pd.DataFrame(records)
    output_csv = os.path.join(input_dir, "dataset_stats.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # 无数据时返回空统计
    if len(df) == 0:
        return {
            "total_files": 0,
            "total_duration": "00:00",
            "avg_duration": "00:00",
            "shortest": {"filename": "", "duration": "00:00"},
            "longest": {"filename": "", "duration": "00:00"},
            "sr_distribution": {},
            "format_distribution": {},
            "total_size_mb": 0.0,
            "output_csv": output_csv,
        }

    # 计算汇总统计信息
    total_duration = float(df["duration_sec"].sum())
    avg_duration = float(df["duration_sec"].mean())

    # 最短和最长文件
    shortest_row = df.loc[df["duration_sec"].idxmin()]
    longest_row = df.loc[df["duration_sec"].idxmax()]

    # 采样率分布
    sr_counts = df["sample_rate"].value_counts()
    sr_distribution = {str(int(k)): int(v) for k, v in sr_counts.items()}

    # 格式分布
    fmt_counts = df["format"].value_counts()
    format_distribution = {str(k): int(v) for k, v in fmt_counts.items()}

    total_size_mb = float(df["file_size_mb"].sum())

    return {
        "total_files": len(df),
        "total_duration": format_time(total_duration),
        "avg_duration": format_time(avg_duration),
        "shortest": {
            "filename": str(shortest_row["filename"]),
            "duration": format_time(float(shortest_row["duration_sec"])),
        },
        "longest": {
            "filename": str(longest_row["filename"]),
            "duration": format_time(float(longest_row["duration_sec"])),
        },
        "sr_distribution": sr_distribution,
        "format_distribution": format_distribution,
        "total_size_mb": round(total_size_mb, 2),
        "output_csv": output_csv,
    }


if __name__ == "__main__":
    # 命令行调试入口
    import argparse

    parser = argparse.ArgumentParser(description="统计音频数据集信息")
    parser.add_argument("input_dir", help="音频文件夹路径")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"文件夹不存在: {args.input_dir}")
        sys.exit(1)

    stats = compute_dataset_stats(args.input_dir)
    print("=" * 50)
    print("数据集统计结果")
    print("=" * 50)
    print(f"总文件数: {stats['total_files']}")
    print(f"总时长: {stats['total_duration']}")
    print(f"平均时长: {stats['avg_duration']}")
    print(f"最短文件: {stats['shortest']['filename']} ({stats['shortest']['duration']})")
    print(f"最长文件: {stats['longest']['filename']} ({stats['longest']['duration']})")
    print(f"采样率分布: {stats['sr_distribution']}")
    print(f"格式分布: {stats['format_distribution']}")
    print(f"总文件大小: {stats['total_size_mb']} MB")
    print(f"输出 CSV: {stats['output_csv']}")
