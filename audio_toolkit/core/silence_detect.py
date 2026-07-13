"""
静音检测模块

检测音频文件中的静音片段，支持单个文件和批量处理。
依赖：librosa（音频加载与能量分析）
"""

import os
import sys

# 导入共享工具模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scan_audio_files, ensure_output_dir, ProgressTracker


def detect_silence(file_path, threshold_db=-40, min_duration_sec=0.5):
    """
    检测单个音频文件中的静音片段。

    使用 librosa 计算短时 RMS 能量，将能量低于 threshold_db（dBFS）
    的帧视为静音，合并连续静音帧为静音段，过滤掉短于
    min_duration_sec 的段。

    参数:
        file_path (str): 音频文件路径
        threshold_db (float): 静音阈值（dBFS），低于此值视为静音，默认 -40
        min_duration_sec (float): 静音段最短时长（秒），短于此值的段被忽略，默认 0.5

    返回:
        dict: 包含以下键:
            - filename: 文件名
            - silences: 静音段列表，每项为 {"start": 秒, "end": 秒, "duration": 秒}
            - total_silence_sec: 静音总时长（秒）
            - silence_ratio: 静音占总时长的比例（0~1）
    """
    import librosa
    import numpy as np

    # 加载音频，保留原始采样率
    y, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    # 计算短时 RMS 能量
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # 将 RMS 转为 dBFS（参考满刻度 1.0）
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))

    # 静音帧掩码：低于阈值的帧
    silent_mask = rms_db < threshold_db

    # 将连续静音帧合并为静音段
    silences = []
    in_silence = False
    start_frame = 0

    for i, is_silent in enumerate(silent_mask):
        if is_silent and not in_silence:
            # 静音段开始
            in_silence = True
            start_frame = i
        elif not is_silent and in_silence:
            # 静音段结束
            in_silence = False
            start_sec = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end_sec = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
            duration = end_sec - start_sec
            if duration >= min_duration_sec:
                silences.append({
                    "start": round(start_sec, 3),
                    "end": round(end_sec, 3),
                    "duration": round(duration, 3),
                })

    # 处理文件末尾的静音段
    if in_silence:
        start_sec = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        end_sec = librosa.frames_to_time(len(silent_mask), sr=sr, hop_length=hop_length)
        duration = end_sec - start_sec
        if duration >= min_duration_sec:
            silences.append({
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "duration": round(duration, 3),
            })

    total_silence = sum(s["duration"] for s in silences)
    silence_ratio = total_silence / total_duration if total_duration > 0 else 0.0

    return {
        "filename": os.path.basename(file_path),
        "silences": silences,
        "total_silence_sec": round(total_silence, 3),
        "silence_ratio": round(silence_ratio, 4),
    }


def detect_silence_batch(input_dir, output_csv, threshold_db=-40, min_duration_sec=0.5, progress_callback=None):
    """
    批量检测音频文件中的静音片段，导出 CSV 报告。

    对输入目录中的每个音频文件执行静音检测，将每个文件的静音
    段汇总信息导出为 CSV 报告。

    参数:
        input_dir (str): 输入音频文件夹路径
        output_csv (str): 输出 CSV 文件路径
        threshold_db (float): 静音阈值（dBFS），默认 -40
        min_duration_sec (float): 静音段最短时长（秒），默认 0.5
        progress_callback (callable, optional): 进度回调函数，
            签名为 callback(current, total, message)

    返回:
        dict: 包含以下键:
            - total_files: 处理的文件总数
            - output_csv: 输出 CSV 文件路径
            - errors: 错误列表，每项为 {"file": 路径, "error": 信息}
    """
    import pandas as pd

    # 扫描音频文件
    files = scan_audio_files(input_dir)
    tracker = ProgressTracker(len(files), progress_callback)

    records = []
    errors = []

    for file_path in files:
        try:
            result = detect_silence(
                file_path, threshold_db=threshold_db, min_duration_sec=min_duration_sec
            )
            # 将每个文件的静音信息汇总为一行
            silence_count = len(result["silences"])
            # 将静音段列表转为可读字符串
            silence_str = "; ".join(
                f"[{s['start']:.2f}-{s['end']:.2f}]" for s in result["silences"]
            )
            records.append({
                "filename": result["filename"],
                "silence_count": silence_count,
                "total_silence_sec": result["total_silence_sec"],
                "silence_ratio": result["silence_ratio"],
                "silence_segments": silence_str,
            })
        except Exception as exc:
            errors.append({"file": file_path, "error": str(exc)})
        tracker.update(os.path.basename(file_path))

    # 导出 CSV 报告
    ensure_output_dir(os.path.dirname(os.path.abspath(output_csv)))
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return {
        "total_files": len(files),
        "output_csv": output_csv,
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口
    import argparse

    parser = argparse.ArgumentParser(description="检测音频文件中的静音片段")
    parser.add_argument("path", help="音频文件或文件夹路径")
    parser.add_argument("-o", "--output", default="silence_report.csv", help="输出 CSV 路径（批量模式）")
    parser.add_argument("-t", "--threshold", type=float, default=-40, help="静音阈值 dBFS（默认 -40）")
    parser.add_argument("-m", "--min-duration", type=float, default=0.5, help="最短静音时长秒（默认 0.5）")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        result = detect_silence_batch(
            args.path, args.output,
            threshold_db=args.threshold, min_duration_sec=args.min_duration,
        )
        print(f"处理完成: {result['total_files']} 个文件")
        print(f"输出 CSV: {result['output_csv']}")
        if result["errors"]:
            print(f"错误数: {len(result['errors'])}")
            for err in result["errors"]:
                print(f"  - {err['file']}: {err['error']}")
    elif os.path.isfile(args.path):
        result = detect_silence(
            args.path, threshold_db=args.threshold, min_duration_sec=args.min_duration
        )
        print(f"文件: {result['filename']}")
        print(f"静音段数: {len(result['silences'])}")
        print(f"静音总时长: {result['total_silence_sec']} 秒")
        print(f"静音比例: {result['silence_ratio']:.2%}")
        for s in result["silences"]:
            print(f"  [{s['start']:.2f} - {s['end']:.2f}] 时长 {s['duration']:.2f}s")
    else:
        print(f"路径不存在: {args.path}")
