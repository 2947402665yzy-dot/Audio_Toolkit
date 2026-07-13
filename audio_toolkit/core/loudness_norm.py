"""
响度标准化模块

将音频文件响度标准化到目标 dBFS 值，支持单个文件和批量处理。
依赖：pydub（音频加载与增益调整）
"""

import os
import sys

# 导入共享工具模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scan_audio_files, ensure_output_dir, ProgressTracker


def normalize_loudness(file_path, output_dir, target_dbfs=-16.0):
    """
    将单个音频文件的响度标准化到目标 dBFS。

    使用 pydub 加载音频，计算当前 dBFS，通过 apply_gain 调整
    增益使响度达到目标值，导出时保持原格式，文件名前缀 norm_。

    参数:
        file_path (str): 输入音频文件路径
        output_dir (str): 输出目录路径
        target_dbfs (float): 目标响度（dBFS），默认 -16.0

    返回:
        dict: 包含以下键:
            - input: 输入文件路径
            - output: 输出文件路径
            - original_dbfs: 原始响度（dBFS）
            - normalized_dbfs: 标准化后响度（dBFS）

    抛出:
        ValueError: 音频为静音（dBFS 为 -inf），无法标准化
    """
    from pydub import AudioSegment

    # 加载音频（pydub 自动识别格式）
    audio = AudioSegment.from_file(file_path)

    original_dbfs = audio.dBFS

    # 如果音频完全静音，dBFS 为 -inf，无法标准化
    if original_dbfs == float("-inf"):
        raise ValueError(f"音频文件为静音，无法计算响度: {file_path}")

    # 计算需要调整的增益并应用
    delta_db = target_dbfs - original_dbfs
    normalized = audio.apply_gain(delta_db)

    # 确保输出目录存在
    ensure_output_dir(output_dir)

    # 构建输出文件名: norm_{原文件名}
    basename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"norm_{basename}")

    # 导出，保持原格式
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    normalized.export(output_path, format=ext)

    return {
        "input": file_path,
        "output": output_path,
        "original_dbfs": round(original_dbfs, 2),
        "normalized_dbfs": round(normalized.dBFS, 2),
    }


def normalize_loudness_batch(input_dir, output_dir, target_dbfs=-16.0, progress_callback=None):
    """
    批量标准化音频文件响度。

    扫描输入目录中的所有音频文件，逐个标准化到目标 dBFS，
    输出到指定目录，文件名前缀 norm_。

    参数:
        input_dir (str): 输入音频文件夹路径
        output_dir (str): 输出目录路径
        target_dbfs (float): 目标响度（dBFS），默认 -16.0
        progress_callback (callable, optional): 进度回调函数，
            签名为 callback(current, total, message)

    返回:
        dict: 包含以下键:
            - total_files: 处理的文件总数
            - errors: 错误列表，每项为 {"file": 路径, "error": 信息}
    """
    # 扫描音频文件
    files = scan_audio_files(input_dir)
    tracker = ProgressTracker(len(files), progress_callback)

    errors = []

    for file_path in files:
        try:
            normalize_loudness(file_path, output_dir, target_dbfs=target_dbfs)
        except Exception as exc:
            errors.append({"file": file_path, "error": str(exc)})
        tracker.update(os.path.basename(file_path))

    return {
        "total_files": len(files),
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口
    import argparse

    parser = argparse.ArgumentParser(description="音频响度标准化")
    parser.add_argument("path", help="音频文件或文件夹路径")
    parser.add_argument("-o", "--output", default="./normalized", help="输出目录路径")
    parser.add_argument("-t", "--target", type=float, default=-16.0, help="目标 dBFS（默认 -16.0）")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        result = normalize_loudness_batch(args.path, args.output, target_dbfs=args.target)
        print(f"处理完成: {result['total_files']} 个文件")
        print(f"输出目录: {args.output}")
        if result["errors"]:
            print(f"错误数: {len(result['errors'])}")
            for err in result["errors"]:
                print(f"  - {err['file']}: {err['error']}")
    elif os.path.isfile(args.path):
        result = normalize_loudness(args.path, args.output, target_dbfs=args.target)
        print(f"输入文件: {result['input']}")
        print(f"输出文件: {result['output']}")
        print(f"原始响度: {result['original_dbfs']} dBFS")
        print(f"标准化响度: {result['normalized_dbfs']} dBFS")
    else:
        print(f"路径不存在: {args.path}")
