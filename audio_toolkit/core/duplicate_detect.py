"""
重复文件检测模块

通过计算文件哈希值检测音频文件夹中的重复文件。
依赖：hashlib（哈希计算）、pandas（CSV 导出）
"""

import os
import sys
import hashlib

# 导入共享工具模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scan_audio_files, ProgressTracker


def compute_file_hash(file_path, algorithm="md5"):
    """
    计算文件的哈希值。

    采用分块读取方式计算哈希，避免大文件一次性读入内存导致
    内存占用过高。

    参数:
        file_path (str): 文件路径
        algorithm (str): 哈希算法名称，如 "md5"、"sha1"、"sha256"，默认 "md5"

    返回:
        str: 十六进制哈希字符串

    抛出:
        ValueError: 不支持的哈希算法
    """
    hash_func = hashlib.new(algorithm)
    chunk_size = 65536  # 64KB 分块读取

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_func.update(chunk)

    return hash_func.hexdigest()


def detect_duplicates(input_dir, progress_callback=None):
    """
    检测文件夹中的重复音频文件。

    扫描输入目录中的所有音频文件，计算每个文件的哈希值，
    将哈希值相同的文件归为一组，即为重复文件组。同时导出
    CSV 报告到 input_dir/duplicate_report.csv。

    参数:
        input_dir (str): 输入音频文件夹路径
        progress_callback (callable, optional): 进度回调函数，
            签名为 callback(current, total, message)

    返回:
        dict: 包含以下键:
            - total_files: 总文件数
            - unique_files: 唯一文件数（不同哈希值的数量）
            - duplicate_groups: 重复组数量
            - duplicates: 重复组列表，每项为 {"hash": 哈希值, "files": [路径列表]}
            - output_csv: 输出 CSV 路径
    """
    import pandas as pd

    # 扫描音频文件
    files = scan_audio_files(input_dir)
    tracker = ProgressTracker(len(files), progress_callback)

    # 哈希值到文件路径列表的映射
    hash_map = {}

    for file_path in files:
        try:
            file_hash = compute_file_hash(file_path)
            if file_hash not in hash_map:
                hash_map[file_hash] = []
            hash_map[file_hash].append(file_path)
        except Exception:
            pass
        tracker.update(os.path.basename(file_path))

    # 找出重复组（哈希值对应多个文件）
    duplicates = [
        {"hash": h, "files": paths}
        for h, paths in hash_map.items()
        if len(paths) > 1
    ]

    # 构建 CSV 记录，每个重复文件占一行
    csv_columns = ["hash", "group_id", "file_index", "filename", "file_path", "duplicate_count"]
    records = []
    for group_id, dup in enumerate(duplicates, 1):
        for file_index, path in enumerate(dup["files"], 1):
            records.append({
                "hash": dup["hash"],
                "group_id": group_id,
                "file_index": file_index,
                "filename": os.path.basename(path),
                "file_path": path,
                "duplicate_count": len(dup["files"]),
            })

    # 导出 CSV 报告（即使无重复也写入表头）
    output_csv = os.path.join(input_dir, "duplicate_report.csv")
    df = pd.DataFrame(records, columns=csv_columns)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return {
        "total_files": len(files),
        "unique_files": len(hash_map),
        "duplicate_groups": len(duplicates),
        "duplicates": duplicates,
        "output_csv": output_csv,
    }


if __name__ == "__main__":
    # 命令行调试入口
    import argparse

    parser = argparse.ArgumentParser(description="检测重复音频文件")
    parser.add_argument("input_dir", help="音频文件夹路径")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"文件夹不存在: {args.input_dir}")
        sys.exit(1)

    result = detect_duplicates(args.input_dir)
    print("=" * 50)
    print("重复文件检测结果")
    print("=" * 50)
    print(f"总文件数: {result['total_files']}")
    print(f"唯一文件数: {result['unique_files']}")
    print(f"重复组数: {result['duplicate_groups']}")
    print(f"输出 CSV: {result['output_csv']}")
    if result["duplicates"]:
        print("\n重复文件详情:")
        for dup in result["duplicates"]:
            print(f"  哈希: {dup['hash']}")
            for path in dup["files"]:
                print(f"    - {path}")
    else:
        print("\n未发现重复文件")
