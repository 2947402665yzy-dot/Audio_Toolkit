"""
Metadata 提取模块

提供单个音频文件和批量音频文件的元数据提取功能。
依赖：mutagen（标签读取）、librosa（补充音频信息）、pandas（CSV 导出）
"""

import os
import sys

# 导入共享工具模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scan_audio_files, ensure_output_dir, ProgressTracker


def extract_metadata(file_path):
    """
    提取单个音频文件的元数据，返回字典。

    使用 mutagen 读取标签信息（标题、艺术家、专辑等）和音频信息
    （时长、比特率、采样率、声道数）。若 mutagen 无法获取时长或采样率，
    则使用 librosa 补充。

    参数:
        file_path (str): 音频文件的完整路径

    返回:
        dict: 包含以下键的元数据字典:
            - filename: 文件名
            - title: 标题
            - artist: 艺术家
            - album: 专辑
            - year: 年份
            - genre: 流派
            - duration_sec: 时长（秒）
            - bitrate: 比特率（bps）
            - sample_rate: 采样率（Hz）
            - channels: 声道数
            - file_size_kb: 文件大小（KB）
            - format: 文件格式（扩展名）
    """
    # 初始化元数据默认值
    metadata = {
        "filename": os.path.basename(file_path),
        "title": "",
        "artist": "",
        "album": "",
        "year": "",
        "genre": "",
        "duration_sec": 0.0,
        "bitrate": 0,
        "sample_rate": 0,
        "channels": 0,
        "file_size_kb": 0.0,
        "format": "",
    }

    # 文件大小和格式（不依赖第三方库）
    try:
        file_size = os.path.getsize(file_path)
        metadata["file_size_kb"] = round(file_size / 1024, 2)
    except OSError:
        pass
    metadata["format"] = os.path.splitext(file_path)[1].lower().lstrip(".")

    # 用 mutagen 加载并提取标签与音频信息
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(file_path, easy=True)
        if audio is not None:
            # 提取文本标签（easy 模式下键名为小写通用名）
            for tag_key, meta_key in [
                ("title", "title"),
                ("artist", "artist"),
                ("album", "album"),
                ("date", "year"),
                ("genre", "genre"),
            ]:
                value = audio.get(tag_key)
                if value:
                    metadata[meta_key] = str(value[0])

            # 提取音频技术信息
            if hasattr(audio, "info"):
                info = audio.info
                if hasattr(info, "length") and info.length:
                    metadata["duration_sec"] = round(float(info.length), 2)
                if hasattr(info, "bitrate") and info.bitrate:
                    metadata["bitrate"] = int(info.bitrate)
                if hasattr(info, "sample_rate") and info.sample_rate:
                    metadata["sample_rate"] = int(info.sample_rate)
                if hasattr(info, "channels") and info.channels:
                    metadata["channels"] = int(info.channels)
    except Exception:
        pass

    # 如果时长或采样率仍为 0，用 librosa 补充
    if metadata["duration_sec"] == 0 or metadata["sample_rate"] == 0:
        try:
            import librosa

            # mono=False 保留原始声道结构
            y, sr = librosa.load(file_path, sr=None, mono=False)
            if metadata["duration_sec"] == 0:
                metadata["duration_sec"] = round(librosa.get_duration(y=y, sr=sr), 2)
            if metadata["sample_rate"] == 0:
                metadata["sample_rate"] = int(sr)
            if metadata["channels"] == 0:
                metadata["channels"] = 1 if y.ndim == 1 else int(y.shape[0])
        except Exception:
            pass

    return metadata


def extract_metadata_batch(input_dir, output_csv, progress_callback=None):
    """
    批量提取音频元数据并导出 CSV 文件。

    扫描输入目录中的所有音频文件，逐个提取元数据，汇总后导出为
    UTF-8-SIG 编码的 CSV 文件（兼容 Excel 中文显示）。

    参数:
        input_dir (str): 输入音频文件夹路径
        output_csv (str): 输出 CSV 文件路径
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
            meta = extract_metadata(file_path)
            records.append(meta)
        except Exception as exc:
            errors.append({"file": file_path, "error": str(exc)})
        tracker.update(os.path.basename(file_path))

    # 导出 CSV
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

    parser = argparse.ArgumentParser(description="提取音频文件元数据")
    parser.add_argument("path", help="音频文件或文件夹路径")
    parser.add_argument("-o", "--output", default="metadata.csv", help="输出 CSV 路径（批量模式）")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        result = extract_metadata_batch(args.path, args.output)
        print(f"处理完成: {result['total_files']} 个文件")
        print(f"输出 CSV: {result['output_csv']}")
        if result["errors"]:
            print(f"错误数: {len(result['errors'])}")
            for err in result["errors"]:
                print(f"  - {err['file']}: {err['error']}")
    elif os.path.isfile(args.path):
        meta = extract_metadata(args.path)
        for key, value in meta.items():
            print(f"  {key}: {value}")
    else:
        print(f"路径不存在: {args.path}")
