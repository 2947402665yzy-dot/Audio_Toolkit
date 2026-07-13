"""
BPM 检测模块
基于 librosa.beat.beat_track 检测音频 tempo（BPM）。
纯逻辑模块，不包含任何 GUI 代码。

注意：部分旧版 librosa 依赖 scipy.signal.hann，而新版 scipy 已将其移至
scipy.signal.windows.hann，这里做兼容性修复，避免 ImportError。
"""

import sys
import os

# 将上级目录加入搜索路径，以便导入共享工具模块 utils.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scan_audio_files, ensure_output_dir, ProgressTracker, format_time, safe_filename

# scipy.signal.hann 兼容性修复：新版 scipy 把 hann 移到 windows 子模块
try:
    import scipy.signal
    import scipy.signal.windows
    if not hasattr(scipy.signal, "hann"):
        scipy.signal.hann = scipy.signal.windows.hann
except Exception:
    # scipy 未安装或异常时不阻断导入，实际调用 librosa 时再抛错
    pass

import librosa
import numpy as np


def detect_bpm(file_path):
    """
    检测单个文件的 BPM（每分钟节拍数）。

    用 librosa.load 以原始采样率加载音频，再用 librosa.beat.beat_track
    估计 tempo。librosa 0.10+ 中 beat_track 返回 (tempo, frames) 元组。

    参数:
        file_path: 音频文件完整路径

    返回:
        dict: {"filename": 文件名, "bpm": BPM 浮点值, "tempo": tempo 浮点值}
              bpm 与 tempo 含义相同，同时返回以兼容不同版本叫法。

    异常:
        处理失败时抛出对应异常，由批量函数捕获记录。
    """
    # 加载音频，sr=None 保留原始采样率，保证节拍检测精度
    y, sr = librosa.load(file_path, sr=None)

    # 检测节拍，返回 tempo（标量或数组）和 beat 帧
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # 不同版本 librosa 返回类型可能不同，统一转成 Python 浮点数
    # 新版可能返回 numpy 数组（shape=(1,)），取第一个元素
    if isinstance(tempo, np.ndarray):
        tempo_val = float(tempo.item()) if tempo.size == 1 else float(tempo.flat[0])
    else:
        tempo_val = float(tempo)

    return {
        "filename": os.path.basename(file_path),
        "bpm": tempo_val,
        "tempo": tempo_val,
    }


def detect_bpm_batch(input_dir, output_csv, progress_callback=None):
    """
    批量检测文件夹内所有音频文件的 BPM，并导出 CSV。

    用 pandas 收集检测结果，导出为 utf-8-sig 编码的 CSV（带 BOM，
    方便 Excel 正确识别中文）。CSV 包含列：filename, bpm。

    参数:
        input_dir: 输入文件夹路径
        output_csv: 输出 CSV 文件路径
        progress_callback: 进度回调函数，签名为 callback(current, total, message)，可为 None

    返回:
        dict: {"total_files": 处理文件总数, "output_csv": CSV 路径, "errors": [失败文件信息列表]}
              errors 中每项为 {"file": 文件路径, "error": 错误描述}
    """
    import pandas as pd

    # 扫描输入目录下所有音频文件
    audio_files = scan_audio_files(input_dir)
    total = len(audio_files)

    # 初始化进度跟踪器
    tracker = ProgressTracker(total, progress_callback)

    records = []
    errors = []

    for file_path in audio_files:
        filename = os.path.basename(file_path)
        try:
            result = detect_bpm(file_path)
            records.append({
                "filename": result["filename"],
                "bpm": result["bpm"],
            })
            # 上报进度，附带检测到的 BPM
            tracker.update(f"已检测: {filename} (BPM={result['bpm']:.1f})")
        except Exception as e:
            # 捕获异常并记录，不中断后续文件处理
            errors.append({"file": file_path, "error": str(e)})
            tracker.update(f"失败: {filename} ({e})")

    # 用 pandas 构建 DataFrame 并导出 CSV
    df = pd.DataFrame(records, columns=["filename", "bpm"])

    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_csv))
    if output_dir:
        ensure_output_dir(output_dir)

    # 导出 utf-8-sig CSV（带 BOM，Excel 打开中文不乱码）
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return {
        "total_files": total,
        "output_csv": output_csv,
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口：python bpm_detect.py <输入文件或文件夹> [输出CSV路径]
    import sys

    if len(sys.argv) < 2:
        print("用法: python bpm_detect.py <输入文件或文件夹> [输出CSV路径]")
        print("示例: python bpm_detect.py ./input bpm_result.csv")
        sys.exit(1)

    src = sys.argv[1]

    if os.path.isfile(src):
        # 单文件调试
        result = detect_bpm(src)
        print("单文件 BPM 检测结果:", result)
    else:
        # 文件夹批量调试，打印进度
        out_csv = sys.argv[2] if len(sys.argv) > 2 else "bpm_result.csv"

        def cb(current, total, message):
            print(f"[{current}/{total}] {message}")

        result = detect_bpm_batch(src, out_csv, progress_callback=cb)
        print("批量 BPM 检测结果:", result)
