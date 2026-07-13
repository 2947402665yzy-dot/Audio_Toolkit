"""
音频格式转换模块
基于 pydub 实现单个/批量音频格式转换，支持 wav/mp3/flac/aac/ogg 等常见格式。
纯逻辑模块，不包含任何 GUI 代码。
"""

import sys
import os

# 将上级目录加入搜索路径，以便导入共享工具模块 utils.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scan_audio_files, ensure_output_dir, ProgressTracker, format_time, safe_filename

from pydub import AudioSegment


def convert_format(file_path, output_dir, target_format="wav", target_sr=None):
    """
    转换单个文件的格式。

    使用 pydub.AudioSegment.from_file 自动识别输入格式并加载，
    可选重采样后导出为目标格式。文件名保持原名，仅替换扩展名。

    参数:
        file_path: 输入音频文件的完整路径
        output_dir: 输出目录路径（不存在会自动创建）
        target_format: 目标格式，如 "wav"/"mp3"/"flac"/"aac"/"ogg"，默认 "wav"
        target_sr: 目标采样率（Hz），为 None 时保持原采样率，默认 None

    返回:
        dict: {"input": 原文件路径, "output": 输出文件路径, "format": 目标格式}

    异常:
        处理失败时抛出对应异常，由批量函数捕获记录。
    """
    # 确保输出目录存在
    ensure_output_dir(output_dir)

    # pydub 会根据文件头自动识别格式，无需手动指定
    audio = AudioSegment.from_file(file_path)

    # 如果指定了目标采样率则重采样，否则保持原始采样率
    if target_sr is not None:
        audio = audio.set_frame_rate(int(target_sr))

    # 取原文件名（不含扩展名），拼接目标格式扩展名
    original_name = os.path.splitext(os.path.basename(file_path))[0]
    # 清理文件名中的非法字符，避免在不同操作系统下写出失败
    safe_name = safe_filename(original_name)
    target_format_lower = str(target_format).lower()
    output_filename = f"{safe_name}.{target_format_lower}"
    output_path = os.path.join(output_dir, output_filename)

    # 导出为目标格式；部分格式（如 mp3）可附加比特率等参数，这里使用默认值
    audio.export(output_path, format=target_format_lower)

    return {
        "input": file_path,
        "output": output_path,
        "format": target_format_lower,
    }


def convert_format_batch(input_dir, output_dir, target_format="wav", target_sr=None, progress_callback=None):
    """
    批量转换文件夹内所有音频文件的格式。

    扫描输入目录下的音频文件，逐个调用 convert_format 进行转换，
    通过 ProgressTracker 上报进度，异常文件记录到 errors 列表。

    参数:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        target_format: 目标格式，默认 "wav"
        target_sr: 目标采样率（Hz），为 None 时保持原采样率，默认 None
        progress_callback: 进度回调函数，签名为 callback(current, total, message)，可为 None

    返回:
        dict: {"total_files": 处理文件总数, "errors": [失败文件信息列表]}
              errors 中每项为 {"file": 文件路径, "error": 错误描述}
    """
    # 扫描输入目录下所有音频文件
    audio_files = scan_audio_files(input_dir)
    total = len(audio_files)

    # 初始化进度跟踪器
    tracker = ProgressTracker(total, progress_callback)

    errors = []

    for file_path in audio_files:
        filename = os.path.basename(file_path)
        try:
            result = convert_format(
                file_path,
                output_dir,
                target_format=target_format,
                target_sr=target_sr,
            )
            # 上报进度，附带输出文件名
            tracker.update(f"已转换: {os.path.basename(result['output'])}")
        except Exception as e:
            # 捕获异常并记录，不中断后续文件处理
            errors.append({"file": file_path, "error": str(e)})
            tracker.update(f"失败: {filename} ({e})")

    return {
        "total_files": total,
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口：python format_convert.py <输入文件或文件夹> <输出目录> [目标格式] [采样率]
    import sys

    if len(sys.argv) < 3:
        print("用法: python format_convert.py <输入文件或文件夹> <输出目录> [目标格式] [采样率]")
        print("示例: python format_convert.py ./input ./output wav 44100")
        sys.exit(1)

    src = sys.argv[1]
    out = sys.argv[2]
    fmt = sys.argv[3] if len(sys.argv) > 3 else "wav"
    sr = int(sys.argv[4]) if len(sys.argv) > 4 else None

    if os.path.isfile(src):
        # 单文件调试
        result = convert_format(src, out, target_format=fmt, target_sr=sr)
        print("单文件转换结果:", result)
    else:
        # 文件夹批量调试，打印进度
        def cb(current, total, message):
            print(f"[{current}/{total}] {message}")

        result = convert_format_batch(src, out, target_format=fmt, target_sr=sr, progress_callback=cb)
        print("批量转换结果:", result)
