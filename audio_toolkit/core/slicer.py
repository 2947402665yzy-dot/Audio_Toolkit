"""
音频切割模块
将单个音频文件按指定时长切片，导出为 wav 等格式。
重构自原始脚本 Slice_audio.py（librosa 解码 + pydub 组装 + 时间轴切片）。
"""

import os
import sys

# 将上级目录（audio_toolkit）加入搜索路径，以便导入共享工具 utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
from pydub import AudioSegment

from utils import scan_audio_files, ensure_output_dir, ProgressTracker, safe_filename


def slice_audio(file_path, output_dir, slice_length_sec=30, output_format="wav"):
    """
    处理单个音频文件，按 slice_length_sec 切片并导出。

    处理流程：
        1. 用 librosa.load 加载音频（sr=None 保持原始采样率）
        2. 将浮点波形转为 16-bit PCM，用 pydub.AudioSegment 组装
        3. 按 slice_length_sec 切片
        4. 导出为 output_format

    参数:
        file_path: 输入音频文件路径
        output_dir: 切片输出目录（不存在会自动创建）
        slice_length_sec: 每个切片的时长（秒），默认 30
        output_format: 输出格式（如 "wav"、"mp3"），默认 "wav"

    返回:
        dict: {"slices": [切片文件名列表], "count": 切片数}
    """
    # 确保输出目录存在
    ensure_output_dir(output_dir)

    # 原始文件名（不含扩展名），用于生成切片文件名，并清理非法字符
    base_name = safe_filename(os.path.splitext(os.path.basename(file_path))[0])

    # 用 librosa 加载音频，sr=None 保持原始采样率（不降频，确保音质）
    # 原始脚本借助 librosa 绕过 pydub 直接读 MP3 时对 ffprobe 的依赖
    y, sr = librosa.load(file_path, sr=None)

    # 将浮点数波形数据转换为 16-bit PCM 整数（音频 CD 标准格式）
    y_int16 = (y * 32767).astype(np.int16)

    # 把整数阵列塞进 pydub，在内存里组装出无损音频流（单声道，16-bit）
    song = AudioSegment(
        y_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,   # 2 字节 = 16 bit
        channels=1,       # librosa 默认读取为单声道
    )

    slice_length_ms = int(slice_length_sec * 1000)
    total_length_ms = len(song)

    slice_names = []
    start_time = 0
    counter = 1

    # 时间轴扫描切片
    while start_time < total_length_ms:
        end_time = start_time + slice_length_ms
        if end_time > total_length_ms:
            end_time = total_length_ms

        chunk = song[start_time:end_time]
        # 文件名格式: {原文件名}_slice_{序号:03d}.{format}
        chunk_name = f"{base_name}_slice_{counter:03d}.{output_format}"
        chunk_path = os.path.join(output_dir, chunk_name)

        chunk.export(chunk_path, format=output_format)
        slice_names.append(chunk_name)

        start_time += slice_length_ms
        counter += 1

    return {"slices": slice_names, "count": len(slice_names)}


def slice_audio_batch(input_dir, output_dir, slice_length_sec=30, output_format="wav",
                      progress_callback=None):
    """
    批量处理文件夹内所有音频文件。

    每个文件的切片存放在以原文件名命名的子文件夹中，避免不同文件的切片互相覆盖。

    参数:
        input_dir: 输入文件夹
        output_dir: 输出根目录
        slice_length_sec: 每个切片的时长（秒），默认 30
        output_format: 输出格式，默认 "wav"
        progress_callback: 进度回调函数，签名为 callback(current, total, message)

    返回:
        dict: {"total_files": N, "total_slices": N, "errors": []}
    """
    # 用共享工具扫描所有音频文件
    files = scan_audio_files(input_dir)
    tracker = ProgressTracker(len(files), progress_callback)

    ensure_output_dir(output_dir)

    total_slices = 0
    errors = []

    for file_path in files:
        filename = os.path.basename(file_path)
        base_name = safe_filename(os.path.splitext(filename)[0])

        # 每个文件单独创建子文件夹存放切片
        sub_dir = os.path.join(output_dir, base_name)
        try:
            result = slice_audio(
                file_path,
                sub_dir,
                slice_length_sec=slice_length_sec,
                output_format=output_format,
            )
            total_slices += result["count"]
            tracker.update(f"已切片: {filename} ({result['count']} 段)")
        except Exception as e:
            errors.append({"file": filename, "error": str(e)})
            tracker.update(f"切片失败: {filename} - {e}")

    return {
        "total_files": len(files),
        "total_slices": total_slices,
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口: python slicer.py <音频文件路径> [切片时长秒]
    if len(sys.argv) < 2:
        print("用法: python slicer.py <音频文件路径> [切片时长秒]")
        sys.exit(1)

    audio_path = sys.argv[1]
    length = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    out_dir = "./slices_output"

    print(f"正在切片: {audio_path} (每段 {length} 秒) -> {out_dir}")
    result = slice_audio(audio_path, out_dir, slice_length_sec=length)
    print(f"完成！共切出 {result['count']} 段:")
    for name in result["slices"]:
        print(f"  - {name}")
