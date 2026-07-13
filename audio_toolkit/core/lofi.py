"""
Lo-fi 音频转换模块
对音频进行降质处理（降低采样率、比特深度），并可叠加高通/低通滤波器。
重构自原始脚本 lofi_maker.py，新增 scipy 滤波器（可同时开启高通与低通，形成带通效果）。
"""

import os
import sys

# 将上级目录（audio_toolkit）加入搜索路径，以便导入共享工具 utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, filtfilt

from utils import scan_audio_files, ensure_output_dir, ProgressTracker, safe_filename


# 比特深度字符串到 sample_width（字节数）的映射
_BIT_DEPTH_MAP = {
    "8-bit": 1,
    "16-bit": 2,
    "24-bit": 3,
}


def _segment_to_float(segment):
    """
    将 pydub AudioSegment 转为 float64 numpy 数组，支持多声道与 8/16/24/32 位。

    参数:
        segment: pydub.AudioSegment

    返回:
        np.ndarray: 形状为 [n_samples]（单声道）或 [n_samples, n_channels]（多声道）
    """
    sw = segment.sample_width
    channels = segment.channels
    raw = segment.raw_data

    if sw == 1:
        arr = np.frombuffer(raw, dtype=np.int8)
    elif sw == 2:
        arr = np.frombuffer(raw, dtype=np.int16)
    elif sw == 4:
        arr = np.frombuffer(raw, dtype=np.int32)
    elif sw == 3:
        # 24-bit 小端手动解析为 int32
        # （pydub 的 get_array_of_samples 不支持 3 字节，故用 raw_data 自行解析）
        bytes_arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        arr = (bytes_arr[:, 0].astype(np.int32)
               | (bytes_arr[:, 1].astype(np.int32) << 8)
               | (bytes_arr[:, 2].astype(np.int32) << 16))
        # 符号扩展：最高位为 1 时转为负数
        arr = np.where(arr & 0x800000, arr - 0x1000000, arr)
    else:
        arr = np.frombuffer(raw, dtype=np.int16)

    if channels > 1:
        arr = arr.reshape(-1, channels)

    return arr.astype(np.float64)


def _float_to_segment(samples, frame_rate, sample_width, channels):
    """
    将 float64 numpy 数组转回 pydub AudioSegment。

    参数:
        samples: float64 数组，形状 [n_samples] 或 [n_samples, channels]
        frame_rate: 采样率
        sample_width: 每采样字节数（1/2/3/4）
        channels: 声道数

    返回:
        AudioSegment: 由 numpy 数组重建的音频段
    """
    sw = sample_width
    max_val = float(2 ** (8 * sw - 1) - 1)
    samples = np.clip(np.round(samples), -max_val, max_val)

    if sw == 1:
        data = samples.astype(np.int8).tobytes()
    elif sw == 2:
        data = samples.astype(np.int16).tobytes()
    elif sw == 4:
        data = samples.astype(np.int32).tobytes()
    elif sw == 3:
        # int32 -> 24-bit 小端打包
        int32 = samples.astype(np.int32)
        b0 = (int32 & 0xFF).astype(np.uint8)
        b1 = ((int32 >> 8) & 0xFF).astype(np.uint8)
        b2 = ((int32 >> 16) & 0xFF).astype(np.uint8)
        data = np.stack([b0, b1, b2], axis=-1).tobytes()
    else:
        data = samples.astype(np.int16).tobytes()

    return AudioSegment(
        data,
        frame_rate=frame_rate,
        sample_width=sw,
        channels=channels,
    )


def _apply_butterworth(segment, cutoff, btype, order):
    """
    对 AudioSegment 应用 Butterworth 滤波器（高通或低通）。

    将 pydub 音频转为 numpy 数组，用 scipy.signal.filtfilt 做零相位滤波后转回 pydub。

    参数:
        segment: pydub.AudioSegment
        cutoff: 截止频率（Hz）
        btype: 滤波器类型，"high"（高通）或 "low"（低通）
        order: 滤波器阶数

    返回:
        AudioSegment: 滤波后的音频
    """
    nyq = 0.5 * segment.frame_rate
    normal_cutoff = cutoff / nyq
    # 截止频率需在 (0, 1) 范围内（归一化奈奎斯特），否则跳过滤波
    if normal_cutoff >= 1.0 or normal_cutoff <= 0.0:
        return segment

    b, a = butter(order, normal_cutoff, btype=btype, analog=False)

    samples = _segment_to_float(segment)  # [n] 或 [n, ch]
    channels = segment.channels

    if channels > 1:
        # 多声道：逐声道滤波
        filtered = np.empty_like(samples)
        for ch in range(channels):
            filtered[:, ch] = filtfilt(b, a, samples[:, ch])
    else:
        filtered = filtfilt(b, a, samples)

    return _float_to_segment(
        filtered,
        frame_rate=segment.frame_rate,
        sample_width=segment.sample_width,
        channels=channels,
    )


def lofi_convert(file_path, output_dir, target_sr=22050, bit_depth="8-bit", bitrate="32k",
                 hp_cutoff=None, lp_cutoff=None, filter_order=4, output_format="mp3"):
    """
    对单个音频文件进行 Lo-fi 降质转换，可叠加高通/低通滤波器。

    处理流程：
        1. 用 pydub 加载音频
        2. set_frame_rate 降低采样率
        3. （可选）高通滤波 + （可选）低通滤波（可同时开启，形成带通效果）
        4. set_sample_width 降低比特深度
        5. 导出为 output_format（mp3 时设置 bitrate）

    参数:
        file_path: 输入音频文件路径
        output_dir: 输出目录（不存在会自动创建）
        target_sr: 目标采样率（Hz），默认 22050
        bit_depth: 目标比特深度，"8-bit"/"16-bit"/"24-bit"，默认 "8-bit"
        bitrate: MP3 比特率（如 "32k"），默认 "32k"
        hp_cutoff: 高通滤波器截止频率（Hz），None 表示不启用，默认 None
        lp_cutoff: 低通滤波器截止频率（Hz），None 表示不启用，默认 None
            （高通与低通可同时开启，形成带通效果）
        filter_order: 滤波器阶数，默认 4
        output_format: 输出格式（如 "mp3"、"wav"），默认 "mp3"

    返回:
        dict: {"output": 输出文件路径}
    """
    ensure_output_dir(output_dir)

    # 用 pydub 加载音频
    sound = AudioSegment.from_file(file_path)

    # 降低采样率（滤波在目标采样率下进行，截止频率相对 target_sr 有意义）
    sound = sound.set_frame_rate(target_sr)

    # 高通滤波
    if hp_cutoff is not None:
        sound = _apply_butterworth(sound, hp_cutoff, "high", filter_order)

    # 低通滤波（可与高通同时开启，形成带通效果）
    if lp_cutoff is not None:
        sound = _apply_butterworth(sound, lp_cutoff, "low", filter_order)

    # 降低比特深度：8-bit=1, 16-bit=2, 24-bit=3
    target_width = _BIT_DEPTH_MAP.get(bit_depth, 1)
    sound = sound.set_sample_width(target_width)

    # 输出文件名: lofi_{原文件名}.{format}
    base_name = safe_filename(os.path.splitext(os.path.basename(file_path))[0])
    out_filename = f"lofi_{base_name}.{output_format}"
    out_path = os.path.join(output_dir, out_filename)

    # 导出（mp3 设置比特率）
    if output_format.lower() == "mp3":
        sound.export(out_path, format="mp3", bitrate=bitrate)
    else:
        sound.export(out_path, format=output_format)

    return {"output": out_path}


def lofi_convert_batch(input_dir, output_dir, target_sr=22050, bit_depth="8-bit", bitrate="32k",
                       hp_cutoff=None, lp_cutoff=None, filter_order=4, output_format="mp3",
                       progress_callback=None):
    """
    批量对文件夹内所有音频文件进行 Lo-fi 降质转换。

    参数:
        input_dir: 输入文件夹
        output_dir: 输出目录
        target_sr: 目标采样率（Hz），默认 22050
        bit_depth: 目标比特深度，默认 "8-bit"
        bitrate: MP3 比特率，默认 "32k"
        hp_cutoff: 高通截止频率（Hz），None 不启用，默认 None
        lp_cutoff: 低通截止频率（Hz），None 不启用，默认 None
        filter_order: 滤波器阶数，默认 4
        output_format: 输出格式，默认 "mp3"
        progress_callback: 进度回调函数，签名为 callback(current, total, message)

    返回:
        dict: {"total_files": N, "errors": []}
    """
    # 用共享工具扫描所有音频文件
    files = scan_audio_files(input_dir)
    tracker = ProgressTracker(len(files), progress_callback)

    ensure_output_dir(output_dir)

    errors = []

    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            result = lofi_convert(
                file_path,
                output_dir,
                target_sr=target_sr,
                bit_depth=bit_depth,
                bitrate=bitrate,
                hp_cutoff=hp_cutoff,
                lp_cutoff=lp_cutoff,
                filter_order=filter_order,
                output_format=output_format,
            )
            tracker.update(f"已转换: {filename} -> {os.path.basename(result['output'])}")
        except Exception as e:
            errors.append({"file": filename, "error": str(e)})
            tracker.update(f"转换失败: {filename} - {e}")

    return {
        "total_files": len(files),
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口: python lofi.py <音频文件路径> [目标采样率] [比特深度]
    if len(sys.argv) < 2:
        print("用法: python lofi.py <音频文件路径> [目标采样率] [比特深度]")
        print("示例: python lofi.py test.mp3 22050 8-bit")
        sys.exit(1)

    audio_path = sys.argv[1]
    sr = int(sys.argv[2]) if len(sys.argv) > 2 else 22050
    bd = sys.argv[3] if len(sys.argv) > 3 else "8-bit"
    out_dir = "./lofi_output"

    print(f"正在 Lo-fi 转换: {audio_path} (sr={sr}, {bd}) -> {out_dir}")
    result = lofi_convert(audio_path, out_dir, target_sr=sr, bit_depth=bd)
    print(f"完成！输出文件: {result['output']}")
