"""
Mel Spectrogram 导出模块
基于 librosa 计算 Mel 频谱图，使用 matplotlib（Agg 后端）渲染为 PNG 图片。
纯逻辑模块，不包含任何 GUI 代码。

注意：matplotlib 必须在导入 pyplot 之前切换到 Agg 后端，避免依赖 GUI 显示环境。
"""

import sys
import os

# 将上级目录加入搜索路径，以便导入共享工具模块 utils.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scan_audio_files, ensure_output_dir, ProgressTracker, format_time, safe_filename

# 必须在导入 pyplot 前切换 Agg 后端，否则在无显示环境的服务器上会报错
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import librosa
import librosa.display
import numpy as np

# 设置中文字体，避免图表中文标签显示为方块
# 优先使用常见的中文字体，找不到则使用默认字体（不报错）
for _font in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei"]:
    try:
        from matplotlib.font_manager import FontProperties
        fp = FontProperties(family=_font)
        # 能找到该字体就采用
        if fp.get_name() != _font and not any(_font.lower() in f.name.lower() for f in matplotlib.font_manager.fontManager.ttflist):
            continue
        plt.rcParams["font.sans-serif"] = [_font] + plt.rcParams.get("font.sans-serif", [])
        break
    except Exception:
        continue
# 负号正常显示
plt.rcParams["axes.unicode_minus"] = False


def export_mel_spectrogram(file_path, output_dir, n_mels=128, fig_size=(10, 4), sr=None):
    """
    导出单个音频的 Mel Spectrogram 图片。

    用 librosa 加载音频，计算 n_mels 个 Mel 频带的功率谱并转为分贝（dB）刻度，
    再用 matplotlib 绘制热力图保存为 PNG。输出文件名格式为 {原文件名}_melspec.png。

    参数:
        file_path: 输入音频文件完整路径
        output_dir: 输出目录路径（不存在会自动创建）
        n_mels: Mel 频带数量，默认 128
        fig_size: 图片尺寸（宽, 高），单位英寸，默认 (10, 4)
        sr: 加载时重采样的目标采样率，为 None 时使用音频原始采样率，默认 None

    返回:
        dict: {"input": 原文件路径, "output": 输出图片路径, "n_mels": Mel 频带数}

    异常:
        处理失败时抛出对应异常，由批量函数捕获记录。
    """
    # 确保输出目录存在
    ensure_output_dir(output_dir)

    # 加载音频，sr=None 表示保留原始采样率
    y, sr_actual = librosa.load(file_path, sr=sr)

    # 计算 Mel 频谱图（功率谱）
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr_actual, n_mels=n_mels)

    # 将功率谱转换为分贝刻度，便于可视化观察动态范围
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 创建画布并绘制热力图
    fig, ax = plt.subplots(figsize=fig_size)
    img = librosa.display.specshow(
        mel_spec_db,
        x_axis="time",
        y_axis="mel",
        sr=sr_actual,
        ax=ax,
    )
    ax.set_title(f"Mel Spectrogram - {os.path.basename(file_path)}")
    ax.set_xlabel("时间")
    ax.set_ylabel("Mel 频率")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # 紧凑布局，避免标签被裁切
    fig.tight_layout()

    # 构造输出文件名：{原文件名}_melspec.png
    original_name = os.path.splitext(os.path.basename(file_path))[0]
    safe_name = safe_filename(original_name)
    output_filename = f"{safe_name}_melspec.png"
    output_path = os.path.join(output_dir, output_filename)

    # 保存为 PNG，dpi=150 平衡清晰度与文件大小
    fig.savefig(output_path, dpi=150, format="png")
    # 关闭画图对象，释放内存，避免批量处理时内存泄漏
    plt.close(fig)

    return {
        "input": file_path,
        "output": output_path,
        "n_mels": n_mels,
    }


def export_mel_spectrogram_batch(input_dir, output_dir, n_mels=128, fig_size=(10, 4), progress_callback=None):
    """
    批量导出文件夹内所有音频的 Mel Spectrogram 图片。

    扫描输入目录下的音频文件，逐个调用 export_mel_spectrogram 生成图片，
    通过 ProgressTracker 上报进度，异常文件记录到 errors 列表。

    参数:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        n_mels: Mel 频带数量，默认 128
        fig_size: 图片尺寸（宽, 高），单位英寸，默认 (10, 4)
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
            result = export_mel_spectrogram(
                file_path,
                output_dir,
                n_mels=n_mels,
                fig_size=fig_size,
            )
            # 上报进度，附带输出图片名
            tracker.update(f"已导出: {os.path.basename(result['output'])}")
        except Exception as e:
            # 捕获异常并记录，不中断后续文件处理
            errors.append({"file": file_path, "error": str(e)})
            tracker.update(f"失败: {filename} ({e})")

    return {
        "total_files": total,
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口：python mel_spectrogram.py <输入文件或文件夹> <输出目录> [n_mels]
    import sys

    if len(sys.argv) < 3:
        print("用法: python mel_spectrogram.py <输入文件或文件夹> <输出目录> [n_mels]")
        print("示例: python mel_spectrogram.py ./input ./output 128")
        sys.exit(1)

    src = sys.argv[1]
    out = sys.argv[2]
    mels = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    if os.path.isfile(src):
        # 单文件调试
        result = export_mel_spectrogram(src, out, n_mels=mels)
        print("单文件导出结果:", result)
    else:
        # 文件夹批量调试，打印进度
        def cb(current, total, message):
            print(f"[{current}/{total}] {message}")

        result = export_mel_spectrogram_batch(src, out, n_mels=mels, progress_callback=cb)
        print("批量导出结果:", result)
