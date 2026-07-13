"""
调性检测模块（实验性）
使用 librosa 提取 chroma（色度）特征，配合 Krumhansl-Schmuckler
调性轮廓匹配算法推断音频的调性（大调 / 小调）。

纯逻辑模块，不包含任何 GUI / tkinter 代码，print 仅用于 __main__ 命令行调试。

注意：部分旧版 librosa 依赖 scipy.signal.hann，而新版 scipy 已将其移至
scipy.signal.windows.hann，这里做兼容性修复，避免 ImportError。
"""

import os
import sys

# 将上级目录（audio_toolkit）加入搜索路径，以便导入共享工具模块 utils.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scan_audio_files, ensure_output_dir, ProgressTracker

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

# Krumhansl-Schmuckler 调性轮廓模板（12 个半音的能量分布权重）
# 大调模板：主音(I)和属音(V)权重最高，体现了大调音阶的特征
MAJOR_TEMPLATE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
# 小调模板：主音(i)和属音(v)权重较高，同时小调的特征音级也有体现
MINOR_TEMPLATE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# 音名映射：索引 0=C, 1=C#, 2=D, ..., 11=B
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _safe_corrcoef(a, b):
    """安全计算两个向量的皮尔逊相关系数（实验性）。

    当某一方方差为零（例如 chroma 向量全相同）时，
    np.corrcoef 会返回 nan，此处将其降级为 0.0，保证算法不中断。

    参数:
        a: 一维 numpy 数组
        b: 一维 numpy 数组（长度需与 a 相同）

    返回:
        float: 相关系数，范围 [-1, 1]，异常时返回 0.0
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # 任一方标准差为零时无法计算有意义的相关系数
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def detect_key(file_path):
    """检测单个文件的调性（实验性）

    使用 librosa 提取 chroma（色度）特征，然后用 Krumhansl-Schmuckler
    调性轮廓匹配算法推断调性。

    算法步骤:
        1. librosa.load(sr=None) 以原始采样率加载音频
        2. librosa.feature.chroma_cqt 提取色度特征（12 维 × 帧数）
        3. 对 12 个半音求时间轴平均，得到 12 维 chroma 向量
        4. 定义 Krumhansl-Schmuckler 大调和小调模板
        5. 对每个可能的调性（12 大调 + 12 小调 = 24 种），
           将 chroma 向量旋转匹配，计算与模板的皮尔逊相关系数
        6. 取相关系数最高的作为调性结果
        7. 音名映射: 0=C, 1=C#, 2=D, ..., 11=B
        8. confidence = 最高相关系数
        9. alternatives = 排序后前 3 个候选

    参数:
        file_path: 音频文件完整路径

    返回:
        dict: {"filename": 文件名,
               "key": "C major" 或 "A minor" 等调性名称,
               "confidence": 0.0-1.0 的置信度,
               "alternatives": ["G major", ...] 前 3 个候选调性}

    异常:
        处理失败时抛出对应异常，由批量函数捕获记录。
    """
    # 1. 以原始采样率加载音频，保证音高信息不被降采样破坏
    y, sr = librosa.load(file_path, sr=None)

    # 2. 提取色度（chroma）特征，基于常数 Q 变换，对音高更鲁棒
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # 3. 沿时间轴求均值，把整首歌压成 12 维半音能量分布向量
    chroma_vector = np.mean(chroma, axis=1)  # shape: (12,)

    # 4-5. 对 24 种调性逐一计算相关系数
    scores = []  # 每项: (调性名称, 相关系数)
    for tonic in range(12):
        # 旋转 chroma 向量，使当前主音对齐到索引 0 位置
        # np.roll(arr, -tonic) 将索引 tonic 的元素移到索引 0
        rotated = np.roll(chroma_vector, -tonic)

        # 与大调模板计算相关系数 -> "X major"
        major_corr = _safe_corrcoef(rotated, MAJOR_TEMPLATE)
        scores.append((f"{NOTE_NAMES[tonic]} major", major_corr))

        # 与小调模板计算相关系数 -> "X minor"
        minor_corr = _safe_corrcoef(rotated, MINOR_TEMPLATE)
        scores.append((f"{NOTE_NAMES[tonic]} minor", minor_corr))

    # 6. 按相关系数从高到低排序
    scores.sort(key=lambda x: x[1], reverse=True)

    # 8. 最高相关系数作为置信度，并限制在 [0, 1] 范围内
    best_key, best_score = scores[0]
    confidence = max(0.0, min(1.0, best_score))

    # 9. 取第 2~4 名作为备选调性
    alternatives = [s[0] for s in scores[1:4]]

    return {
        "filename": os.path.basename(file_path),
        "key": best_key,
        "confidence": confidence,
        "alternatives": alternatives,
    }


def detect_key_batch(input_dir, output_csv, progress_callback=None):
    """批量检测调性，导出 CSV（实验性）

    扫描输入文件夹内所有音频文件，逐一调用 detect_key 进行调性检测，
    将结果收集后用 pandas 导出为 utf-8-sig 编码的 CSV（带 BOM，
    方便 Excel 正确识别中文）。

    CSV 列: filename, key, confidence, alternatives

    参数:
        input_dir: 输入文件夹路径
        output_csv: 输出 CSV 文件路径
        progress_callback: 进度回调函数，签名为
            callback(current, total, message)，可为 None

    返回:
        dict: {"total_files": 处理文件总数,
               "output_csv": CSV 路径,
               "errors": [失败文件信息列表]}
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
            result = detect_key(file_path)
            records.append({
                "filename": result["filename"],
                "key": result["key"],
                "confidence": round(result["confidence"], 4),
                # 备选调性用分号拼接，便于 CSV 单列存储
                "alternatives": "; ".join(result["alternatives"]),
            })
            tracker.update(
                f"已检测: {filename} -> {result['key']} "
                f"(置信度={result['confidence']:.2f})"
            )
        except Exception as e:
            # 捕获异常并记录，不中断后续文件处理
            errors.append({"file": file_path, "error": str(e)})
            tracker.update(f"检测失败: {filename} - {e}")

    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_csv))
    ensure_output_dir(output_dir)

    # 构建 DataFrame 并导出 CSV
    df = pd.DataFrame(records, columns=["filename", "key", "confidence", "alternatives"])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return {
        "total_files": total,
        "output_csv": output_csv,
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口：
    #   python key_detect.py <音频文件路径>            单文件检测
    #   python key_detect.py <输入文件夹> <输出CSV>     批量检测
    if len(sys.argv) < 2:
        print("用法: python key_detect.py <音频文件路径>")
        print("      python key_detect.py <输入文件夹> <输出CSV路径>")
        sys.exit(1)

    src = sys.argv[1]

    if os.path.isfile(src):
        # 单文件调试
        print(f"正在检测调性: {src}")
        result = detect_key(src)
        print(f"文件名:    {result['filename']}")
        print(f"调性:      {result['key']}")
        print(f"置信度:    {result['confidence']:.4f}")
        print(f"备选调性:  {result['alternatives']}")
    else:
        # 文件夹批量调试，打印进度
        out_csv = sys.argv[2] if len(sys.argv) > 2 else "key_result.csv"

        def cb(current, total, message):
            print(f"[{current}/{total}] {message}")

        result = detect_key_batch(src, out_csv, progress_callback=cb)
        print(f"\n批量检测完成:")
        print(f"  总文件数: {result['total_files']}")
        print(f"  输出 CSV: {result['output_csv']}")
        print(f"  失败数:   {len(result['errors'])}")
        for err in result["errors"]:
            print(f"    - {err['file']}: {err['error']}")
