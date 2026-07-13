"""
音乐风格检测模块（实验性）
提取多种声学特征（MFCC、频谱特征、过零率、节拍）作为风格分类的输入。
当前版本为框架搭建，只做特征提取，不进行实际分类。
后期整合开源模型后，将由 extract_genre_features 的输出驱动分类。

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
import pandas as pd


def extract_genre_features(file_path):
    """提取用于风格分类的声学特征（实验性）

    提取多种特征作为风格分类的输入:
        - MFCC 均值与方差（13 + 13 = 26 维）
        - 频谱质心（均值）
        - 频谱带宽（均值）
        - 频谱滚降（均值）
        - 过零率（均值）
        - 节拍速度 BPM

    特征提取步骤:
        1. librosa.load(sr=22050) 统一采样率加载
        2. MFCC: librosa.feature.mfcc，求均值和方差（13+13=26 维）
        3. 频谱质心: librosa.feature.spectral_centroid，求均值
        4. 频谱带宽: librosa.feature.spectral_bandwidth，求均值
        5. 频谱滚降: librosa.feature.spectral_rolloff，求均值
        6. 过零率: librosa.feature.zero_crossing_rate，求均值
        7. 节拍: librosa.beat.beat_track 估计 BPM
        8. 合并为特征向量

    参数:
        file_path: 音频文件完整路径

    返回:
        dict: {"filename": 文件名,
               "features": {特征名: 值},
               "feature_vector": [数值列表]}

    异常:
        处理失败时抛出对应异常，由批量函数捕获记录。
    """
    # 1. 统一采样率 22050 Hz 加载音频
    y, sr = librosa.load(file_path, sr=22050)

    # 2. MFCC 特征（13 维），计算均值和方差
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)   # shape: (13,)
    mfcc_var = np.var(mfcc, axis=1)     # shape: (13,)

    # 3. 频谱质心：描述声音的"亮度"，值越大高频成分越多
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = float(np.mean(centroid))

    # 4. 频谱带宽：描述频谱能量分布的集中程度
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = float(np.mean(bandwidth))

    # 5. 频谱滚降：85%（默认）能量所在频率，反映高频边界
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = float(np.mean(rolloff))

    # 6. 过零率：信号穿过零点的频率，与声音的清浊度相关
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = float(np.mean(zcr))

    # 7. 节拍速度：用 beat_track 估计 BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # 不同版本 librosa 返回类型可能不同，统一转成 Python 浮点数
    if isinstance(tempo, np.ndarray):
        tempo_val = float(tempo.item()) if tempo.size == 1 else float(tempo.flat[0])
    else:
        tempo_val = float(tempo)

    # 8. 组装特征字典
    features = {}
    for i in range(13):
        features[f"mfcc_{i}_mean"] = float(mfcc_mean[i])
        features[f"mfcc_{i}_var"] = float(mfcc_var[i])
    features["spectral_centroid"] = centroid_mean
    features["spectral_bandwidth"] = bandwidth_mean
    features["spectral_rolloff"] = rolloff_mean
    features["zero_crossing_rate"] = zcr_mean
    features["tempo"] = tempo_val

    # 合并为有序特征向量，方便后续模型输入
    feature_vector = [
        *mfcc_mean.tolist(),
        *mfcc_var.tolist(),
        centroid_mean,
        bandwidth_mean,
        rolloff_mean,
        zcr_mean,
        tempo_val,
    ]

    return {
        "filename": os.path.basename(file_path),
        "features": features,
        "feature_vector": feature_vector,
    }


def detect_genre(file_path):
    """检测单个文件的音乐风格（实验性 - 当前为占位实现）

    当前版本只提取特征，不进行分类。
    后期整合开源模型后，此函数将返回风格分类结果。

    参数:
        file_path: 音频文件完整路径

    返回:
        dict: {"filename": 文件名,
               "genre": "待模型支持",
               "confidence": 0.0,
               "features": {特征名: 值}}

    异常:
        处理失败时抛出对应异常，由批量函数捕获记录。
    """
    # 调用特征提取函数
    feat_result = extract_genre_features(file_path)

    # 当前为占位实现：genre 字段返回固定占位文本，置信度为 0
    return {
        "filename": feat_result["filename"],
        "genre": "待模型支持",
        "confidence": 0.0,
        "features": feat_result["features"],
    }


def detect_genre_batch(input_dir, output_csv, progress_callback=None):
    """批量提取风格特征，导出 CSV（实验性）

    扫描输入文件夹内所有音频文件，逐一调用 detect_genre 提取声学特征，
    将结果收集后用 pandas 导出为 utf-8-sig 编码的 CSV（带 BOM，
    方便 Excel 正确识别中文）。

    CSV 列: filename, genre, confidence, 以及各特征列。
    其中 genre 列固定为 "待模型支持"，confidence 列固定为 0.0，
    待后期整合分类模型后替换为实际预测值。

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
            result = detect_genre(file_path)
            # 将特征字典铺平为 CSV 列
            record = {
                "filename": result["filename"],
                "genre": result["genre"],
                "confidence": result["confidence"],
            }
            record.update(result["features"])
            records.append(record)
            tracker.update(f"已提取特征: {filename}")
        except Exception as e:
            # 捕获异常并记录，不中断后续文件处理
            errors.append({"file": file_path, "error": str(e)})
            tracker.update(f"提取失败: {filename} - {e}")

    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_csv))
    ensure_output_dir(output_dir)

    # 构建 DataFrame 并导出 CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return {
        "total_files": total,
        "output_csv": output_csv,
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口：
    #   python genre_detect.py <音频文件路径>            单文件特征提取
    #   python genre_detect.py <输入文件夹> <输出CSV>     批量提取
    if len(sys.argv) < 2:
        print("用法: python genre_detect.py <音频文件路径>")
        print("      python genre_detect.py <输入文件夹> <输出CSV路径>")
        sys.exit(1)

    src = sys.argv[1]

    if os.path.isfile(src):
        # 单文件调试：提取特征并打印
        print(f"正在提取风格特征: {src}")
        result = detect_genre(src)
        print(f"文件名:   {result['filename']}")
        print(f"风格:     {result['genre']}")
        print(f"置信度:   {result['confidence']}")
        print(f"特征详情 ({len(result['features'])} 项):")
        for key, val in result["features"].items():
            print(f"  {key}: {val:.4f}")
    else:
        # 文件夹批量调试，打印进度
        out_csv = sys.argv[2] if len(sys.argv) > 2 else "genre_features.csv"

        def cb(current, total, message):
            print(f"[{current}/{total}] {message}")

        result = detect_genre_batch(src, out_csv, progress_callback=cb)
        print(f"\n批量提取完成:")
        print(f"  总文件数: {result['total_files']}")
        print(f"  输出 CSV: {result['output_csv']}")
        print(f"  失败数:   {len(result['errors'])}")
        for err in result["errors"]:
            print(f"    - {err['file']}: {err['error']}")
