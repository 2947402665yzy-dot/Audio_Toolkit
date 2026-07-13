"""
MFCC 特征提取模块
提取音频的 MFCC（梅尔频率倒谱系数）特征，支持基础 13 维与详细 78 维两种模式。
重构自原始脚本 extract_mfcc.py（13 维均值）与 pipeline_pandas.py（78 维鲁棒特征），
合并为带 detailed 参数的统一函数。
"""

import os
import sys

# 将上级目录（audio_toolkit）加入搜索路径，以便导入共享工具 utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# scipy.signal.hann 兼容性修复（复制自原始脚本）
# 新版 scipy 将 hann 移至 scipy.signal.windows，此处做向后兼容补丁
import scipy.signal
import scipy.signal.windows
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

import librosa
import numpy as np
import pandas as pd

from utils import scan_audio_files, ensure_output_dir, ProgressTracker


def extract_mfcc(file_path, n_mfcc=13, detailed=False):
    """
    提取单个音频文件的 MFCC 特征。

    参数:
        file_path: 音频文件路径
        n_mfcc: MFCC 维度，默认 13
        detailed: 是否计算 78 维鲁棒特征。
            - False: 只计算 13 维 MFCC 均值
            - True:  计算 78 维鲁棒特征
              （13 MFCC × {Mean, MAD} + 13 Delta × {AbsMean, MAD} + 13 Delta2 × {AbsMean, MAD}）

    返回:
        dict:
            - detailed=False: {"filename": ..., "sr": ..., "mfcc_mean": [13 个值]}
            - detailed=True:  {"filename": ..., "features": {78 个键值对}}
    """
    filename = os.path.basename(file_path)

    # 高保真加载音频，sr=None 保持原始采样率（绝不加高切滤镜）
    y, sr = librosa.load(file_path, sr=None)

    if not detailed:
        # 基础模式：13 维 MFCC 均值（来自 extract_mfcc.py）
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # 沿时间轴取均值，把整首歌压成 13 维音色指纹
        mean_mfccs = np.mean(mfccs, axis=1)
        return {
            "filename": filename,
            "sr": sr,
            "mfcc_mean": mean_mfccs.tolist(),
        }

    # 详细模式：78 维鲁棒特征（来自 pipeline_pandas.py）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)    # 一阶差分（速度）
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)   # 二阶差分（加速度）

    features = {}

    for i in range(n_mfcc):
        # A. 原生音色流派：均值 + 平均绝对差 (MAD)
        raw_mean = np.mean(mfcc[i])
        features[f"MFCC_{i}_Mean"] = raw_mean
        features[f"MFCC_{i}_MAD"] = np.mean(np.abs(mfcc[i] - raw_mean))

        # B. 动态速度流派：绝对值均值 + MAD
        abs_mean_d1 = np.mean(np.abs(mfcc_delta[i]))
        features[f"Delta_{i}_AbsMean"] = abs_mean_d1
        features[f"Delta_{i}_MAD"] = np.mean(np.abs(np.abs(mfcc_delta[i]) - abs_mean_d1))

        # C. 瞬态加速度流派：绝对值均值 + MAD
        abs_mean_d2 = np.mean(np.abs(mfcc_delta2[i]))
        features[f"Delta2_{i}_AbsMean"] = abs_mean_d2
        features[f"Delta2_{i}_MAD"] = np.mean(np.abs(np.abs(mfcc_delta2[i]) - abs_mean_d2))

    return {
        "filename": filename,
        "features": features,
    }


def extract_mfcc_batch(input_dir, output_csv, n_mfcc=13, detailed=False, progress_callback=None):
    """
    批量提取文件夹内所有音频文件的 MFCC 特征，导出为 CSV。

    参数:
        input_dir: 输入文件夹
        output_csv: 输出 CSV 文件路径
        n_mfcc: MFCC 维度，默认 13
        detailed: 是否计算 78 维鲁棒特征，默认 False
        progress_callback: 进度回调函数，签名为 callback(current, total, message)

    返回:
        dict: {"total_files": N, "output_csv": 路径, "errors": []}
    """
    # 用共享工具扫描所有音频文件
    files = scan_audio_files(input_dir)
    tracker = ProgressTracker(len(files), progress_callback)

    # 确保输出目录存在
    ensure_output_dir(os.path.dirname(os.path.abspath(output_csv)))

    records = []
    errors = []

    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            result = extract_mfcc(file_path, n_mfcc=n_mfcc, detailed=detailed)
            if detailed:
                # 详细模式：filename + 78 维特征铺平成列
                record = {"filename": result["filename"]}
                record.update(result["features"])
            else:
                # 基础模式：filename + sr + 13 维均值
                record = {"filename": result["filename"], "sr": result["sr"]}
                for i, val in enumerate(result["mfcc_mean"]):
                    record[f"MFCC_{i}_Mean"] = val
            records.append(record)
            tracker.update(f"已提取: {filename}")
        except Exception as e:
            errors.append({"file": filename, "error": str(e)})
            tracker.update(f"提取失败: {filename} - {e}")

    # 用 pandas.DataFrame 收集所有记录，导出 CSV（utf-8-sig 编码，兼容 Excel 直接打开）
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return {
        "total_files": len(files),
        "output_csv": output_csv,
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口: python mfcc.py <音频文件路径> [--detailed]
    if len(sys.argv) < 2:
        print("用法: python mfcc.py <音频文件路径> [--detailed]")
        sys.exit(1)

    audio_path = sys.argv[1]
    use_detailed = "--detailed" in sys.argv

    print(f"正在提取 MFCC: {audio_path} (detailed={use_detailed})")
    result = extract_mfcc(audio_path, detailed=use_detailed)

    if use_detailed:
        print(f"文件名: {result['filename']}")
        print(f"特征维度: {len(result['features'])}")
        for key, val in result["features"].items():
            print(f"  {key}: {val:.4f}")
    else:
        print(f"文件名: {result['filename']}")
        print(f"采样率: {result['sr']} Hz")
        print(f"13 维 MFCC 均值: {[f'{x:.4f}' for x in result['mfcc_mean']]}")
