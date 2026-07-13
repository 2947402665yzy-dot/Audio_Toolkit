"""
乐段识别与切分模块（实验性）
使用 librosa 提取 MFCC 特征，构建自相似矩阵，再通过层次聚类
（agglomerative clustering）识别音频的乐段边界，并支持按边界切分导出。

纯逻辑模块，不包含任何 GUI / tkinter 代码，print 仅用于 __main__ 命令行调试。

注意：部分旧版 librosa 依赖 scipy.signal.hann，而新版 scipy 已将其移至
scipy.signal.windows.hann，这里做兼容性修复，避免 ImportError。
"""

import os
import sys

# 将上级目录（audio_toolkit）加入搜索路径，以便导入共享工具模块 utils.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scan_audio_files, ensure_output_dir, ProgressTracker, safe_filename

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
from pydub import AudioSegment


def _cosine_similarity(a, b):
    """计算两个向量的余弦相似度（实验性）。

    用于比较不同乐段的 MFCC 质心，判断是否为相似段落（如重复的副歌）。

    参数:
        a: 一维 numpy 数组
        b: 一维 numpy 数组（长度需与 a 相同）

    返回:
        float: 余弦相似度，范围 [-1, 1]，零向量时返回 0.0
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _assign_segment_labels(centroids):
    """根据 MFCC 质心的余弦相似度为各段分配标签（实验性）。

    使用动态阈值：计算所有段两两之间的相似度，取中位数作为阈值。
    超过动态阈值的段落复用标签（表示重复段落，如两次副歌都标 "B"），
    否则分配新标签。

    参数:
        centroids: 各段 MFCC 质心列表，每个元素为一维 numpy 数组

    返回:
        list: 各段的标签 ID（整数，从 0 开始）
    """
    if not centroids:
        return []
    if len(centroids) == 1:
        return [0]

    n = len(centroids)

    # 计算所有段两两之间的余弦相似度
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_cosine_similarity(centroids[i], centroids[j]))

    if not sims:
        return [0] * n

    # 动态阈值：取所有相似度的上四分位数（即前25%最相似的视为同一段落）
    sims_arr = np.array(sims)
    threshold = np.percentile(sims_arr, 75)

    # 确保阈值不会太低（至少 0.5）
    threshold = max(threshold, 0.5)

    labels = [0] * n
    current_label = 0
    label_centroids = {0: centroids[0]}

    for i in range(1, n):
        best_sim = -1.0
        best_label = -1
        for lbl, centroid in label_centroids.items():
            sim = _cosine_similarity(centroids[i], centroid)
            if sim > best_sim:
                best_sim = sim
                best_label = lbl

        if best_sim >= threshold and best_label >= 0:
            labels[i] = best_label
            label_centroids[best_label] = (label_centroids[best_label] + centroids[i]) / 2
        else:
            current_label += 1
            labels[i] = current_label
            label_centroids[current_label] = centroids[i]

    return labels


def detect_segments(file_path, n_segments=None):
    """识别单个文件的乐段结构（实验性）

    使用 librosa 提取 MFCC 特征，构建自相似矩阵，通过新颖性曲线
    （novelty curve）自动检测乐段边界，再根据特征相似度合并相似段落。

    算法步骤:
        1. librosa.load(sr=22050) 加载（统一采样率便于分析）
        2. 提取 MFCC 特征: librosa.feature.mfcc
        3. 计算自相似矩阵: librosa.segment.recurrence_matrix
        4. 从自相似矩阵提取新颖性曲线，找到峰值作为边界
        5. 边界数量由音乐结构自动决定，无需手动指定段数
        6. 为每段分配标签（A, B, C...），根据特征相似度合并相同段
        7. 返回结构化的段落信息

    参数:
        file_path: 音频文件完整路径
        n_segments: 已废弃，保留参数仅为向后兼容。段数现在由算法自动检测。

    返回:
        dict: {"filename": 文件名,
               "segments": [{"label": "A", "start": 秒, "end": 秒, "duration": 秒}, ...],
               "boundaries": [秒列表]}

    异常:
        处理失败时抛出对应异常，由批量函数捕获记录。
    """
    # 1. 统一采样率 22050 Hz 加载
    y, sr = librosa.load(file_path, sr=22050)

    # 2. 提取 MFCC 特征（20 维）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # 特征归一化
    mfcc = librosa.util.normalize(mfcc, axis=1)

    n_frames = mfcc.shape[1]

    if n_frames < 10:
        # 音频太短，整首歌作为一个段落
        bound_times = [0.0, float(librosa.frames_to_time(n_frames, sr=sr))]
        segments = [{
            "label": "A",
            "start": 0.0,
            "end": bound_times[1],
            "duration": bound_times[1],
        }]
        return {
            "filename": os.path.basename(file_path),
            "segments": segments,
            "boundaries": bound_times,
        }

    # 3. 计算自相似矩阵
    # 使用 librosa 的 recurrence matrix，基于 MFCC 特征
    rec = librosa.segment.recurrence_matrix(mfcc, k=None, metric="cosine")

    # 4. 从自相似矩阵提取新颖性曲线
    # 沿对角线方向计算偏移矩阵的差异，峰值即为段落边界
    # lag matrix: 将自相似矩阵转换为时滞矩阵
    lag = librosa.segment.recurrence_to_lag(rec)

    # 计算新颖性曲线：时滞矩阵每行的和，检测边界
    novelty = np.diff(np.sum(lag, axis=0), prepend=0)
    # 取绝对值并平滑
    novelty = np.abs(novelty)

    # 5. 自动检测边界峰值
    # 使用相对阈值：取新颖性曲线的前 80% 分位数作为阈值
    if novelty.max() > 0:
        threshold = np.percentile(novelty[novelty > 0], 60) if np.any(novelty > 0) else 0
    else:
        threshold = 0

    # 找到超过阈值的峰值，用 scipy.signal.find_peaks
    from scipy.signal import find_peaks
    # 最小段长：约 5 秒（避免过度切分）
    min_seg_frames = int(5 * sr / 512)  # 512 是 librosa 默认 hop_length
    peaks, _ = find_peaks(novelty, height=threshold, distance=min_seg_frames)

    # 6. 构建边界帧列表
    bound_frames = [0] + list(peaks) + [n_frames]
    bound_frames = sorted(set(bound_frames))

    # 如果自动检测出的段数太少（只有1段），尝试用 fallback 方法
    if len(bound_frames) < 3:
        # Fallback: 用 MFCC 均值的突变点检测
        # 按固定窗口计算局部 MFCC 均值，检测突变
        window = max(min_seg_frames, 20)
        n_windows = max(n_frames // window, 2)
        local_means = []
        for i in range(n_windows):
            start_f = i * window
            end_f = min((i + 1) * window, n_frames)
            local_means.append(np.mean(mfcc[:, start_f:end_f], axis=1))
        local_means = np.array(local_means)

        # 计算相邻窗口的差异
        diffs = np.linalg.norm(np.diff(local_means, axis=0), axis=1)
        if len(diffs) > 0 and diffs.max() > 0:
            diff_threshold = np.percentile(diffs, 50)
            diff_peaks, _ = find_peaks(diffs, height=diff_threshold, distance=2)
            bound_frames = [0] + [p * window for p in diff_peaks] + [n_frames]
            bound_frames = sorted(set(bound_frames))

    # 7. 将边界帧索引转为时间（秒）
    bound_times = librosa.frames_to_time(bound_frames, sr=sr)

    # 8. 计算每段的 MFCC 质心，用于判断段落相似性
    seg_centroids = []
    for i in range(len(bound_frames) - 1):
        start_f, end_f = bound_frames[i], bound_frames[i + 1]
        if end_f > start_f:
            seg_centroids.append(np.mean(mfcc[:, start_f:end_f], axis=1))
        else:
            seg_centroids.append(np.zeros(mfcc.shape[0]))

    # 根据余弦相似度合并相似段，分配聚类 ID
    seg_cluster_ids = _assign_segment_labels(seg_centroids)

    # 9. 将聚类 ID 映射为字母标签（A, B, C, ...）
    unique_ids = sorted(set(seg_cluster_ids))
    id_to_letter = {cid: chr(ord("A") + idx) for idx, cid in enumerate(unique_ids)}

    # 10. 构建结构化的段落信息
    segments = []
    for i in range(len(bound_frames) - 1):
        start_time = float(bound_times[i])
        end_time = float(bound_times[i + 1])
        seg_label = id_to_letter[seg_cluster_ids[i]]
        segments.append({
            "label": seg_label,
            "start": start_time,
            "end": end_time,
            "duration": end_time - start_time,
        })

    return {
        "filename": os.path.basename(file_path),
        "segments": segments,
        "boundaries": [float(t) for t in bound_times],
    }


def segment_batch(input_dir, output_dir, n_segments=None, progress_callback=None):
    """批量识别并切分乐段，导出 CSV + 切片文件（实验性）

    对输入文件夹内的每个音频文件:
        1. 调用 detect_segments 获取段落信息（段数自动检测）
        2. 用 pydub 按段落边界切分音频，保存到子文件夹
        3. 切片文件名: {原文件名}_{标签}_{序号}.wav（如 song1_A_01.wav）
        4. 导出段落信息 CSV 到 output_dir/segment_info.csv

    参数:
        input_dir: 输入文件夹路径
        output_dir: 输出根目录路径
        n_segments: 已废弃，保留仅为向后兼容。段数由算法自动检测。
        progress_callback: 进度回调函数，签名为
            callback(current, total, message)，可为 None

    返回:
        dict: {"total_files": 处理文件总数,
               "errors": [失败文件信息列表]}
              errors 中每项为 {"file": 文件路径, "error": 错误描述}
    """
    import pandas as pd

    # 扫描输入目录下所有音频文件
    audio_files = scan_audio_files(input_dir)
    total = len(audio_files)

    # 初始化进度跟踪器
    tracker = ProgressTracker(total, progress_callback)

    # 确保输出根目录存在
    ensure_output_dir(output_dir)

    all_records = []
    errors = []

    for file_path in audio_files:
        filename = os.path.basename(file_path)
        # 清理文件名用于创建子文件夹和切片文件名
        base_name = safe_filename(os.path.splitext(filename)[0])

        try:
            # 1. 识别乐段结构
            result = detect_segments(file_path, n_segments=n_segments)

            # 2. 创建该文件的切片子文件夹
            sub_dir = os.path.join(output_dir, base_name)
            ensure_output_dir(sub_dir)

            # 用 pydub 加载原始音频用于切分（保留原始音质和声道）
            audio = AudioSegment.from_file(file_path)

            # 3. 按段落边界切分并导出
            for idx, seg in enumerate(result["segments"], 1):
                start_ms = int(seg["start"] * 1000)
                end_ms = int(seg["end"] * 1000)

                # 边界保护：确保切片范围合法
                if end_ms <= start_ms:
                    continue
                if end_ms > len(audio):
                    end_ms = len(audio)

                chunk = audio[start_ms:end_ms]
                # 切片文件名: {原文件名}_{标签}_{序号:02d}.wav
                chunk_name = f"{base_name}_{seg['label']}_{idx:02d}.wav"
                chunk_path = os.path.join(sub_dir, chunk_name)
                chunk.export(chunk_path, format="wav")

                # 记录段落信息用于 CSV 导出
                all_records.append({
                    "filename": filename,
                    "label": seg["label"],
                    "segment_index": idx,
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "duration": round(seg["duration"], 3),
                    "slice_file": chunk_name,
                })

            tracker.update(
                f"已切分: {filename} ({len(result['segments'])} 段)"
            )
        except Exception as e:
            # 捕获异常并记录，不中断后续文件处理
            errors.append({"file": file_path, "error": str(e)})
            tracker.update(f"切分失败: {filename} - {e}")

    # 4. 导出段落信息 CSV
    csv_path = os.path.join(output_dir, "segment_info.csv")
    df = pd.DataFrame(all_records, columns=[
        "filename", "label", "segment_index",
        "start", "end", "duration", "slice_file",
    ])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {
        "total_files": total,
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口：
    #   python segment.py <音频文件路径> [段数]       单文件识别
    #   python segment.py <输入文件夹> <输出目录> [段数]  批量切分
    if len(sys.argv) < 2:
        print("用法: python segment.py <音频文件路径> [段数]")
        print("      python segment.py <输入文件夹> <输出目录> [段数]")
        sys.exit(1)

    src = sys.argv[1]

    if os.path.isfile(src):
        # 单文件调试：只识别段落，不切分
        n_seg = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        print(f"正在识别乐段: {src} (期望 {n_seg} 段)")
        result = detect_segments(src, n_segments=n_seg)
        print(f"文件名: {result['filename']}")
        print(f"边界时间: {[f'{t:.2f}s' for t in result['boundaries']]}")
        print(f"段落详情 ({len(result['segments'])} 段):")
        for seg in result["segments"]:
            print(
                f"  {seg['label']}: {seg['start']:.2f}s - {seg['end']:.2f}s "
                f"(时长 {seg['duration']:.2f}s)"
            )
    else:
        # 文件夹批量调试：识别 + 切分
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "./segments_output"
        n_seg = int(sys.argv[3]) if len(sys.argv) > 3 else 6

        def cb(current, total, message):
            print(f"[{current}/{total}] {message}")

        result = segment_batch(src, out_dir, n_segments=n_seg, progress_callback=cb)
        print(f"\n批量切分完成:")
        print(f"  总文件数: {result['total_files']}")
        print(f"  输出目录: {out_dir}")
        print(f"  段落信息: {os.path.join(out_dir, 'segment_info.csv')}")
        print(f"  失败数:   {len(result['errors'])}")
        for err in result["errors"]:
            print(f"    - {err['file']}: {err['error']}")
