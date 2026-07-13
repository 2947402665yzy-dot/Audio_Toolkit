"""
文件重命名模块
根据模板对音频文件进行批量重命名，支持计数器、原名、BPM、调性等变量。
纯 Python 实现（os, re, csv），不依赖第三方音频库。
纯逻辑模块，不包含任何 GUI 代码。
"""

import sys
import os
import re
import csv

# 将上级目录加入搜索路径，以便导入共享工具模块 utils.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scan_audio_files, ensure_output_dir, ProgressTracker, format_time, safe_filename

# 模板中支持的所有变量名，用于判断是否需要读取 CSV 元数据
TEMPLATE_VAR_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?::[^}]*)?\}")


def _load_metadata_csv(metadata_csv):
    """
    读取元数据 CSV，返回以文件名为键的字典。

    CSV 需包含 filename 列作为主键，其余列（如 bpm、key）作为元数据。
    若文件不存在或格式异常，返回空字典。

    参数:
        metadata_csv: CSV 文件路径

    返回:
        dict: {filename: {列名: 值, ...}}，读取失败时返回 {}
    """
    metadata = {}
    if not metadata_csv or not os.path.isfile(metadata_csv):
        return metadata

    try:
        # 用 utf-8-sig 读取，兼容带 BOM 的 CSV（如 bpm_detect 导出的文件）
        with open(metadata_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 以 filename 列作为主键
                fname = row.get("filename")
                if not fname:
                    continue
                metadata[fname] = row
    except Exception:
        # 读取失败时返回空字典，调用方会用 "unknown" 兜底
        return {}

    return metadata


def _format_template(template, counter, original_name, metadata_row):
    """
    根据模板和上下文变量生成新文件名（不含扩展名）。

    支持的模板变量：
        {counter:03d} - 计数器，支持格式说明符（如 03d 表示补零三位）
        {original}    - 原文件名（不含扩展名）
        {bpm}         - BPM 值，从元数据读取，无则用 "unknown"
        {key}         - 调性，从元数据读取，无则用 "unknown"

    参数:
        template: 命名模板字符串
        counter: 当前计数器值（整数）
        original_name: 原文件名（不含扩展名）
        metadata_row: 该文件对应的元数据字典（可能为 None 或空）

    返回:
        str: 替换变量后的新文件名（不含扩展名）
    """
    # 找出模板中实际用到的变量名
    used_vars = set(TEMPLATE_VAR_PATTERN.findall(template))

    # 准备各变量值
    values = {}

    # counter 支持格式说明符，用 format 映射方式替换
    if "counter" in used_vars:
        values["counter"] = counter

    if "original" in used_vars:
        values["original"] = original_name

    # bpm / key 从元数据行读取，缺失则用 "unknown"
    if "bpm" in used_vars:
        bpm_val = "unknown"
        if metadata_row:
            bpm_raw = metadata_row.get("bpm")
            if bpm_raw not in (None, "", "nan", "NaN"):
                # 若是浮点数尝试转成更简洁的整数字符串
                try:
                    bpm_f = float(bpm_raw)
                    bpm_val = str(int(bpm_f)) if bpm_f.is_integer() else str(bpm_f)
                except (ValueError, TypeError):
                    bpm_val = str(bpm_raw)
        values["bpm"] = bpm_val

    if "key" in used_vars:
        key_val = "unknown"
        if metadata_row:
            key_raw = metadata_row.get("key")
            if key_raw not in (None, "", "nan", "NaN"):
                key_val = str(key_raw)
        values["key"] = key_val

    # 使用 str.format_map 替换变量；自定义缺省字典避免 KeyError
    class _SafeDict(dict):
        def __missing__(self, key):
            return "unknown"

    safe_values = _SafeDict(values)

    try:
        new_name = template.format_map(safe_values)
    except (KeyError, ValueError, IndexError):
        # 模板格式异常时退回原名，避免崩溃
        new_name = original_name

    return new_name


def rename_file(file_path, output_dir, template="{counter:03d}", counter=1, metadata_csv=None):
    """
    根据模板重命名单个文件。

    将文件复制/移动到输出目录并按模板命名，保持原扩展名不变。
    若模板包含 {bpm} 或 {key}，会从 metadata_csv 读取对应元数据；
    CSV 不存在或文件不在其中时，对应变量用 "unknown" 替代。

    参数:
        file_path: 输入文件完整路径
        output_dir: 输出目录路径（不存在会自动创建）
        template: 命名模板，支持 {counter:03d}/{original}/{bpm}/{key}，默认 "{counter:03d}"
        counter: 当前计数器值，默认 1
        metadata_csv: 元数据 CSV 路径（含 filename/bpm/key 等列），可为 None

    返回:
        dict: {"original": 原文件名, "new_name": 新文件名(含扩展名), "new_path": 新文件完整路径}

    异常:
        处理失败时抛出对应异常，由批量函数捕获记录。
    """
    # 确保输出目录存在
    ensure_output_dir(output_dir)

    # 解析原文件名与扩展名
    original_basename = os.path.basename(file_path)
    original_name, ext = os.path.splitext(original_basename)

    # 模板含 bpm/key 时才读取 CSV，避免无谓 IO
    used_vars = set(TEMPLATE_VAR_PATTERN.findall(template))
    metadata_row = None
    if ("bpm" in used_vars or "key" in used_vars) and metadata_csv:
        metadata = _load_metadata_csv(metadata_csv)
        # 元数据以 filename 为键，尝试匹配
        metadata_row = metadata.get(original_basename) or metadata.get(original_name)

    # 根据模板生成新文件名（不含扩展名）
    new_name_body = _format_template(template, counter, original_name, metadata_row)

    # 清理文件名非法字符，确保跨平台合法
    new_name_body = safe_filename(new_name_body)
    # 清理后若为空字符串，退回原名避免覆盖
    if not new_name_body:
        new_name_body = original_name

    # 拼接新文件名（含原扩展名）
    new_name = f"{new_name_body}{ext}"
    new_path = os.path.join(output_dir, new_name)

    # 处理重名：若目标已存在，追加计数后缀避免覆盖
    if os.path.exists(new_path):
        dup = 1
        while os.path.exists(new_path):
            new_name = f"{new_name_body}_{dup}{ext}"
            new_path = os.path.join(output_dir, new_name)
            dup += 1

    # 复制文件到输出目录（不修改原始文件，与其他模块保持一致）
    import shutil
    shutil.copy2(file_path, new_path)

    return {
        "original": original_basename,
        "new_name": new_name,
        "new_path": new_path,
    }


def rename_batch(input_dir, output_dir, template="{counter:03d}", metadata_csv=None, progress_callback=None):
    """
    批量重命名文件夹内所有音频文件。

    counter 从 1 开始递增，扫描顺序按文件名排序。
    通过 ProgressTracker 上报进度，异常文件记录到 errors 列表。
    用 safe_filename 确保生成的文件名合法。

    参数:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        template: 命名模板，默认 "{counter:03d}"
        metadata_csv: 元数据 CSV 路径，可为 None
        progress_callback: 进度回调函数，签名为 callback(current, total, message)，可为 None

    返回:
        dict: {"total_files": 处理文件总数, "renamed": [重命名结果列表], "errors": [失败文件信息列表]}
              renamed 中每项为 rename_file 的返回值；
              errors 中每项为 {"file": 文件路径, "error": 错误描述}
    """
    # 扫描输入目录下所有音频文件（已排序）
    audio_files = scan_audio_files(input_dir)
    total = len(audio_files)

    # 初始化进度跟踪器
    tracker = ProgressTracker(total, progress_callback)

    renamed = []
    errors = []

    # counter 从 1 开始递增
    counter = 1
    for file_path in audio_files:
        filename = os.path.basename(file_path)
        try:
            result = rename_file(
                file_path,
                output_dir,
                template=template,
                counter=counter,
                metadata_csv=metadata_csv,
            )
            renamed.append(result)
            # 上报进度，附带新文件名
            tracker.update(f"已重命名: {result['original']} -> {result['new_name']}")
        except Exception as e:
            # 捕获异常并记录，不中断后续文件处理
            errors.append({"file": file_path, "error": str(e)})
            tracker.update(f"失败: {filename} ({e})")
        # 无论成功失败都递增计数器，保持序号连续
        counter += 1

    return {
        "total_files": total,
        "renamed": renamed,
        "errors": errors,
    }


if __name__ == "__main__":
    # 命令行调试入口：python rename.py <输入文件夹> <输出目录> [模板] [元数据CSV]
    import sys

    if len(sys.argv) < 3:
        print("用法: python rename.py <输入文件夹> <输出目录> [模板] [元数据CSV]")
        print("示例: python rename.py ./input ./output \"{counter:03d}_{original}\"")
        print("示例: python rename.py ./input ./output \"{counter:03d}_{bpm}bpm\" metadata.csv")
        sys.exit(1)

    src = sys.argv[1]
    out = sys.argv[2]
    tpl = sys.argv[3] if len(sys.argv) > 3 else "{counter:03d}"
    csv_path = sys.argv[4] if len(sys.argv) > 4 else None

    # 批量调试，打印进度
    def cb(current, total, message):
        print(f"[{current}/{total}] {message}")

    result = rename_batch(src, out, template=tpl, metadata_csv=csv_path, progress_callback=cb)
    print("批量重命名结果:", result)
