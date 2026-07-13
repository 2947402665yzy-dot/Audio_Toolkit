"""
Audio Data Toolkit - 共享工具模块
提供所有功能模块共用的公共函数，避免重复代码。
"""

import os
import sys

# 支持的音频格式
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.aif', '.aiff', '.aac', '.ogg', '.m4a')


def scan_audio_files(folder):
    """
    扫描文件夹，返回所有音频文件的完整路径列表。

    参数:
        folder: 要扫描的文件夹路径

    返回:
        list: 音频文件完整路径列表，按文件名排序
    """
    result = []
    if not os.path.isdir(folder):
        return result
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(AUDIO_EXTENSIONS):
            result.append(os.path.join(folder, filename))
    return result


def ensure_output_dir(path):
    """
    确保输出目录存在，不存在则递归创建。

    参数:
        path: 输出目录路径

    返回:
        str: 输出目录路径（创建后的）
    """
    if path and not os.path.exists(path):
        os.makedirs(path)
    return path


def get_audio_duration(file_path):
    """
    获取音频文件时长（秒）。

    参数:
        file_path: 音频文件路径

    返回:
        float: 时长（秒），失败返回 0
    """
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    except Exception:
        return 0.0


def format_time(seconds):
    """
    将秒数格式化为 mm:ss 或 hh:mm:ss 字符串。

    参数:
        seconds: 秒数

    返回:
        str: 格式化后的时间字符串
    """
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def safe_filename(name):
    """
    清理文件名中的非法字符。

    参数:
        name: 原始文件名

    返回:
        str: 安全的文件名
    """
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        name = name.replace(char, '_')
    return name.strip()


class ProgressTracker:
    """
    进度跟踪器，供批量处理函数使用。
    GUI 通过传入 callback 实现实时更新进度条。
    """

    def __init__(self, total, callback=None):
        """
        参数:
            total: 总文件数
            callback: 回调函数，签名为 callback(current, total, message)
        """
        self.total = total
        self.current = 0
        self.callback = callback

    def update(self, message=""):
        """更新进度，current 自动 +1"""
        self.current += 1
        if self.callback:
            self.callback(self.current, self.total, message)

    def is_done(self):
        """是否已完成"""
        return self.current >= self.total
