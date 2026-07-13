# -*- coding: utf-8 -*-
"""
Audio Data Toolkit - 图形界面模块

基于 tkinter + ttk 实现，采用配置驱动架构：
    - 通过 TAB_CONFIGS 列表描述 15 个功能标签页的全部信息
    - 通用 create_tab 方法根据配置生成每个标签页的「输入区 / 参数区 / 执行区」
    - 输入/输出路径通过共享的 tkinter.StringVar 在所有标签页间同步
    - 处理逻辑在后台线程执行，进度回调通过 root.after(0, ...) 安全更新界面

core 模块采用动态导入（importlib），这样即使某个 core 模块依赖未安装，
GUI 仍可正常启动，仅在实际点击「开始处理」时给出友好的错误提示。
"""

import os
import sys
import threading
import importlib
import inspect

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 确保 core 包能被导入（audio_toolkit 自身目录加入搜索路径）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 标签页配置：每个元素描述一个功能标签页的全部信息
#
# 字段说明:
#   key             - 唯一标识
#   title           - 标签页标题（实验性功能标题已含“(实验性)”）
#   module          - core 模块路径（如 "core.slicer"）
#   func            - 处理函数名（如 "slice_audio_batch"）
#   output_path_type- 输出路径类型:
#                       "folder" = 输出文件夹（共享 output_folder）
#                       "csv"    = 输出 CSV 文件（共享 output_csv）
#                       None     = 无需输出路径（结果自动保存到输入目录）
#   experimental    - 是否为实验性功能
#   warning         - 实验性功能在界面顶部显示的红色提示文字
#   note            - 无输出路径时显示的说明文字（可选）
#   params          - 参数控件列表（无参数时为空列表）
#
# 参数控件字段说明:
#   key         - 参数名（对应处理函数的关键字参数）
#   label       - 界面显示标签
#   type        - 控件类型: number / combobox / text / checkbox / file
#   default     - 默认值（number 类型留空时用 ""）
#   value_type  - 取值类型: int / float / str（默认 str，仅 number/combobox 使用）
#   allow_empty - number / file 是否允许留空（留空时转为 None）
#   options     - combobox 可选项列表
# ---------------------------------------------------------------------------
TAB_CONFIGS = [
    # 1. 音频切割
    {
        "key": "slicer",
        "title": "音频切割",
        "module": "core.slicer",
        "func": "slice_audio_batch",
        "output_path_type": "folder",
        "experimental": False,
        "warning": None,
        "note": None,
        "params": [
            {"key": "slice_length_sec", "label": "切片时长(秒)", "type": "number",
             "default": 30, "value_type": "int"},
            {"key": "output_format", "label": "输出格式", "type": "combobox",
             "default": "wav", "options": ["wav", "mp3"], "value_type": "str"},
        ],
    },
    # 2. MFCC 提取
    {
        "key": "mfcc",
        "title": "MFCC 提取",
        "module": "core.mfcc",
        "func": "extract_mfcc_batch",
        "output_path_type": "csv",
        "experimental": False,
        "warning": None,
        "note": None,
        "params": [
            {"key": "n_mfcc", "label": "MFCC维度", "type": "number",
             "default": 13, "value_type": "int"},
            {"key": "detailed", "label": "详细模式(维度×6)", "type": "checkbox",
             "default": False},
        ],
    },
    # 3. Lo-fi 转换
    {
        "key": "lofi",
        "title": "Lo-fi 转换",
        "module": "core.lofi",
        "func": "lofi_convert_batch",
        "output_path_type": "folder",
        "experimental": False,
        "warning": None,
        "note": None,
        "params": [
            {"key": "target_sr", "label": "采样率(Hz)", "type": "combobox",
             "default": "22050", "options": ["44100", "22050", "11025", "8000"],
             "value_type": "int"},
            {"key": "bit_depth", "label": "比特深度", "type": "combobox",
             "default": "8-bit", "options": ["8-bit", "16-bit", "24-bit"],
             "value_type": "str"},
            {"key": "bitrate", "label": "MP3比特率", "type": "combobox",
             "default": "32k", "options": ["320k", "128k", "64k", "32k"],
             "value_type": "str"},
            {"key": "hp_cutoff", "label": "高通截止频率(Hz, 留空关闭)", "type": "number",
             "default": "", "value_type": "float", "allow_empty": True},
            {"key": "lp_cutoff", "label": "低通截止频率(Hz, 留空关闭)", "type": "number",
             "default": "", "value_type": "float", "allow_empty": True},
            {"key": "filter_order", "label": "滤波器阶数", "type": "number",
             "default": 4, "value_type": "int"},
            {"key": "output_format", "label": "输出格式", "type": "combobox",
             "default": "mp3", "options": ["mp3", "wav"], "value_type": "str"},
        ],
    },
    # 4. Metadata 提取
    {
        "key": "metadata",
        "title": "Metadata 提取",
        "module": "core.metadata",
        "func": "extract_metadata_batch",
        "output_path_type": "csv",
        "experimental": False,
        "warning": None,
        "note": None,
        "params": [],
    },
    # 5. 数据集统计
    {
        "key": "dataset_stats",
        "title": "数据集统计",
        "module": "core.dataset_stats",
        "func": "compute_dataset_stats",
        "output_path_type": None,
        "experimental": False,
        "warning": None,
        "note": "本功能无需输出路径，统计结果将自动保存到输入文件夹下（dataset_stats.csv）。",
        "params": [],
    },
    # 6. 静音检测
    {
        "key": "silence_detect",
        "title": "静音检测",
        "module": "core.silence_detect",
        "func": "detect_silence_batch",
        "output_path_type": "csv",
        "experimental": False,
        "warning": None,
        "note": None,
        "params": [
            {"key": "threshold_db", "label": "静音阈值(dB)", "type": "number",
             "default": -40, "value_type": "float"},
            {"key": "min_duration_sec", "label": "最短静音时长(秒)", "type": "number",
             "default": 0.5, "value_type": "float"},
        ],
    },
    # 7. 重复检测
    {
        "key": "duplicate_detect",
        "title": "重复检测",
        "module": "core.duplicate_detect",
        "func": "detect_duplicates",
        "output_path_type": None,
        "experimental": False,
        "warning": None,
        "note": "本功能无需输出路径，重复检测报告将自动保存到输入文件夹下（duplicate_report.csv）。",
        "params": [],
    },
    # 8. 响度标准化
    {
        "key": "loudness_norm",
        "title": "响度标准化",
        "module": "core.loudness_norm",
        "func": "normalize_loudness_batch",
        "output_path_type": "folder",
        "experimental": False,
        "warning": None,
        "note": None,
        "params": [
            {"key": "target_dbfs", "label": "目标响度(dBFS)", "type": "number",
             "default": -16.0, "value_type": "float"},
        ],
    },
    # 9. 音频格式转换
    {
        "key": "format_convert",
        "title": "音频格式转换",
        "module": "core.format_convert",
        "func": "convert_format_batch",
        "output_path_type": "folder",
        "experimental": False,
        "warning": None,
        "note": None,
        "params": [
            {"key": "target_format", "label": "目标格式", "type": "combobox",
             "default": "wav", "options": ["wav", "mp3", "flac", "aac"],
             "value_type": "str"},
            {"key": "target_sr", "label": "采样率(Hz, 留空保持原样)", "type": "number",
             "default": "", "value_type": "int", "allow_empty": True},
        ],
    },
    # 10. Mel Spectrogram 导出
    {
        "key": "mel_spectrogram",
        "title": "Mel Spectrogram 导出",
        "module": "core.mel_spectrogram",
        "func": "export_mel_spectrogram_batch",
        "output_path_type": "folder",
        "experimental": False,
        "warning": None,
        "note": None,
        "params": [
            {"key": "n_mels", "label": "Mel频带数", "type": "number",
             "default": 128, "value_type": "int"},
        ],
    },
    # 11. 文件重命名
    {
        "key": "rename",
        "title": "文件重命名",
        "module": "core.rename",
        "func": "rename_batch",
        "output_path_type": "folder",
        "experimental": False,
        "warning": None,
        "note": None,
        "params": [
            {"key": "template", "label": "命名模板", "type": "text",
             "default": "{counter:03d}"},
            {"key": "metadata_csv", "label": "元数据CSV(用于{bpm}/{key}变量)", "type": "file",
             "default": "", "allow_empty": True},
        ],
    },
    # 12. BPM 检测
    {
        "key": "bpm_detect",
        "title": "BPM 检测",
        "module": "core.bpm_detect",
        "func": "detect_bpm_batch",
        "output_path_type": "csv",
        "experimental": False,
        "warning": None,
        "note": None,
        "params": [],
    },
    # 13. 调性检测 (实验性)
    {
        "key": "key_detect",
        "title": "调性检测 (实验性)",
        "module": "core.key_detect",
        "func": "detect_key_batch",
        "output_path_type": "csv",
        "experimental": True,
        "warning": "⚠ 实验性功能：结果可能不准确",
        "note": None,
        "params": [],
    },
    # 14. 乐段识别与切分 (实验性)
    {
        "key": "segment",
        "title": "乐段识别 (实验性)",
        "module": "core.segment",
        "func": "segment_batch",
        "output_path_type": "folder",
        "experimental": True,
        "warning": "⚠ 实验性功能：段数由算法自动检测，无需手动指定",
        "note": None,
        "params": [],
    },
    # 15. 音乐风格检测 (实验性)
    {
        "key": "genre_detect",
        "title": "风格检测 (实验性)",
        "module": "core.genre_detect",
        "func": "detect_genre_batch",
        "output_path_type": "csv",
        "experimental": True,
        "warning": "⚠ 实验性功能：当前仅提取特征，风格分类待模型支持",
        "note": None,
        "params": [],
    },
]


class AudioToolkitGUI:
    """音频处理工具箱主界面。

    通过 TAB_CONFIGS 配置驱动生成 15 个标签页，所有标签页共享输入/输出路径，
    处理逻辑在后台线程执行，避免界面卡顿。
    """

    def __init__(self, root):
        self.root = root

        # 输入文件夹路径共享（所有标签页共用）
        self.input_folder = tk.StringVar()

        # 输出路径改为每个标签页独立，避免不同模块互相覆盖
        self.output_folders = {}   # key -> tk.StringVar
        self.output_csvs = {}      # key -> tk.StringVar
        for cfg in TAB_CONFIGS:
            self.output_folders[cfg["key"]] = tk.StringVar()
            self.output_csvs[cfg["key"]] = tk.StringVar()

        # 全局处理状态标志（同一时刻仅允许一个任务运行）
        self.is_processing = False

        # 配置索引：key -> config，便于快速查找
        self._config_map = {cfg["key"]: cfg for cfg in TAB_CONFIGS}

        # 存储每个标签页的控件引用
        # 结构: { key: {"config", "button", "progress", "log", "param_vars"} }
        self.tabs = {}

        # 顶部说明
        top_label = ttk.Label(
            root,
            text='Audio Data Toolkit —— 在左侧选择功能，设置路径与参数后点击「开始处理」。',
            anchor="w",
        )
        top_label.pack(fill="x", padx=10, pady=(6, 2))

        # 主体：左侧功能列表 + 右侧内容区
        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        # 左侧功能列表
        left_frame = ttk.Frame(body, width=180)
        left_frame.pack(side="left", fill="y", padx=(0, 4))
        left_frame.pack_propagate(False)  # 固定宽度

        # 列表标题
        ttk.Label(left_frame, text="功能列表", font=("", 11, "bold")).pack(
            fill="x", padx=8, pady=(6, 2)
        )

        # 用 Listbox 显示功能名称，选中后切换右侧内容
        self.func_listbox = tk.Listbox(left_frame, width=20, height=25,
                                       font=("", 11), selectmode="single")
        self.func_listbox.pack(fill="both", expand=True, padx=4, pady=4)
        for config in TAB_CONFIGS:
            self.func_listbox.insert("end", config["title"])
        self.func_listbox.bind("<<ListboxSelect>>", self._on_func_select)

        # 右侧内容区：用一个 Frame，切换内容时清空重建
        self.content_frame = ttk.Frame(body)
        self.content_frame.pack(side="right", fill="both", expand=True)

        # 当前选中的标签页 key
        self.current_tab_key = None

        # 创建所有标签页内容（隐藏不选中的）
        for config in TAB_CONFIGS:
            self.create_tab(config)

        # 默认选中第一个
        self.func_listbox.selection_set(0)
        self._show_tab(TAB_CONFIGS[0]["key"])

    # ------------------------------------------------------------------
    # 标签页创建与切换
    # ------------------------------------------------------------------
    def create_tabs(self):
        """根据 TAB_CONFIGS 创建所有标签页。"""
        for config in TAB_CONFIGS:
            self.create_tab(config)

    def _on_func_select(self, event):
        """左侧列表选中时，切换右侧显示的标签页内容。"""
        selection = self.func_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        config = TAB_CONFIGS[idx]
        self._show_tab(config["key"])

    def _show_tab(self, key):
        """显示指定的标签页，隐藏其他所有标签页。"""
        if self.current_tab_key == key:
            return
        # 隐藏当前显示的
        if self.current_tab_key and self.current_tab_key in self.tabs:
            old_frame = self.tabs[self.current_tab_key].get("frame")
            if old_frame:
                old_frame.pack_forget()
        # 显示新的
        if key in self.tabs:
            frame = self.tabs[key].get("frame")
            if frame:
                frame.pack(fill="both", expand=True, in_=self.content_frame)
        self.current_tab_key = key

    def create_tab(self, config):
        """通用方法：根据单个配置创建一个标签页。

        每个标签页统一包含三块结构：
            1. 路径设置（输入区）：输入文件夹 + 输出路径（如需要）
            2. 参数设置（参数区）：根据 config["params"] 生成控件
            3. 执行区：开始处理按钮 + 进度条 + 日志区

        每个标签页创建为独立的 Frame，初始不打包(pack)，
        由 _show_tab 方法控制显示/隐藏。
        """
        key = config["key"]
        tab = ttk.Frame(self.content_frame)
        # 记录 frame 引用，用于切换显示
        self.tabs[key] = {"frame": tab, "config": config, "param_vars": {}}

        # 实验性功能顶部红色提示
        if config.get("warning"):
            warn = tk.Label(tab, text=config["warning"], fg="red", anchor="w")
            warn.pack(fill="x", padx=10, pady=(6, 2))

        # ===== 区块 1：路径设置（输入区）=====
        path_frame = ttk.LabelFrame(
            tab, text="路径设置（输入文件夹共享，输出路径各功能独立）"
        )
        path_frame.pack(fill="x", padx=8, pady=5)

        row = 0
        # 输入文件夹（所有标签页共享）
        self._create_path_row(
            path_frame, row, "输入文件夹：",
            self.input_folder, self.select_input_folder,
        )
        row += 1

        # 输出路径（按类型生成，每个标签页独立）
        output_type = config["output_path_type"]
        if output_type == "folder":
            self._create_path_row(
                path_frame, row, "输出文件夹：",
                self.output_folders[key], self.select_output_folder,
            )
            row += 1
        elif output_type == "csv":
            self._create_path_row(
                path_frame, row, "输出CSV文件：",
                self.output_csvs[key], self.select_output_csv, file_mode=True,
            )
            row += 1
        else:
            # 无输出路径，显示说明文字
            note_text = config.get("note") or "本功能不需要单独的输出路径。"
            ttk.Label(path_frame, text=note_text, foreground="gray").grid(
                row=row, column=0, columnspan=3, sticky="w", padx=5, pady=6
            )
            row += 1

        # ===== 区块 2：参数设置（参数区）=====
        params_frame = ttk.LabelFrame(tab, text="参数设置")
        params_frame.pack(fill="x", padx=8, pady=5)

        param_vars = {}
        if config["params"]:
            for i, param in enumerate(config["params"]):
                pvar = self._create_param_control(params_frame, i, param)
                param_vars[param["key"]] = pvar
        else:
            ttk.Label(
                params_frame,
                text="本功能无需额外参数，直接点击下方 [开始处理] 即可。",
                foreground="gray",
            ).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=10)

        # ===== 区块 3：执行区 =====
        exec_frame = ttk.LabelFrame(tab, text="执行")
        exec_frame.pack(fill="both", expand=True, padx=8, pady=5)

        # 按钮行
        btn_row = ttk.Frame(exec_frame)
        btn_row.pack(fill="x", padx=5, pady=5)
        start_btn = ttk.Button(
            btn_row, text="开始处理",
            command=lambda k=key: self.start_processing(k),
        )
        start_btn.pack(side="left")
        ttk.Button(
            btn_row, text="清空日志",
            command=lambda k=key: self.clear_log(k),
        ).pack(side="left", padx=10)

        # 进度条
        progress = ttk.Progressbar(exec_frame, mode="determinate", maximum=100)
        progress.pack(fill="x", padx=5, pady=2)

        # 日志区（只读 Text + 滚动条，自动滚动到底部）
        log_frame = ttk.Frame(exec_frame)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side="right", fill="y")
        log_text = tk.Text(
            log_frame, height=10, state="disabled",
            yscrollcommand=scrollbar.set, wrap="word",
        )
        log_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=log_text.yview)

        # 保存控件引用（保留 frame 引用）
        self.tabs[key].update({
            "config": config,
            "button": start_btn,
            "progress": progress,
            "log": log_text,
            "param_vars": param_vars,
        })

    def _create_path_row(self, parent, row, label_text, var, pick_command, file_mode=False):
        """在路径设置区创建一行：标签 + 输入框 + 浏览按钮。

        参数:
            parent       - 父容器（使用 grid 布局）
            row          - grid 行号
            label_text   - 标签文字
            var          - 绑定的 tkinter 变量（共享路径）
            pick_command - 浏览按钮触发的回调
            file_mode    - True 表示文件选择（输入框跨两列），False 表示文件夹选择
        """
        ttk.Label(parent, text=label_text).grid(
            row=row, column=0, sticky="w", padx=5, pady=4
        )
        if file_mode:
            sub = ttk.Frame(parent)
            sub.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=4)
            entry = ttk.Entry(sub, textvariable=var)
            entry.pack(side="left", fill="x", expand=True)
            ttk.Button(sub, text="浏览...", command=pick_command).pack(
                side="left", padx=(5, 0)
            )
        else:
            entry = ttk.Entry(parent, textvariable=var)
            entry.grid(row=row, column=1, sticky="ew", padx=5, pady=4)
            ttk.Button(parent, text="浏览...", command=pick_command).grid(
                row=row, column=2, padx=5, pady=4
            )
        parent.columnconfigure(1, weight=1)

    def _create_param_control(self, parent, row, param):
        """在参数设置区创建一个参数控件。

        根据 param["type"] 生成对应控件：
            number   - ttk.Entry（数值输入，支持留空转 None）
            combobox - ttk.Combobox（下拉选择，readonly）
            text     - ttk.Entry（文本输入）
            checkbox - ttk.Checkbutton（复选框）
            file     - ttk.Entry + 浏览按钮（文件路径选择）

        返回:
            (var, type, param_spec) 三元组，供 collect_params 收集值时使用
        """
        ptype = param["type"]
        ttk.Label(parent, text=param["label"] + "：").grid(
            row=row, column=0, sticky="w", padx=5, pady=4
        )

        var = None
        if ptype == "number":
            default = param.get("default", "")
            var = tk.StringVar(value="" if default == "" else str(default))
            ttk.Entry(parent, textvariable=var, width=22).grid(
                row=row, column=1, sticky="w", padx=5, pady=4
            )
        elif ptype == "combobox":
            var = tk.StringVar(value=str(param["default"]))
            ttk.Combobox(
                parent, textvariable=var, values=param["options"],
                state="readonly", width=20,
            ).grid(row=row, column=1, sticky="w", padx=5, pady=4)
        elif ptype == "text":
            var = tk.StringVar(value=str(param["default"]))
            ttk.Entry(parent, textvariable=var, width=32).grid(
                row=row, column=1, sticky="w", padx=5, pady=4
            )
        elif ptype == "checkbox":
            var = tk.BooleanVar(value=bool(param["default"]))
            ttk.Checkbutton(parent, variable=var).grid(
                row=row, column=1, sticky="w", padx=5, pady=4
            )
        elif ptype == "file":
            default = param.get("default", "")
            var = tk.StringVar(value="" if default == "" else str(default))
            sub = ttk.Frame(parent)
            sub.grid(row=row, column=1, sticky="ew", padx=5, pady=4)
            ttk.Entry(sub, textvariable=var).pack(side="left", fill="x", expand=True)
            ttk.Button(
                sub, text="浏览...",
                command=lambda v=var: self._pick_file(v),
            ).pack(side="left", padx=(5, 0))

        parent.columnconfigure(1, weight=1)
        # 返回 (变量, 控件类型, 参数配置)，便于后续收集与类型转换
        return (var, ptype, param)

    # ------------------------------------------------------------------
    # 路径选择回调
    # ------------------------------------------------------------------
    def select_input_folder(self):
        """选择输入文件夹，更新共享变量。"""
        folder = filedialog.askdirectory(title="选择输入文件夹")
        if folder:
            self.input_folder.set(folder)

    def select_output_folder(self):
        """选择输出文件夹，更新当前标签页的输出路径。"""
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            # 更新当前激活标签页的 output_folder
            if self.current_tab_key and self.current_tab_key in self.output_folders:
                self.output_folders[self.current_tab_key].set(folder)

    def select_output_csv(self):
        """选择输出 CSV 文件保存路径，更新当前标签页的输出路径。"""
        path = filedialog.asksaveasfilename(
            title="选择输出CSV文件",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        if path:
            # 更新当前激活标签页的 output_csv
            if self.current_tab_key and self.current_tab_key in self.output_csvs:
                self.output_csvs[self.current_tab_key].set(path)

    def _pick_file(self, var):
        """通用文件选择（用于参数区文件路径控件），选中后更新对应变量。"""
        path = filedialog.askopenfilename(
            title="选择文件",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        if path:
            var.set(path)

    # ------------------------------------------------------------------
    # 参数收集
    # ------------------------------------------------------------------
    def collect_params(self, key):
        """从标签页控件读取所有参数值，并做类型转换。

        - number: 按 value_type 转 int/float；allow_empty 且为空时转 None
        - combobox: 按 value_type 转 int/float/str
        - text: 直接取字符串
        - checkbox: 转 bool
        - file: 取字符串；allow_empty 且为空时转 None

        参数非法时抛出 ValueError，由调用方捕获。
        """
        params = {}
        for pkey, (var, ptype, spec) in self.tabs[key]["param_vars"].items():
            if ptype == "number":
                raw = var.get().strip()
                if spec.get("allow_empty") and raw == "":
                    params[pkey] = None
                else:
                    vtype = spec.get("value_type", "int")
                    if vtype == "float":
                        params[pkey] = float(raw)
                    else:
                        params[pkey] = int(raw)
            elif ptype == "combobox":
                raw = var.get()
                vtype = spec.get("value_type", "str")
                if vtype == "int":
                    params[pkey] = int(raw)
                elif vtype == "float":
                    params[pkey] = float(raw)
                else:
                    params[pkey] = raw
            elif ptype == "text":
                params[pkey] = var.get()
            elif ptype == "checkbox":
                params[pkey] = bool(var.get())
            elif ptype == "file":
                raw = var.get().strip()
                if spec.get("allow_empty") and raw == "":
                    params[pkey] = None
                else:
                    params[pkey] = raw
        return params

    # ------------------------------------------------------------------
    # 处理流程：启动线程 / 后台执行 / 进度回调 / 完成
    # ------------------------------------------------------------------
    def start_processing(self, key):
        """点击「开始处理」按钮的入口：校验路径、收集参数、启动后台线程。"""
        if self.is_processing:
            messagebox.showwarning("提示", "已有任务正在处理中，请等待完成。")
            return

        config = self._config_map.get(key)
        if config is None:
            messagebox.showerror("错误", "未找到该功能的配置。")
            return

        # 校验输入路径
        input_dir = self.input_folder.get().strip()
        if not input_dir:
            messagebox.showwarning("提示", "请先选择输入文件夹。")
            return

        # 校验输出路径（每个标签页独立）
        output_type = config["output_path_type"]
        if output_type == "folder" and not self.output_folders[key].get().strip():
            messagebox.showwarning("提示", "请先选择输出文件夹。")
            return
        if output_type == "csv" and not self.output_csvs[key].get().strip():
            messagebox.showwarning("提示", "请先选择输出CSV文件路径。")
            return

        # 收集并校验参数
        try:
            params = self.collect_params(key)
        except ValueError as e:
            messagebox.showerror("参数错误", f"参数解析失败，请检查数值输入：\n{e}")
            return

        # 在主线程中一次性解析好所有路径字符串，避免后台线程读取 tkinter 变量
        output_dir = self.output_folders[key].get().strip()
        output_csv = self.output_csvs[key].get().strip()

        # 进入处理状态：禁用按钮、清空进度条、记录日志
        self.is_processing = True
        btn = self.tabs[key]["button"]
        btn.config(text="处理中...", state="disabled")
        self.tabs[key]["progress"]["value"] = 0

        self.log(key, "===== 开始处理: {} =====".format(config["title"]))
        self.log(key, "输入文件夹: {}".format(input_dir))
        if output_type == "folder":
            self.log(key, "输出文件夹: {}".format(output_dir))
        elif output_type == "csv":
            self.log(key, "输出CSV: {}".format(output_csv))
        if params:
            self.log(key, "参数: {}".format(params))

        # 启动后台线程执行处理逻辑（界面不卡顿）
        # 路径以字符串形式传入，后台线程不再访问任何 tkinter 对象
        thread = threading.Thread(
            target=self._run_in_thread,
            args=(key, params, input_dir, output_dir, output_csv),
            daemon=True,
        )
        thread.start()

    def _run_in_thread(self, key, params, input_dir, output_dir, output_csv):
        """后台线程执行体：动态导入并调用对应的处理函数。

        所有 UI 更新均通过 root.after(0, ...) 调度到主线程，保证线程安全。
        本方法不直接访问任何 tkinter 对象，所需路径均已由主线程以字符串传入。
        """
        config = self._config_map[key]

        # 动态导入 core 模块与处理函数（GUI 启动时不强制依赖）
        try:
            module = importlib.import_module(config["module"])
            func = getattr(module, config["func"])
        except Exception as e:
            msg = "无法加载模块 {}.{}: {}".format(config["module"], config["func"], e)
            self.root.after(0, lambda: self._on_error(key, msg))
            return

        # 组装调用参数（路径使用主线程传入的字符串）
        kwargs = dict(params)
        kwargs["input_dir"] = input_dir

        output_type = config["output_path_type"]
        if output_type == "folder":
            kwargs["output_dir"] = output_dir
        elif output_type == "csv":
            kwargs["output_csv"] = output_csv

        # 仅当函数声明了 progress_callback（或含 **kwargs）时才传入
        try:
            sig = inspect.signature(func)
            accepts_pc = (
                "progress_callback" in sig.parameters
                or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
            )
        except (ValueError, TypeError):
            accepts_pc = True
        if accepts_pc:
            kwargs["progress_callback"] = self._make_progress_callback(key)

        # 执行处理（异常捕获后回传主线程显示）
        try:
            result = func(**kwargs)
            self.root.after(0, lambda: self._on_done(key, result, config))
        except Exception as e:
            msg = "处理出错: {}".format(e)
            self.root.after(0, lambda: self._on_error(key, msg))

    def _make_progress_callback(self, key):
        """构造进度回调函数，捕获目标标签页 key。

        回调签名: callback(current, total, message)
        通过 root.after(0, ...) 将 UI 更新调度到主线程。
        """
        def callback(current, total, message=""):
            self.root.after(
                0, lambda: self._update_progress(key, current, total, message)
            )
        return callback

    def _update_progress(self, key, current, total, message):
        """主线程中更新进度条与日志（由进度回调调度）。"""
        progress = self.tabs[key]["progress"]
        # total 为 0（无文件）时避免除零，进度条保持 0
        if total and total > 0:
            progress["value"] = current / total * 100
        else:
            progress["value"] = 0

        if message:
            if total and total > 0:
                self.log(key, "[{}/{}] {}".format(current, total, message))
            else:
                self.log(key, message)

    def _on_done(self, key, result, config):
        """处理完成的回调（主线程）：汇总结果并恢复按钮。"""
        parts = ["[{}] 处理完成".format(config["title"])]
        if isinstance(result, dict):
            if "total_files" in result:
                parts.append("共处理 {} 个文件".format(result["total_files"]))
            if "total_slices" in result:
                parts.append("共切出 {} 段".format(result["total_slices"]))
            if "duplicate_groups" in result:
                parts.append("重复组 {} 个".format(result["duplicate_groups"]))
            if "errors" in result and result["errors"]:
                parts.append("失败 {} 个".format(len(result["errors"])))
            if "output_csv" in result and result.get("output_csv"):
                parts.append("输出CSV: {}".format(result["output_csv"]))
        self.log(key, " | ".join(parts))
        self.log(key, "-" * 50)
        self._finish(key)

    def _on_error(self, key, message):
        """处理出错的回调（主线程）：记录日志并弹窗提示，恢复按钮。"""
        self.log(key, "错误: " + message)
        self.log(key, "-" * 50)
        messagebox.showerror("处理出错", message)
        self._finish(key)

    def _finish(self, key):
        """恢复按钮状态并清除处理中标志。"""
        btn = self.tabs[key]["button"]
        btn.config(text="开始处理", state="normal")
        self.is_processing = False

    # ------------------------------------------------------------------
    # 日志区操作
    # ------------------------------------------------------------------
    def log(self, key, message):
        """向指定标签页的日志区追加一条消息，并自动滚动到底部。

        日志区为只读：写入前临时切到 normal，写完再切回 disabled。
        """
        text_widget = self.tabs[key]["log"]
        text_widget.config(state="normal")
        text_widget.insert("end", str(message) + "\n")
        text_widget.see("end")
        text_widget.config(state="disabled")

    def clear_log(self, key):
        """清空指定标签页的日志区。"""
        text_widget = self.tabs[key]["log"]
        text_widget.config(state="normal")
        text_widget.delete("1.0", "end")
        text_widget.config(state="disabled")


if __name__ == "__main__":
    # 直接运行本文件时也可启动界面
    root = tk.Tk()
    root.title("Audio Data Toolkit")
    root.geometry("700x600")
    AudioToolkitGUI(root)
    root.mainloop()
