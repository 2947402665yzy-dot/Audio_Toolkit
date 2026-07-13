# -*- coding: utf-8 -*-
"""Audio Data Toolkit 启动入口"""
import sys
import os

# 确保能找到模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from gui import AudioToolkitGUI


def main():
    root = tk.Tk()
    root.title("Audio Data Toolkit")
    root.geometry("700x600")
    app = AudioToolkitGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
