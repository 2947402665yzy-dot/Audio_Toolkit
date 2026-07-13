#!/bin/bash
# Audio Data Toolkit 启动脚本
# 双击此文件即可启动 GUI

# 获取脚本所在目录（即 audio_toolkit 所在目录）
cd "$(dirname "$0")"

# 用 python3 启动 main.py，nohup 让程序独立运行，不依赖终端
nohup python3 main.py > /dev/null 2>&1 &

# 等待一秒让窗口弹出
sleep 1
