#!/bin/bash

# 相机符号链接安装脚本
# 用于安装udev规则以创建永久的相机设备符号链接

echo "正在安装相机符号链接规则..."

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then
    echo "请使用sudo运行此脚本"
    exit 1
fi

# 复制udev规则文件到系统目录
cp 99-camera-symlinks.rules /etc/udev/rules.d/

# 重新加载udev规则
udevadm control --reload-rules
udevadm trigger

echo "相机符号链接规则已安装完成！"
echo ""
echo "现在系统将自动创建以下三个符号链接："
echo "  /dev/camera_left  -> /dev/video12"
echo "  /dev/camera_right -> /dev/video2" 
echo "  /dev/camera_high  -> /dev/video0"
echo ""
echo "请重新插拔相机设备或重启系统以使规则生效。"
echo "您可以使用以下命令检查符号链接是否创建成功："
echo "  ls -la /dev/camera_*" 
