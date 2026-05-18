#!/bin/bash

echo "=== 相机设备映射测试 ==="
echo "当前时间: $(date)"
echo

echo "1. 检查符号链接状态:"
echo "------------------------"
ls -la /dev/camera* 2>/dev/null || echo "未找到相机符号链接"

echo
echo "2. 检查实际设备映射:"
echo "------------------------"
for camera in camera_left camera_right camera_high; do
    if [ -L "/dev/$camera" ]; then
        target=$(readlink "/dev/$camera")
        echo "$camera -> $target"
        
        # 检查目标设备是否存在
        if [ -e "/dev/$target" ]; then
            echo "  ✓ 目标设备存在"
        else
            echo "  ✗ 目标设备不存在"
        fi
    else
        echo "$camera -> 符号链接不存在"
    fi
done

echo
echo "3. 检查 USB 端口分配:"
echo "------------------------"
for dev in video0 video2 video12; do
    if [ -e "/dev/$dev" ]; then
        devpath=$(udevadm info -a -n "/dev/$dev" | grep "ATTRS{devpath}" | head -1 | cut -d'"' -f2)
        echo "$dev -> USB 端口: $devpath"
    fi
done

echo
echo "4. 验证 udev 规则匹配:"
echo "------------------------"
for dev in video0 video2 video12; do
    if [ -e "/dev/$dev" ]; then
        echo "测试 $dev:"
        udevadm test "/sys/class/video4linux/$dev" 2>&1 | grep -E "(camera_left|camera_right|camera_high)" || echo "  未匹配到相机规则"
    fi
done

echo
echo "=== 测试完成 ===" 