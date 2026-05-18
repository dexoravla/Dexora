q退出，s保存图像为参考，o切换参考图像叠加，n下一张图 传入--json来只使用json文件读取和保存参考位姿，此时参考图像叠加不可用

v4l2-ctl --list-devices 确认各个相机id，修改{}config.json，camera_ref为拍摄参考图片的相机的id

camera_ref: 键:值     拍摄时图片所用相机id ： 拍摄reference-image所用相机id

TODO处是运行前要修改的参数 相机连接后的id（/dev/video?）可能会变，可以手动固定