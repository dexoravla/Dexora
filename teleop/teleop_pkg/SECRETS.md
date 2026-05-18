# 灵巧手鉴权 / 密钥文件说明

`receive_from_vision_pro.py` 等脚本会通过 `XHandTeleOps("config.yaml")` 读取 `config.yaml`，
其中默认引用了同目录下两个**未随项目分发**的文件：

| 文件 | 说明 | 怎么拿 |
| ---- | ---- | ------ |
| `auth_info.json` | xhand_tele_ops SDK 鉴权信息 | 向 RobotEra / xhand SDK 提供方索取 |
| `key.dat`        | xhand_tele_ops SDK 授权密钥 | 同上 |

迁移本项目到新机器时：

1. 把厂家提供的 `auth_info.json` 和 `key.dat` 复制到 `teleop_pkg/` 目录下
   （与 `config.yaml` 同级）。
2. 在 `config.yaml`、`config_without_xhand.yaml` 里检查
   `auth_info_file_path` / `key_file_path` 字段，确保指向上面这两个文件。
3. **不要把这两个文件提交进 git**，本项目根目录的 `.gitignore` 已经把它们排除。

如果新机器是 aarch64 / Jetson 平台，记得替换为对应的 wheel：
`teleop_pkg/xhand_tele_ops-*.whl` 默认是 `cp38-linux_x86_64` 版本。
