#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 BSON 数据中的 action 与 observation 对调。

对 xhand_control_data.bson 还会额外执行：
- 在互换之前，将每帧 observation 中的数值（left_hand/right_hand）从角度制转换为弧度制

默认行为（不带任何参数运行）：
- 就地覆盖修改当前目录下的 `episode_0.bson` 与 `xhand_control_data.bson`

支持两类数据：
1) episode_0.bson：顶层字段包含 id/timestamp/metadata/data，topic 名含 /action/ 与 /observation/
2) xhand_control_data.bson：顶层字段包含 frames，每帧包含 action/observation
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


LOGGER = logging.getLogger(__name__)


JsonLike = Union[Dict[str, Any], List[Any]]

DEFAULT_INPUT_FILES: List[str] = [
    "episode_0.bson",
    "xhand_control_data.bson",
]


def _decode_bson_file(path: Path) -> JsonLike:
    try:
        import bson  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "未安装 bson 依赖。请执行：pip install pymongo"
        ) from exc

    raw = path.read_bytes()
    try:
        return bson.BSON(raw).decode()
    except Exception:
        # 兼容“多个 BSON 文档拼接”的情况（本项目里一般是单文档）
        docs: List[Dict[str, Any]] = []
        offset = 0
        raw_len = len(raw)
        while offset + 4 <= raw_len:
            doc_len = int.from_bytes(raw[offset : offset + 4], "little", signed=False)
            if doc_len <= 0 or offset + doc_len > raw_len:
                break
            docs.append(bson.BSON(raw[offset : offset + doc_len]).decode())
            offset += doc_len
        return docs


def _encode_bson_doc(doc: Dict[str, Any]) -> bytes:
    try:
        import bson  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "未安装 bson 依赖。请执行：pip install pymongo"
        ) from exc
    return bson.BSON.encode(doc)


def _swap_topic_prefix(topic: str) -> str:
    if topic.startswith("/action/"):
        return "/observation/" + topic[len("/action/") :]
    if topic.startswith("/observation/"):
        return "/action/" + topic[len("/observation/") :]
    return topic


def _swap_episode(doc: Dict[str, Any]) -> Dict[str, Any]:
    # metadata.topics
    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        topics = metadata.get("topics")
        if isinstance(topics, dict):
            new_topics: Dict[str, Any] = {}
            for k, v in topics.items():
                new_topics[_swap_topic_prefix(str(k))] = v
            metadata["topics"] = new_topics

    # data
    data = doc.get("data")
    if isinstance(data, dict):
        new_data: Dict[str, Any] = {}
        for k, v in data.items():
            new_data[_swap_topic_prefix(str(k))] = v
        doc["data"] = new_data

    return doc


def _swap_xhand(doc: Dict[str, Any]) -> Dict[str, Any]:
    frames = doc.get("frames")
    if not isinstance(frames, list):
        return doc

    def _deg_list_to_rad(values: Iterable[float]) -> List[float]:
        return [float(v) * math.pi / 180.0 for v in values]

    for frame in frames:
        if not isinstance(frame, dict):
            continue

        action = frame.get("action")
        observation = frame.get("observation")

        # 1) 先把 observation 中的数值从角度制转换为弧度制
        if isinstance(observation, dict):
            for hand_key in ("left_hand", "right_hand"):
                arr = observation.get(hand_key)
                if isinstance(arr, list) and arr and all(
                    isinstance(x, (int, float)) for x in arr
                ):
                    observation[hand_key] = _deg_list_to_rad(arr)

        # 2) 再进行 action / observation 对调
        frame["action"] = observation
        frame["observation"] = action
    return doc


def _swap_action_observation(doc: Dict[str, Any]) -> Dict[str, Any]:
    if "frames" in doc:
        return _swap_xhand(doc)
    if "data" in doc and "metadata" in doc:
        return _swap_episode(doc)
    # 兜底：如果结构不匹配，就原样返回
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Swap action and observation in BSON files. "
            "默认不带参数会覆盖修改 episode_0.bson 与 xhand_control_data.bson。"
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help="输入 BSON 文件路径（可多个）。不传则默认处理 episode_0.bson 与 xhand_control_data.bson",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="不覆盖原文件，输出为 *_swapped.bson（与输入同目录）",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别：DEBUG/INFO/WARNING/ERROR",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_items = args.inputs or DEFAULT_INPUT_FILES

    for input_item in input_items:
        in_path = Path(input_item).expanduser().resolve()
        if not in_path.exists():
            LOGGER.error("文件不存在：%s", in_path)
            continue

        decoded = _decode_bson_file(in_path)
        if not isinstance(decoded, dict):
            LOGGER.error("不支持的 BSON 顶层结构（不是单文档 dict）：%s", in_path)
            continue

        swapped = _swap_action_observation(decoded)
        if args.no_overwrite:
            out_path = in_path.with_name(f"{in_path.stem}_swapped{in_path.suffix}")
        else:
            out_path = in_path

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(_encode_bson_doc(swapped))
        LOGGER.info("已输出：%s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

