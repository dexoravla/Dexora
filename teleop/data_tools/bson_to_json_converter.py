#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BSON到JSON转换器
将指定的BSON文件转换为JSON格式并保存到当前目录
"""

import bson
import json
import os
from pathlib import Path

def bson_to_json(bson_file_path, json_file_path):
    """
    将BSON文件转换为JSON文件
    
    Args:
        bson_file_path (str): BSON文件路径
        json_file_path (str): 输出JSON文件路径
    """
    try:
        print(f"正在转换: {bson_file_path}")
        
        # 读取BSON文件
        with open(bson_file_path, 'rb') as f:
            bson_data = f.read()
        
        # 尝试直接解析整个BSON文件
        try:
            # 方法1: 尝试解析为单个BSON文档
            doc = bson.BSON(bson_data).decode()
            json_data = doc
            print("成功解析为单个BSON文档")
        except Exception as e1:
            print(f"单文档解析失败: {e1}")
            try:
                # 方法2: 尝试解析为BSON数组
                documents = []
                offset = 0
                
                while offset < len(bson_data):
                    try:
                        # 获取文档长度（前4个字节）
                        if offset + 4 > len(bson_data):
                            break
                        doc_len = int.from_bytes(bson_data[offset:offset+4], 'little')
                        
                        if doc_len <= 0 or offset + doc_len > len(bson_data):
                            break
                        
                        # 解析单个文档
                        doc_data = bson_data[offset:offset+doc_len]
                        doc = bson.BSON(doc_data).decode()
                        documents.append(doc)
                        offset += doc_len
                        
                    except Exception as e:
                        print(f"解析文档时出错: {e}")
                        break
                
                if documents:
                    json_data = documents
                    print(f"成功解析为 {len(documents)} 个BSON文档")
                else:
                    # 方法3: 尝试其他解析方式
                    print("尝试其他解析方式...")
                    # 这里可以添加其他解析逻辑
                    json_data = {"error": "无法解析BSON文件", "raw_size": len(bson_data)}
                    
            except Exception as e2:
                print(f"多文档解析失败: {e2}")
                json_data = {"error": "无法解析BSON文件", "raw_size": len(bson_data)}
        
        # 将数据转换为JSON格式并保存
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"转换完成: {json_file_path}")
        
    except Exception as e:
        print(f"转换文件 {bson_file_path} 时出错: {e}")

def main():
    """主函数"""
    # 定义要转换的文件
    files_to_convert = [
        "data_collection/action37/episode_7/episode_0.bson",
        "data_collection/action37/episode_7/xhand_control_data.bson"
    ]
    
    print("开始BSON到JSON转换...")
    print("=" * 50)
    
    for bson_file in files_to_convert:
        # 检查源文件是否存在
        if not os.path.exists(bson_file):
            print(f"错误: 文件不存在 - {bson_file}")
            continue
        
        # 生成输出文件名
        bson_path = Path(bson_file)
        json_filename = bson_path.stem + ".json"
        json_file = json_filename
        
        print(f"\n处理文件: {bson_file}")
        print(f"输出文件: {json_file}")
        
        # 执行转换
        bson_to_json(bson_file, json_file)
    
    print("\n" + "=" * 50)
    print("所有转换完成！")

if __name__ == "__main__":
    main() 