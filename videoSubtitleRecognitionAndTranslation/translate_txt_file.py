#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
转录文件翻译工具
将whisper-transcription.py生成的txt文件转换为SRT格式并进行翻译
"""

import sys
import os
import re
from pathlib import Path


def convert_txt_to_srt(txt_file: Path) -> Path:
    """
    将txt文件转换为SRT格式
    
    Args:
        txt_file: 输入的txt文件路径
        
    Returns:
        转换后的SRT文件路径
    """
    try:
        # 读取txt文件内容
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过文件头信息（前几行）
        content_lines = []
        for line in lines:
            if line.strip() and '=' not in line and not line.startswith('视频:') and not line.startswith('模型:'):
                content_lines.append(line.strip())
        
        # 解析时间戳和文本
        srt_entries = []
        entry_index = 1
        
        for line in content_lines:
            if line.startswith('[') and ']' in line:
                # 解析时间戳行，如: [00:01:23 - 00:01:45] 文本内容
                time_part, text_part = line.split(']', 1)
                time_part = time_part[1:]  # 去掉开头的[
                
                if ' - ' in time_part:
                    start_time, end_time = time_part.split(' - ', 1)
                    
                    # 将时间格式转换为SRT格式（HH:MM:SS,mmm）
                    def convert_time_format(time_str):
                        # transcription.txt文件中的时间戳已经是正确格式：00:00:02,719
                        # 直接返回原始时间戳，不需要额外转换
                        return time_str.strip()
                    
                    srt_start = convert_time_format(start_time.strip())
                    srt_end = convert_time_format(end_time.strip())
                    
                    # 创建SRT条目
                    srt_entry = f"{entry_index}\n{srt_start} --> {srt_end}\n{text_part.strip()}\n"
                    srt_entries.append(srt_entry)
                    entry_index += 1
        
        # 生成SRT文件
        srt_file = txt_file.with_suffix('.srt')
        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_entries))
        
        print(f"✅ 成功将 {txt_file.name} 转换为 {srt_file.name}")
        print(f"   发现 {len(srt_entries)} 个字幕块")
        
        return srt_file
        
    except Exception as e:
        print(f"❌ 转换txt到SRT时出错: {e}")
        raise


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python translate_txt_file.py <txt文件路径>")
        print("示例: python translate_txt_file.py temp/视频名称/transcription.txt")
        return 1
    
    txt_file_path = Path(sys.argv[1])
    
    if not txt_file_path.exists():
        print(f"❌ 文件不存在: {txt_file_path}")
        return 1
    
    if txt_file_path.suffix.lower() != '.txt':
        print(f"❌ 文件必须是txt格式: {txt_file_path}")
        return 1
    
    print(f"📖 开始处理转录文件: {txt_file_path}")
    
    try:
        # 步骤1: 转换为SRT格式
        print("\n" + "=" * 60)
        print("[阶段1] 转换txt到SRT格式")
        print("=" * 60)
        
        srt_file = convert_txt_to_srt(txt_file_path)
        
        # 步骤2: 使用srt-translation.py进行翻译
        print("\n" + "=" * 60)
        print("[阶段2] 开始SRT翻译")
        print("=" * 60)
        
        # 构建翻译命令
        script_dir = Path(__file__).parent
        srt_translation_script = script_dir / "srt-translation.py"
        
        command = [
            sys.executable, str(srt_translation_script),
            str(srt_file),
            "--source-lang", "ja",
            "--target-lang", "zh-CN"
        ]
        
        print(f"   输入文件: {srt_file.name}")
        print(f"   源语言: ja (日语)")
        print(f"   目标语言: zh-CN (简体中文)")
        print(f"   文件名处理: 输出文件将保持原文件名，原文件将备份为.back.srt")
        
        # 运行翻译命令
        import subprocess
        result = subprocess.run(command, capture_output=False, text=True, cwd=script_dir)
        
        if result.returncode == 0:
            print("\n✨ 翻译完成!")
            print(f"✅ 翻译后的文件: {srt_file}")
            print(f"💾 原文件备份: {srt_file.parent / f'{srt_file.stem}.back.srt'}")
            return 0
        else:
            print("\n💥 翻译失败!")
            return 1
            
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
