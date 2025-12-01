#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新翻译工具
用于批量重新翻译temp目录下srt文件中翻译失败的条目
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入项目中的翻译模块
from translator import (
    set_current_video_name, 
    load_translation_cache, 
    save_translation_cache,
    batch_translate
)
from config import validate_config

def parse_srt_file(file_path):
    """
    解析SRT文件，提取所有字幕条目
    
    Args:
        file_path: SRT文件路径
        
    Returns:
        list: 字幕条目列表，每个条目包含index, timestamp, original_text, translated_text
    """
    subtitles = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 按字幕块分割内容
        blocks = re.split(r'\n\n+', content.strip())
        
        for block in blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
                
            # 解析字幕索引
            try:
                index = int(lines[0])
            except ValueError:
                continue
                
            # 解析时间戳
            timestamp_line = lines[1]
            if ' --> ' not in timestamp_line:
                continue
                
            # 解析文本内容
            text_lines = lines[2:]
            original_text = ""
            translated_text = ""
            
            for line in text_lines:
                # 提取日文原文（黄色字体）
                yellow_match = re.search(r'<font size="12" color="#FFD700">(.*?)</font>', line)
                if yellow_match:
                    original_text = yellow_match.group(1)
                
                # 提取中文翻译（白色字体）
                white_match = re.search(r'<font size="16" color="#FFFFFF">(.*?)</font>', line)
                if white_match:
                    translated_text = white_match.group(1)
            
            subtitles.append({
                'index': index,
                'timestamp': timestamp_line,
                'original_text': original_text,
                'translated_text': translated_text,
                'needs_translation': translated_text == original_text or translated_text.startswith('[翻译失败]') or not translated_text
            })
            
    except Exception as e:
        print(f"❌ 解析SRT文件失败 {file_path}: {e}")
        
    return subtitles

def format_subtitle_block(index, timestamp, original_text, translated_text):
    """
    格式化字幕块
    
    Args:
        index: 字幕索引
        timestamp: 时间戳
        original_text: 日文原文
        translated_text: 中文翻译
        
    Returns:
        str: 格式化的字幕块
    """
    return f"{index}\n{timestamp}\n<font size=\"12\" color=\"#FFD700\">{original_text}</font>\n<font size=\"16\" color=\"#FFFFFF\">{translated_text}</font>"

def retranslate_srt_file(file_path):
    """
    重新翻译SRT文件中失败的条目
    
    Args:
        file_path: SRT文件路径
        
    Returns:
        bool: 是否成功重新翻译
    """
    print(f"📄 处理文件: {file_path}")
    
    # 设置当前视频名称
    video_name = Path(file_path).stem
    set_current_video_name(file_path)
    
    # 加载翻译缓存
    load_translation_cache()
    
    # 解析SRT文件
    subtitles = parse_srt_file(file_path)
    if not subtitles:
        print(f"⚠️  未找到字幕条目: {file_path}")
        return False
    
    print(f"📊 总共找到 {len(subtitles)} 个字幕条目")
    
    # 收集需要重新翻译的文本
    need_translate = [sub for sub in subtitles if sub['needs_translation']]
    if not need_translate:
        print(f"✅ 文件 {file_path} 中没有需要重新翻译的条目")
        return True
    
    print(f"🔄 需要重新翻译 {len(need_translate)} 个条目")
    
    # 提取需要翻译的原文
    texts_to_translate = [sub['original_text'] for sub in need_translate]
    
    # 执行批量翻译（使用batch_translate方法）
    print(f"🌐 开始批量翻译...")
    print(f"📋 示例文本: {texts_to_translate[:3]}")
    print(f"📊 待翻译文本总数: {len(texts_to_translate)}")
    
    # 直接调用batch_translate方法
    translated_results = batch_translate(
        texts_to_translate, 
        adult_content=False, 
        show_individual_logs=True  # 启用详细日志
    )
    
    print(f"📊 翻译完成，总共: {len(translated_results)} 条结果")
    
    # 更新翻译结果
    success_count = 0
    failed_count = 0
    manual_translation_count = 0
    
    # 统计信息
    failed_texts = []
    
    for i, sub in enumerate(need_translate):
        if i < len(translated_results):
            translated_text = translated_results[i]
            
            # 清理翻译结果
            translated_text = translated_text.strip()
            original_text = sub['original_text']
            
            # 检查是否需要手动翻译简单的日期和数字
            manual_translated = False
            if original_text.endswith('日目'):
                # 处理日期格式
                match = re.match(r'(\d+)日目', original_text)
                if match:
                    day = match.group(1)
                    translated_text = f"第{day}天"
                    manual_translated = True
                    manual_translation_count += 1
                    if manual_translation_count <= 5:  # 只显示前5个手动翻译示例
                        print(f"🔧 手动翻译日期: {original_text} -> {translated_text}")
            elif len(original_text) <= 3 and original_text.isdigit():
                # 处理纯数字
                translated_text = original_text
                manual_translated = True
                manual_translation_count += 1
                if manual_translation_count <= 3:  # 只显示前3个数字保留示例
                    print(f"🔧 保留数字: {original_text}")
            
            # 更新字幕条目的翻译
            sub['translated_text'] = translated_text
            
            # 检查是否翻译成功
            if translated_text != original_text and not translated_text.startswith('[翻译失败]') and translated_text:
                success_count += 1
                # 只显示前10个成功翻译示例，避免日志过多
                if success_count <= 10:
                    print(f"✅ 已更新翻译: {original_text[:20]}{'...' if len(original_text) > 20 else ''} -> {translated_text[:20]}{'...' if len(translated_text) > 20 else ''}")
            else:
                failed_count += 1
                failed_texts.append(original_text)
                # 只显示前5个失败翻译示例，避免日志过多
                if failed_count <= 5:
                    print(f"⚠️  翻译仍失败: {original_text[:20]}{'...' if len(original_text) > 20 else ''}")
    
    # 显示统计信息
    print(f"📊 翻译更新完成: 成功 {success_count}/{len(need_translate)}")
    if manual_translation_count > 0:
        print(f"🔧 手动翻译处理: {manual_translation_count} 个")
    if failed_count > 0:
        print(f"⚠️  翻译失败: {failed_count} 个")
        if len(failed_texts) > 0:
            print(f"📋 失败文本示例: {failed_texts[:3]}")
    
    # 保存翻译缓存
    save_translation_cache()
    
    # 重新生成SRT文件内容
    new_content = []
    for sub in subtitles:
        block = format_subtitle_block(
            sub['index'],
            sub['timestamp'],
            sub['original_text'],
            sub['translated_text']
        )
        new_content.append(block)
    
    # 写入文件（添加时间戳备份）
    backup_path = f"{file_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        # 创建备份
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(file_path, 'r', encoding='utf-8') as original:
                f.write(original.read())
        print(f"💾 已创建备份文件: {backup_path}")
        
        # 写入更新后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(new_content))
        
        print(f"✅ 文件已更新: {file_path}")
        print(f"📊 翻译统计: 成功 {success_count}/{len(need_translate)}")
        return True
        
    except Exception as e:
        print(f"❌ 写入文件失败 {file_path}: {e}")
        return False

def clear_error_cache(video_name):
    """清除错误的翻译缓存"""
    print(f"🧹 开始清除错误缓存: {video_name}")
    
    # 导入translator模块
    from translator import (
        set_current_video_name,
        get_video_specific_cache_file,
        load_translation_cache,
        save_translation_cache
    )
    
    # 设置当前视频名称，确保能正确获取缓存文件
    set_current_video_name(video_name)
    
    # 使用translator模块的方法获取缓存文件
    cache_file = get_video_specific_cache_file()
    print(f"📁 缓存文件路径: {cache_file}")
    
    if os.path.exists(cache_file):
        # 加载现有缓存
        cache_data = load_translation_cache()
        print(f"📊 当前缓存条目数: {len(cache_data)}")
        
        # 统计错误缓存条目
        error_count = 0
        keys_to_remove = []
        error_examples = []  # 存储错误缓存示例
        
        for key, value in cache_data.items():
            # 检查是否是错误的缓存（原文等于译文）
            if isinstance(value, str):
                # 从缓存键中提取原文
                # 缓存键格式: "jp:zh:原文"
                if key.startswith("jp:zh:"):
                    original_text = key[6:]  # 去掉"jp:zh:"前缀
                    if value == original_text:
                        keys_to_remove.append(key)
                        error_count += 1
                        # 只记录前3个错误示例
                        if error_count <= 3:
                            error_examples.append(f"{original_text[:20]}{'...' if len(original_text) > 20 else ''} -> {value[:20]}{'...' if len(value) > 20 else ''}")
            elif isinstance(value, dict):
                # 处理字典格式的缓存
                if 'result' in value and value['result'] == key[6:]:
                    keys_to_remove.append(key)
                    error_count += 1
                    # 只记录前3个错误示例
                    if error_count <= 3:
                        error_examples.append(f"{key[6:][:20]}{'...' if len(key[6:]) > 20 else ''} -> {value['result'][:20]}{'...' if len(value['result']) > 20 else ''}")
        
        # 显示错误缓存示例
        if error_examples:
            print(f"❌ 发现错误缓存示例:")
            for example in error_examples:
                print(f"  - {example}")
        
        # 移除错误缓存
        for key in keys_to_remove:
            del cache_data[key]
        
        # 保存清理后的缓存
        save_translation_cache(cache_data)
        print(f"✅ 已清除 {error_count} 个错误缓存条目")
        return error_count
    else:
        print("ℹ️  未找到缓存文件，无需清理")
        return 0

def main():
    """
    主函数
    """
    print(f"🚀 开始重新翻译任务")
    print(f"🕒 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 验证配置
    errors = validate_config()
    if errors:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        print("请先配置百度翻译API的appid和key")
        return 1
    
    # 先清除错误缓存
    import json
    
    # 获取temp目录
    temp_dir = Path("temp")
    if not temp_dir.exists():
        print(f"❌ temp目录不存在: {temp_dir}")
        return 1
    
    # 查找所有SRT文件
    srt_files = list(temp_dir.glob("*.srt"))
    if not srt_files:
        print(f"❌ 在 {temp_dir} 目录下未找到SRT文件")
        return 1
    
    print(f"📁 找到 {len(srt_files)} 个SRT文件")
    
    # 为每个视频清除错误缓存
    total_error_count = 0
    for srt_file in srt_files:
        video_name = Path(srt_file).stem
        error_count = clear_error_cache(video_name)
        total_error_count += error_count
    
    print(f"📊 缓存清理完成: 总共清除 {total_error_count} 个错误缓存条目")
    
    # 处理每个SRT文件
    success_count = 0
    file_results = []
    
    for i, srt_file in enumerate(srt_files):
        print(f"\n{'-' * 50}")
        print(f"📄 处理文件 ({i+1}/{len(srt_files)}): {srt_file.name}")
        
        if retranslate_srt_file(srt_file):
            success_count += 1
            file_results.append(f"✅ {srt_file.name}")
        else:
            file_results.append(f"❌ {srt_file.name}")
    
    # 打印汇总信息
    print(f"\n{'-' * 50}")
    print(f"📊 任务完成")
    print(f"✅ 成功处理: {success_count}/{len(srt_files)}")
    
    # 显示文件处理结果摘要
    if len(file_results) <= 10:
        print(f"📋 文件处理结果:")
        for result in file_results:
            print(f"  {result}")
    else:
        print(f"📋 文件处理结果摘要 (显示前10个):")
        for result in file_results[:10]:
            print(f"  {result}")
        print(f"  ... 还有 {len(file_results) - 10} 个文件")
    
    print(f"🕒 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
