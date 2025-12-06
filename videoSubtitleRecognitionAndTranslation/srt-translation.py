#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT文件翻译工具
使用免费的Google翻译接口将SRT文件翻译成双语字幕
"""

import re
import time
import requests
import json
from pathlib import Path
from typing import List, Dict, Optional
import urllib.parse


class TimeTracker:
    """耗时跟踪器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.checkpoints = {}
    
    def checkpoint(self, stage_name: str):
        """记录检查点耗时"""
        current_time = time.time()
        stage_duration = current_time - self.last_checkpoint
        total_duration = current_time - self.start_time
        
        self.checkpoints[stage_name] = {
            'stage_duration': stage_duration,
            'total_duration': total_duration
        }
        
        print(f"[耗时] [{stage_name}] 阶段耗时: {stage_duration:.2f}s, 累计耗时: {total_duration:.2f}s")
        
        # 更新最后检查点时间
        self.last_checkpoint = current_time
        
        return stage_duration, total_duration


class SRTTranslator:
    """SRT文件翻译器"""
    
    def __init__(self, source_lang: str = "ja", target_lang: str = "zh-CN"):
        """
        初始化翻译器
        
        Args:
            source_lang: 源语言代码 (ja=日语, en=英语等)
            target_lang: 目标语言代码 (zh-CN=简体中文)
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.session = requests.Session()
        
        # 设置缓存文件路径
        self.cache_dir = Path("translation_caches")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / f"translation_cache_{source_lang}_{target_lang}.json"
        self.translation_cache = self._load_cache()
        
        # 设置请求头，模拟浏览器
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _load_cache(self) -> Dict[str, str]:
        """
        加载翻译缓存
        
        Returns:
            翻译缓存字典
        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    print(f"📥 加载翻译缓存: {len(cache)} 条记录")
                    return cache
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}")
        
        return {}
    
    def _save_cache(self):
        """保存翻译缓存到文件"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)
            print(f"💾 保存翻译缓存: {len(self.translation_cache)} 条记录")
        except Exception as e:
            print(f"⚠️ 保存缓存失败: {e}")
    
    def translate_text(self, text: str, max_retries: int = 3) -> Optional[str]:
        """
        使用Google翻译接口翻译文本
        
        Args:
            text: 要翻译的文本
            max_retries: 最大重试次数
            
        Returns:
            翻译后的文本，失败返回None
        """
        if not text.strip():
            return ""
        
        # 清理文本，移除HTML标签
        clean_text = re.sub(r'<[^>]+>', '', text).strip()
        if not clean_text:
            return ""
        
        # 检查缓存
        cache_key = f"{self.source_lang}_{self.target_lang}_{clean_text}"
        if cache_key in self.translation_cache:
            cached_result = self.translation_cache[cache_key]
            print(f"📚 使用缓存翻译: '{clean_text[:50]}...' -> '{cached_result[:50]}...'")
            return cached_result
        
        for attempt in range(max_retries):
            try:
                # 使用Google翻译的免费接口
                url = f"https://translate.googleapis.com/translate_a/single"
                params = {
                    'client': 'gtx',
                    'sl': self.source_lang,
                    'tl': self.target_lang,
                    'dt': 't',
                    'q': clean_text
                }
                
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    # 解析返回的JSON数据
                    data = response.json()
                    if data and len(data) > 0:
                        # 提取翻译结果
                        translated_parts = []
                        for part in data[0]:
                            if part[0]:
                                translated_parts.append(part[0])
                        
                        if translated_parts:
                            translated_text = ' '.join(translated_parts)
                            print(f"✅ 翻译成功: '{clean_text[:50]}...' -> '{translated_text[:50]}...'")
                            
                            # 保存到缓存
                            self.translation_cache[cache_key] = translated_text
                            return translated_text
                
                # 如果失败，等待后重试
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ 翻译失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)
        
        print(f"⚠️ 无法翻译文本: '{clean_text[:100]}...'")
        return None
    
    def parse_srt(self, srt_content: str) -> List[Dict]:
        """
        解析SRT文件内容
        
        Args:
            srt_content: SRT文件内容
            
        Returns:
            字幕块列表
        """
        blocks = []
        
        # 分割字幕块（空行分隔）
        raw_blocks = re.split(r'\n\s*\n', srt_content.strip())
        
        for block in raw_blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    # 序号
                    index = int(lines[0].strip())
                    
                    # 时间戳
                    timestamp = lines[1].strip()
                    
                    # 文本内容（可能有多行）
                    text_lines = lines[2:]
                    text = '\n'.join(text_lines).strip()
                    
                    blocks.append({
                        'index': index,
                        'timestamp': timestamp,
                        'text': text,
                        'original_text': text  # 保存原始文本
                    })
                except (ValueError, IndexError):
                    # 跳过格式错误的块
                    continue
        
        return blocks
    
    def format_bilingual_subtitle(self, original_text: str, translated_text: str) -> str:
        """
        格式化双语字幕，参考JUQ-587-C.srt的格式
        
        Args:
            original_text: 原文
            translated_text: 译文
            
        Returns:
            格式化的双语字幕文本
        """
        # 清理文本，移除可能存在的HTML标签
        original_clean = re.sub(r'<[^>]+>', '', original_text).strip()
        translated_clean = re.sub(r'<[^>]+>', '', translated_text).strip() if translated_text else ""
        
        # 如果翻译失败，只显示原文
        if not translated_clean:
            return f"<font size=\"16\" color=\"#FFFFFF\">{original_clean}</font>"
        
        # 添加自动换行功能：每行最多显示指定字符数
        def add_line_breaks(text, max_chars=20):
            if not text:
                return text
            
            lines = []
            current_line = ""
            
            # 按字符逐个处理
            for char in text:
                # 如果当前行长度未超过限制，继续添加字符
                if len(current_line) < max_chars:
                    current_line += char
                else:
                    # 当前行已达到限制，添加到结果列表
                    lines.append(current_line)
                    current_line = char
            
            # 添加最后一行
            if current_line:
                lines.append(current_line)
            
            return '\\n'.join(lines)
        
        # 对原文和译文都进行换行处理
        original_with_breaks = add_line_breaks(original_clean, 20)
        translated_with_breaks = add_line_breaks(translated_clean, 25)
        
        # 根据文本长度动态调整字号
        def get_font_size_by_length(text):
            if not text:
                return 16
            
            # 计算文本总长度（不考虑换行符）
            total_length = len(text.replace("\\n", ""))
            
            # 根据长度调整字号
            if total_length <= 20:
                return 16  # 短文本使用正常字号
            elif total_length <= 40:
                return 14  # 中等长度文本稍小
            elif total_length <= 60:
                return 12  # 较长文本再小一些
            else:
                return 10  # 很长文本使用最小字号
        
        # 获取原文和译文的合适字号
        # 原文使用比译文小两号的字体
        original_font_size = get_font_size_by_length(original_clean)
        translated_font_size = get_font_size_by_length(translated_clean)
        
        # 确保原文字体比译文小两号，最小为8号字体
        original_font_size = max(8, translated_font_size - 2)
        
        # 格式化双语字幕（参考JUQ-587-C.srt格式）
        formatted = f"<font size=\"{original_font_size}\" color=\"#FFD700\">{original_with_breaks}</font>\\n"
        formatted += f"<font size=\"{translated_font_size}\" color=\"#FFFFFF\">{translated_with_breaks}</font>"
        
        return formatted
    
    def translate_srt_file(self, input_file: str, output_file: Optional[str] = None,
                          batch_size: int = 10, delay: float = 1.0) -> bool:
        """
        翻译SRT文件
        
        Args:
            input_file: 输入SRT文件路径
            output_file: 输出SRT文件路径（默认在原文件名后加_translated）
            batch_size: 批量翻译大小
            delay: 翻译间隔（秒），避免请求过快
            
        Returns:
            是否成功
        """
        # 初始化耗时跟踪器
        time_tracker = TimeTracker()
        
        try:
            # 检查输入文件
            input_path = Path(input_file)
            if not input_path.exists():
                print(f"❌ 输入文件不存在: {input_file}")
                return False
            
            # 设置输出文件
            if output_file is None:
                # 保持原文件名不变，将原文件重命名为.back.srt
                output_path = input_path
                backup_path = input_path.parent / f"{input_path.stem}.back.srt"
                
                # 如果原文件存在，先备份
                if input_path.exists():
                    import shutil
                    shutil.copy2(input_path, backup_path)
                    print(f"💾 备份原文件: {backup_path}")
            else:
                output_path = Path(output_file)
                backup_path = None
            
            time_tracker.checkpoint("文件准备")
            
            # 读取SRT文件
            print(f"📖 读取SRT文件: {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            
            time_tracker.checkpoint("文件读取")
            
            # 解析SRT文件
            blocks = self.parse_srt(srt_content)
            print(f"📊 发现 {len(blocks)} 个字幕块")
            
            time_tracker.checkpoint("SRT解析")
            
            if not blocks:
                print("❌ 未找到有效的字幕块")
                return False
            
            # 检测是否已经是双语字幕（幂等性检查）
            def is_bilingual_subtitle(text):
                """检测文本是否已经是双语字幕格式"""
                # 检查是否包含双语字幕的典型特征：font标签和换行符
                return '<font' in text and '\\n' in text
            
            # 翻译字幕块（批量处理）
            translated_blocks = []
            success_count = 0
            fail_count = 0
            already_translated_count = 0
            
            # 按batch_size分批处理
            for batch_start in range(0, len(blocks), batch_size):
                batch_end = min(batch_start + batch_size, len(blocks))
                batch_blocks = blocks[batch_start:batch_end]
                
                print(f"\n📦 处理批次 {batch_start // batch_size + 1}/{(len(blocks) - 1) // batch_size + 1} (块 {batch_start + 1}-{batch_end})")
                
                # 跟踪批次中是否使用了网络翻译
                batch_used_network = False
                
                for i, block in enumerate(batch_blocks):
                    block_index = batch_start + i
                    print(f"🔍 处理第 {block_index + 1}/{len(blocks)} 个字幕块")
                    
                    # 幂等性检查：如果已经是双语字幕，直接跳过
                    if is_bilingual_subtitle(block['text']):
                        print(f"✅ 跳过已翻译的字幕块")
                        translated_blocks.append(block)
                        already_translated_count += 1
                        continue
                    
                    # 清理文本，移除HTML标签，获取纯文本用于翻译
                    clean_text = re.sub(r'<[^>]+>', '', block['text']).strip()
                    cache_key = f"{self.source_lang}_{self.target_lang}_{clean_text}"
                    
                    # 如果在缓存中，直接使用缓存翻译
                    if cache_key in self.translation_cache:
                        translated_text = self.translation_cache[cache_key]
                        print(f"📚 使用缓存翻译: '{clean_text[:50]}...' -> '{translated_text[:50]}...'")
                        
                        # 格式化双语字幕（确保应用字符限制）
                        formatted_text = self.format_bilingual_subtitle(block['text'], translated_text)
                        
                        # 更新块内容
                        block['text'] = formatted_text
                        block['translated'] = True
                        success_count += 1
                    else:
                        # 需要网络翻译
                        batch_used_network = True
                        translated_text = self.translate_text(block['text'])
                        
                        if translated_text:
                            # 格式化双语字幕（应用字符限制）
                            formatted_text = self.format_bilingual_subtitle(block['text'], translated_text)
                            
                            # 更新块内容
                            block['text'] = formatted_text
                            block['translated'] = True
                            success_count += 1
                        else:
                            # 翻译失败，只显示原文
                            formatted_text = self.format_bilingual_subtitle(block['text'], "")
                            block['text'] = formatted_text
                            block['translated'] = False
                            fail_count += 1
                    
                    translated_blocks.append(block)
                
                # 只有在批次中使用了网络翻译时才添加延迟
                if batch_end < len(blocks) and batch_used_network:
                    print(f"⏳ 批次处理完成（使用了网络翻译），等待 {delay} 秒...")
                    time.sleep(delay)
                elif batch_end < len(blocks):
                    print(f"✅ 批次处理完成（完全使用缓存），无需等待")
            
            time_tracker.checkpoint("翻译处理")
            
            # 生成翻译后的SRT内容
            output_content = self.generate_srt_content(translated_blocks)
            
            time_tracker.checkpoint("内容生成")
            
            # 保存文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            time_tracker.checkpoint("文件保存")
            
            # 保存翻译缓存
            self._save_cache()
            
            time_tracker.checkpoint("缓存保存")
            
            # 显示总耗时统计
            total_duration = time.time() - time_tracker.start_time
            print(f"\n🎉 翻译完成!")
            print(f"✅ 成功翻译: {success_count} 个")
            print(f"❌ 翻译失败: {fail_count} 个")
            print(f"💾 已翻译跳过: {already_translated_count} 个")
            print(f"💾 输出文件: {output_path}")
            print(f"📚 缓存记录: {len(self.translation_cache)} 条")
            print(f"⏱️  总耗时: {total_duration:.2f}秒")
            
            # 显示各阶段耗时详情
            print("\n📊 各阶段耗时详情:")
            for stage, times in time_tracker.checkpoints.items():
                print(f"   {stage}: {times['stage_duration']:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ 翻译过程出错: {e}")
            return False
    
    def generate_srt_content(self, blocks: List[Dict]) -> str:
        """
        生成SRT文件内容
        
        Args:
            blocks: 字幕块列表
            
        Returns:
            SRT文件内容
        """
        content = []
        
        for i, block in enumerate(blocks):
            # 序号（从1开始连续编号）
            content.append(str(i + 1))
            
            # 时间戳
            content.append(block['timestamp'])
            
            # 文本内容
            content.append(block['text'])
            
            # 块之间用空行分隔
            content.append("")
        
        return '\n'.join(content)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SRT文件翻译工具')
    parser.add_argument('input_file', help='输入SRT文件路径')
    parser.add_argument('-o', '--output', help='输出SRT文件路径')
    parser.add_argument('--source-lang', default='ja', help='源语言代码 (默认: ja=日语)')
    parser.add_argument('--target-lang', default='zh-CN', help='目标语言代码 (默认: zh-CN=简体中文)')
    parser.add_argument('--batch-size', type=int, default=10, help='批量翻译大小 (默认: 10)')
    parser.add_argument('--delay', type=float, default=1.0, help='翻译间隔秒数 (默认: 1.0)')
    
    args = parser.parse_args()
    
    # 创建翻译器
    translator = SRTTranslator(
        source_lang=args.source_lang,
        target_lang=args.target_lang
    )
    
    # 执行翻译
    success = translator.translate_srt_file(
        input_file=args.input_file,
        output_file=args.output,
        batch_size=args.batch_size,
        delay=args.delay
    )
    
    if success:
        print("\n✨ SRT文件翻译完成!")
    else:
        print("\n💥 SRT文件翻译失败!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
