#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频翻译工具
传入视频地址，先调用whisper-transcription.py得到SRT字幕，再调用srt-translation.py得到双语字幕
"""

import os
import sys
import time
import subprocess
import hashlib
import json  # 添加json导入
from pathlib import Path
from typing import Dict, Optional
import send2trash  # 新增导入，用于将文件移动到回收站


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
    
    def print_summary(self):
        """打印耗时总结"""
        total_duration = time.time() - self.start_time
        print(f"\n[统计] 总耗时统计:")
        print(f"   总耗时: {total_duration:.2f}秒")
        print(f"\n[详情] 各阶段耗时详情:")
        for stage, times in self.checkpoints.items():
            print(f"   {stage}: {times['stage_duration']:.2f}秒")


class VideoTranslator:
    """视频翻译器"""
    
    def __init__(self, whisper_model: str = "base", device: str = "cpu", 
                 source_lang: str = "ja", target_lang: str = "zh-CN", 
                 use_vad: bool = False):
        """
        初始化视频翻译器
        
        Args:
            whisper_model: Whisper模型大小 (tiny, base, small, medium, large, large-v1, lage-v2, large-v3, large-v3-turbo, turbo)
            device: 运行设备 (cpu, cuda)
            source_lang: 源语言代码 (ja=日语, en=英语等)
            target_lang: 目标语言代码 (zh-CN=简体中文)
            use_vad: 是否使用VAD（语音活动检测）转录
        """
        self.whisper_model = whisper_model
        self.device = device
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.use_vad = use_vad
        
        # 设置工作目录
        self.workspace_dir = Path("temp")
        self.workspace_dir.mkdir(exist_ok=True)
        
        # 获取脚本所在目录
        self.script_dir = Path(__file__).parent
        # 根据use_vad选择转录脚本
        if self.use_vad:
            self.whisper_script = self.script_dir / "whisper-transcription.vad.py"
        else:
            self.whisper_script = self.script_dir / "whisper-transcription.py"
        self.srt_translation_script = self.script_dir / "srt-translation.py"
        
        # 验证脚本文件存在
        if not self.whisper_script.exists():
            raise FileNotFoundError(f"Whisper转录脚本不存在: {self.whisper_script}")
        if not self.srt_translation_script.exists():
            raise FileNotFoundError(f"SRT翻译脚本不存在: {self.srt_translation_script}")
    
    def _check_existing_transcription(self, video_path: str) -> Dict:
        """
        检查是否已有转录文件或转录状态
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            返回包含状态信息的字典
        """
        video_name = Path(video_path).stem
        temp_dir = Path("temp")
        
        if not temp_dir.exists():
            return {"should_continue": False, "reason": "temp目录不存在"}
        
        print(f"[状态检查] 检查已有转录状态...")
        print(f"   视频名称: {video_name}")
        print(f"   模型: {self.whisper_model}")
        
        # 查找匹配的状态文件
        state_files = []
        for subdir in temp_dir.iterdir():
            if subdir.is_dir() and video_name in subdir.name and self.whisper_model in subdir.name:
                state_file = subdir / "transcription_state.json"
                if state_file.exists():
                    state_files.append(state_file)
                    print(f"   ✓ 找到转录状态文件: {state_file}")
        
        if not state_files:
            return {"should_continue": False, "reason": "未找到转录状态文件"}
        
        # 使用最新的状态文件
        state_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        state_file = state_files[0]
        
        try:
            # 读取状态文件
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            processed_segments = state.get("processed_segments", 0)
            total_segments = state.get("total_segments", 0)
            segments = state.get("segments", [])
            
            print(f"   ✓ 转录进度: {processed_segments}/{total_segments} 个片段")
            print(f"   ✓ 已转录有效片段: {len(segments)} 个")
            
            # 检查是否已经完成
            if processed_segments >= total_segments:
                print(f"   ✓ 所有片段已处理完成")
                
                # 检查是否有最终的转录文本文件
                transcription_txt = state_file.parent / "transcription.txt"
                if transcription_txt.exists():
                    print(f"   ✓ 找到完整转录文件: {transcription_txt}")
                    return {
                        "completed": True,
                        "reason": "转录已完成",
                        "transcription_file": transcription_txt,
                        "state_file": state_file
                    }
                else:
                    print(f"   ℹ️ 转录已标记完成但未生成transcription.txt文件")
                    return {
                        "completed": False,
                        "reason": "转录状态异常",
                        "state_file": state_file
                    }
            else:
                print(f"   ℹ️ 发现未完成的转录工作")
                print(f"   ℹ️ 将继续从第 {processed_segments + 1} 个片段开始")
                return {
                    "should_continue": True,
                    "reason": "继续未完成的转录",
                    "current_segment": processed_segments,
                    "total_segments": total_segments,
                    "state_file": state_file
                }
                
        except Exception as e:
            print(f"   ✗ 处理状态文件时出错: {e}")
            return {"should_continue": False, "reason": f"状态文件错误: {str(e)}"}
    
    def _convert_txt_to_srt(self, txt_file: Path) -> Optional[Path]:
        """
        将whisper-transcription.py生成的txt文件转换为SRT格式
        
        Args:
            txt_file: 输入的txt文件路径
            
        Returns:
            转换后的SRT文件路径，失败返回None
        """
        try:
            # 读取txt文件内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查内容是否为空
            if not content.strip():
                print(f"   ✗ 转录文件内容为空")
                return None
            
            # 解析文件，支持两种格式：
            # 1. 带有时间戳的行：[00:01:23 - 00:01:45] 文本内容
            # 2. 直接是分段文本
            
            lines = content.split('\n')
            srt_entries = []
            entry_index = 1
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 跳过文件头信息行
                if any(line.startswith(prefix) for prefix in ['视频:', '模型:', '语言:', '音频预处理:', '转录时间:', '音频时长:', '原始片段数:', '过滤后片段数:', '去重后片段数:', '=']):
                    continue
                
                # 尝试解析时间戳行
                if line.startswith('[') and ']' in line:
                    try:
                        # 解析时间戳行，如: [00:01:23 - 00:01:45] 文本内容
                        time_part, text_part = line.split(']', 1)
                        time_part = time_part[1:]  # 去掉开头的[
                        
                        if ' - ' in time_part:
                            start_time, end_time = time_part.split(' - ', 1)
                            
                            # 将时间格式转换为SRT格式（HH:MM:SS,mmm）
                            def convert_time_format(time_str):
                                time_str = time_str.strip()
                                # 如果已经是SRT格式（有逗号），直接返回
                                if ',' in time_str:
                                    return time_str
                                # 否则添加毫秒部分
                                # 处理可能的毫秒部分（如00:01:23.456）
                                if '.' in time_str:
                                    parts = time_str.split('.')
                                    time_part = parts[0]
                                    millis = parts[1][:3].ljust(3, '0')
                                    return f"{time_part},{millis}"
                                # 没有毫秒的情况
                                return f"{time_str},000"
                            
                            srt_start = convert_time_format(start_time.strip())
                            srt_end = convert_time_format(end_time.strip())
                            
                            # 获取文本内容
                            cleaned_text = text_part.strip()
                            
                            # 移除开头的[弱]标记
                            if cleaned_text.startswith("[弱]"):
                                cleaned_text = cleaned_text[3:].strip()
                            
                            # 创建SRT条目
                            if cleaned_text:  # 确保文本不为空
                                srt_entry = f"{entry_index}\n{srt_start} --> {srt_end}\n{cleaned_text}\n"
                                srt_entries.append(srt_entry)
                                entry_index += 1
                                
                    except Exception as e:
                        print(f"   警告: 解析行失败 '{line[:50]}...': {e}")
                        continue
            
            # 如果没有找到时间戳格式，尝试其他格式
            if not srt_entries:
                # 尝试直接使用所有非空行作为字幕
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not any(line.startswith(prefix) for prefix in ['视频:', '模型:', '语言:', '音频预处理:', '转录时间:', '音频时长:']):
                        # 为每行创建简单的时间戳（每行1秒）
                        start_seconds = i - 1
                        end_seconds = i
                        
                        def seconds_to_srt(seconds):
                            hours = int(seconds // 3600)
                            minutes = int((seconds % 3600) // 60)
                            secs = int(seconds % 60)
                            millis = int((seconds - int(seconds)) * 1000)
                            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
                        
                        srt_start = seconds_to_srt(start_seconds)
                        srt_end = seconds_to_srt(end_seconds)
                        
                        srt_entry = f"{i}\n{srt_start} --> {srt_end}\n{line}\n"
                        srt_entries.append(srt_entry)
            
            # 生成SRT文件
            if srt_entries:
                srt_file = txt_file.with_suffix('.srt')
                with open(srt_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(srt_entries))
                
                print(f"   ✓ 成功转换 {len(srt_entries)} 个字幕块")
                return srt_file
            else:
                print(f"   ✗ 没有找到可转换的字幕内容")
                return None
                
        except Exception as e:
            print(f"❌ 转换txt到SRT时出错: {e}")
            return None
    
    def run_whisper_transcription(self, video_path: str, output_dir: Optional[str] = None, 
                                  enable_memory_optimization: bool = False, max_chunk_duration: int = 180, 
                                  use_vad: bool = False, test_percentage: int = 0) -> Optional[str]:
        """
        运行Whisper转录，生成SRT字幕文件
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            enable_memory_optimization: 是否启用内存优化
            max_chunk_duration: 最大分块时长（秒）
            
        Returns:
            SRT文件路径，失败返回None
        """
        try:
            # 首先检查转录状态
            status = self._check_existing_transcription(video_path)
            
            if status.get("completed"):
                print(f"[转录] ✓ 转录已完成，复用现有转录文件")
                transcription_file = Path(status["transcription_file"])
                
                # 将转录文件转换为SRT
                srt_file = self._convert_txt_to_srt(transcription_file)
                if srt_file:
                    # 移动到输出目录
                    if output_dir:
                        output_path = Path(output_dir)
                        output_path.mkdir(exist_ok=True)
                        video_name = Path(video_path).stem
                        final_srt_file = output_path / f"{video_name}.srt"
                        
                        if final_srt_file.exists():
                            final_srt_file.unlink()
                        srt_file.rename(final_srt_file)
                        print(f"[转录]   转录完成: {final_srt_file.name}")
                        return str(final_srt_file)
                    else:
                        print(f"[转录]   转录完成: {srt_file.name}")
                        return str(srt_file)
                else:
                    print(f"[转录] ✗ 转录文件转换失败，将重新转录")
            
            # 执行转录（无论是否发现未完成的工作，都运行转录脚本，它会自动从断点继续）
            print(f"[转录] 开始转录...")
            print(f"   视频文件: {Path(video_path).name}")
            print(f"   模型: {self.whisper_model}")
            print(f"   语言: {self.source_lang}")
            print(f"   分段时长: {max_chunk_duration}秒")
            print(f"   转录脚本: {'whisper-transcription.vad.py (VAD模式)' if use_vad else 'whisper-transcription.py (标准模式)'}")
            
            # 构建命令行参数
            if self.use_vad:
                cmd = [
                    sys.executable, 'whisper-transcription.vad.py',
                    video_path,
                    '--model', self.whisper_model,
                    '--language', self.source_lang
                ]
                if test_percentage > 0:
                    cmd.extend(['--test', str(test_percentage)])
            else:
                cmd = [
                    sys.executable, 'whisper-transcription.py',
                    video_path,
                    '--model', self.whisper_model,
                    '--language', self.source_lang,
                    '--segment-duration', str(max_chunk_duration)
                ]
                if test_percentage > 0:
                    cmd.extend(['--test', str(test_percentage)])
            
            # 执行转录
            print(f"[转录] 运行转录命令...")
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', cwd=self.script_dir)
            
            print(f"[转录] 转录命令执行完成，返回码: {result.returncode}")
            
            # 检查是否成功
            if result.returncode != 0:
                print(f"[转录] ✗ 转录脚本返回错误代码: {result.returncode}")
                print(f"[转录] 标准输出: {result.stdout[:500]}...")
                print(f"[转录] 错误输出: {result.stderr[:500]}...")
                
                # 检查返回码是否为内存访问冲突（常见于Windows）
                if result.returncode == 3221225620:
                    print(f"[转录] ✗ 检测到内存访问冲突，可能是内存不足或模型太大")
                    print(f"[转录] ✗ 建议: 使用更小的模型，或增加系统内存")
                
                # 即使失败，也尝试查找是否生成了部分结果
                print(f"[转录] ℹ️ 尝试查找已生成的部分转录文件...")
            
            # 转录完成后，查找结果
            video_name = Path(video_path).stem
            
            # 清理文件名
            safe_video_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in video_name)
            safe_video_name = safe_video_name[:50]
            
            # 查找转录结果文件
            transcription_txt = None
            video_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
            
            # 尝试多种目录格式
            possible_dirs = [
                Path("temp") / f"{safe_video_name}_{video_hash}_{self.whisper_model}",
                Path("temp") / f"{safe_video_name}_{self.whisper_model}",
                Path("temp") / f"{safe_video_name}__{self.whisper_model}",
            ]
            
            for temp_dir in possible_dirs:
                if temp_dir.exists():
                    candidate_txt = temp_dir / "transcription.txt"
                    if candidate_txt.exists():
                        transcription_txt = candidate_txt
                        print(f"[转录] 找到转录文件: {transcription_txt}")
                        break
            
            # 如果找不到，使用通配符搜索
            if transcription_txt is None:
                temp_dir = Path("temp")
                if temp_dir.exists():
                    pattern = f"*{safe_video_name}*{self.whisper_model}*"
                    for model_dir in temp_dir.glob(pattern):
                        if model_dir.is_dir():
                            candidate_txt = model_dir / "transcription.txt"
                            if candidate_txt.exists():
                                transcription_txt = candidate_txt
                                print(f"[转录] 找到转录文件（通配符）: {transcription_txt}")
                                break
            
            if transcription_txt is None:
                print(f"[转录] ✗ 未找到转录文件")
                
                # 检查是否转录被中断
                # 查找状态文件检查进度
                state_file = None
                for temp_dir in possible_dirs:
                    if temp_dir.exists():
                        candidate_state = temp_dir / "transcription_state.json"
                        if candidate_state.exists():
                            state_file = candidate_state
                            break
                
                if state_file:
                    try:
                        with open(state_file, 'r', encoding='utf-8') as f:
                            state = json.load(f)
                        
                        processed = state.get("processed_segments", 0)
                        total = state.get("total_segments", 0)
                        
                        if processed < total:
                            print(f"[转录] ℹ️ 转录被中断，进度: {processed}/{total}")
                            print(f"[转录] ℹ️ 请重新运行命令以继续转录")
                        elif processed >= total:
                            print(f"[转录] ℹ️ 转录已完成但未生成transcription.txt文件")
                    except:
                        pass
                
                return None
            
            # 将转录文件转换为SRT
            srt_file = self._convert_txt_to_srt(transcription_txt)
            if not srt_file:
                print(f"[转录] ✗ 转录文件转换失败")
                return None
            
            # 移动到输出目录
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
                final_srt_file = output_path / f"{video_name}.srt"
                
                if final_srt_file.exists():
                    final_srt_file.unlink()
                
                srt_file.rename(final_srt_file)
                print(f"[转录] SRT文件已保存: {final_srt_file.name}")
                return str(final_srt_file)
            else:
                print(f"[转录] SRT文件已保存: {srt_file.name}")
                return str(srt_file)
                
        except Exception as e:
            print(f"❌ 运行Whisper转录时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_srt_translation(self, srt_path: str) -> bool:
        """
        运行SRT翻译，生成双语字幕
        
        Args:
            srt_path: SRT文件路径
            
        Returns:
            是否成功
        """
        # 确保SRT文件路径是绝对路径
        srt_abs_path = Path(srt_path).absolute()
        
        # 检查SRT文件是否存在
        if not srt_abs_path.exists():
            print(f"❌ SRT文件不存在: {srt_abs_path}")
            return False
        
        # 构建SRT翻译命令
        # 不指定输出文件名，让srt-translation.py自动处理：
        # 输出文件名改为原文件名，原文名改为.back.srt
        command = [
            sys.executable, str(self.srt_translation_script),
            str(srt_abs_path),
            "--source-lang", self.source_lang,
            "--target-lang", self.target_lang
        ]
        
        print(f"[翻译] 开始SRT翻译...")
        print(f"   输入文件: {srt_abs_path.name}")
        print(f"   文件路径: {srt_abs_path}")
        print(f"   源语言: {self.source_lang}")
        print(f"   目标语言: {self.target_lang}")
        print(f"   文件名处理: 输出文件将保持原文件名，原文件将备份为.back.srt")
        
        try:
            # 运行翻译命令，实时显示输出
            result = subprocess.run(command, capture_output=False, text=True, cwd=self.script_dir)
            
            if result.returncode == 0:
                print(f"[成功] SRT翻译完成")
                return True
            else:
                print(f"[失败] SRT翻译失败")
                return False
                
        except Exception as e:
            print(f"❌ 运行SRT翻译时出错: {e}")
            return False
    
    def translate_video(self, video_path: str, output_dir: Optional[str] = None, 
                        enable_memory_optimization: bool = False, max_chunk_duration: int = 180, 
                        use_vad: bool = False, test_percentage: int = 0) -> Dict:
        """
        完整的视频翻译流程
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            enable_memory_optimization: 是否启用内存优化
            max_chunk_duration: 最大分块时长（秒）
            
        Returns:
            翻译结果信息
        """
        # 初始化耗时跟踪器
        time_tracker = TimeTracker()
        
        # 验证视频文件存在
        video_file = Path(video_path)
        if not video_file.exists():
            print(f"[失败] 视频文件不存在: {video_path}")
            return {"error": "视频文件不存在", "success": False}
        
        # 设置输出目录
        if output_dir is None:
            output_dir = self.workspace_dir
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        time_tracker.checkpoint("初始化")
        
        result = {
            "video_path": video_path,
            "output_dir": str(output_path),
            "success": False,
            "stages": {}
        }
        
        # 阶段1: Whisper转录
        print("=" * 60)
        print(f"[开始] 开始视频翻译流程")
        print(f"   视频: {video_file.name}")
        print(f"   输出目录: {output_path}")
        if use_vad:
            print(f"   转录模式: VAD（语音活动检测）模式")
        else:
            print(f"   转录模式: 标准模式")
        if test_percentage > 0:
            print(f"   测试模式: 仅处理前 {test_percentage}%")
        print("=" * 60)
        
        # 检查转录状态
        status = self._check_existing_transcription(video_path)
        if status.get("should_continue"):
            print(f"[转录] ℹ️ 发现未完成的转录工作")
            print(f"[转录] ℹ️ 将继续从第 {status['current_segment'] + 1}/{status['total_segments']} 个片段开始")
        
        srt_file = self.run_whisper_transcription(video_path, output_dir, 
                                                  enable_memory_optimization, max_chunk_duration, 
                                                  use_vad, test_percentage)
        time_tracker.checkpoint("Whisper转录")
        
        if not srt_file:
            result["error"] = "Whisper转录失败"
            time_tracker.print_summary()
            return result
        
        result["original_srt"] = srt_file
        result["stages"]["transcription"] = True
        
        # 阶段2: SRT翻译
        print("\n" + "=" * 60)
        print("[开始] 开始字幕翻译阶段")
        print("=" * 60)
        
        # 不指定输出文件名，让srt-translation.py自动处理：
        # 输出文件名改为原文件名，原文名改为.back.srt
        translation_success = self.run_srt_translation(srt_file)
        time_tracker.checkpoint("SRT翻译")
        
        if not translation_success:
            result["error"] = "SRT翻译失败"
            time_tracker.print_summary()
            return result
        
        # 翻译后的文件将保持原文件名，原文件备份为.back.srt
        result["translated_srt"] = srt_file
        result["backup_srt"] = str(Path(srt_file).parent / f"{Path(srt_file).stem}.back.srt")
        result["stages"]["translation"] = True
        result["success"] = True
        
        # 阶段3: 清理和总结
        print("\n" + "=" * 60)
        print("[完成] 视频翻译完成!")
        print("=" * 60)
        
        # 显示结果文件
        print(f"[文件] 生成的文件:")
        print(f"   双语SRT: {Path(srt_file).name} (原文件已备份为.back.srt)")
        print(f"   备份文件: {Path(srt_file).stem}.back.srt")
        
        # 打印耗时总结
        time_tracker.print_summary()
        result["processing_time"] = time.time() - time_tracker.start_time
        
        return result


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="视频翻译工具")
    parser.add_argument("video_path", help="视频文件路径")
    parser.add_argument("--model", "-m", default="base", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo"],
                       help="Whisper模型大小 (默认: base)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="运行设备 (默认: cpu)")
    parser.add_argument("--source-lang", default="ja", 
                       help="源语言代码 (默认: ja=日语)")
    parser.add_argument("--target-lang", default="zh-CN", 
                       help="目标语言代码 (默认: zh-CN=简体中文)")
    parser.add_argument("--output-dir", help="输出目录 (默认: temp)")
    
    # 内存优化参数
    parser.add_argument("--enable-memory-optimization", action="store_true",
                       help="启用内存优化模式，支持large-v2模型在16GB内存机器上运行")
    parser.add_argument("--max-chunk-duration", type=int, default=180,
                       help="内存优化模式下的最大分块时长（秒） (默认: 180)")
    
    # VAD参数
    parser.add_argument("--vad", action="store_true",
                       help="使用VAD（语音活动检测）模式进行转录，使用whisper-transcription.vad.py脚本")
    
    # 测试参数
    parser.add_argument("--test", type=int, default=0,
                       help="测试模式：仅转录前百分之N的音频 (默认: 0=禁用，10=转录前10%)")
    
    args = parser.parse_args()
    
    # 验证视频文件存在
    if not Path(args.video_path).exists():
        print(f"[失败] 视频文件不存在: {args.video_path}")
        return 1
    
    # 初始化视频翻译器
    try:
        translator = VideoTranslator(
            whisper_model=args.model,
            device=args.device,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            use_vad=args.vad
        )
    except Exception as e:
        print(f"[失败] 初始化失败: {e}")
        return 1
    
    # 执行视频翻译
    result = translator.translate_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        enable_memory_optimization=args.enable_memory_optimization,
        max_chunk_duration=args.max_chunk_duration,
        use_vad=args.vad,
        test_percentage=args.test
    )
    
    if result["success"]:
        print(f"\n[成功] 视频翻译成功完成!")
        return 0
    else:
        print(f"\n[失败] 视频翻译失败: {result.get('error', '未知错误')}")
        return 1


if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) == 1:
        print("🎬 视频翻译工具")
        print("=" * 50)
        print("使用方法:")
        print("  python video-translation.py <视频文件路径> [选项]")
        print("")
        print("选项:")
        print("  --model MODEL        Whisper模型 (tiny, base, small, medium, large, large-v1, large-v2, large-v3, large-v3-turbo, turbo)")
        print("  --device DEVICE      运行设备 (cpu, cuda)")
        print("  --source-lang LANG   源语言代码 (ja, zh, en等)")
        print("  --target-lang LANG   目标语言代码 (zh-CN, en等)")
        print("  --output-dir DIR     输出目录")
        print("")
        print("示例:")
        print("  python video-translation.py my_video.mp4 --model base --source-lang ja --target-lang zh-CN")
        print("  python video-translation.py video.avi --model large-v4 --device cuda")
        print("")
        
        # 测试示例
        test_video = input("输入测试视频路径 (或按回车跳过): ").strip()
        if test_video and Path(test_video).exists():
            print(f"\n[开始] 开始测试处理: {test_video}")
            
            translator = VideoTranslator()
            result = translator.translate_video(video_path=test_video)
        else:
            print("❌ 未提供有效的测试视频路径")
    else:
        # 正常命令行执行
        exit(main())
