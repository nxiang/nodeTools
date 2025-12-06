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
from pathlib import Path
from typing import Dict, Optional


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
                 source_lang: str = "ja", target_lang: str = "zh-CN"):
        """
        初始化视频翻译器
        
        Args:
            whisper_model: Whisper模型大小 (tiny, base, small, medium, large, large-v1, lage-v2, large-v3, larrge-v3-turbo, turbo)
            device: 运行设备 (cpu, cuda)
            source_lang: 源语言代码 (ja=日语, en=英语等)
            target_lang: 目标语言代码 (zh-CN=简体中文)
        """
        self.whisper_model = whisper_model
        self.device = device
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # 设置工作目录
        self.workspace_dir = Path("temp")
        self.workspace_dir.mkdir(exist_ok=True)
        
        # 获取脚本所在目录
        self.script_dir = Path(__file__).parent
        self.whisper_script = self.script_dir / "whisper-transcription.py"
        self.srt_translation_script = self.script_dir / "srt-translation.py"
        
        # 验证脚本文件存在
        if not self.whisper_script.exists():
            raise FileNotFoundError(f"Whisper转录脚本不存在: {self.whisper_script}")
        if not self.srt_translation_script.exists():
            raise FileNotFoundError(f"SRT翻译脚本不存在: {self.srt_translation_script}")
    
    def _check_existing_transcription(self, video_path: str) -> Optional[Path]:
        """
        检查是否已有转录文件可以复用
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            如果找到可复用的转录文件，返回SRT文件路径，否则返回None
        """
        video_name = Path(video_path).stem
        temp_dir = Path("temp")
        
        if not temp_dir.exists():
            return None
        
        print(f"[复用检查] 检查已有转录文件...")
        print(f"   视频名称: {video_name}")
        print(f"   模型: {self.whisper_model}")
        
        # 查找匹配的转录文件
        txt_files = []
        for subdir in temp_dir.iterdir():
            if subdir.is_dir() and video_name in subdir.name and self.whisper_model in subdir.name:
                txt_file = subdir / "transcription.txt"
                if txt_file.exists():
                    txt_files.append(txt_file)
                    print(f"   ✓ 找到转录文件: {txt_file}")
                    break  # 找到第一个匹配的就退出
        
        if txt_files:
            txt_file = txt_files[0]
            
            # 检查转录文件是否完整（有足够的内容）
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否有实际转录内容（排除文件头信息）
                lines_with_content = [line for line in content.split('\n') 
                                    if line.strip() and '=' not in line 
                                    and not line.startswith('视频:') 
                                    and not line.startswith('模型:') 
                                    and line.startswith('[') and ']' in line]
                
                if len(lines_with_content) > 0:
                    print(f"   ✓ 转录文件包含 {len(lines_with_content)} 个有效字幕块")
                    
                    # 将txt文件转换为SRT格式
                    srt_file = self._convert_txt_to_srt(txt_file)
                    if srt_file:
                        print(f"   ✓ 成功转换为SRT格式: {srt_file.name}")
                        return srt_file
                    else:
                        print(f"   ✗ SRT转换失败")
                        return None
                else:
                    print(f"   ✗ 转录文件为空或格式不正确")
                    return None
                    
            except Exception as e:
                print(f"   ✗ 检查转录文件时出错: {e}")
                return None
        
        print(f"   ✗ 未找到可复用的转录文件")
        return None
    
    def run_whisper_transcription(self, video_path: str, output_dir: Optional[str] = None, enable_memory_optimization: bool = False, max_chunk_duration: int = 180) -> Optional[str]:
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
            # 首先检查是否有可复用的转录文件
            existing_srt = self._check_existing_transcription(video_path)
            if existing_srt:
                print(f"[复用] ✓ 复用已有转录文件，跳过转录过程")
                
                # 将SRT文件移动到输出目录
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(exist_ok=True)
                    video_name = Path(video_path).stem
                    final_srt_file = output_path / f"{video_name}.srt"
                    existing_srt.rename(final_srt_file)
                    print(f"[复用] 转录完成: {final_srt_file.name}")
                    return str(final_srt_file)
                else:
                    print(f"[复用] 转录完成: {existing_srt.name}")
                    return str(existing_srt)
            
            # 如果没有可复用的文件，执行转录
            print(f"[Whisper] 开始转录...")
            print(f"   视频文件: {Path(video_path).name}")
            print(f"   模型: {self.whisper_model}")
            print(f"   语言: {self.source_lang}")
            print(f"   分段时长: {max_chunk_duration}秒")
            
            # 构建命令行参数 - 只传递whisper-transcription.py支持的参数
            cmd = [
                sys.executable, 'whisper-transcription.py',
                video_path,
                '--model', self.whisper_model,
                '--language', self.source_lang,
                '--segment-duration', str(max_chunk_duration)
            ]
            
            # 执行转录，实时显示输出
            result = subprocess.run(cmd, capture_output=False, text=True, encoding='utf-8', cwd=self.script_dir)
            
            if result.returncode == 0:
                # whisper-transcription.py生成的文件路径格式: temp/{video_name}_{hash}_{model}/transcription.txt
                video_name = Path(video_path).stem
                
                # 查找temp目录下的转录文件
                temp_dir = Path("temp")
                if not temp_dir.exists():
                    print(f"[Whisper] temp目录不存在: {temp_dir}")
                    return None
                
                # 查找匹配的转录文件
                # 由于文件名可能包含特殊字符（如方括号），使用更安全的搜索方式
                txt_files = []
                for subdir in temp_dir.iterdir():
                    if subdir.is_dir() and video_name in subdir.name and self.whisper_model in subdir.name:
                        txt_file = subdir / "transcription.txt"
                        if txt_file.exists():
                            txt_files.append(txt_file)
                            break  # 找到第一个匹配的就退出
                
                if txt_files:
                    txt_file = txt_files[0]  # 取第一个匹配的文件
                    
                    # 将txt文件转换为SRT格式
                    srt_file = self._convert_txt_to_srt(txt_file)
                    if srt_file:
                        # 将SRT文件移动到输出目录
                        if output_dir:
                            output_path = Path(output_dir)
                            output_path.mkdir(exist_ok=True)
                            final_srt_file = output_path / f"{video_name}.srt"
                            srt_file.rename(final_srt_file)
                            print(f"[Whisper] 转录完成: {final_srt_file.name}")
                            return str(final_srt_file)
                        else:
                            print(f"[Whisper] 转录完成: {srt_file.name}")
                            return str(srt_file)
                    else:
                        print(f"[Whisper] SRT文件转换失败")
                        return None
                else:
                    print(f"[Whisper] 转录文件未找到")
                    return None
            else:
                # 由于capture_output=False，stderr不会被捕获，显示通用错误信息
                print(f"[Whisper] 转录过程出现错误，但可能已有部分转录内容")
                print(f"[Whisper] 返回码: {result.returncode}")
                
                # 检查是否已经生成了部分转录文件
                video_name = Path(video_path).stem
                temp_dir = Path("temp")
                
                # 由于文件名可能包含特殊字符（如方括号），使用更安全的搜索方式
                # 先找到所有包含视频名称的目录，然后在这些目录中查找transcription.txt
                txt_files = []
                for subdir in temp_dir.iterdir():
                    if subdir.is_dir() and video_name in subdir.name:
                        txt_file = subdir / "transcription.txt"
                        if txt_file.exists():
                            txt_files.append(txt_file)
                            break  # 找到第一个匹配的就退出
                
                if txt_files:
                    print(f"[Whisper] 发现部分转录文件，可能仍有可用内容")
                    txt_file = txt_files[0]
                    srt_file = self._convert_txt_to_srt(txt_file)
                    if srt_file:
                        # 重命名为partial.srt以表示部分转录
                        partial_srt = srt_file.with_name(f"{video_name}_partial.srt")
                        srt_file.rename(partial_srt)
                        print(f"[Whisper] 部分转录转换完成: {partial_srt.name}")
                        
                        # 将SRT文件移动到输出目录
                        if output_dir:
                            output_path = Path(output_dir)
                            output_path.mkdir(exist_ok=True)
                            final_srt_file = output_path / f"{video_name}_partial.srt"
                            partial_srt.rename(final_srt_file)
                            print(f"[Whisper] 已生成部分转录文件: {final_srt_file.name}")
                        else:
                            print(f"[Whisper] 已生成部分转录文件: {partial_srt.name}")
                            return str(partial_srt)
                    else:
                        print("[Whisper] 部分转录转换失败")
                        return None
                else:
                    print(f"[Whisper] 转录失败，返回码: {result.returncode}")
                    return None
                
        except Exception as e:
            print(f"❌ 运行Whisper转录时出错: {e}")
            return None
    
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
                            # 格式: HH:MM:SS -> HH:MM:SS,000
                            parts = time_str.split(':')
                            if len(parts) == 3:
                                return f"{parts[0]}:{parts[1]}:{parts[2]},000"
                            return time_str
                        
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
            
            return srt_file
            
        except Exception as e:
            print(f"❌ 转换txt到SRT时出错: {e}")
            return None
    
    def run_srt_translation(self, srt_path: str) -> bool:
        """
        运行SRT翻译，生成双语字幕
        
        Args:
            srt_path: SRT文件路径
            
        Returns:
            是否成功
        """
        # 构建SRT翻译命令
        # 不指定输出文件名，让srt-translation.py自动处理：
        # 输出文件名改为原文件名，原文名改为.back.srt
        command = [
            sys.executable, str(self.srt_translation_script),
            srt_path,
            "--source-lang", self.source_lang,
            "--target-lang", self.target_lang
        ]
        
        print(f"[翻译] 开始SRT翻译...")
        print(f"   输入文件: {Path(srt_path).name}")
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
    
    def translate_video(self, video_path: str, output_dir: Optional[str] = None, enable_memory_optimization: bool = False, max_chunk_duration: int = 180) -> Dict:
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
        if enable_memory_optimization:
            print(f"   内存优化: 已启用，分块时长: {max_chunk_duration}秒")
        print("=" * 60)
        
        srt_file = self.run_whisper_transcription(video_path, output_dir, enable_memory_optimization, max_chunk_duration)
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
            target_lang=args.target_lang
        )
    except Exception as e:
        print(f"[失败] 初始化失败: {e}")
        return 1
    
    # 执行视频翻译
    result = translator.translate_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        enable_memory_optimization=args.enable_memory_optimization,
        max_chunk_duration=args.max_chunk_duration
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
