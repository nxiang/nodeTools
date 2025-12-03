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
            whisper_model: Whisper模型大小 (tiny, base, small, medium, large, large-v2, large-v3)
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
    
    def run_whisper_transcription(self, video_path: str, output_dir: Optional[str] = None, enable_memory_optimization: bool = False, max_chunk_duration: int = 60) -> Optional[str]:
        """
        运行Whisper转录，生成SRT文件
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            enable_memory_optimization: 是否启用内存优化
            max_chunk_duration: 最大分块时长（秒）
            
        Returns:
            SRT文件路径，失败返回None
        """
        if output_dir is None:
            output_dir = self.workspace_dir
        
        # 构建Whisper转录命令
        command = [
            sys.executable, str(self.whisper_script),
            video_path,
            "--model", self.whisper_model,
            "--device", self.device,
            "--language", self.source_lang,
            "--output-dir", str(output_dir),
            "--clean"  # 清理临时文件
        ]
        
        # 添加内存优化参数
        if enable_memory_optimization:
            command.extend(["--enable-memory-optimization"])
            command.extend(["--max-chunk-duration", str(max_chunk_duration)])
        
        print(f"[转录] 开始Whisper转录...")
        print(f"   视频文件: {Path(video_path).name}")
        print(f"   模型: {self.whisper_model}")
        print(f"   设备: {self.device}")
        print(f"   语言: {self.source_lang}")
        if enable_memory_optimization:
            print(f"   内存优化: 已启用，分块时长: {max_chunk_duration}秒")
        
        try:
            # 运行转录命令，实时显示输出
            result = subprocess.run(command, capture_output=False, text=True, cwd=self.script_dir)
            
            # 由于capture_output=False，输出会直接显示，不需要额外处理
            
            if result.returncode == 0:
                # 转录成功，查找生成的SRT文件
                video_stem = Path(video_path).stem
                srt_file = Path(output_dir) / f"{video_stem}.srt"
                
                if srt_file.exists():
                    print(f"[成功] Whisper转录完成: {srt_file}")
                    return str(srt_file)
                else:
                    print(f"[失败] 未找到生成的SRT文件: {srt_file}")
                    return None
            else:
                print(f"[失败] Whisper转录失败 (返回码: {result.returncode}):")
                # 尝试直接运行whisper-transcription.py来调试
                debug_command = command + ["--help"]
                debug_result = subprocess.run(debug_command, capture_output=True, text=True, cwd=self.script_dir)
                if debug_result.returncode == 0:
                    print(f"   调试: whisper-transcription.py 可以正常运行")
                else:
                    print(f"   调试: whisper-transcription.py 也存在问题")
                return None
                
        except Exception as e:
            print(f"[失败] 运行Whisper转录时出错: {e}")
            return None
    
    def run_srt_translation(self, srt_path: str, output_path: Optional[str] = None) -> bool:
        """
        运行SRT翻译，生成双语字幕
        
        Args:
            srt_path: SRT文件路径
            output_path: 输出文件路径
            
        Returns:
            是否成功
        """
        # 构建SRT翻译命令
        command = [
            sys.executable, str(self.srt_translation_script),
            srt_path,
            "--source-lang", self.source_lang,
            "--target-lang", self.target_lang
        ]
        
        if output_path:
            command.extend(["-o", output_path])
        
        print(f"[翻译] 开始SRT翻译...")
        print(f"   输入文件: {Path(srt_path).name}")
        print(f"   源语言: {self.source_lang}")
        print(f"   目标语言: {self.target_lang}")
        
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
    
    def translate_video(self, video_path: str, output_dir: Optional[str] = None, enable_memory_optimization: bool = False, max_chunk_duration: int = 60) -> Dict:
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
        
        # 生成翻译后的SRT文件路径
        translated_srt = output_path / f"{video_file.stem}_translated.srt"
        
        translation_success = self.run_srt_translation(srt_file, str(translated_srt))
        time_tracker.checkpoint("SRT翻译")
        
        if not translation_success:
            result["error"] = "SRT翻译失败"
            time_tracker.print_summary()
            return result
        
        result["translated_srt"] = str(translated_srt)
        result["stages"]["translation"] = True
        result["success"] = True
        
        # 阶段3: 清理和总结
        print("\n" + "=" * 60)
        print("[完成] 视频翻译完成!")
        print("=" * 60)
        
        # 显示结果文件
        print(f"[文件] 生成的文件:")
        print(f"   原始SRT: {Path(srt_file).name}")
        print(f"   双语SRT: {translated_srt.name}")
        
        # 打印耗时总结
        time_tracker.print_summary()
        result["processing_time"] = time.time() - time_tracker.start_time
        
        return result


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="视频翻译工具")
    parser.add_argument("video_path", help="视频文件路径")
    parser.add_argument("--model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
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
    parser.add_argument("--max-chunk-duration", type=int, default=60,
                       help="内存优化模式下的最大分块时长（秒） (默认: 60)")
    
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
        print("  --model MODEL        Whisper模型 (tiny, base, small, medium, large, large-v2, large-v3)")
        print("  --device DEVICE      运行设备 (cpu, cuda)")
        print("  --source-lang LANG   源语言代码 (ja, zh, en等)")
        print("  --target-lang LANG   目标语言代码 (zh-CN, en等)")
        print("  --output-dir DIR     输出目录")
        print("")
        print("示例:")
        print("  python video-translation.py my_video.mp4 --model base --source-lang ja --target-lang zh-CN")
        print("  python video-translation.py video.avi --model large-v3 --device cuda")
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
