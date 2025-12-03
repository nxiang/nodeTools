import whisper
import json
import os
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import librosa
import gc
import psutil
import subprocess
import tempfile
import math
import wave
import struct
from datetime import timedelta

warnings.filterwarnings("ignore")


class TimeTracker:
    """耗时跟踪器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.checkpoints = {}
    
    def checkpoint(self, stage_name: str):
        """记录阶段耗时"""
        current_time = time.time()
        stage_duration = current_time - self.last_checkpoint
        total_duration = current_time - self.start_time
        
        self.checkpoints[stage_name] = {
            'stage_duration': stage_duration,
            'total_duration': total_duration
        }
        
        # 实时显示耗时
        print(f"[耗时] [{stage_name}] 阶段耗时: {stage_duration:.2f}s | 累计耗时: {total_duration:.2f}s")
        
        self.last_checkpoint = current_time
        
    def print_summary(self):
        """打印耗时总结"""
        total_time = time.time() - self.start_time
        print(f"\n[统计] 耗时统计总结:")
        print(f"总耗时: {total_time:.2f}秒")
        print("各阶段耗时详情:")
        for stage, times in self.checkpoints.items():
            print(f"  {stage}: {times['stage_duration']:.2f}s ({times['stage_duration']/total_time*100:.1f}%)")
        print("=" * 50)


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_usage_mb(self) -> float:
        """获取当前进程内存使用（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_available_memory_mb(self) -> float:
        """获取系统可用内存（MB）"""
        return psutil.virtual_memory().available / 1024 / 1024
    
    def get_total_memory_mb(self) -> float:
        """获取系统总内存（MB）"""
        return psutil.virtual_memory().total / 1024 / 1024
    
    def log_memory_usage(self, stage: str = ""):
        """记录内存使用情况"""
        used = self.get_memory_usage_mb()
        available = self.get_available_memory_mb()
        total = self.get_total_memory_mb()
        
        print(f"内存使用[{stage}]: 进程{used:.0f}MB, "
              f"系统可用{available:.0f}MB/{total:.0f}MB")


class MemoryEfficientWhisper:
    """
    内存高效的Whisper转录器，专为有限内存环境设计
    支持large-v2模型在16GB内存机器上运行
    """
    
    def __init__(self, model_or_size="large-v2", device="cpu", 
                 max_chunk_duration=60,  # 默认60秒一块，防止内存溢出
                 checkpoint_dir="memory_safe_checkpoints"):
        """
        初始化内存高效转录器
        
        Args:
            model_or_size: 模型大小或已加载的模型对象，对于16GB内存推荐"large-v2"
            device: 运行设备，16GB内存建议用cpu
            max_chunk_duration: 最大分块时长（秒），根据内存调整
            checkpoint_dir: 检查点保存目录
        """
        # 检查传入的是模型对象还是模型名称
        if isinstance(model_or_size, str):
            # 传入的是模型名称，需要加载模型
            print(f"加载 {model_or_size} 模型...")
            self.model = whisper.load_model(model_or_size, device=device)
        else:
            # 传入的是已加载的模型对象，直接使用
            print(f"内存优化转录器重用已加载的模型")
            self.model = model_or_size
            
        self.device = device
        self.max_chunk_duration = max_chunk_duration
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 监控内存使用
        self.memory_monitor = MemoryMonitor()
        
        print(f"内存优化转录器初始化完成。设备: {device}")
        print(f"最大分块时长: {max_chunk_duration}秒")
        
    def get_checkpoint_path(self, audio_path: str) -> Path:
        """获取检查点文件路径"""
        audio_name = Path(audio_path).stem
        return self.checkpoint_dir / f"{audio_name}_memory_safe_checkpoint.json"
    
    def optimize_chunk_size(self, audio_duration: float, available_memory_mb: float) -> int:
        """
        根据音频长度和可用内存动态调整分块大小
        
        Args:
            audio_duration: 音频总时长（秒）
            available_memory_mb: 可用内存（MB）
            
        Returns:
            优化的分块时长（秒）
        """
        # large-v2模型本身需要约4000MB
        model_memory = 4000
        
        # 每60秒音频大约需要100MB处理内存
        memory_per_minute = 100 * 1024 / 60  # 转换为每秒
        
        # 计算安全内存
        safe_memory = available_memory_mb - model_memory - 2000  # 保留2GB给系统
        
        if safe_memory <= 0:
            return 30  # 最小分块
        
        # 计算理论最大分块时长
        max_possible = safe_memory / memory_per_minute
        
        # 限制在合理范围内
        chunk_size = min(self.max_chunk_duration, max_possible)
        chunk_size = max(30, chunk_size)  # 至少30秒
        
        # 如果音频很短，不需要分块
        if audio_duration <= 120:  # 2分钟以内
            chunk_size = audio_duration
        
        print(f"内存优化: 可用{available_memory_mb:.0f}MB, "
              f"建议分块{chunk_size:.0f}秒")
        
        return int(chunk_size)
    
    def transcribe_with_memory_safety(self, audio_path: str, **transcribe_kwargs) -> Dict:
        """
        内存安全的转录方法，支持断点续传
        
        Args:
            audio_path: 音频文件路径
            **transcribe_kwargs: Whisper转录参数
            
        Returns:
            转录结果
        """
        print(f"\n开始内存安全转录: {Path(audio_path).name}")
        
        # 初始化耗时跟踪器
        time_tracker = TimeTracker()
        time_tracker.checkpoint("初始化")
        
        # 获取音频信息
        audio_duration = self.get_audio_duration(audio_path)
        available_memory = self.memory_monitor.get_available_memory_mb()
        time_tracker.checkpoint("音频信息获取")
        
        # 优化分块大小
        chunk_duration = self.optimize_chunk_size(audio_duration, available_memory)
        time_tracker.checkpoint("分块优化")
        
        # 检查点路径
        checkpoint_path = self.get_checkpoint_path(audio_path)
        
        # 尝试从检查点恢复
        if checkpoint_path.exists():
            print(f"发现内存安全检查点，尝试恢复...")
            result = self.resume_from_checkpoint(audio_path, checkpoint_path, 
                                              chunk_duration, **transcribe_kwargs)
        else:
            print(f"开始新的内存安全转录，总时长: {audio_duration:.1f}秒")
            result = self.new_transcription(audio_path, checkpoint_path,
                                         chunk_duration, **transcribe_kwargs)
        
        # 打印耗时总结
        time_tracker.checkpoint("转录完成")
        time_tracker.print_summary()
        
        return result
    
    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration
        except:
            # 备用方法
            audio, sr = librosa.load(audio_path, sr=None)
            return len(audio) / sr
    
    def new_transcription(self, audio_path: str, checkpoint_path: Path,
                         chunk_duration: int, **transcribe_kwargs) -> Dict:
        """开始新的转录"""
        # 初始化检查点
        checkpoint = {
            "audio_path": audio_path,
            "chunk_duration": chunk_duration,
            "processed_chunks": [],
            "results": [],
            "current_chunk": 0,
            "total_chunks": 0,
            "transcribe_kwargs": transcribe_kwargs,
            "start_time": time.time()
        }
        
        # 初始化耗时跟踪器
        time_tracker = TimeTracker()
        time_tracker.checkpoint("检查点初始化")
        
        # 加载音频
        print("加载音频文件...")
        audio, sr = librosa.load(audio_path, sr=16000)
        total_samples = len(audio)
        time_tracker.checkpoint("音频加载")
        
        # 计算分块
        chunk_samples = chunk_duration * sr
        total_chunks = int(np.ceil(total_samples / chunk_samples))
        checkpoint["total_chunks"] = total_chunks
        
        print(f"音频总长度: {total_samples/sr:.1f}秒")
        print(f"分块数量: {total_chunks} (每块{chunk_duration}秒)")
        time_tracker.checkpoint("分块计算")
        
        all_segments = []
        all_text = []
        
        for chunk_idx in range(total_chunks):
            print(f"\n处理分块 {chunk_idx + 1}/{total_chunks}")
            
            # 检查内存
            self.memory_monitor.log_memory_usage(f"分块{chunk_idx+1}开始前")
            
            # 计算当前块的范围
            start_sample = chunk_idx * chunk_samples
            end_sample = min((chunk_idx + 1) * chunk_samples, total_samples)
            
            # 提取音频块
            chunk_audio = audio[start_sample:end_sample]
            
            # 转换为Whisper格式
            chunk_audio_whisper = whisper.pad_or_trim(chunk_audio)
            time_tracker.checkpoint(f"分块{chunk_idx+1}_音频准备")
            
            # 转录当前块
            try:
                # 过滤掉Whisper不支持的参数，并确保启用时间戳功能
                whisper_params = {k: v for k, v in transcribe_kwargs.items() 
                                 if k not in ["segment_duration", "overlap", "use_segmented", "use_memory_optimization"]}
                
                # 确保启用时间戳功能
                if "word_timestamps" not in whisper_params:
                    whisper_params["word_timestamps"] = True
                if "no_speech_threshold" not in whisper_params:
                    whisper_params["no_speech_threshold"] = 0.6
                if "logprob_threshold" not in whisper_params:
                    whisper_params["logprob_threshold"] = -1.0
                
                chunk_result = self.model.transcribe(
                    chunk_audio_whisper,
                    **whisper_params
                )
                time_tracker.checkpoint(f"分块{chunk_idx+1}_转录")
                
                # 调整时间戳
                chunk_start_time = start_sample / sr
                for segment in chunk_result.get("segments", []):
                    segment["start"] += chunk_start_time
                    segment["end"] += chunk_start_time
                    segment["chunk_id"] = chunk_idx
                
                # 保存结果
                all_segments.extend(chunk_result.get("segments", []))
                all_text.append(chunk_result.get("text", ""))
                
                # 更新检查点
                checkpoint["processed_chunks"].append(chunk_idx)
                checkpoint["current_chunk"] = chunk_idx + 1
                checkpoint["results"].append({
                    "chunk_id": chunk_idx,
                    "text": chunk_result.get("text", ""),
                    "segments": chunk_result.get("segments", []),  # 保存完整的segments数据
                    "num_segments": len(chunk_result.get("segments", [])),
                    "start_time": chunk_start_time
                })
                
                # 保存检查点
                self.save_checkpoint(checkpoint_path, checkpoint)
                time_tracker.checkpoint(f"分块{chunk_idx+1}_保存检查点")
                
                print(f"✓ 分块 {chunk_idx + 1} 完成")
                
                # 清理内存
                del chunk_audio, chunk_audio_whisper, chunk_result
                gc.collect()
                
            except MemoryError:
                print(f"内存不足！减小分块大小并重试...")
                # 减小分块大小重新尝试
                return self.retry_with_smaller_chunks(audio_path, chunk_duration//2)
            except KeyboardInterrupt:
                print(f"\n转录被用户中断")
                self.save_checkpoint(checkpoint_path, checkpoint)
                print(f"检查点已保存，下次可从分块 {chunk_idx + 1} 继续")
                return None
            except Exception as e:
                print(f"分块 {chunk_idx + 1} 处理失败: {e}")
                # 跳过错误的分块，继续处理下一个
                continue
        
        # 合并结果
        time_tracker.checkpoint("结果合并")
        final_result = {
            "text": " ".join(all_text),
            "segments": all_segments,
            "language": transcribe_kwargs.get("language", "unknown"),
            "total_duration": total_samples / sr,
            "num_chunks": total_chunks,
            "processing_time": time.time() - checkpoint["start_time"]
        }
        
        # 保存最终结果
        self.save_final_result(audio_path, final_result)
        time_tracker.checkpoint("结果保存")
        
        # 删除检查点
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        print(f"\n[完成] 分段转录完成!")
        print(f"[统计] 总时长: {total_samples / sr:.2f} 秒")
        print(f"[统计] 处理段数: {len(all_segments)}")
        print(f"[统计] 总文本长度: {len(final_result['text'])} 字符")
        
        # 打印耗时总结
        time_tracker.print_summary()
        
        return final_result
    
    def resume_from_checkpoint(self, audio_path: str, checkpoint_path: Path,
                              chunk_duration: int, **transcribe_kwargs) -> Dict:
        """从检查点恢复转录"""
        print("加载检查点...")
        
        # 初始化耗时跟踪器
        time_tracker = TimeTracker()
        time_tracker.checkpoint("初始化")
        
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        
        # 验证音频文件
        if checkpoint.get("audio_path") != audio_path:
            print("检查点对应的音频文件不匹配，开始新的转录")
            return self.new_transcription(audio_path, checkpoint_path,
                                         chunk_duration, **transcribe_kwargs)
        
        # 合并参数
        saved_kwargs = checkpoint.get("transcribe_kwargs", {})
        saved_kwargs.update(transcribe_kwargs)
        time_tracker.checkpoint("参数合并")
        
        print(f"从分块 {checkpoint['current_chunk']}/{checkpoint['total_chunks']} 恢复")
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        total_samples = len(audio)
        chunk_samples = chunk_duration * sr
        time_tracker.checkpoint("音频加载")
        
        all_segments = []
        all_text = []
        
        # 恢复已处理的结果
        for result in checkpoint.get("results", []):
            all_text.append(result.get("text", ""))
            # 恢复segments数据（从检查点文件中的segments字段）
            if "segments" in result:
                all_segments.extend(result["segments"])
        time_tracker.checkpoint("结果恢复")
        
        # 继续处理剩余分块
        for chunk_idx in range(checkpoint["current_chunk"], checkpoint["total_chunks"]):
            print(f"\n处理分块 {chunk_idx + 1}/{checkpoint['total_chunks']}")
            
            # 计算当前块的范围
            start_sample = chunk_idx * chunk_samples
            end_sample = min((chunk_idx + 1) * chunk_samples, total_samples)
            
            # 提取音频块
            chunk_audio = audio[start_sample:end_sample]
            chunk_audio_whisper = whisper.pad_or_trim(chunk_audio)
            time_tracker.checkpoint(f"分块{chunk_idx+1}_音频准备")
            
            # 转录当前块
            try:
                # 过滤掉Whisper不支持的参数，并确保启用时间戳功能
                whisper_params = {k: v for k, v in saved_kwargs.items() 
                                 if k not in ["segment_duration", "overlap", "use_segmented", "use_memory_optimization"]}
                
                # 确保启用时间戳功能
                if "word_timestamps" not in whisper_params:
                    whisper_params["word_timestamps"] = True
                if "no_speech_threshold" not in whisper_params:
                    whisper_params["no_speech_threshold"] = 0.6
                if "logprob_threshold" not in whisper_params:
                    whisper_params["logprob_threshold"] = -1.0
                
                chunk_result = self.model.transcribe(
                    chunk_audio_whisper,
                    **whisper_params
                )
                time_tracker.checkpoint(f"分块{chunk_idx+1}_转录")
                
                # 调整时间戳
                chunk_start_time = start_sample / sr
                for segment in chunk_result.get("segments", []):
                    segment["start"] += chunk_start_time
                    segment["end"] += chunk_start_time
                    segment["chunk_id"] = chunk_idx
                
                # 保存结果
                all_segments.extend(chunk_result.get("segments", []))
                all_text.append(chunk_result.get("text", ""))
                
                # 更新检查点
                checkpoint["processed_chunks"].append(chunk_idx)
                checkpoint["current_chunk"] = chunk_idx + 1
                checkpoint["results"].append({
                    "chunk_id": chunk_idx,
                    "text": chunk_result.get("text", ""),
                    "segments": chunk_result.get("segments", []),  # 保存完整的segments数据
                    "num_segments": len(chunk_result.get("segments", [])),
                    "start_time": chunk_start_time
                })
                
                # 保存检查点
                self.save_checkpoint(checkpoint_path, checkpoint)
                time_tracker.checkpoint(f"分块{chunk_idx+1}_保存检查点")
                
                print(f"✓ 分块 {chunk_idx + 1} 完成")
                
                # 清理内存
                del chunk_audio, chunk_audio_whisper, chunk_result
                gc.collect()
                
            except KeyboardInterrupt:
                print(f"\n转录被用户中断")
                self.save_checkpoint(checkpoint_path, checkpoint)
                print(f"检查点已保存，下次可从分块 {chunk_idx + 1} 继续")
                return None
            except Exception as e:
                print(f"分块 {chunk_idx + 1} 处理失败: {e}")
                continue
        
        # 合并结果
        time_tracker.checkpoint("结果合并")
        final_result = {
            "text": " ".join(all_text),
            "segments": all_segments,
            "language": saved_kwargs.get("language", "unknown"),
            "total_duration": total_samples / sr,
            "num_chunks": checkpoint["total_chunks"],
            "processing_time": time.time() - checkpoint["start_time"]
        }
        
        # 保存最终结果
        self.save_final_result(audio_path, final_result)
        time_tracker.checkpoint("结果保存")
        
        # 保留检查点文件（即使成功完成转录）
        if checkpoint_path.exists():
            print(f"✓ 检查点文件已保留: {checkpoint_path}")
        
        print(f"\n转录完成！")
        print(f"总处理时间: {final_result['processing_time']:.1f}秒")
        
        # 打印耗时总结
        time_tracker.print_summary()
        
        return final_result
    
    def retry_with_smaller_chunks(self, audio_path: str, new_chunk_duration: int) -> Dict:
        """使用更小的分块重试"""
        print(f"使用更小的分块: {new_chunk_duration}秒")
        self.max_chunk_duration = new_chunk_duration
        
        # 删除旧的检查点
        checkpoint_path = self.get_checkpoint_path(audio_path)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        # 重新开始
        return self.new_transcription(audio_path, checkpoint_path, new_chunk_duration,
                                     language="ja", fp16=False)
    
    def save_checkpoint(self, checkpoint_path: Path, checkpoint_data: Dict):
        """保存检查点"""
        checkpoint_data["last_save"] = time.time()
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    def save_final_result(self, audio_path: str, result: Dict):
        """保存最终结果"""
        audio_name = Path(audio_path).stem
        
        # 保存JSON格式的完整结果
        json_file = Path(audio_path).parent / f"{audio_name}_memory_safe_transcript.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"内存安全转录结果已保存到: {json_file}")


class WhisperSegmentResume:
    """
    Whisper分段转录与断点续传
    将长音频分成多个小段，逐段转录，实现真正的断点续传
    """
    
    def __init__(self, model_or_name, device: str = "cpu"):
        try:
            # 检查传入的是模型对象还是模型名称
            if isinstance(model_or_name, str):
                # 传入的是模型名称，需要加载模型
                print(f"[加载] 正在为分段转录器加载Whisper模型: {model_or_name}")
                self.model = whisper.load_model(model_or_name, device=device)
            else:
                # 传入的是已加载的模型对象，直接使用
                print(f"[加载] 分段转录器重用已加载的模型")
                self.model = model_or_name
            
            self.device = device
            self.segments_cache = []
            print(f"[成功] 分段转录器初始化成功")
        except Exception as e:
            # 将详细错误信息写入文件
            error_file = Path("error_log.txt")
            with open(error_file, "w", encoding="utf-8") as f:
                import traceback
                f.write(f"分段转录器初始化失败: {e}\n")
                f.write("完整错误堆栈:\n")
                f.write(traceback.format_exc())
            print(f"[失败] 分段转录器初始化失败，详细信息已保存到 error_log.txt")
            print(f"[失败] 分段转录器初始化失败: {e}")
            raise
    
    def transcribe_long_audio(
        self,
        audio_path: str,
        segment_duration: int = 300,  # 每段5分钟
        overlap: int = 5,  # 段间重叠5秒
        checkpoint_dir: str = "whisper_checkpoints",
        **transcribe_kwargs
    ) -> Dict:
        """
        转录长音频，支持真正的断点续传
        
        Args:
            audio_path: 音频文件路径
            segment_duration: 每段时长（秒）
            overlap: 段间重叠时长（秒）
            checkpoint_dir: 检查点目录
            **transcribe_kwargs: Whisper转录参数
            
        Returns:
            完整的转录结果
        """
        # 初始化耗时跟踪器
        time_tracker = TimeTracker()
        time_tracker.checkpoint("初始化")
        
        # 创建检查点目录
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)
        
        audio_file = Path(audio_path)
        checkpoint_file = checkpoint_path / f"{audio_file.stem}_segments.json"
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        total_samples = len(audio)
        segment_samples = segment_duration * sr
        overlap_samples = overlap * sr
        time_tracker.checkpoint("音频加载")
        
        # 计算总段数
        num_segments = math.ceil(total_samples / segment_samples)
        time_tracker.checkpoint("段数计算")
        
        # 尝试加载检查点
        processed_segments = []
        if checkpoint_file.exists():
            print(f"[加载] 加载检查点: {checkpoint_file}")
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    processed_segments = checkpoint_data.get('segments', [])
                    last_processed = checkpoint_data.get('last_processed', 0)
                    print(f"[统计] 已处理 {len(processed_segments)} 段，上次处理到第 {last_processed} 段")
            except Exception as e:
                print(f"[警告] 检查点加载失败，从头开始: {e}")
                last_processed = 0
        else:
            last_processed = 0
        time_tracker.checkpoint("检查点加载")
        
        # 逐段处理
        all_segments = []
        
        for seg_idx in range(last_processed, num_segments):
            print(f"\n[处理] 处理第 {seg_idx + 1}/{num_segments} 段...")
            
            # 计算当前段的起始和结束位置
            start_sample = max(0, seg_idx * segment_samples - (overlap_samples if seg_idx > 0 else 0))
            end_sample = min((seg_idx + 1) * segment_samples + overlap_samples, total_samples)
            
            # 提取音频段
            segment_audio = audio[start_sample:end_sample]
            
            # 转换为适合Whisper的格式
            segment_audio_whisper = whisper.pad_or_trim(segment_audio)
            time_tracker.checkpoint(f"第{seg_idx+1}段_音频准备")
            
            # 转录当前段
            try:
                # 转录
                segment_result = self.model.transcribe(
                    segment_audio_whisper,
                    **transcribe_kwargs
                )
                time_tracker.checkpoint(f"第{seg_idx+1}段_转录")
                
                # 调整时间戳
                segment_start_time = start_sample / sr
                for seg in segment_result.get("segments", []):
                    seg["start"] += segment_start_time
                    seg["end"] += segment_start_time
                    seg["segment_id"] = seg_idx
                    seg["segment_start_sample"] = start_sample
                    seg["segment_end_sample"] = end_sample
                
                # 添加到结果
                all_segments.extend(segment_result.get("segments", []))
                processed_segments.append({
                    "segment_id": seg_idx,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_time": segment_start_time,
                    "num_segments": len(segment_result.get("segments", [])),
                    "text": segment_result.get("text", "")
                })
                
                # 保存检查点
                checkpoint_data = {
                    "audio_path": str(audio_path),
                    "total_segments": num_segments,
                    "last_processed": seg_idx + 1,
                    "segments": processed_segments,
                    "timestamp": time.time(),
                    "transcribe_kwargs": transcribe_kwargs
                }
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                time_tracker.checkpoint(f"第{seg_idx+1}段_保存检查点")
                
                print(f"[完成] 第 {seg_idx + 1} 段完成，已保存检查点")
                
            except KeyboardInterrupt:
                print("\n[暂停] 转录被用户中断")
                print(f"[保存] 检查点已保存，下次可从第 {seg_idx + 1} 段继续")
                return None
                
            except Exception as e:
                print(f"[失败] 第 {seg_idx + 1} 段处理失败: {e}")
                # 继续处理下一段
        
        # 合并所有段的文本
        full_text = " ".join([seg.get("text", "") for seg in all_segments])
        time_tracker.checkpoint("结果合并")
        
        # 最终结果
        final_result = {
            "text": full_text,
            "segments": all_segments,
            "language": transcribe_kwargs.get("language", "unknown"),
            "total_duration": total_samples / sr,
            "num_segments_processed": len(processed_segments),
            "segment_duration": segment_duration,
            "overlap": overlap
        }
        
        # 保存最终结果
        result_file = audio_file.parent / f"{audio_file.stem}_full_transcript.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        time_tracker.checkpoint("结果保存")
        
        # 删除检查点
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        
        print(f"\n[完成] 分段转录完成!")
        print(f"[统计] 总时长: {total_samples / sr:.2f} 秒")
        print(f"[统计] 处理段数: {len(processed_segments)}")
        print(f"[统计] 总文本长度: {len(full_text)} 字符")
        
        # 打印耗时总结
        time_tracker.print_summary()
        
        return final_result


class VideoSubtitleGenerator:
    """
    视频字幕生成器，支持本地视频文件输入
    基于Whisper实现音频转录和SRT字幕生成，支持长音频分段转录和断点续传
    新增内存优化功能，支持large-v2模型在16GB内存机器上运行
    """
    
    def __init__(self, model_name: str = "base", device: str = "cpu", 
                 enable_memory_optimization: bool = False,
                 max_chunk_duration: int = 60):
        """
        初始化字幕生成器
        
        Args:
            model_name: Whisper模型名称 (base, small, medium, large, large-v2, large-v3)
            device: 运行设备 (cpu, cuda)
            enable_memory_optimization: 是否启用内存优化模式
            max_chunk_duration: 内存优化模式下的最大分块时长（秒）
        """
        print(f"[初始化] 初始化视频字幕生成器...")
        print(f"[模型] 模型名称: {model_name}")
        print(f"[设备] 设备: {device}")
        
        # 确保temp目录存在
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # 保存模型参数，延迟加载模型
        self.model_name = model_name
        self.device = device
        self.enable_memory_optimization = enable_memory_optimization
        self.max_chunk_duration = max_chunk_duration
        
        # 延迟加载的模型和转录器
        self.model = None
        self.segment_transcriber = None
        self.memory_efficient_transcriber = None
        
        print(f"[成功] 字幕生成器初始化完成（模型延迟加载）")
    
    def _lazy_load_model(self):
        """
        延迟加载Whisper模型和转录器
        只有在需要转录时才加载模型，避免不必要的资源消耗
        """
        if self.model is not None:
            return  # 模型已经加载
            
        print(f"[加载] 延迟加载Whisper模型: {self.model_name}")
        
        # 加载Whisper模型
        try:
            print(f"[加载] 正在加载Whisper模型: {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            print(f"[成功] Whisper模型加载成功")
        except Exception as e:
            print(f"[失败] Whisper模型加载失败: {e}")
            raise
        
        # 初始化分段转录器（重用已加载的模型）
        try:
            print(f"[加载] 正在初始化分段转录器")
            self.segment_transcriber = WhisperSegmentResume(self.model, self.device)
            print(f"[成功] 分段转录器初始化成功")
        except Exception as e:
            print(f"[失败] 分段转录器初始化失败: {e}")
            raise
        
        # 初始化内存优化转录器（仅在需要时加载）
        if self.enable_memory_optimization:
            print(f"[内存优化] 启用内存优化模式")
            print(f"[参数] 最大分块时长: {self.max_chunk_duration}秒")
            try:
                self.memory_efficient_transcriber = MemoryEfficientWhisper(
                    self.model,
                    device=self.device,
                    max_chunk_duration=self.max_chunk_duration,
                    checkpoint_dir=str(self.temp_dir / "memory_safe_checkpoints")
                )
                print(f"[成功] 内存优化转录器初始化成功")
            except Exception as e:
                print(f"[失败] 内存优化转录器初始化失败: {e}")
                raise
        
        print(f"[成功] 模型和转录器加载完成")
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """
        从视频文件中提取音频
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            提取的音频文件路径
        """
        print(f"[音频] 检查音频文件...")
        
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 生成音频文件名
        audio_filename = f"{video_file.stem}_audio.wav"
        audio_path = self.temp_dir / audio_filename
        
        # 检查音频文件是否已存在
        if audio_path.exists():
            print(f"[跳过] 音频文件已存在: {audio_path}")
            return str(audio_path)
        
        print(f"[音频] 从视频提取音频...")
        
        # 使用ffmpeg提取音频
        try:
            cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                str(audio_path)
            ]
            
            # 使用universal_newlines=False避免编码问题，手动处理输出
            result = subprocess.run(cmd, capture_output=True, text=False)
            if result.returncode != 0:
                # 尝试使用UTF-8编码解码错误信息
                try:
                    error_msg = result.stderr.decode('utf-8', errors='ignore')
                except:
                    error_msg = result.stderr.decode('gbk', errors='ignore')
                raise RuntimeError(f"音频提取失败: {error_msg}")
            
            print(f"[完成] 音频提取完成: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            print(f"[失败] 音频提取失败: {e}")
            # 备用方案：使用librosa直接读取视频音频
            try:
                print("[备用] 尝试备用音频提取方案...")
                audio, sr = librosa.load(video_path, sr=16000)
                librosa.output.write_wav(str(audio_path), audio, sr)
                print(f"[完成] 备用方案音频提取完成: {audio_path}")
                return str(audio_path)
            except Exception as fallback_e:
                raise RuntimeError(f"所有音频提取方法均失败: {fallback_e}")
    
    def transcribe_audio(self, audio_path: str, **transcribe_kwargs) -> Dict:
        """
        转录音频文件，支持分段转录、断点续传和内存优化
        
        Args:
            audio_path: 音频文件路径
            **transcribe_kwargs: Whisper转录参数
            
        Returns:
            转录结果字典
        """
        print(f"[转录] 开始音频转录...")
        
        # 延迟加载模型（只有在需要转录时才加载）
        self._lazy_load_model()
        
        # 默认转录参数
        default_params = {
            "language": "ja",  # 默认日语转录
            "task": "transcribe",
            "fp16": False,
            "verbose": False,
            "temperature": 0.0,
            "best_of": 1,
            "beam_size": 1,
            "word_timestamps": True,
            "suppress_tokens": [-1],
            "initial_prompt": None
        }
        
        # 合并参数
        params = {**default_params, **transcribe_kwargs}
        
        # 检查是否需要内存优化转录
        use_memory_optimization = transcribe_kwargs.get("use_memory_optimization", False)
        
        # 如果启用了内存优化模式，优先使用内存优化转录
        if self.enable_memory_optimization and use_memory_optimization:
            print(f"🧠 启用内存优化转录模式")
            
            try:
                # 使用内存优化转录器
                result = self.memory_efficient_transcriber.transcribe_with_memory_safety(
                    audio_path, **params
                )
                
                if result is None:
                    raise KeyboardInterrupt("转录被用户中断")
                
                # 添加元数据
                result["audio_path"] = audio_path
                result["transcription_time"] = time.time()
                result["transcription_params"] = params
                result["transcription_mode"] = "memory_optimized"
                
                print(f"✅ 内存优化音频转录完成")
                print(f"[统计] 识别片段数: {len(result.get('segments', []))}")
                print(f"[文本] 总文本长度: {len(result.get('text', ''))} 字符")
                
                return result
                
            except Exception as e:
                print(f"[警告] 内存优化转录失败，回退到标准模式: {e}")
        
        # 检查是否需要分段转录
        segment_duration = transcribe_kwargs.get("segment_duration", 0)
        use_segmented = transcribe_kwargs.get("use_segmented", False)
        
        # 获取音频时长
        try:
            audio_info = librosa.load(audio_path, sr=16000)
            audio_duration = len(audio_info[0]) / audio_info[1]
            
            # 如果音频时长超过10分钟或明确指定使用分段转录，则启用分段模式
            if audio_duration > 600 or use_segmented:  # 10分钟
                print(f"[统计] 音频时长: {audio_duration:.2f} 秒 ({timedelta(seconds=int(audio_duration))})")
                print(f"[模式] 启用分段转录模式")
                
                # 设置分段参数
                segment_duration = segment_duration if segment_duration > 0 else 300  # 默认5分钟一段
                overlap = transcribe_kwargs.get("overlap", 5)  # 默认重叠5秒
                checkpoint_dir = str(self.temp_dir / "whisper_checkpoints")
                
                print(f"[参数] 分段参数: 每段 {segment_duration} 秒，重叠 {overlap} 秒")
                
                # 过滤掉分段转录相关的参数，只保留Whisper转录参数
                whisper_params = {k: v for k, v in params.items() 
                                 if k not in ["segment_duration", "overlap", "use_segmented", "use_memory_optimization"]}
                
                # 使用分段转录
                result = self.segment_transcriber.transcribe_long_audio(
                    audio_path=audio_path,
                    segment_duration=segment_duration,
                    overlap=overlap,
                    checkpoint_dir=checkpoint_dir,
                    **whisper_params
                )
                
                if result is None:
                    raise KeyboardInterrupt("转录被用户中断")
                
                # 添加元数据
                result["audio_path"] = audio_path
                result["transcription_time"] = time.time()
                result["transcription_params"] = params
                result["transcription_mode"] = "segmented"
                
                print(f"✅ 分段音频转录完成")
                print(f"[统计] 识别片段数: {len(result.get('segments', []))}")
                print(f"[文本] 总文本长度: {len(result.get('text', ''))} 字符")
                
                return result
            
        except Exception as e:
            print(f"[警告] 音频时长检测失败，使用标准转录模式: {e}")
        
        # 标准转录模式
        try:
            print(f"[模式] 使用标准转录模式")
            # 过滤掉非Whisper参数
            whisper_params = {k: v for k, v in params.items() 
                             if k not in ["segment_duration", "overlap", "use_segmented", "use_memory_optimization"]}
            result = self.model.transcribe(audio_path, **whisper_params)
            
            # 添加元数据
            result["audio_path"] = audio_path
            result["transcription_time"] = time.time()
            result["transcription_params"] = params
            result["transcription_mode"] = "standard"
            
            print(f"✅ 音频转录完成")
            print(f"[统计] 识别片段数: {len(result.get('segments', []))}")
            print(f"[文本] 总文本长度: {len(result.get('text', ''))} 字符")
            
            return result
            
        except Exception as e:
            print(f"[失败] 音频转录失败: {e}")
            raise
    
    def generate_srt_content(self, transcription_result: Dict) -> str:
        """
        生成SRT格式的字幕内容
        
        Args:
            transcription_result: 转录结果
            
        Returns:
            SRT格式的字幕内容
        """
        print(f"📝 生成SRT字幕内容...")
        
        segments = transcription_result.get("segments", [])
        text = transcription_result.get("text", "").strip()
        
        if not segments and not text:
            raise ValueError("转录结果中没有有效的片段或文本内容")
        
        srt_content = ""
        
        if segments:
            # 如果有时间戳片段，使用片段信息
            for i, segment in enumerate(segments):
                # 格式化时间戳 (SRT格式: HH:MM:SS,mmm)
                start_time = self.format_time_srt(segment["start"])
                end_time = self.format_time_srt(segment["end"])
                
                # 获取文本内容
                text = segment.get("text", "").strip()
                
                # 构建SRT条目
                srt_content += f"{i+1}\n"
                srt_content += f"{start_time} --> {end_time}\n"
                srt_content += f"{text}\n\n"
            
            print(f"✅ SRT内容生成完成，共 {len(segments)} 个字幕条目")
        else:
            # 如果没有时间戳片段但有文本内容，生成虚拟时间戳
            print(f"[警告] 转录结果中没有时间戳片段，生成虚拟时间戳")
            
            # 分割文本为段落
            paragraphs = [p.strip() for p in text.split('。') if p.strip()]
            
            # 计算总时长
            total_duration = transcription_result.get("total_duration", 300)  # 默认5分钟
            
            # 为每个段落分配时间
            for i, paragraph in enumerate(paragraphs):
                # 计算每个段落的持续时间（平均分配）
                segment_duration = total_duration / max(len(paragraphs), 1)
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, total_duration)
                
                # 格式化时间戳
                start_time_str = self.format_time_srt(start_time)
                end_time_str = self.format_time_srt(end_time)
                
                # 构建SRT条目
                srt_content += f"{i+1}\n"
                srt_content += f"{start_time_str} --> {end_time_str}\n"
                srt_content += f"{paragraph}。\n\n"
            
            print(f"✅ SRT内容生成完成，共 {len(paragraphs)} 个虚拟字幕条目")
        
        return srt_content
    
    def format_time_srt(self, seconds: float) -> str:
        """
        将秒数格式化为SRT时间格式 (HH:MM:SS,mmm)
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"
    
    def save_srt_file(self, srt_content: str, video_path: str, output_dir: str = None) -> str:
        """
        保存SRT文件
        
        Args:
            srt_content: SRT内容
            video_path: 原始视频路径（用于生成文件名）
            output_dir: 输出目录（默认使用temp目录）
            
        Returns:
            保存的SRT文件路径
        """
        if output_dir is None:
            output_dir = self.temp_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        video_file = Path(video_path)
        srt_filename = f"{video_file.stem}.srt"
        srt_path = output_dir / srt_filename
        
        # 写入文件
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        print(f"[保存] SRT文件已保存: {srt_path}")
        return str(srt_path)
    
    def save_transcription_result(self, result: Dict, video_path: str) -> str:
        """
        保存转录结果到JSON文件
        
        Args:
            result: 转录结果
            video_path: 原始视频路径
            
        Returns:
            JSON文件路径
        """
        video_file = Path(video_path)
        json_filename = f"{video_file.stem}_transcription.json"
        json_path = self.temp_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"[保存] 转录结果已保存: {json_path}")
        return str(json_path)
    
    def cleanup_temp_files(self, keep_audio: bool = False, keep_json: bool = False, keep_srt: bool = False):
        """
        清理临时文件
        
        Args:
            keep_audio: 是否保留音频文件
            keep_json: 是否保留JSON转录结果
            keep_srt: 是否保留SRT字幕文件
        """
        print(f"🧹 清理临时文件...")
        
        temp_files = list(self.temp_dir.glob("*"))
        
        for file_path in temp_files:
            if file_path.is_file():
                if keep_audio and "_audio.wav" in file_path.name:
                    continue
                if keep_json and "_transcription.json" in file_path.name:
                    continue
                if keep_srt and file_path.suffix == ".srt":
                    continue
                
                try:
                    file_path.unlink()
                    print(f"[删除] 已删除: {file_path.name}")
                except Exception as e:
                    print(f"[警告] 删除失败 {file_path.name}: {e}")
            elif file_path.is_dir():
                # 清理检查点目录
                if file_path.name in ["memory_safe_checkpoints", "whisper_checkpoints"]:
                    try:
                        # 删除目录及其所有内容
                        import shutil
                        shutil.rmtree(file_path)
                        print(f"[删除] 已删除检查点目录: {file_path.name}")
                    except Exception as e:
                        print(f"[警告] 删除检查点目录失败 {file_path.name}: {e}")
        
        print(f"[完成] 临时文件清理完成")
    
    def generate_subtitles(self, video_path: str, 
                          output_srt: bool = True,
                          output_json: bool = True,
                          cleanup: bool = True,
                          segment_duration: int = 0,
                          overlap: int = 5,
                          use_segmented: bool = False,
                          use_memory_optimization: bool = False,
                          **transcribe_kwargs) -> Dict:
        """
        生成视频字幕的完整流程，支持分段转录和内存优化
        
        Args:
            video_path: 视频文件路径
            output_srt: 是否输出SRT文件
            output_json: 是否输出JSON转录结果
            cleanup: 是否清理临时文件（仅在开始执行前清理，不清理翻译缓存）
            segment_duration: 分段时长（秒），0表示自动判断
            overlap: 段间重叠时长（秒）
            use_segmented: 强制使用分段转录
            use_memory_optimization: 强制使用内存优化转录
            **transcribe_kwargs: Whisper转录参数
            
        Returns:
            包含所有结果的字典
        """
        print("=" * 60)
        print(f"[开始] 开始处理视频: {Path(video_path).name}")
        print("=" * 60)
        
        # 初始化耗时跟踪器
        time_tracker = TimeTracker()
        time_tracker.checkpoint("初始化")
        
        start_time = time.time()
        result = {
            "video_path": video_path,
            "processing_start_time": start_time,
            "steps": {}
        }
        
        # 步骤0: 如果指定了--clean参数，在开始执行前清理临时文件
        if cleanup:
            cleanup_start = time.time()
            print(f"[清理] 开始执行前清理临时文件...")
            # 只清理临时文件，保留翻译缓存文件
            self.cleanup_temp_files(keep_audio=True, keep_json=True, keep_srt=True)
            result["steps"]["pre_cleanup"] = time.time() - cleanup_start
            time_tracker.checkpoint("执行前清理")
        
        try:
            # 步骤1: 提取音频（在模型加载之前）
            audio_start = time.time()
            audio_path = self.extract_audio_from_video(video_path)
            result["audio_path"] = audio_path
            result["steps"]["audio_extraction"] = time.time() - audio_start
            time_tracker.checkpoint("音频提取")
            
            # 步骤2: 转录音频
            transcribe_start = time.time()
            
            # 如果启用了内存优化功能，自动启用内存优化转录模式
            if self.enable_memory_optimization and not use_memory_optimization:
                use_memory_optimization = True
                print(f"[内存优化] 检测到内存优化功能已启用，自动启用内存优化转录模式")
            
            # 添加分段转录和内存优化参数
            transcribe_params = {
                **transcribe_kwargs,
                "segment_duration": segment_duration,
                "overlap": overlap,
                "use_segmented": use_segmented,
                "use_memory_optimization": use_memory_optimization
            }
            
            transcription_result = self.transcribe_audio(audio_path, **transcribe_params)
            result["transcription_result"] = transcription_result
            result["steps"]["audio_transcription"] = time.time() - transcribe_start
            time_tracker.checkpoint("音频转录")
            
            # 记录转录模式
            result["transcription_mode"] = transcription_result.get("transcription_mode", "unknown")
            
            # 步骤3: 生成SRT内容
            srt_start = time.time()
            srt_content = self.generate_srt_content(transcription_result)
            result["srt_content"] = srt_content
            result["steps"]["srt_generation"] = time.time() - srt_start
            time_tracker.checkpoint("SRT内容生成")
            
            # 步骤4: 保存文件
            save_start = time.time()
            
            if output_srt:
                srt_path = self.save_srt_file(srt_content, video_path)
                result["srt_path"] = srt_path
            
            if output_json:
                json_path = self.save_transcription_result(transcription_result, video_path)
                result["json_path"] = json_path
            
            result["steps"]["file_saving"] = time.time() - save_start
            time_tracker.checkpoint("文件保存")
            
            # 步骤5: 不再在处理完成后清理临时文件，保留所有生成的文件
            # --clean参数只在开始执行前清理临时文件，不清理翻译缓存文件
            
            # 计算总时间
            total_time = time.time() - start_time
            result["processing_end_time"] = time.time()
            result["total_processing_time"] = total_time
            
            # 输出统计信息
            print("=" * 60)
            print(f"[完成] 处理完成!")
            print("=" * 60)
            print(f"[统计] 处理统计:")
            print(f"   视频文件: {Path(video_path).name}")
            print(f"   转录模式: {result.get('transcription_mode', 'unknown')}")
            print(f"   总处理时间: {total_time:.2f}秒")
            print(f"   识别片段数: {len(transcription_result.get('segments', []))}")
            print(f"   总文本长度: {len(transcription_result.get('text', ''))} 字符")
            
            # 如果是分段转录，显示分段信息
            if result.get('transcription_mode') == 'segmented':
                print(f"   分段参数: {segment_duration}秒/段，重叠{overlap}秒")
            
            # 如果是内存优化转录，显示内存优化信息
            if result.get('transcription_mode') == 'memory_optimized':
                # 从转录结果中获取实际使用的分块时长
                actual_chunk_duration = transcription_result.get('num_chunks', 0)
                if actual_chunk_duration > 0:
                    total_duration = transcription_result.get('total_duration', 0)
                    avg_chunk_duration = total_duration / actual_chunk_duration
                    print(f"   内存优化: 已启用，平均分块时长{avg_chunk_duration:.1f}秒")
                else:
                    print(f"   内存优化: 已启用")
            
            if output_srt:
                print(f"   SRT文件: {result.get('srt_path', '未生成')}")
            if output_json:
                print(f"   JSON文件: {result.get('json_path', '未生成')}")
            
            print(f"[耗时] 各步骤耗时:")
            for step, duration in result["steps"].items():
                print(f"     {step}: {duration:.2f}秒")
            
            # 打印耗时总结
            time_tracker.print_summary()
            
            return result
            
        except Exception as e:
            print(f"[失败] 处理失败: {e}")
            result["error"] = str(e)
            result["processing_end_time"] = time.time()
            result["total_processing_time"] = time.time() - start_time
            
            # 发生错误时保留临时文件以便调试
            print(f"[警告] 发生错误，临时文件将保留在 {self.temp_dir}")
            
            return result


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    # 初始化总耗时跟踪器
    total_time_tracker = TimeTracker()
    total_time_tracker.checkpoint("程序启动")
    
    parser = argparse.ArgumentParser(description="视频字幕生成工具")
    parser.add_argument("video_path", help="视频文件路径")
    parser.add_argument("--model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper模型大小 (默认: base)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="运行设备 (默认: cpu)")
    parser.add_argument("--language", default="ja", 
                       help="转录语言代码 (默认: ja - 日语)")
    parser.add_argument("--output-dir", help="输出目录 (默认: temp)")
    parser.add_argument("--no-srt", action="store_true", help="不生成SRT文件")
    parser.add_argument("--no-json", action="store_true", help="不生成JSON文件")
    parser.add_argument("--clean", action="store_true", help="清理临时文件")
    parser.add_argument("--keep-audio", action="store_true", help="保留音频文件")
    
    # 分段转录参数
    parser.add_argument("--segment-duration", type=int, default=0,
                       help="分段时长（秒），0表示自动判断 (默认: 0)")
    parser.add_argument("--overlap", type=int, default=5,
                       help="段间重叠时长（秒） (默认: 5)")
    parser.add_argument("--force-segmented", action="store_true",
                       help="强制使用分段转录模式")
    
    # 内存优化参数
    parser.add_argument("--enable-memory-optimization", action="store_true",
                       help="启用内存优化模式，支持large-v2模型在16GB内存机器上运行")
    parser.add_argument("--max-chunk-duration", type=int, default=60,
                       help="内存优化模式下的最大分块时长（秒） (默认: 60)")
    parser.add_argument("--force-memory-optimized", action="store_true",
                       help="强制使用内存优化转录模式")
    
    args = parser.parse_args()
    total_time_tracker.checkpoint("参数解析")
    
    # 验证视频文件存在
    if not Path(args.video_path).exists():
        print(f"[失败] 视频文件不存在: {args.video_path}")
        return 1
    
    # 初始化字幕生成器
    try:
        generator = VideoSubtitleGenerator(
            model_name=args.model,
            device=args.device,
            enable_memory_optimization=args.enable_memory_optimization,
            max_chunk_duration=args.max_chunk_duration
        )
        total_time_tracker.checkpoint("字幕生成器初始化")
    except Exception as e:
        print(f"[失败] 初始化失败: {e}")
        return 1
    
    # 设置输出目录
    if args.output_dir:
        generator.temp_dir = Path(args.output_dir)
        generator.temp_dir.mkdir(exist_ok=True)
    
    # 转录参数
    transcribe_params = {
        "language": args.language
    }
    
    # 生成字幕
    result = generator.generate_subtitles(
        video_path=args.video_path,
        output_srt=not args.no_srt,
        output_json=not args.no_json,
        cleanup=args.clean,  # 只有当用户指定--clean时才清理临时文件
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        use_segmented=args.force_segmented,
        use_memory_optimization=args.force_memory_optimized,
        **transcribe_params
    )
    total_time_tracker.checkpoint("字幕生成完成")
    
    # 处理结果
    if "error" in result:
        print(f"[失败] 字幕生成失败: {result['error']}")
        total_time_tracker.checkpoint("处理失败")
        total_time_tracker.print_summary()
        return 1
    else:
        print(f"[成功] 字幕生成成功!")
        total_time_tracker.checkpoint("处理成功")
        total_time_tracker.print_summary()
        return 0


if __name__ == "__main__":
    # 示例用法
    if len(os.sys.argv) == 1:
        print("[工具] 视频字幕生成工具")
        print("=" * 50)
        print("使用方法:")
        print("  pythonwhisper-transcription.py <视频文件路径> [选项]")
        print("")
        print("选项:")
        print("  --model MODEL        Whisper模型 (tiny, base, small, medium, large, large-v2, large-v3)")
        print("  --device DEVICE      运行设备 (cpu, cuda)")
        print("  --language LANG      转录语言代码 (ja, zh, en等)")
        print("  --output-dir DIR     输出目录")
        print("  --no-srt             不生成SRT文件")
        print("  --no-json            不生成JSON文件")
        print("  --clean              清理临时文件（默认保留）")
        print("  --keep-audio         保留音频文件")
        print("")
        print("分段转录选项 (用于处理长视频):")
        print("  --segment-duration SEC  分段时长（秒），0表示自动判断")
        print("  --overlap SEC           段间重叠时长（秒）")
        print("  --force-segmented       强制使用分段转录模式")
        print("")
        print("内存优化选项 (用于有限内存环境):")
        print("  --enable-memory-optimization  启用内存优化模式")
        print("  --max-chunk-duration SEC      最大分块时长（秒）")
        print("  --force-memory-optimized      强制使用内存优化转录")
        print("")
        print("示例:")
        print("  pythonwhisper-transcription.py my_video.mp4 --model base --language ja")
        print("  pythonwhisper-transcription.py video.avi --model large-v3 --device cuda")
        print("  pythonwhisper-transcription.py long_movie.mp4 --segment-duration 300 --overlap 10")
        print("  pythonwhisper-transcription.py lecture.mp4 --force-segmented --segment-duration 600")
        print("  pythonwhisper-transcription.py big_video.mp4 --model large-v2 --enable-memory-optimization")
        print("  pythonwhisper-transcription.py hd_video.mp4 --model large-v3 --force-memory-optimized --max-chunk-duration 30")
        print("")
        
        # 测试示例
        test_video = input("输入测试视频路径 (或按回车跳过): ").strip()
        if test_video and Path(test_video).exists():
            print(f"\n[开始] 开始测试处理: {test_video}")
            
            generator = VideoSubtitleGenerator(model_name="base", device="cpu")
            result = generator.generate_subtitles(
                video_path=test_video,
                output_srt=True,
                output_json=True,
                cleanup=False,  # 测试时保留文件（默认行为）
                language="ja"
            )
        else:
            print("[失败] 未提供有效的测试视频路径")
    else:
        # 正常命令行执行
        exit(main())
