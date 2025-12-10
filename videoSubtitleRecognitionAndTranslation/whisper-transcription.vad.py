import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import whisper
from whisper.utils import get_writer
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
import traceback
from datetime import datetime
import hashlib

# 尝试导入 faster-whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# 你的VAD模块导入（根据实际情况调整）
# 假设原来从 VAD 模块导入相关功能
# from vad_utils import VADProcessor, get_speech_timestamps, collect_chunks

class ProgressTracker:
    """进度跟踪器，支持断点续传"""
    
    def __init__(self, audio_path: str, output_dir: str):
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, "progress.json")
        self.state = self._load_progress()
        
        # 创建音频指纹用于验证续传的匹配性
        self.audio_fingerprint = self._get_audio_fingerprint()
    
    def _get_audio_fingerprint(self):
        """生成音频文件指纹"""
        try:
            file_stat = os.stat(self.audio_path)
            fingerprint = hashlib.md5(f"{self.audio_path}_{file_stat.st_size}_{file_stat.st_mtime}".encode()).hexdigest()
            return fingerprint
        except:
            return None
    
    def _load_progress(self) -> Dict:
        """加载进度文件"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logging.info(f"找到进度文件: {self.progress_file}")
                return state
            except Exception as e:
                logging.warning(f"无法加载进度文件: {e}")
        
        return {
            "audio_path": self.audio_path,
            "fingerprint": None,
            "total_segments": 0,
            "processed_segments": 0,
            "completed_segments": [],
            "current_position": 0,
            "start_time": None,
            "last_update": None,
            "results": []
        }
    
    def save_progress(self, segment_results: List[Dict] = None, current_pos: float = None):
        """保存进度"""
        if segment_results:
            self.state["results"] = segment_results
        
        if current_pos is not None:
            self.state["current_position"] = current_pos
        
        self.state["fingerprint"] = self.audio_fingerprint
        self.state["last_update"] = datetime.now().isoformat()
        self.state["processed_segments"] = len(self.state["results"])
        
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存进度失败: {e}")
    
    def can_resume(self) -> bool:
        """检查是否可以续传"""
        if not self.state.get("fingerprint") or not self.audio_fingerprint:
            return False
        
        # 检查指纹是否匹配
        fingerprint_match = self.state["fingerprint"] == self.audio_fingerprint
        
        # 检查音频路径是否匹配
        path_match = self.state.get("audio_path") == self.audio_path
        
        # 检查是否有未完成的工作
        has_progress = len(self.state.get("results", [])) > 0
        
        return fingerprint_match and path_match and has_progress
    
    def get_resume_point(self) -> Tuple[float, List[Dict]]:
        """获取续传点"""
        return self.state.get("current_position", 0), self.state.get("results", [])
    
    def mark_complete(self):
        """标记任务完成"""
        self.state["completed"] = True
        self.state["end_time"] = datetime.now().isoformat()
        self.save_progress()
        
        # 将进度文件重命名为完成状态
        try:
            completed_file = os.path.join(self.output_dir, "completed.json")
            os.rename(self.progress_file, completed_file)
        except:
            pass

class OptimizedWhisperTranscriber:
    """优化版Whisper转录器，支持用户参数和断点续传"""
    
    def __init__(self, args):
        self.args = args
        self._setup_logging()
        
        # 初始化进度跟踪器
        self.progress = ProgressTracker(args.input_file, args.output_dir)
        
        # 根据参数选择模型后端
        self.use_faster = FASTER_WHISPER_AVAILABLE and not args.no_faster_whisper
        
        # 加载模型
        self.model = self._load_model()
        
        # 初始化VAD（根据你的实际VAD代码调整）
        self.vad_processor = self._load_vad_processor()
        
        logging.info(f"初始化完成: faster-whisper={self.use_faster}, 设备={args.device}")
    
    def _setup_logging(self):
        """设置日志"""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.args.output_dir, "transcription.log")),
                logging.StreamHandler()
            ]
        )
    
    def _load_model(self):
        """加载模型"""
        model_size = self.args.model
        
        if self.use_faster:
            logging.info(f"使用 faster-whisper，模型: {model_size}")
            
            # 设置计算类型
            compute_type = "float16" if self.args.device == "cuda" else "int8"
            if self.args.compute_type:
                compute_type = self.args.compute_type
            
            try:
                model = WhisperModel(
                    model_size,
                    device=self.args.device,
                    compute_type=compute_type,
                    num_workers=self.args.threads,
                    download_root=self.args.model_dir,
                )
                return model
            except Exception as e:
                logging.error(f"加载 faster-whisper 失败: {e}")
                self.use_faster = False
                # 回退到原版whisper
        
        # 使用原版whisper
        logging.info(f"使用原版 whisper，模型: {model_size}")
        model = whisper.load_model(
            model_size,
            device=self.args.device,
            download_root=self.args.model_dir
        )
        return model
    
    def _load_vad_processor(self):
        """加载VAD处理器"""
        # 这里根据你的VAD实现调整
        # 假设你的VAD处理器初始化需要这些参数
        try:
            from vad_utils import VADProcessor  # 你的VAD模块
            return VADProcessor(
                aggressiveness=self.args.vad_aggressiveness,
                min_silence_duration_ms=self.args.vad_min_silence_duration,
                speech_pad_ms=self.args.vad_speech_pad_ms,
                threshold=self.args.vad_threshold,
            )
        except ImportError:
            logging.warning("无法导入VAD模块，将使用简化VAD或Whisper内置VAD")
            return None
    
    def process_audio(self):
        """处理音频"""
        logging.info(f"开始处理音频: {self.args.input_file}")
        
        # 检查是否可以从断点续传
        if self.progress.can_resume() and not self.args.no_resume:
            logging.info("检测到未完成的任务，尝试续传...")
            resume_position, previous_results = self.progress.get_resume_point()
            all_results = previous_results
            
            # 加载音频并从断点开始
            audio = self._load_audio_from_position(resume_position)
            start_time = resume_position
        else:
            # 从头开始
            logging.info("开始新的转录任务")
            audio, sr = self._load_audio(self.args.input_file)
            all_results = []
            start_time = 0
            
            # 保存初始进度
            self.progress.save_progress(all_results, start_time)
        
        # 检测语音片段
        speech_segments = self._detect_speech_segments(audio)
        total_segments = len(speech_segments)
        logging.info(f"检测到 {total_segments} 个语音片段")
        
        # 处理每个片段
        for i, segment in enumerate(speech_segments):
            # 计算全局时间
            segment_start = start_time + segment[0]
            segment_end = start_time + segment[1]
            
            logging.info(f"处理片段 {i+1}/{total_segments}: {segment_start:.2f}s - {segment_end:.2f}s")
            
            # 提取音频片段
            audio_segment = audio[segment[0]:segment[1]]
            
            # 转录片段
            result = self._transcribe_segment(audio_segment, segment_start)
            
            if result:
                all_results.append(result)
                
                # 定期保存进度
                if (i + 1) % self.args.save_every == 0:
                    self.progress.save_progress(all_results, segment_end)
                    logging.info(f"已保存进度，完成 {len(all_results)}/{total_segments} 个片段")
        
        # 最终保存
        self.progress.save_progress(all_results)
        self.progress.mark_complete()
        
        # 生成最终输出
        final_result = self._format_final_result(all_results)
        self._save_results(final_result)
        
        logging.info("转录完成！")
        return final_result
    
    def _load_audio(self, audio_path):
        """加载音频"""
        logging.info(f"加载音频文件: {audio_path}")
        
        if self.use_faster:
            # faster-whisper 需要 librosa
            try:
                import librosa
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                return audio, sr
            except ImportError:
                logging.warning("未安装librosa，使用原版音频加载")
                return whisper.load_audio(audio_path), 16000
        else:
            # 原版whisper
            return whisper.load_audio(audio_path), 16000
    
    def _load_audio_from_position(self, position_seconds):
        """从指定位置加载音频"""
        logging.info(f"从 {position_seconds:.2f} 秒处加载音频")
        
        if self.use_faster:
            try:
                import librosa
                # 计算开始样本
                start_sample = int(position_seconds * 16000)
                audio, sr = librosa.load(
                    self.args.input_file,
                    sr=16000,
                    mono=True,
                    offset=position_seconds
                )
                return audio, sr
            except Exception as e:
                logging.error(f"从位置加载音频失败: {e}")
                # 回退到全量加载
                return self._load_audio(self.args.input_file)
        else:
            # 原版whisper需要全量加载然后切片
            audio, sr = self._load_audio(self.args.input_file)
            start_sample = int(position_seconds * sr)
            return audio[start_sample:], sr
    
    def _detect_speech_segments(self, audio):
        """检测语音片段"""
        logging.info("使用VAD检测语音片段")
        
        # 根据你的VAD实现调整
        if self.vad_processor:
            # 假设你的VAD处理器有这个接口
            segments = self.vad_processor.get_speech_segments(
                audio,
                sr=16000,
                min_silence_duration=self.args.vad_min_silence_duration,
                speech_pad_ms=self.args.vad_speech_pad_ms,
                threshold=self.args.vad_threshold,
            )
            return segments
        elif self.use_faster:
            # faster-whisper内置VAD
            # 这里我们返回整个音频作为一个片段，让faster-whisper处理VAD
            return [(0, len(audio) / 16000)]
        else:
            # 原版whisper，返回整个音频
            return [(0, len(audio) / 16000)]
    
    def _transcribe_segment(self, audio_segment, segment_start):
        """转录单个片段"""
        if self.use_faster:
            return self._transcribe_with_faster_whisper(audio_segment, segment_start)
        else:
            return self._transcribe_with_original_whisper(audio_segment, segment_start)
    
    def _transcribe_with_faster_whisper(self, audio, segment_start):
        """使用faster-whisper转录"""
        # 设置VAD参数（faster-whisper内置）
        vad_params = {
            "threshold": self.args.vad_threshold,
            "min_speech_duration_ms": 500,
            "max_speech_duration_s": 30,
            "min_silence_duration_ms": self.args.vad_min_silence_duration,
            "window_size_samples": 512,
            "speech_pad_ms": self.args.vad_speech_pad_ms,
        }
        
        # 转录参数
        whisper_params = {
            "language": self.args.language,
            "beam_size": self.args.beam_size,
            "best_of": self.args.best_of,
            "temperature": [self.args.temperature],
            "patience": self.args.patience,
            "initial_prompt": self.args.initial_prompt,
            "word_timestamps": self.args.word_timestamps,
            "condition_on_previous_text": False,  # 加速处理
            "vad_filter": True,
            "vad_parameters": vad_params,
        }
        
        try:
            segments, info = self.model.transcribe(audio, **whisper_params)
            
            result_segments = []
            for segment in segments:
                segment_dict = {
                    "start": segment_start + segment.start,
                    "end": segment_start + segment.end,
                    "text": segment.text.strip(),
                    "confidence": getattr(segment, 'confidence', None),
                }
                result_segments.append(segment_dict)
            
            return {
                "start": segment_start,
                "end": segment_start + (len(audio) / 16000),
                "segments": result_segments,
                "language": info.language if info else self.args.language,
            }
            
        except Exception as e:
            logging.error(f"转录失败: {e}")
            return None
    
    def _transcribe_with_original_whisper(self, audio, segment_start):
        """使用原版whisper转录"""
        try:
            result = self.model.transcribe(
                audio,
                language=self.args.language,
                task=self.args.task,
                temperature=self.args.temperature,
                beam_size=self.args.beam_size,
                best_of=self.args.best_of,
                patience=self.args.patience,
                length_penalty=self.args.length_penalty,
                initial_prompt=self.args.initial_prompt,
                condition_on_previous_text=self.args.condition_on_previous_text,
                word_timestamps=self.args.word_timestamps,
                prepend_punctuations=self.args.prepend_punctuations,
                append_punctuations=self.args.append_punctuations,
            )
            
            # 调整时间戳
            for segment in result["segments"]:
                segment["start"] += segment_start
                segment["end"] += segment_start
            
            return {
                "start": segment_start,
                "end": segment_start + (len(audio) / 16000),
                "segments": result["segments"],
                "language": result.get("language", self.args.language),
            }
            
        except Exception as e:
            logging.error(f"转录失败: {e}")
            return None
    
    def _format_final_result(self, all_results):
        """格式化最终结果"""
        all_segments = []
        all_text = []
        
        for result in all_results:
            if result and "segments" in result:
                all_segments.extend(result["segments"])
                for seg in result["segments"]:
                    all_text.append(seg.get("text", ""))
        
        return {
            "text": " ".join(all_text).strip(),
            "segments": all_segments,
            "language": self.args.language,
            "info": {
                "model": self.args.model,
                "processing_time": datetime.now().isoformat(),
                "audio_file": self.args.input_file,
            }
        }
    
    def _save_results(self, result):
        """保存结果"""
        output_dir = self.args.output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 基础文件名
        base_name = Path(self.args.input_file).stem
        
        # 1. 保存为JSON
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logging.info(f"结果已保存为JSON: {json_path}")
        
        # 2. 保存为TXT
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        
        # 3. 保存为SRT（如果启用了时间戳）
        if self.args.word_timestamps or self.args.output_format in ["srt", "all"]:
            srt_path = os.path.join(output_dir, f"{base_name}.srt")
            self._save_as_srt(result["segments"], srt_path)
            logging.info(f"字幕已保存为SRT: {srt_path}")
        
        # 4. 保存为其他格式（根据args.output_format）
        if self.args.output_format in ["vtt", "tsv", "all"]:
            try:
                writer = get_writer(self.args.output_format, output_dir)
                writer(result, base_name)
            except Exception as e:
                logging.warning(f"保存 {self.args.output_format} 格式失败: {e}")
        
        return {
            "json": json_path,
            "txt": txt_path,
            "srt": srt_path if (self.args.word_timestamps or self.args.output_format in ["srt", "all"]) else None,
        }
    
    def _save_as_srt(self, segments, output_path):
        """保存为SRT格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start = self._format_timestamp(segment["start"])
                end = self._format_timestamp(segment["end"])
                text = segment.get("text", "").strip()
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
    
    def _format_timestamp(self, seconds):
        """格式化时间戳为SRT格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def main():
    """主函数，解析参数并运行转录"""
    parser = argparse.ArgumentParser(description="优化版Whisper日语视频转录工具")
    
    # 输入输出参数
    parser.add_argument("input_file", type=str, help="输入音频/视频文件路径")
    parser.add_argument("--output_dir", "-o", type=str, default="output", help="输出目录")
    parser.add_argument("--output_format", type=str, default="all", 
                       choices=["txt", "json", "srt", "vtt", "tsv", "all"], help="输出格式")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="large-v2",
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper模型大小")
    parser.add_argument("--model_dir", type=str, help="模型缓存目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       choices=["cpu", "cuda"], help="运行设备")
    parser.add_argument("--compute_type", type=str, choices=["float16", "int8", "int8_float16"],
                       help="计算类型（仅faster-whisper）")
    parser.add_argument("--no_faster_whisper", action="store_true", 
                       help="禁用faster-whisper，使用原版whisper")
    
    # 转录参数
    parser.add_argument("--language", type=str, default="ja", help="音频语言")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                       help="任务类型：转录或翻译")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--beam_size", type=int, default=3, help="束搜索大小")
    parser.add_argument("--best_of", type=int, default=1, help="最佳采样数")
    parser.add_argument("--patience", type=float, default=1.0, help="耐心参数")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="长度惩罚")
    parser.add_argument("--initial_prompt", type=str, help="初始提示词")
    parser.add_argument("--condition_on_previous_text", action="store_true", 
                       help="使用上文作为条件")
    parser.add_argument("--word_timestamps", action="store_true", help="生成词级时间戳")
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-",
                       help="前置标点符号")
    parser.add_argument("--append_punctuations", type=str, default="\"\'.。,，!！?？:：”)]}、",
                       help="后置标点符号")
    parser.add_argument("--threads", type=int, default=0, 
                       help="线程数（仅faster-whisper）")
    
    # VAD参数
    parser.add_argument("--vad_aggressiveness", type=int, default=2, choices=[0, 1, 2, 3],
                       help="VAD激进程度（0-3，越大越敏感）")
    parser.add_argument("--vad_min_silence_duration", type=int, default=300,
                       help="最小静音时长（毫秒）")
    parser.add_argument("--vad_speech_pad_ms", type=int, default=200,
                       help="语音片段填充（毫秒）")
    parser.add_argument("--vad_threshold", type=float, default=0.3,
                       help="VAD阈值（0-1）")
    
    # 进度和性能参数
    parser.add_argument("--no_resume", action="store_true", help="禁用断点续传")
    parser.add_argument("--save_every", type=int, default=10, 
                       help="每处理多少个片段保存一次进度")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 检查faster-whisper是否可用
    if not FASTER_WHISPER_AVAILABLE and not args.no_faster_whisper:
        logging.warning("faster-whisper 不可用，将使用原版 whisper")
        logging.warning("安装 faster-whisper: pip install faster-whisper")
        args.no_faster_whisper = True
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 创建转录器
        transcriber = OptimizedWhisperTranscriber(args)
        
        # 开始处理
        start_time = time.time()
        result = transcriber.process_audio()
        end_time = time.time()
        
        # 输出统计信息
        duration = end_time - start_time
        audio_duration = len(result["segments"]) * 10  # 粗略估计，实际需要计算
        speed_factor = audio_duration / duration if duration > 0 else 0
        
        logging.info(f"任务完成！")
        logging.info(f"总耗时: {duration:.2f}秒")
        logging.info(f"处理速度: {speed_factor:.2f}x 实时速度")
        logging.info(f"转录文本长度: {len(result['text'])} 字符")
        logging.info(f"输出文件保存在: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"处理失败: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
