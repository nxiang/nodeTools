import os
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import hashlib
from datetime import datetime
import subprocess
import re

# 导入torch
import torch

# 尝试导入 faster-whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("错误: 请安装 faster-whisper: pip install faster-whisper")
    sys.exit(1)

# 尝试导入Silero VAD
try:
    from silero_vad import load_silero_vad, get_speech_timestamps, read_audio, save_audio
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False
    print("提示: 如需使用更准确的Silero VAD，请安装: pip install silero-vad")

class TimeFormatter:
    """时间格式化工具类"""
    
    @staticmethod
    def format_seconds(seconds: float) -> str:
        """将秒数格式化为 HH:MM:SS 格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """格式化时间戳为 SRT 格式 HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

class PerformanceTracker:
    """性能追踪器"""
    
    def __init__(self):
        self.stages = {}
        self.start_time = time.time()
        self.current_stage = None
        self.current_stage_start = None
    
    def start_stage(self, stage_name: str):
        """开始一个阶段"""
        if self.current_stage and self.current_stage_start:
            elapsed = time.time() - self.current_stage_start
            self.stages[self.current_stage] = elapsed
        
        self.current_stage = stage_name
        self.current_stage_start = time.time()
        logging.info(f"开始阶段: {stage_name}")
    
    def end_stage(self):
        """结束当前阶段"""
        if self.current_stage and self.current_stage_start:
            elapsed = time.time() - self.current_stage_start
            self.stages[self.current_stage] = elapsed
            self.current_stage = None
            self.current_stage_start = None
    
    def get_summary(self) -> Dict:
        """获取性能摘要"""
        total = time.time() - self.start_time
        summary = {
            "各阶段耗时": {k: TimeFormatter.format_seconds(v) for k, v in self.stages.items()},
            "总耗时": TimeFormatter.format_seconds(total),
            "实时因子": f"{total / self.stages.get('转录处理', total):.2f}x",
        }
        return summary
    
    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("性能摘要")
        print("="*50)
        print(f"总耗时: {summary['总耗时']}")
        print(f"实时因子: {summary['实时因子']}")
        print("\n各阶段耗时:")
        for stage, duration in summary["各阶段耗时"].items():
            print(f"  {stage}: {duration}")
        print("="*50)

class EnhancedVADProcessor:
    """增强版VAD处理器，专注减少幻觉文本"""
    
    def __init__(self, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.available = False
        
        if not SILERO_VAD_AVAILABLE:
            logging.warning("Silero VAD不可用")
            return
        
        try:
            logging.info(f"正在加载Silero VAD模型 (设备: {self.device})...")
            
            # 尝试多种加载方式以适应不同版本的silero_vad
            try:
                # 方法1: 直接使用torch.hub加载（最兼容的方式）
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
                
                # 将模型移动到指定设备
                self.model = model.to(self.device)
                self.model.eval()
                
                # 工具函数
                (get_speech_timestamps, _, read_audio, _, _) = utils
                self.get_speech_timestamps_fn = get_speech_timestamps
                self.available = True
                
                logging.info("Silero VAD加载成功 (使用torch.hub)")
                
            except Exception as hub_error:
                logging.warning(f"torch.hub加载失败，尝试其他方式: {hub_error}")
                
                # 方法2: 尝试导入silero_vad模块
                try:
                    from silero_vad import load_silero_vad
                    
                    # 尝试不同的参数组合
                    try:
                        # 无参数
                        model, utils = load_silero_vad()
                    except TypeError:
                        try:
                            # 带torchscript参数
                            model, utils = load_silero_vad(torchscript=False)
                        except TypeError:
                            # 带model_path参数
                            model, utils = load_silero_vad(model_path=None)
                    
                    self.model = model.to(self.device)
                    self.model.eval()
                    (get_speech_timestamps, _, read_audio, _, _) = utils
                    self.get_speech_timestamps_fn = get_speech_timestamps
                    self.available = True
                    
                    logging.info("Silero VAD加载成功 (使用silero_vad模块)")
                    
                except Exception as module_error:
                    logging.error(f"所有Silero VAD加载方式均失败: {module_error}")
                    self.available = False
            
        except Exception as e:
            logging.error(f"加载Silero VAD失败: {e}")
            self.available = False
    
    def detect_speech_with_filters(self, audio: np.ndarray, sr: int = 16000, 
                                  vad_threshold: float = 0.4,  # 提高阈值减少误检
                                  min_speech_duration_ms: int = 400,  # 增加最小时长
                                  max_speech_duration_s: float = 8.0,  # 添加最大时长限制
                                  frequency_filter: bool = True) -> List[Dict[str, float]]:
        """检测语音段，带多种过滤器减少幻觉"""
        if not self.available or self.model is None:
            return []
        
        try:
            # 确保音频是单声道
            if len(audio.shape) > 1:
                audio = audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]
            
            # 确保音频是float32类型
            audio = audio.astype(np.float32)
            
            # 1. 音频能量分析，过滤极低能量段
            audio_energy = np.mean(np.abs(audio))
            if audio_energy < 0.001:  # 能量太低，可能全是静音
                logging.debug(f"音频能量过低 ({audio_energy:.6f})，跳过VAD检测")
                return []
            
            # 2. 转换为PyTorch张量并移动到设备
            audio_tensor = torch.from_numpy(audio).to(self.device)
            
            # 3. 使用较严格的VAD参数检测语音时间戳
            vad_start = time.time()
            
            timestamps = get_speech_timestamps(
                audio_tensor, 
                self.model,
                threshold=vad_threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=200,  # 增加静音时长
                window_size_samples=512,
                speech_pad_ms=100,
                return_seconds=True
            )
            
            vad_time = time.time() - vad_start
            
            if not timestamps:
                return []
            
            # 4. 过滤过长的语音段（很可能是背景音乐/噪声）
            filtered_timestamps = []
            for ts in timestamps:
                duration = ts['end'] - ts['start']
                
                # 过滤标准
                if duration > max_speech_duration_s:
                    logging.debug(f"过滤过长的语音段: {duration:.2f}s > {max_speech_duration_s}s")
                    continue
                
                if duration < 0.4:  # 小于400ms的片段
                    logging.debug(f"过滤过短的语音段: {duration:.2f}s")
                    continue
                
                # 5. 提取音频段进行进一步分析
                start_sample = int(ts['start'] * sr)
                end_sample = int(ts['end'] * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # 计算段内特征
                segment_energy = np.mean(np.abs(segment_audio))
                segment_std = np.std(segment_audio)
                
                # 过滤能量太低或变化太小的段（可能是恒定噪声）
                if segment_energy < audio_energy * 0.3:
                    logging.debug(f"过滤低能量语音段: {segment_energy:.6f}")
                    continue
                
                if segment_std < 0.01:  # 变化太小
                    logging.debug(f"过滤低变化语音段: {segment_std:.6f}")
                    continue
                
                filtered_timestamps.append(ts)
            
            logging.debug(f"VAD检测: 原始 {len(timestamps)} 个，过滤后 {len(filtered_timestamps)} 个，耗时 {vad_time:.3f}秒")
            
            return filtered_timestamps
            
        except Exception as e:
            logging.error(f"VAD检测失败: {e}")
            return []

class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, audio_path: str, temp_dir: str, overlap_seconds: float = 2.0):
        self.audio_path = audio_path
        self.temp_dir = temp_dir
        self.overlap_seconds = overlap_seconds  # 重叠时长（秒）
        self.audio_info = self._get_audio_info()
    
    def _get_audio_info(self) -> Dict:
        """获取音频信息"""
        try:
            # 使用ffprobe获取音频信息
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', self.audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                raise ValueError(f"ffprobe执行失败: {result.stderr}")
                
            info = json.loads(result.stdout)
            
            # 查找音频流
            audio_stream = None
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                raise ValueError("未找到音频流")
            
            duration = float(info['format']['duration'])
            if duration <= 0:
                raise ValueError(f"音频时长为0或无效: {duration}")
            
            return {
                'duration': duration,
                'sample_rate': int(audio_stream.get('sample_rate', 16000)),
                'channels': int(audio_stream.get('channels', 1)),
                'codec': audio_stream.get('codec_name', 'unknown')
            }
        except Exception as e:
            logging.warning(f"无法获取音频信息: {e}")
            # 尝试使用librosa获取音频时长
            try:
                import librosa
                audio, sr = librosa.load(self.audio_path, sr=None, mono=True, duration=30)
                duration = len(audio) / sr if len(audio) > 0 else 0
                
                if duration > 0:
                    logging.info(f"使用librosa获取音频时长: {duration:.2f}秒")
                    return {
                        'duration': duration,
                        'sample_rate': sr,
                        'channels': 1,
                        'codec': 'unknown'
                    }
            except Exception as librosa_error:
                logging.warning(f"librosa获取音频信息也失败: {librosa_error}")
            
            # 最终默认值
            logging.error("无法获取有效的音频信息，请检查音频文件")
            return {
                'duration': 0,
                'sample_rate': 16000,
                'channels': 1,
                'codec': 'unknown'
            }
    
    def extract_audio_chunks(self, chunk_duration: int = 180) -> List[Tuple[int, str, float]]:
        """提取音频分块 (3分钟 = 180秒) 支持重叠"""
        chunks = []
        total_duration = self.audio_info['duration']
        overlap = self.overlap_seconds
        
        # 创建chunks目录
        chunks_dir = os.path.join(self.temp_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # 计算分块数量（考虑重叠）
        step_duration = chunk_duration - overlap  # 每个chunk前进的时长
        num_chunks = max(1, int(np.ceil((total_duration - overlap) / step_duration)))
        
        logging.info(f"音频总时长: {TimeFormatter.format_seconds(total_duration)}")
        logging.info(f"将分割为 {num_chunks} 个分块 (每个 {chunk_duration} 秒, 重叠 {overlap} 秒)")
        
        # 预检查已存在的chunks
        existing_chunks = []
        for i in range(num_chunks):
            chunk_path = os.path.join(chunks_dir, f"chunk_{i:04d}.wav")
            if os.path.exists(chunk_path):
                start_time = i * step_duration
                existing_chunks.append((i, chunk_path, start_time))
        
        if existing_chunks:
            logging.info(f"发现 {len(existing_chunks)} 个已存在的chunks，跳过生成")
        
        # 使用ffmpeg分割音频（仅生成缺失的chunks）
        start_time_total = time.time()
        
        for i in range(num_chunks):
            start_time = i * step_duration
            chunk_path = os.path.join(chunks_dir, f"chunk_{i:04d}.wav")
            
            # 检查是否已存在
            if os.path.exists(chunk_path):
                chunks.append((i, chunk_path, start_time))
                continue
            
            chunk_start_time = time.time()
            
            # 性能优化：每10个chunk后添加短暂延迟，避免系统资源竞争
            if i > 0 and i % 10 == 0:
                elapsed_time = time.time() - start_time_total
                avg_time_per_chunk = elapsed_time / i
                remaining_chunks = num_chunks - i
                estimated_remaining_time = avg_time_per_chunk * remaining_chunks
                
                logging.info(f"进度: {i}/{num_chunks} chunks, 平均时间: {avg_time_per_chunk:.2f}s/chunk, 预计剩余时间: {estimated_remaining_time/60:.1f}分钟")
                
                # 每10个chunk后休息1秒，让系统恢复
                time.sleep(1)
            
            # 计算实际结束时间（不超过音频总时长）
            end_time = min(start_time + chunk_duration, total_duration)
            actual_chunk_duration = end_time - start_time
            
            # 如果时长太短（小于1秒），跳过这个chunk
            if actual_chunk_duration < 1.0:
                logging.warning(f"chunk {i} 时长过短 ({actual_chunk_duration:.2f}秒)，跳过")
                continue
            
            # 优化：越到后面使用更快的seek方式
            if i < 10:
                # 精确seek：先-i后-ss，精度高但慢
                cmd = [
                    'ffmpeg', '-i', self.audio_path,
                    '-ss', str(start_time),
                    '-t', str(actual_chunk_duration),
                    '-ac', '1',  # 单声道
                    '-ar', '16000',  # 16kHz采样率
                    '-acodec', 'pcm_s16le',  # PCM 16-bit
                    '-y',  # 覆盖输出文件
                    '-loglevel', 'error',
                    '-threads', '1',  # 单线程避免竞争
                    '-nostdin',  # 禁用标准输入，避免阻塞
                    chunk_path
                ]
            else:
                # 快速seek：先-ss后-i，速度快但精度稍低
                cmd = [
                    'ffmpeg', '-ss', str(start_time),
                    '-i', self.audio_path,
                    '-t', str(actual_chunk_duration),
                    '-ac', '1',  # 单声道
                    '-ar', '16000',  # 16kHz采样率
                    '-acodec', 'pcm_s16le',  # PCM 16-bit
                    '-y',  # 覆盖输出文件
                    '-loglevel', 'error',
                    '-threads', '1',  # 单线程避免竞争
                    '-nostdin',  # 禁用标准输入，避免阻塞
                    chunk_path
                ]
            
            try:
                # 使用Popen而不是run，避免阻塞
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(timeout=120)
                
                chunk_elapsed_time = time.time() - chunk_start_time
                
                if process.returncode == 0:
                    chunks.append((i, chunk_path, start_time))
                    logging.info(f"已创建分块 {i+1}/{num_chunks}: {chunk_path} (开始: {start_time:.1f}s, 时长: {actual_chunk_duration:.1f}s, 耗时: {chunk_elapsed_time:.2f}s)")
                else:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    logging.error(f"创建分块 {i} 失败 (耗时: {chunk_elapsed_time:.2f}s): {error_msg}")
                    
                    # 尝试使用不同的ffmpeg参数重新生成
                    if "Invalid data found" in error_msg or "moov atom not found" in error_msg:
                        logging.warning(f"尝试使用备用参数重新生成chunk {i}")
                        if self._retry_extract_chunk(i, start_time, actual_chunk_duration, chunk_path):
                            chunks.append((i, chunk_path, start_time))
                            logging.info(f"备用参数成功创建chunk {i}")
            except subprocess.TimeoutExpired:
                chunk_elapsed_time = time.time() - chunk_start_time
                logging.error(f"创建分块 {i} 超时 (耗时: {chunk_elapsed_time:.2f}s)，终止进程")
                process.kill()
                stdout, stderr = process.communicate()
                
                # 尝试跳过问题chunk，继续处理后续chunks
                logging.warning(f"跳过问题chunk {i}，继续处理后续chunks")
            except Exception as e:
                chunk_elapsed_time = time.time() - chunk_start_time
                logging.error(f"创建分块 {i} 时出错 (耗时: {chunk_elapsed_time:.2f}s): {e}")
                # 跳过问题chunk，继续处理后续chunks
                logging.warning(f"跳过问题chunk {i}，继续处理后续chunks")
        
        # 按开始时间排序
        chunks.sort(key=lambda x: x[2])
        logging.info(f"音频分块完成，共 {len(chunks)} 个chunks (重叠 {overlap} 秒)")
        
        return chunks
    
    def _retry_extract_chunk(self, chunk_id: int, start_time: float, chunk_duration: int, chunk_path: str) -> bool:
        """尝试使用备用参数重新生成chunk"""
        try:
            # 备用参数1：使用不同的seek方式
            cmd1 = [
                'ffmpeg', '-ss', str(start_time), '-i', self.audio_path,
                '-t', str(chunk_duration),
                '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le',
                '-y', '-loglevel', 'error', '-threads', '1', '-nostdin',
                chunk_path
            ]
            
            process1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout1, stderr1 = process1.communicate(timeout=90)
            
            if process1.returncode == 0:
                logging.info(f"备用参数1成功创建chunk {chunk_id}")
                return True
            
            # 备用参数2：使用更简单的参数
            cmd2 = [
                'ffmpeg', '-i', self.audio_path,
                '-ss', str(start_time), '-t', str(chunk_duration),
                '-ac', '1', '-ar', '8000',  # 降低采样率
                '-y', '-loglevel', 'error', '-threads', '1', '-nostdin',
                chunk_path
            ]
            
            process2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout2, stderr2 = process2.communicate(timeout=90)
            
            if process2.returncode == 0:
                logging.info(f"备用参数2成功创建chunk {chunk_id}")
                return True
                
            logging.error(f"所有备用参数均失败，无法创建chunk {chunk_id}")
            return False
            
        except Exception as e:
            logging.error(f"重试生成chunk {chunk_id}失败: {e}")
            return False

class ProgressManager:
    """进度管理器"""
    
    def __init__(self, audio_path: str, model_name: str, temp_dir: str):
        self.audio_path = audio_path
        self.model_name = model_name
        self.temp_dir = temp_dir
        self.progress_file = os.path.join(temp_dir, "progress.json")
        self.state = self._load_progress()
        self.audio_fingerprint = self._create_fingerprint()
    
    def _create_fingerprint(self) -> str:
        """创建音频指纹"""
        try:
            stat = os.stat(self.audio_path)
            key = f"{self.audio_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(key.encode()).hexdigest()[:16]
        except:
            return "unknown"
    
    def _load_progress(self) -> Dict:
        """加载进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "audio_path": self.audio_path,
            "fingerprint": "",
            "model": "",
            "total_chunks": 0,
            "completed_chunks": [],
            "current_chunk": 0,
            "results": {},
            "start_time": "",
            "last_update": ""
        }
    
    def save_progress(self, chunk_id: Optional[int] = None, result: Optional[Dict] = None):
        """保存进度"""
        if chunk_id is not None:
            self.state["completed_chunks"].append(chunk_id)
            self.state["current_chunk"] = chunk_id + 1
        
        if result is not None and chunk_id is not None:
            self.state["results"][str(chunk_id)] = result
        
        self.state["fingerprint"] = self.audio_fingerprint
        self.state["model"] = self.model_name
        self.state["last_update"] = datetime.now().isoformat()
        
        if not self.state.get("start_time"):
            self.state["start_time"] = self.state["last_update"]
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
    
    def can_resume(self) -> bool:
        """检查是否可以续传"""
        if not self.state.get("fingerprint") or not self.audio_fingerprint:
            return False
        if self.state.get("audio_path") != self.audio_path:
            return False
        if self.state.get("model") != self.model_name:
            return False
        return self.state["fingerprint"] == self.audio_fingerprint
    
    def get_completed_chunks(self) -> List[int]:
        """获取已完成的chunk ID"""
        return self.state.get("completed_chunks", [])
    
    def get_completed_results(self) -> Dict:
        """获取已完成的结果"""
        return self.state.get("results", {})
    
    def mark_complete(self):
        """标记完成"""
        self.state["completed"] = True
        self.state["end_time"] = datetime.now().isoformat()
        self.save_progress()

class AntiHallucinationTranscriber:
    """抗幻觉转录器 - 专门解决幻觉文本问题"""
    
    def __init__(self, input_file: str, model_name: str, language: str = "ja", 
                 overlap_seconds: float = 2.0, use_silero_vad: bool = True,
                 vad_threshold: float = 0.4, min_speech_duration_ms: int = 400,
                 max_speech_duration_s: float = 8.0,
                 adult_mode: bool = True, filter_hallucinations: bool = True):
        self.input_file = os.path.abspath(input_file)
        self.model_name = model_name
        self.language = language
        self.overlap_seconds = overlap_seconds
        self.use_silero_vad = use_silero_vad
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.adult_mode = adult_mode
        self.filter_hallucinations = filter_hallucinations
        
        # 幻觉文本黑名单（日语）
        self.hallucination_patterns = [
            # 常见的结束语幻觉
            r'ご視聴ありがとうございました[。.]*$',
            r'ご視聴ありがとうございます[。.]*$',
            r'ありがとうございました[。.]*$',
            r'ご覧いただきありがとうございました[。.]*$',
            r'本日の放送はこれで終了です[。.]*$',
            r'お楽しみいただけましたでしょうか[。.]*$',
            
            # 常见的开场白幻觉
            r'これは日本語の音声です[。.]*$',
            r'こんにちは[。.]*$',
            r'よろしくお願いします[。.]*$',
            
            # 其他常见幻觉
            r'お疲れ様でした[。.]*$',
            r'失礼します[。.]*$',
            r'どうもありがとうございました[。.]*$',
            r'以上です[。.]*$',
            
            # 成人内容特定幻觉（根据您的数据）
            r'音声です[。.]*$',  # 您的数据中有"音声です"
        ]
        
        # 创建临时目录
        audio_name = Path(self.input_file).stem
        self.temp_dir = f"temp/{audio_name}_{model_name}"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.performance = PerformanceTracker()
        self.audio_processor = AudioProcessor(self.input_file, self.temp_dir, overlap_seconds)
        self.progress = ProgressManager(self.input_file, model_name, self.temp_dir)
        self.model = None
        self.vad_processor = None
        
        # 初始化Silero VAD
        if self.use_silero_vad and SILERO_VAD_AVAILABLE:
            self.performance.start_stage("VAD模型加载")
            self.vad_processor = EnhancedVADProcessor()
            if self.vad_processor.available:
                logging.info("成功启用增强版VAD，专注减少幻觉")
                self.use_silero_vad = True
            else:
                logging.warning("Silero VAD不可用，将回退到faster-whisper内置VAD")
                self.use_silero_vad = False
            self.performance.end_stage()
        else:
            logging.info("使用faster-whisper内置VAD")
        
        logging.info(f"初始化完成: 输入={self.input_file}, 模型={model_name}, 语言={language}")
        logging.info(f"抗幻觉模式: {filter_hallucinations}")
        logging.info(f"VAD设置: 阈值={vad_threshold}, 最小语音时长={min_speech_duration_ms}ms, 最大语音时长={max_speech_duration_s}s")
        logging.info(f"临时目录: {self.temp_dir}")
    
    def _setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.temp_dir, "transcription.log")
        
        # 移除所有现有的处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _load_model(self):
        """加载Whisper模型"""
        self.performance.start_stage("Whisper模型加载")
        
        # 自动选择设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        logging.info(f"使用设备: {device}, 计算类型: {compute_type}")
        
        try:
            model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
                num_workers=1,
                download_root=None,
            )
            
            self.performance.end_stage()
            return model
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            raise
    
    def _get_transcription_options(self, use_external_vad: bool = False):
        """获取转录参数 - 针对抗幻觉优化"""
        options = {
            "language": self.language,
            "beam_size": 3,  # 降低束搜索宽度，减少幻觉
            "best_of": 3,    # 减少采样次数
            "temperature": 0.0,  # 只使用确定性采样
            "patience": 1.0,
            "condition_on_previous_text": False,  # 关闭上下文依赖，减少重复幻觉
            "word_timestamps": False,
            "compression_ratio_threshold": 1.8,  # 降低压缩比阈值，过滤长文本
            "no_speech_threshold": 0.7,  # 提高无语音阈值，减少在静音上的幻觉
            "suppress_tokens": [-1],  # 不抑制特定token
        }
        
        # 关键：根据是否使用外部VAD设置vad_filter
        if use_external_vad:
            options["vad_filter"] = False  # 使用外部VAD时关闭内置VAD
        else:
            options["vad_filter"] = True  # 使用内置VAD
            
        # 关键：使用抗幻觉提示词
        if self.language == "ja":
            # 使用更中性的提示词，避免引导模型产生特定内容
            options["initial_prompt"] = "音声を正確に書き起こしてください。"
        
        return options
    
    def _transcribe_with_enhanced_vad(self, audio: np.ndarray, chunk_start_time: float, 
                                     chunk_id: int, sr: int = 16000) -> Dict:
        """使用增强版VAD进行转录 - 专注抗幻觉"""
        try:
            # 使用增强版VAD检测语音段
            vad_start = time.time()
            
            speech_timestamps = self.vad_processor.detect_speech_with_filters(
                audio, sr=sr,
                vad_threshold=self.vad_threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                max_speech_duration_s=self.max_speech_duration_s,
                frequency_filter=True
            )
            
            vad_time = time.time() - vad_start
            
            if not speech_timestamps:
                logging.info(f"chunk {chunk_id}: 未检测到有效语音段")
                return self._create_empty_result(chunk_id, chunk_start_time, audio, sr)
            
            logging.info(f"chunk {chunk_id}: 检测到 {len(speech_timestamps)} 个有效语音段")
            
            # 转录每个语音段
            all_segments = []
            transcription_options = self._get_transcription_options(use_external_vad=True)
            
            for i, ts in enumerate(speech_timestamps):
                # 提取音频段
                start_sample = int(ts['start'] * sr)
                end_sample = int(ts['end'] * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # 计算绝对时间
                absolute_start = chunk_start_time + ts['start']
                absolute_end = chunk_start_time + ts['end']
                duration = ts['end'] - ts['start']
                
                # 转录
                segment_start = time.time()
                try:
                    segments, info = self.model.transcribe(
                        segment_audio,
                        **transcription_options  # 已包含vad_filter=False
                    )
                    
                    segment_time = time.time() - segment_start
                    
                    # 处理转录结果
                    segment_texts = []
                    for segment in segments:
                        text = segment.text.strip()
                        
                        # 关键：应用抗幻觉过滤
                        filtered_text = self._filter_hallucination_text(text, duration)
                        
                        if filtered_text:  # 只有非空文本才保留
                            segment_dict = {
                                "start": absolute_start + segment.start,
                                "end": absolute_start + segment.end,
                                "text": filtered_text,
                                "chunk_id": chunk_id,
                                "segment_index": i,
                                "duration": duration,
                                "original_text": text,  # 保留原始文本用于调试
                                "confidence": segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.0,
                                "no_speech_prob": segment.no_speech_prob if hasattr(segment, 'no_speech_prob') else 0.0,
                            }
                            all_segments.append(segment_dict)
                            segment_texts.append(filtered_text)
                    
                    if segment_texts:
                        logging.debug(f"语音段 {i+1}/{len(speech_timestamps)} 转录完成: {' '.join(segment_texts)[:50]}... (耗时: {segment_time:.2f}s)")
                    
                except Exception as e:
                    logging.warning(f"语音段 {i+1} 转录失败: {e}")
                    continue
            
            # 关键：后处理，移除重复和相似的片段
            filtered_segments = self._post_process_segments(all_segments)
            
            return {
                "chunk_id": chunk_id,
                "chunk_start": chunk_start_time,
                "chunk_end": chunk_start_time + len(audio)/sr,
                "segments": filtered_segments,
                "language": self.language,
                "vad_type": "enhanced",
                "vad_time": vad_time,
                "speech_segments": len(speech_timestamps),
                "transcribed_segments": len(filtered_segments),
                "filtered_hallucinations": len(all_segments) - len(filtered_segments),
            }
            
        except Exception as e:
            logging.error(f"增强版VAD转录失败: {e}")
            return None
    
    def _filter_hallucination_text(self, text: str, duration: float) -> str:
        """过滤幻觉文本"""
        if not text or len(text.strip()) == 0:
            return ""
        
        # 0. 如果是非常长的文本（可能是一大段幻觉）
        if len(text) > 100:  # 超过100字符的文本
            logging.debug(f"过滤过长的文本: {text[:50]}...")
            return ""
        
        # 1. 检查是否是幻觉模式
        if self.filter_hallucinations:
            for pattern in self.hallucination_patterns:
                if re.match(pattern, text.strip()):
                    logging.debug(f"过滤幻觉文本: {text}")
                    return ""
        
        # 2. 过滤成人内容中的特定模式（如有需要）
        if self.adult_mode:
            # 可以根据需要添加成人内容特定的过滤
            pass
        
        # 3. 检查文本是否与duration匹配
        # 日语平均语速约为5-7字符/秒
        chars_per_second = len(text) / duration if duration > 0 else 0
        
        # 如果语速异常（太慢或太快），可能是幻觉
        if duration > 2.0 and chars_per_second < 1.0:  # 超过2秒但少于1字符/秒
            logging.debug(f"过滤语速异常文本: {text} (语速: {chars_per_second:.1f}字符/秒)")
            return ""
        
        if chars_per_second > 15.0:  # 超过15字符/秒，异常快
            logging.debug(f"过滤语速异常文本: {text} (语速: {chars_per_second:.1f}字符/秒)")
            return ""
        
        return text.strip()
    
    def _post_process_segments(self, segments: List[Dict]) -> List[Dict]:
        """后处理：移除重复和相似的片段"""
        if not segments:
            return []
        
        # 按开始时间排序
        segments.sort(key=lambda x: x["start"])
        
        filtered = []
        last_segment = None
        
        for segment in segments:
            current_text = segment["text"]
            
            # 跳过空文本
            if not current_text:
                continue
            
            # 如果是第一个片段，直接添加
            if not last_segment:
                filtered.append(segment)
                last_segment = segment
                continue
            
            # 检查是否与上一个片段重叠太多
            overlap = min(last_segment["end"], segment["end"]) - max(last_segment["start"], segment["start"])
            if overlap > 0.5:  # 重叠超过0.5秒
                # 选择置信度更高的片段
                last_confidence = last_segment.get("confidence", 0)
                current_confidence = segment.get("confidence", 0)
                
                if current_confidence > last_confidence:
                    # 用当前片段替换上一个片段
                    filtered[-1] = segment
                    last_segment = segment
                # 否则保持上一个片段
                continue
            
            # 检查文本是否相似（可能是重复的幻觉）
            if self._texts_are_similar(last_segment["text"], current_text):
                # 合并相似的片段
                last_segment["end"] = segment["end"]
                last_segment["text"] = last_segment["text"]  # 保持第一个文本
                # 更新置信度为平均值
                if "confidence" in last_segment and "confidence" in segment:
                    last_segment["confidence"] = (last_segment["confidence"] + segment["confidence"]) / 2
                continue
            
            # 否则添加新片段
            filtered.append(segment)
            last_segment = segment
        
        return filtered
    
    def _texts_are_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """判断两个文本是否相似"""
        # 简单的文本相似度检查
        if text1 == text2:
            return True
        
        # 去除标点
        import difflib
        clean1 = re.sub(r'[、。,!?]', '', text1)
        clean2 = re.sub(r'[、。,!?]', '', text2)
        
        # 使用difflib计算相似度
        similarity = difflib.SequenceMatcher(None, clean1, clean2).ratio()
        
        return similarity > threshold
    
    def _create_empty_result(self, chunk_id: int, chunk_start_time: float, 
                            audio: np.ndarray, sr: int) -> Dict:
        """创建空结果"""
        return {
            "chunk_id": chunk_id,
            "chunk_start": chunk_start_time,
            "chunk_end": chunk_start_time + len(audio)/sr,
            "segments": [],
            "language": self.language,
            "vad_type": "enhanced",
            "speech_segments": 0,
            "transcribed_segments": 0,
        }
    
    def _transcribe_with_builtin_vad(self, audio: np.ndarray, chunk_start_time: float,
                                    chunk_id: int, sr: int = 16000) -> Dict:
        """使用faster-whisper内置VAD进行转录"""
        try:
            # 转录参数
            transcription_options = self._get_transcription_options(use_external_vad=False)
            
            # 内置VAD参数 - 使用更严格的参数减少幻觉
            vad_params = {
                "threshold": 0.2,  # 提高阈值
                "min_speech_duration_ms": 200,
                "max_speech_duration_s": self.max_speech_duration_s,  # 使用配置的最大时长
                "min_silence_duration_ms": 300,  # 增加静音时长
                "speech_pad_ms": 100,
            }
            
            # 执行转录
            segments, info = self.model.transcribe(
                audio,
                vad_parameters=vad_params,
                **transcription_options  # 已包含vad_filter=True
            )
            
            # 处理结果
            chunk_segments = []
            for segment in segments:
                text = segment.text.strip()
                duration = segment.end - segment.start
                
                # 应用抗幻觉过滤
                filtered_text = self._filter_hallucination_text(text, duration)
                
                if filtered_text:  # 只保留非空文本
                    segment_dict = {
                        "start": chunk_start_time + segment.start,
                        "end": chunk_start_time + segment.end,
                        "text": filtered_text,
                        "chunk_id": chunk_id,
                        "original_text": text,  # 保留原始文本用于调试
                        "confidence": segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.0,
                        "no_speech_prob": segment.no_speech_prob if hasattr(segment, 'no_speech_prob') else 0.0,
                    }
                    chunk_segments.append(segment_dict)
            
            # 后处理
            filtered_segments = self._post_process_segments(chunk_segments)
            
            return {
                "chunk_id": chunk_id,
                "chunk_start": chunk_start_time,
                "chunk_end": chunk_start_time + len(audio)/sr,
                "segments": filtered_segments,
                "language": info.language,
                "language_probability": info.language_probability,
                "vad_type": "builtin",
                "filtered_hallucinations": len(chunk_segments) - len(filtered_segments),
            }
            
        except Exception as e:
            logging.error(f"使用内置VAD转录chunk {chunk_id}失败: {e}")
            return None
    
    def transcribe_chunk(self, chunk_id: int, chunk_path: str, chunk_start_time: float) -> Dict:
        """转录单个chunk"""
        try:
            # 检查chunk文件是否存在且有效
            if not os.path.exists(chunk_path):
                logging.error(f"chunk文件不存在: {chunk_path}")
                return None
                
            if os.path.getsize(chunk_path) == 0:
                logging.error(f"chunk文件为空: {chunk_path}")
                return None
            
            # 加载chunk音频
            import librosa
            try:
                audio, sr = librosa.load(chunk_path, sr=16000, mono=True)
                
                # 检查音频数据是否有效
                if len(audio) == 0 or np.max(np.abs(audio)) < 0.001:
                    logging.warning(f"chunk {chunk_id} 音频数据过小或无效")
                    return {
                        "chunk_id": chunk_id,
                        "chunk_start": chunk_start_time,
                        "chunk_end": chunk_start_time + 180,
                        "segments": [],
                        "language": self.language,
                    }
            except Exception as load_error:
                logging.error(f"加载chunk音频失败: {load_error}")
                return None
            
            # 根据VAD设置选择转录方法
            if self.use_silero_vad and self.vad_processor and self.vad_processor.available:
                result = self._transcribe_with_enhanced_vad(audio, chunk_start_time, chunk_id, sr)
            else:
                result = self._transcribe_with_builtin_vad(audio, chunk_start_time, chunk_id, sr)
            
            # 清理内存
            del audio
            import gc
            gc.collect()
            
            return result
            
        except Exception as e:
            logging.error(f"转录chunk {chunk_id}失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process(self):
        """主处理流程"""
        total_start = time.time()
        
        try:
            # 1. 加载Whisper模型
            self.model = self._load_model()
            
            # 2. 提取音频chunks
            self.performance.start_stage("音频分块")
            chunks = self.audio_processor.extract_audio_chunks()
            self.progress.state["total_chunks"] = len(chunks)
            self.performance.end_stage()
            
            # 3. 检查续传
            completed_chunks = []
            completed_results = {}
            
            if self.progress.can_resume():
                completed_chunks = self.progress.get_completed_chunks()
                completed_results = self.progress.get_completed_results()
                logging.info(f"检测到进度，已完成 {len(completed_chunks)}/{len(chunks)} 个chunks")
            
            # 4. 转录处理
            self.performance.start_stage("转录处理")
            
            # 统计幻觉文本过滤情况
            total_hallucinations_filtered = 0
            
            for chunk_id, chunk_path, start_time in chunks:
                # 跳过已完成的chunk
                if chunk_id in completed_chunks:
                    logging.info(f"跳过已完成的chunk {chunk_id}")
                    continue
                
                # 开始处理当前chunk
                chunk_start = time.time()
                logging.info(f"开始处理chunk {chunk_id+1}/{len(chunks)} (开始时间: {start_time:.1f}s)")
                
                # 转录chunk
                result = self.transcribe_chunk(chunk_id, chunk_path, start_time)
                
                if result:
                    # 保存进度
                    self.progress.save_progress(chunk_id, result)
                    completed_results[str(chunk_id)] = result
                    
                    chunk_time = time.time() - chunk_start
                    
                    # 获取统计信息
                    vad_type = result.get("vad_type", "unknown")
                    speech_segments = result.get("speech_segments", 0)
                    transcribed_segments = result.get("transcribed_segments", len(result.get("segments", [])))
                    filtered_hallucinations = result.get("filtered_hallucinations", 0)
                    total_hallucinations_filtered += filtered_hallucinations
                    
                    logging.info(f"完成chunk {chunk_id}，VAD: {vad_type}, 语音段: {speech_segments}, 转录段: {transcribed_segments}, 过滤幻觉: {filtered_hallucinations}, 耗时: {TimeFormatter.format_seconds(chunk_time)}")
                else:
                    logging.warning(f"chunk {chunk_id} 转录失败，跳过")
            
            self.performance.end_stage()
            
            # 5. 合并结果
            self.performance.start_stage("结果合并")
            final_result = self._merge_results(completed_results)
            self.performance.end_stage()
            
            # 6. 保存最终结果
            self.performance.start_stage("结果保存")
            self._save_results(final_result)
            self.progress.mark_complete()
            self.performance.end_stage()
            
            # 7. 打印统计信息
            total_time = time.time() - total_start
            audio_duration = self.audio_processor.audio_info['duration']
            
            print("\n" + "="*70)
            print(f"转录完成!")
            print("="*70)
            print(f"音频时长: {TimeFormatter.format_seconds(audio_duration)}")
            print(f"总处理时间: {TimeFormatter.format_seconds(total_time)}")
            
            if audio_duration > 0:
                real_time_factor = total_time / audio_duration
                print(f"实时因子: {real_time_factor:.2f}x")
            else:
                print("实时因子: 无法计算（音频时长为0）")
            
            print(f"\n抗幻觉统计:")
            print(f"  过滤幻觉文本总数: {total_hallucinations_filtered}")
            print(f"  最终有效片段数: {final_result.get('info', {}).get('total_segments', 0)}")
            print(f"  VAD类型: {'增强版VAD' if self.use_silero_vad else '内置VAD'}")
            print(f"  VAD阈值: {self.vad_threshold}")
            print(f"  最小语音时长: {self.min_speech_duration_ms}ms")
            print(f"  最大语音时长: {self.max_speech_duration_s}s")
            
            print(f"\n临时文件位置: {self.temp_dir}")
            print("="*70)
            
            # 性能摘要
            self.performance.print_summary()
            
            return True
            
        except Exception as e:
            logging.error(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _merge_results(self, chunk_results: Dict) -> Dict:
        """合并所有chunk的结果"""
        all_segments = []
        
        # 按chunk_id排序
        for chunk_id in sorted(map(int, chunk_results.keys())):
            result = chunk_results[str(chunk_id)]
            if result and "segments" in result:
                all_segments.extend(result["segments"])
        
        # 按开始时间排序
        all_segments.sort(key=lambda x: x["start"])
        
        # 最终过滤：移除过于相似的相邻片段
        filtered_segments = []
        last_segment = None
        
        for segment in all_segments:
            if last_segment and self._texts_are_similar(last_segment["text"], segment["text"]):
                # 合并相似的片段
                last_segment["end"] = segment["end"]
                continue
            
            filtered_segments.append(segment)
            last_segment = segment
        
        # 提取文本
        all_text = [seg.get("text", "") for seg in filtered_segments]
        
        return {
            "text": " ".join(all_text).strip(),
            "segments": filtered_segments,
            "language": self.language,
            "info": {
                "audio_file": self.input_file,
                "model": self.model_name,
                "language": self.language,
                "total_segments": len(filtered_segments),
                "original_segments": len(all_segments),
                "filtered_segments": len(all_segments) - len(filtered_segments),
                "vad_type": "enhanced" if self.use_silero_vad else "builtin",
                "vad_threshold": self.vad_threshold,
                "min_speech_duration_ms": self.min_speech_duration_ms,
                "max_speech_duration_s": self.max_speech_duration_s,
                "adult_mode": self.adult_mode,
                "filter_hallucinations": self.filter_hallucinations,
                "overlap_seconds": self.overlap_seconds,
                "processing_time": datetime.now().isoformat(),
            }
        }
    
    def _save_results(self, result: Dict):
        """保存结果到临时目录"""
        # JSON格式
        json_path = os.path.join(self.temp_dir, "transcription.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # TXT格式（包含时间信息）
        txt_path = os.path.join(self.temp_dir, "transcription.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            # 写入文件头信息
            f.write(f"视频: {result.get('info', {}).get('audio_file', 'Unknown')}\n")
            f.write(f"模型: {result.get('info', {}).get('model', 'Unknown')}\n")
            f.write(f"语言: {result.get('language', 'Unknown')}\n")
            f.write(f"VAD类型: {result.get('info', {}).get('vad_type', 'Unknown')}\n")
            f.write(f"抗幻觉模式: {'启用' if self.filter_hallucinations else '禁用'}\n")
            f.write(f"总片段数: {result.get('info', {}).get('total_segments', 0)}\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入带时间戳的文本内容
            segments = result.get("segments", [])
            for segment in segments:
                start_time = TimeFormatter.format_timestamp(segment["start"])
                end_time = TimeFormatter.format_timestamp(segment["end"])
                text = segment.get("text", "").strip()
                
                if text:
                    f.write(f"[{start_time} - {end_time}] {text}\n")
        
        # SRT格式
        srt_path = os.path.join(self.temp_dir, "transcription.srt")
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result.get("segments", []), 1):
                start = TimeFormatter.format_timestamp(segment["start"])
                end = TimeFormatter.format_timestamp(segment["end"])
                text = segment.get("text", "").strip()
                
                if text:
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")
        
        logging.info(f"结果已保存到: {self.temp_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="日语视频转录工具 - 抗幻觉版")
    parser.add_argument("input_file", type=str, help="输入音频/视频文件路径")
    parser.add_argument("--model", type=str, default="large-v3-turbo",
                       choices=["tiny", "base", "small", "medium", "large-v2", "large-v3-turbo"],
                       help="Whisper模型大小 (默认: large-v3-turbo)")
    parser.add_argument("--language", "-l", default="ja", 
                       help="音频语言代码 (默认: ja - 日语)")
    parser.add_argument("--overlap", "-o", type=float, default=2.0,
                       help="分块重叠时长（秒），避免语句被切断 (默认: 2.0)")
    parser.add_argument("--no-silero-vad", action="store_true",
                       help="禁用Silero VAD，使用faster-whisper内置VAD")
    parser.add_argument("--vad-threshold", type=float, default=0.4,
                       help="VAD检测阈值 (0.2-0.6，越高越严格) (默认: 0.4)")
    parser.add_argument("--min-speech-duration", type=int, default=400,
                       help="最小语音时长（毫秒）(默认: 400)")
    parser.add_argument("--max-speech-duration", type=float, default=8.0,
                       help="最大语音时长（秒）(默认: 8.0)")
    parser.add_argument("--no-filter-hallucinations", action="store_true",
                       help="禁用幻觉文本过滤")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 检查是否安装faster-whisper
    if not FASTER_WHISPER_AVAILABLE:
        print("请先安装 faster-whisper:")
        print("  pip install faster-whisper")
        sys.exit(1)
    
    # 检查Silero VAD
    if not args.no_silero_vad and not SILERO_VAD_AVAILABLE:
        print("提示: Silero VAD未安装，将使用内置VAD")
        print("如需更准确的语音检测，请安装: pip install silero-vad")
        args.no_silero_vad = True
    
    # 检查其他依赖
    try:
        import torch
        import librosa
    except ImportError:
        print("请安装依赖库:")
        print("  pip install torch librosa")
        sys.exit(1)
    
    # 检查ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
    except FileNotFoundError:
        print("警告: 未找到ffmpeg，音频分块功能可能受限")
        print("请安装ffmpeg: https://ffmpeg.org/download.html")
    
    # 创建转录器并开始处理
    use_silero_vad = not args.no_silero_vad
    filter_hallucinations = not args.no_filter_hallucinations
    
    transcriber = AntiHallucinationTranscriber(
        args.input_file, 
        args.model, 
        args.language, 
        args.overlap,
        use_silero_vad,
        args.vad_threshold,
        args.min_speech_duration,
        args.max_speech_duration,
        adult_mode=True,
        filter_hallucinations=filter_hallucinations
    )
    
    print("\n" + "="*70)
    print(f"开始转录: {args.input_file}")
    print(f"使用模型: {args.model}")
    print(f"使用VAD: {'增强版VAD' if use_silero_vad else '内置VAD'}")
    print(f"抗幻觉模式: {'启用' if filter_hallucinations else '禁用'}")
    print(f"VAD阈值: {args.vad_threshold} (越高越严格)")
    print(f"最小语音时长: {args.min_speech_duration}ms")
    print(f"最大语音时长: {args.max_speech_duration}s")
    print(f"重叠时长: {args.overlap}秒")
    print("="*70 + "\n")
    
    success = transcriber.process()
    
    if success:
        print("任务完成!")
        sys.exit(0)
    else:
        print("任务失败，请检查日志文件")
        sys.exit(1)

if __name__ == "__main__":
    main()
