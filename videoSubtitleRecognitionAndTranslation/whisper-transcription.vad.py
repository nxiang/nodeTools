import os
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import hashlib
from datetime import datetime
import subprocess

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

class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, audio_path: str, temp_dir: str):
        self.audio_path = audio_path
        self.temp_dir = temp_dir
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
                audio, sr = librosa.load(self.audio_path, sr=None, mono=True)
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
    
    def extract_audio_chunks(self, chunk_duration: int = 180) -> List[Tuple[int, str]]:
        """提取音频分块 (3分钟 = 180秒)"""
        chunks = []
        total_duration = self.audio_info['duration']
        
        # 创建chunks目录
        chunks_dir = os.path.join(self.temp_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # 计算分块数量
        num_chunks = int(np.ceil(total_duration / chunk_duration))
        
        logging.info(f"音频总时长: {TimeFormatter.format_seconds(total_duration)}")
        logging.info(f"将分割为 {num_chunks} 个分块 (每个 {chunk_duration} 秒)")
        
        # 预检查已存在的chunks
        existing_chunks = []
        for i in range(num_chunks):
            chunk_path = os.path.join(chunks_dir, f"chunk_{i:04d}.wav")
            if os.path.exists(chunk_path):
                existing_chunks.append((i, chunk_path))
        
        if existing_chunks:
            logging.info(f"发现 {len(existing_chunks)} 个已存在的chunks，跳过生成")
        
        # 使用ffmpeg分割音频（仅生成缺失的chunks）
        start_time_total = time.time()
        
        for i in range(num_chunks):
            chunk_path = os.path.join(chunks_dir, f"chunk_{i:04d}.wav")
            
            # 检查是否已存在
            if os.path.exists(chunk_path):
                chunks.append((i, chunk_path))
                continue
            
            start_time = i * chunk_duration
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
            
            # 优化：越到后面使用更快的seek方式
            # 前10个chunk使用精确seek，后面使用快速seek
            if i < 10:
                # 精确seek：先-i后-ss，精度高但慢
                cmd = [
                    'ffmpeg', '-i', self.audio_path,
                    '-ss', str(start_time),
                    '-t', str(chunk_duration),
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
                    '-t', str(chunk_duration),
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
                stdout, stderr = process.communicate(timeout=120)  # 增加超时时间到120秒
                
                chunk_elapsed_time = time.time() - chunk_start_time
                
                if process.returncode == 0:
                    chunks.append((i, chunk_path))
                    logging.info(f"已创建分块 {i+1}/{num_chunks}: {chunk_path} (耗时: {chunk_elapsed_time:.2f}s)")
                else:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    logging.error(f"创建分块 {i} 失败 (耗时: {chunk_elapsed_time:.2f}s): {error_msg}")
                    
                    # 尝试使用不同的ffmpeg参数重新生成
                    if "Invalid data found" in error_msg or "moov atom not found" in error_msg:
                        logging.warning(f"尝试使用备用参数重新生成chunk {i}")
                        if self._retry_extract_chunk(i, start_time, chunk_duration, chunk_path):
                            chunks.append((i, chunk_path))
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
        
        logging.info(f"音频分块完成，共 {len(chunks)} 个chunks")
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

class OptimizedTranscriber:
    """优化转录器"""
    
    def __init__(self, input_file: str, model_name: str, language: str = "ja"):
        self.input_file = os.path.abspath(input_file)
        self.model_name = model_name
        self.language = language
        
        # 创建临时目录
        audio_name = Path(self.input_file).stem
        self.temp_dir = f"temp/{audio_name}_{model_name}"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.performance = PerformanceTracker()
        self.audio_processor = AudioProcessor(self.input_file, self.temp_dir)
        self.progress = ProgressManager(self.input_file, model_name, self.temp_dir)
        self.model = None
        
        logging.info(f"初始化完成: 输入={self.input_file}, 模型={model_name}, 语言={language}")
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
        """加载模型"""
        self.performance.start_stage("模型加载")
        
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
                download_root=None,  # 使用默认缓存
            )
            
            self.performance.end_stage()
            return model
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            raise
    
    def _get_transcription_options(self):
        """获取转录参数"""
        # 基础参数
        options = {
            "language": self.language,
            "beam_size": 3,
            "best_of": 1,
            "temperature": 0.0,
            "patience": 1.0,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        }
        
        # 语言特定的优化
        if self.language == "ja":
            options["initial_prompt"] = "以下是日语转写文本："
        elif self.language == "zh":
            options["initial_prompt"] = "以下是中文转写文本："
        elif self.language == "en":
            options["initial_prompt"] = "以下是英语转写文本："
        else:
            options["initial_prompt"] = f"以下是{self.language}语言转写文本："
        
        return options
    
    def _get_vad_parameters(self):
        """VAD参数优化 - 改进版本，减少环境噪声误识别"""
        # 根据语言调整参数
        if self.language == "ja":
            # 日语参数：更严格的设置，减少环境噪声
            return {
                "threshold": 0.3,  # 提高阈值，过滤掉大部分环境噪声
                "min_speech_duration_ms": 300,  # 增加最小语音时长，过滤短噪声
                "max_speech_duration_s": 60,  # 适当的最大语音时长
                "min_silence_duration_ms": 200,  # 增加静音间隔，更好区分语音
                "speech_pad_ms": 100,  # 适当填充
            }
        elif self.language == "zh":
            # 中文参数：中等严格设置
            return {
                "threshold": 0.25,
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": 60,
                "min_silence_duration_ms": 150,
                "speech_pad_ms": 80,
            }
        else:
            # 通用参数：中等严格设置
            return {
                "threshold": 0.2,  # 提高阈值，过滤环境噪声
                "min_speech_duration_ms": 200,  # 检测较长的语音片段
                "max_speech_duration_s": 60,  # 适当的最大语音时长
                "min_silence_duration_ms": 120,  # 增加静音间隔
                "speech_pad_ms": 60,  # 适当填充
            }
    
    def transcribe_chunk(self, chunk_id: int, chunk_path: str) -> Dict:
        """转录单个chunk"""
        try:
            # 检查chunk文件是否存在且有效
            if not os.path.exists(chunk_path):
                logging.error(f"chunk文件不存在: {chunk_path}")
                return None
                
            if os.path.getsize(chunk_path) == 0:
                logging.error(f"chunk文件为空: {chunk_path}")
                return None
            
            # 加载chunk音频（使用更高效的方式）
            import librosa
            try:
                audio, sr = librosa.load(chunk_path, sr=16000, mono=True, duration=180)
                
                # 检查音频数据是否有效
                if len(audio) == 0 or np.max(np.abs(audio)) < 0.001:
                    logging.warning(f"chunk {chunk_id} 音频数据过小或无效")
                    return {
                        "chunk_id": chunk_id,
                        "chunk_start": chunk_id * 180,
                        "chunk_end": (chunk_id + 1) * 180,
                        "segments": [],
                        "language": self.language,
                        "language_probability": 0.0,
                    }
            except Exception as load_error:
                logging.error(f"加载chunk音频失败: {load_error}")
                return None
            
            # 转录参数
            options = self._get_transcription_options()
            vad_params = self._get_vad_parameters()
            
            # 执行转录
            try:
                segments, info = self.model.transcribe(
                    audio,
                    vad_filter=True,
                    vad_parameters=vad_params,
                    **options
                )
                
                # 处理结果（限制结果数量，避免内存泄漏）
                chunk_segments = []
                chunk_start_time = chunk_id * 180  # 每个chunk 180秒
                segment_count = 0
                
                for segment in segments:
                    if segment_count >= 100:  # 限制每个chunk最多100个segment
                        logging.warning(f"chunk {chunk_id} 超过100个segments，截断处理")
                        break
                        
                    segment_dict = {
                        "start": chunk_start_time + segment.start,
                        "end": chunk_start_time + segment.end,
                        "text": segment.text.strip(),
                    }
                    chunk_segments.append(segment_dict)
                    segment_count += 1
                
                # 强制清理内存
                del audio
                import gc
                gc.collect()
                
                return {
                    "chunk_id": chunk_id,
                    "chunk_start": chunk_start_time,
                    "chunk_end": chunk_start_time + 180,
                    "segments": chunk_segments,
                    "language": info.language,
                    "language_probability": info.language_probability,
                }
                
            except Exception as transcribe_error:
                logging.error(f"转录chunk {chunk_id}失败: {transcribe_error}")
                return None
            
        except Exception as e:
            logging.error(f"转录chunk {chunk_id}失败: {e}")
            return None
    
    def process(self):
        """主处理流程"""
        total_start = time.time()
        
        try:
            # 1. 加载模型
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
            
            # 初始化累计耗时统计
            total_elapsed_time = 0
            
            for chunk_id, chunk_path in chunks:
                # 跳过已完成的chunk
                if chunk_id in completed_chunks:
                    logging.info(f"跳过已完成的chunk {chunk_id}")
                    continue
                
                # 开始处理当前chunk
                chunk_start = time.time()
                logging.info(f"开始处理chunk {chunk_id+1}/{len(chunks)}")
                
                # 转录chunk
                result = self.transcribe_chunk(chunk_id, chunk_path)
                
                if result:
                    # 保存进度
                    self.progress.save_progress(chunk_id, result)
                    completed_results[str(chunk_id)] = result
                    
                    chunk_time = time.time() - chunk_start
                    total_elapsed_time += chunk_time
                    logging.info(f"完成chunk {chunk_id}，耗时: {TimeFormatter.format_seconds(chunk_time)}，累计耗时: {TimeFormatter.format_seconds(total_elapsed_time)}")
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
            
            # 7. 打印详细统计信息
            total_time = time.time() - total_start
            audio_duration = self.audio_processor.audio_info['duration']
            
            print("\n" + "="*60)
            print(f"转录完成!")
            print(f"音频时长: {TimeFormatter.format_seconds(audio_duration)}")
            print(f"总处理时间: {TimeFormatter.format_seconds(total_time)}")
            
            # 防止除零错误
            if audio_duration > 0:
                real_time_factor = total_time / audio_duration
                print(f"实时因子: {real_time_factor:.2f}x")
            else:
                print("实时因子: 无法计算（音频时长为0）")
                
            print(f"临时文件位置: {self.temp_dir}")
            print("="*60)
            
            # 详细耗时统计
            print("\n" + "="*50)
            print("详细耗时统计")
            print("="*50)
            
            # 获取性能追踪器的数据
            performance_data = self.performance.get_summary()
            
            # 打印各阶段耗时
            print("各阶段耗时详情:")
            for stage, duration in performance_data["各阶段耗时"].items():
                print(f"  {stage}: {duration}")
            
            # 计算并打印累计耗时
            cumulative_time = 0
            print("\n累计耗时分析:")
            for stage, duration_str in performance_data["各阶段耗时"].items():
                # 将时间字符串转换回秒数
                time_parts = duration_str.split(':')
                if len(time_parts) == 3:
                    hours, minutes, seconds = map(int, time_parts)
                    stage_seconds = hours * 3600 + minutes * 60 + seconds
                    cumulative_time += stage_seconds
                    print(f"  {stage}: {TimeFormatter.format_seconds(cumulative_time)}")
            
            print(f"\n总累计耗时: {performance_data['总耗时']}")
            print(f"实时因子: {performance_data['实时因子']}")
            print("="*50)
            
            # 性能摘要（保持原有格式）
            self.performance.print_summary()
            
            return True
            
        except Exception as e:
            logging.error(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _filter_noise_segments(self, segments: List[Dict]) -> List[Dict]:
        """过滤环境噪声片段 - 智能后处理"""
        filtered_segments = []
        
        for segment in segments:
            text = segment.get("text", "").strip()
            
            # 跳过空文本
            if not text:
                continue
                
            # 计算语音时长（秒）
            speech_duration = segment["end"] - segment["start"]
            
            # 过滤规则：
            # 1. 文本长度过短（<2个字符）且语音时长过短（<0.5秒）
            if len(text) < 2 and speech_duration < 0.5:
                logging.debug(f"过滤短噪声片段: '{text}' (时长: {speech_duration:.2f}s)")
                continue
                
            # 2. 重复字符检测（如"啊啊啊"、"嗯嗯嗯"等）
            if len(text) >= 3 and len(set(text)) <= 2:
                logging.debug(f"过滤重复字符片段: '{text}'")
                continue
                
            # 3. 常见环境噪声词汇过滤
            noise_words = ["嗯", "啊", "呃", "唔", "哦", "哼", "呼", "嘶", "啪", "咔", "咚"]
            if text in noise_words and speech_duration < 1.0:
                logging.debug(f"过滤环境噪声词汇: '{text}'")
                continue
                
            # 4. 日语特定噪声过滤
            if self.language == "ja":
                japanese_noise = ["え", "あ", "う", "い", "お", "ん", "は", "ふ", "へ", "ほ"]
                if text in japanese_noise and speech_duration < 1.0:
                    logging.debug(f"过滤日语噪声: '{text}'")
                    continue
            
            # 5. 语音时长合理性检查
            if speech_duration > 0.1 and speech_duration < 10.0:  # 合理的语音时长范围
                filtered_segments.append(segment)
            else:
                logging.debug(f"过滤异常时长片段: '{text}' (时长: {speech_duration:.2f}s)")
        
        # 统计过滤结果
        original_count = len(segments)
        filtered_count = len(filtered_segments)
        
        if original_count > 0:
            logging.info(f"噪声过滤完成: {filtered_count}/{original_count} 片段保留 (过滤率: {(1 - filtered_count/original_count)*100:.1f}%)")
        
        return filtered_segments
    
    def _analyze_audio_quality(self, chunk_path: str, segment_start: float, segment_end: float) -> Dict:
        """分析音频片段质量"""
        try:
            import librosa
            import numpy as np
            
            # 加载音频片段
            audio, sr = librosa.load(chunk_path, sr=None, mono=True)
            
            # 计算片段在音频中的位置
            start_sample = int(segment_start * sr)
            end_sample = int(segment_end * sr)
            
            # 确保索引在有效范围内
            start_sample = max(0, min(start_sample, len(audio)-1))
            end_sample = max(start_sample + 1, min(end_sample, len(audio)))
            
            segment_audio = audio[start_sample:end_sample]
            
            if len(segment_audio) == 0:
                return {"energy": 0, "is_silent": True}
            
            # 计算音频能量（RMS）
            energy = np.sqrt(np.mean(segment_audio**2))
            
            # 判断是否为静音（能量低于阈值）
            is_silent = energy < 0.01  # 可调整的阈值
            
            return {
                "energy": energy,
                "is_silent": is_silent,
                "duration": len(segment_audio) / sr
            }
            
        except Exception as e:
            logging.debug(f"音频质量分析失败: {e}")
            return {"energy": 0, "is_silent": False}
    
    def _merge_results(self, chunk_results: Dict) -> Dict:
        """合并所有chunk的结果"""
        all_segments = []
        all_text = []
        
        # 按chunk_id排序
        for chunk_id in sorted(map(int, chunk_results.keys())):
            result = chunk_results[str(chunk_id)]
            if result and "segments" in result:
                all_segments.extend(result["segments"])
                for seg in result["segments"]:
                    all_text.append(seg.get("text", ""))
        
        # 按时间排序
        all_segments.sort(key=lambda x: x["start"])
        
        # 应用噪声过滤
        filtered_segments = self._filter_noise_segments(all_segments)
        
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
            f.write(f"总片段数: {result.get('info', {}).get('total_segments', 0)}\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入带时间戳的文本内容
            segments = result.get("segments", [])
            for segment in segments:
                start_time = TimeFormatter.format_timestamp(segment["start"])
                end_time = TimeFormatter.format_timestamp(segment["end"])
                text = segment.get("text", "").strip()
                
                if text:  # 只写入有内容的片段
                    f.write(f"[{start_time} - {end_time}] {text}\n")
        
        # SRT格式
        srt_path = os.path.join(self.temp_dir, "transcription.srt")
        srt_success = self._save_as_srt(result["segments"], srt_path)
        
        logging.info(f"结果已保存到: {self.temp_dir}")
        logging.info(f"  - JSON: {json_path}")
        logging.info(f"  - TXT: {txt_path}")
        if srt_success:
            logging.info(f"  - SRT: {srt_path}")
        else:
            logging.warning(f"  - SRT: 生成失败，请检查日志")
    
    def _save_as_srt(self, segments: List[Dict], output_path: str):
        """保存为SRT格式"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, 1):
                    start = TimeFormatter.format_timestamp(segment["start"])
                    end = TimeFormatter.format_timestamp(segment["end"])
                    text = segment.get("text", "").strip()
                    
                    # 跳过空文本的片段
                    if not text:
                        continue
                        
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")
            
            logging.info(f"SRT文件已成功生成: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"生成SRT文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="日语视频转录工具")
    parser.add_argument("input_file", type=str, help="输入音频/视频文件路径")
    parser.add_argument("--model", type=str, default="large-v3-turbo",
                       choices=["tiny", "base", "small", "medium", "large-v2", "large-v3-turbo"],
                       help="Whisper模型大小 (默认: large-v3-turbo)")
    parser.add_argument("--language", "-l", default="ja", 
                       help="音频语言代码 (默认: ja - 日语)")
    
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
    
    # 检查依赖
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
    transcriber = OptimizedTranscriber(args.input_file, args.model, args.language)
    
    print("\n" + "="*60)
    print(f"开始转录: {args.input_file}")
    print(f"使用模型: {args.model}")
    print("="*60 + "\n")
    
    success = transcriber.process()
    
    if success:
        print("任务完成!")
        sys.exit(0)
    else:
        print("任务失败，请检查日志文件")
        sys.exit(1)

if __name__ == "__main__":
    main()
