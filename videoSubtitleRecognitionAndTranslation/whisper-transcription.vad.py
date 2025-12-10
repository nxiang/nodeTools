#!/usr/bin/env python3
"""
Whisper转录 + 日语成人向视频VAD检测
集成断点续传功能的完整解决方案 - 统一分块处理版本
"""

import os
import json
import logging
import numpy as np
import torch
from scipy import signal
import librosa
import soundfile as sf
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import time
from datetime import datetime
import warnings
import gc
import psutil
import shutil
warnings.filterwarnings("ignore")

# 自定义JSON编码器处理numpy数据类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# 导入现有的whisper相关模块
try:
    import whisper
    from whisper.utils import get_writer
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("警告: whisper未安装，部分功能可能受限")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量记录处理开始时间
_start_time = None
_last_chunk_time = None

def log_memory_usage(prefix="", chunk_duration=None, total_duration=None):
    """记录当前内存使用情况，支持显示耗时和累计耗时"""
    global _start_time, _last_chunk_time
    
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # 获取磁盘使用情况（Windows系统）
        disk_usage = psutil.disk_usage('C:')
        disk_free_gb = disk_usage.free / 1024 / 1024 / 1024
        
        # 获取系统内存使用情况
        system_memory = psutil.virtual_memory()
        system_memory_used_gb = (system_memory.total - system_memory.available) / 1024 / 1024 / 1024
        system_memory_total_gb = system_memory.total / 1024 / 1024 / 1024
        
        # 构建日志消息
        log_message = f"{prefix}进程内存: {memory_mb:.1f}MB, 系统内存: {system_memory_used_gb:.1f}/{system_memory_total_gb:.1f}GB, 磁盘剩余: {disk_free_gb:.1f}GB"
        
        # 添加耗时信息
        current_time = time.time()
        if _start_time is None:
            _start_time = current_time
            _last_chunk_time = current_time
        
        # 计算当前分块耗时和累计耗时
        chunk_elapsed = current_time - _last_chunk_time
        total_elapsed = current_time - _start_time
        
        # 格式化时间为时分秒
        chunk_time_str = format_time(chunk_elapsed)
        total_time_str = format_time(total_elapsed)
        
        # 添加耗时信息到日志消息
        log_message += f", 耗时: {chunk_time_str}, 累计: {total_time_str}"
        
        # 如果提供了分块时长和总时长，显示进度百分比
        if chunk_duration is not None and total_duration is not None:
            # 计算累计进度：当前分块结束时间占总时长的比例
            progress_percent = (chunk_duration / total_duration) * 100
            log_message += f", 进度: {progress_percent:.1f}%"
        
        logger.info(log_message)
        
        # 更新最后分块时间
        _last_chunk_time = current_time
        
        # 如果磁盘空间不足或内存使用过高，发出警告
        if disk_free_gb < 5:
            logger.warning(f"磁盘空间不足！仅剩 {disk_free_gb:.1f}GB")
        if system_memory_used_gb / system_memory_total_gb > 0.9:
            logger.warning(f"系统内存使用率过高: {system_memory_used_gb/system_memory_total_gb*100:.1f}%")
            
    except Exception as e:
        logger.warning(f"内存监控失败: {e}")

def reset_memory_log_timer():
    """重置内存日志计时器"""
    global _start_time, _last_chunk_time
    _start_time = None
    _last_chunk_time = None

def cleanup_memory():
    """清理内存"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("内存清理完成")
    except Exception as e:
        logger.warning(f"内存清理失败: {e}")

def format_time(seconds: float) -> str:
    """将秒数格式化为时分秒格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    elif minutes > 0:
        return f"{minutes:02d}:{seconds:06.3f}"
    else:
        return f"{seconds:06.3f}s"

@dataclass
class VADConfig:
    """针对日语成人视频的VAD配置 - 优化版"""
    # 基础参数
    sample_rate: int = 16000
    frame_duration: int = 40  # 增加帧时长以提高性能（40毫秒）
    threshold: float = 0.3  # 进一步降低阈值以提高成人内容检测灵敏度
    model_name: str = "adult_content"  # 模型名称，用于临时文件隔离
    
    # 成人视频特定参数
    min_speech_duration: float = 0.15  # 更短的最小持续时间以适应快速声音变化
    min_silence_duration: float = 0.1  # 更短的静音持续时间
    
    # 频率范围（针对成人内容特殊声音优化）
    low_freq: int = 60  # 更低的低频以检测喘息声和呻吟声
    high_freq: int = 5000  # 扩展高频范围以检测尖叫和耳语
    
    # 能量阈值（大幅优化）
    energy_threshold: float = 0.0005  # 大幅降低能量阈值以适应成人内容微弱声音
    
    # 特殊声音检测（重点优化）
    detect_moans: bool = True  # 重点检测呻吟声
    detect_whispers: bool = True  # 重点检测耳语
    detect_screams: bool = False  # 禁用尖叫检测以提升性能（成人内容较少）
    moan_freq_range: Tuple[int, int] = (50, 300)  # 扩展呻吟声频率范围
    
    # 后处理
    merge_gap: float = 0.3  # 缩短合并间隙以适应快速场景切换
    padding: float = 0.2  # 减少填充时间
    max_segment_duration: float = 120.0  # 缩短最大段持续时间
    
    # 针对成人内容的特殊参数
    japanese_phoneme_threshold: float = 0.1  # 大幅降低阈值以提高日语检测灵敏度
    vowel_detection: bool = True  # 日语元音检测
    
    # 性能优化参数
    enable_lightweight_features: bool = True  # 启用轻量级特征计算
    skip_mfcc_calculation: bool = True  # 跳过MFCC计算以提升性能
    
    # 分块处理参数
    chunk_duration: float = 120.0  # 缩短分块时长以提高响应速度
    min_chunk_duration: float = 30.0  # 最小30秒
    max_chunk_duration: float = 300.0  # 最大5分钟

@dataclass
class TranscriptionConfig:
    """转录配置"""
    model_size: str = "large"  # whisper模型大小
    language: str = "ja"  # 日语
    task: str = "transcribe"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch.cuda.is_available() else "float32"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    word_timestamps: bool = True
    prepend_punctuations: str = "\"'¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、"
    
    # 输出格式
    output_formats: List[str] = field(default_factory=lambda: ["txt", "srt", "vtt", "tsv", "json"])
    
    # 临时文件目录 - 使用项目目录下的temp
    temp_dir: str = str(Path(__file__).parent / "temp")
    
    # 断点续传
    save_checkpoint_interval: int = 10  # 每10个片段保存一次检查点

# 成人内容专用配置
ADULT_CONTENT_CONFIG = VADConfig(
    # 性能优化参数
    frame_duration=40,
    skip_mfcc_calculation=True,
    enable_lightweight_features=True,
    
    # 成人内容检测优化
    detect_moans=True,
    detect_whispers=True,
    detect_screams=False,  # 禁用尖叫检测提升性能
    
    # 灵敏度优化
    energy_threshold=0.0005,
    japanese_phoneme_threshold=0.1,
    
    # 分块优化
    chunk_duration=180.0,
    min_chunk_duration=60.0,
    max_chunk_duration=600.0
)

# 标准配置（普通视频）
STANDARD_CONFIG = VADConfig(
    # 标准参数
    frame_duration=30,
    skip_mfcc_calculation=False,
    enable_lightweight_features=False,
    
    # 标准检测
    detect_moans=False,
    detect_whispers=False,
    detect_screams=False,
    
    # 标准灵敏度
    energy_threshold=0.001,
    japanese_phoneme_threshold=0.15
)

class JapaneseAdultVAD:
    """日语成人视频专用的VAD检测器 - 统一分块处理版本"""
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._init_filters()
    
    def switch_to_adult_mode(self):
        """切换到成人内容检测模式"""
        self.config = ADULT_CONTENT_CONFIG
        self._init_filters()
        logger.info("已切换到成人内容检测模式（优化性能+高灵敏度）")
        
    def switch_to_standard_mode(self):
        """切换到标准检测模式"""
        self.config = STANDARD_CONFIG
        self._init_filters()
        logger.info("已切换到标准检测模式（快速处理）")
    
    def get_current_mode(self) -> str:
        """获取当前检测模式"""
        if (self.config.detect_moans and 
            self.config.detect_whispers and 
            not self.config.detect_screams and
            self.config.energy_threshold <= 0.0005):
            return "成人内容模式"
        else:
            return "标准模式"
        
    def _init_filters(self):
        """初始化滤波器"""
        nyquist = self.config.sample_rate / 2
        
        # 主带通滤波器（针对日语语音优化）
        if self.config.low_freq < nyquist and self.config.high_freq < nyquist:
            self.bandpass_filter = signal.butter(
                4,
                [self.config.low_freq/nyquist, self.config.high_freq/nyquist],
                btype='band'
            )
        else:
            # 使用默认值
            self.bandpass_filter = signal.butter(4, [80/nyquist, 4000/nyquist], btype='band')
        
        # 呻吟声检测滤波器
        moan_low = max(10, self.config.moan_freq_range[0]) / nyquist
        moan_high = min(nyquist-1, self.config.moan_freq_range[1]) / nyquist
        self.moan_filter = signal.butter(4, [moan_low, moan_high], btype='band')
    
    def _get_audio_chunk_path(self, audio_path: str, chunk_index: int, target_sr: int) -> str:
        """生成音频分块文件路径"""
        audio_name = Path(audio_path).stem
        model_name = self.config.model_name.replace('/', '_')
        
        # 创建项目目录下的temp/视频名_模型名/chunks目录
        temp_dir = Path(__file__).parent / "temp"
        chunk_dir = temp_dir / f"{audio_name}_{model_name}" / "chunks"
        os.makedirs(chunk_dir, exist_ok=True)
        
        # 生成分块文件名
        return str(chunk_dir / f"chunk_{chunk_index:04d}_{target_sr}Hz.wav")
    
    def test_audio_chunk(self, chunk_path: str):
        """测试音频分块并生成报告"""
        try:
            audio, sr = self.load_audio_chunk(chunk_path)
            
            print(f"\n=== 音频分块测试报告 ===")
            print(f"文件: {chunk_path}")
            print(f"采样率: {sr} Hz")
            print(f"时长: {len(audio)/sr:.2f} 秒")
            print(f"样本数: {len(audio)}")
            
            # 基本统计
            print(f"\n基本统计:")
            print(f"  最大值: {np.max(audio):.6f}")
            print(f"  最小值: {np.min(audio):.6f}")
            print(f"  平均值: {np.mean(audio):.6f}")
            print(f"  标准差: {np.std(audio):.6f}")
            
            # 能量统计
            energy = np.mean(audio ** 2)
            print(f"\n能量统计:")
            print(f"  总能量: {energy:.8f}")
            print(f"  对数能量: {np.log(energy + 1e-10):.4f}")
            
            # 频谱分析
            stft = librosa.stft(audio[:min(len(audio), 2048)])
            magnitude = np.abs(stft)
            spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0].mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0].mean()
            
            print(f"\n频谱分析:")
            print(f"  频谱质心: {spectral_centroid:.1f} Hz")
            print(f"  频谱带宽: {spectral_bandwidth:.1f} Hz")
            
            # 检测语音
            vad_result = self.detect_voice_activity(audio, sr)
            print(f"\nVAD检测结果:")
            print(f"  发现 {len(vad_result)} 个语音段")
            for i, (start, end, info) in enumerate(vad_result):
                print(f"    段{i+1}: {start:.2f}s - {end:.2f}s (时长: {end-start:.2f}s)")
            
            return True
        except Exception as e:
            print(f"测试失败: {e}")
            return False
    
    def _get_chunk_checkpoint_path(self, audio_path: str) -> str:
        """生成分块检查点文件路径"""
        audio_name = Path(audio_path).stem
        model_name = self.config.model_name.replace('/', '_')
        
        # 创建项目目录下的temp/视频名_模型名目录
        temp_dir = Path(__file__).parent / "temp"
        checkpoint_dir = temp_dir / f"{audio_name}_{model_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 生成检查点文件名
        return str(checkpoint_dir / f"{audio_name}_chunk_checkpoint.json")
    
    def _get_vad_checkpoint_path(self, audio_path: str) -> str:
        """生成VAD检查点文件路径"""
        audio_name = Path(audio_path).stem
        model_name = self.config.model_name.replace('/', '_')
        
        # 创建项目目录下的temp/视频名_模型名目录
        temp_dir = Path(__file__).parent / "temp"
        checkpoint_dir = temp_dir / f"{audio_name}_{model_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 生成检查点文件名
        return str(checkpoint_dir / f"{audio_name}_vad_checkpoint.json")
    
    def _get_video_duration(self, video_path: str) -> float:
        """获取视频总时长"""
        try:
            import subprocess
            
            ffprobe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                logger.warning(f"ffprobe获取时长失败，使用默认值7200秒")
                return 7200.0
                
        except Exception as e:
            logger.warning(f"获取视频时长失败: {e}，使用默认值7200秒")
            return 7200.0
    
    def _extract_audio_chunk(self, audio_path: str, start_time: float, end_time: float, 
                           target_sr: int, chunk_index: int, force_recreate: bool = False) -> Optional[str]:
        """提取音频分块到文件，返回文件路径"""
        chunk_path = self._get_audio_chunk_path(audio_path, chunk_index, target_sr)
        
        # 如果文件已存在且不需要强制重新创建，则直接返回路径
        if os.path.exists(chunk_path) and not force_recreate:
            logger.info(f"使用现有音频分块: {chunk_path}")
            return chunk_path
        
        try:
            import subprocess
            import tempfile
            
            # 创建临时文件，尝试在与目标目录相同的驱动器上创建
            try:
                # 获取目标驱动器
                target_drive = os.path.splitdrive(chunk_path)[0]
                if target_drive:
                    # 尝试在目标驱动器上创建临时文件
                    temp_dir = os.path.join(target_drive, "Temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix='.wav', 
                        dir=temp_dir,
                        delete=False
                    )
                else:
                    # 如果无法获取驱动器，使用默认临时目录
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            except:
                # 如果失败，回退到默认临时目录
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            
            temp_path = temp_file.name
            temp_file.close()  # 关闭文件句柄，以便ffmpeg可以写入
            
            try:
                # 构建ffmpeg命令
                cmd = [
                    'ffmpeg',
                    '-y',  # 覆盖输出文件
                    '-ss', str(start_time),
                    '-i', audio_path,
                    '-t', str(end_time - start_time),
                    '-ac', '1',
                    '-ar', str(target_sr),
                    '-acodec', 'pcm_s16le',
                    '-f', 'wav',
                    '-loglevel', 'error',  # 减少日志输出
                    temp_path
                ]
                
                # 执行ffmpeg命令
                result = subprocess.run(cmd, capture_output=True, text=False, timeout=300)
                
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ""
                    raise Exception(f"ffmpeg提取失败: {error_msg}")
                
                # 检查文件是否存在且大小合理
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    raise Exception("ffmpeg未生成有效输出文件")
                
                # 确保目标目录存在
                os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
                
                # 使用shutil.copy2复制文件（支持跨驱动器）
                shutil.copy2(temp_path, chunk_path)
                
                logger.info(f"音频分块已保存: {chunk_path} ({end_time-start_time:.1f}s)")
                
                # 删除临时文件
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                return chunk_path
                
            except subprocess.TimeoutExpired:
                # 清理临时文件
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                raise Exception("ffmpeg处理超时（5分钟）")
                
            except Exception as e:
                # 确保清理临时文件
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                raise e
                
        except Exception as e:
            logger.error(f"提取音频分块失败: {e}")
            return None
    
    def _load_chunk_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """加载分块检查点"""
        if not os.path.exists(checkpoint_file):
            return {}
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            logger.info(f"成功加载检查点: {checkpoint_file}")
            return data
            
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
            return {}
    
    def _save_chunk_checkpoint(self, checkpoint_file: str, checkpoint_data: Dict[str, Any]):
        """保存分块检查点"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            
            # 保存检查点数据，使用自定义编码器
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                
            logger.debug(f"检查点已保存: {checkpoint_file}")
            
        except Exception as e:
            logger.warning(f"保存检查点失败: {e}")
    
    def extract_audio_chunks(self, audio_path: str, target_sr: Optional[int] = None, 
                           force_recreate: bool = False) -> Dict[str, Any]:
        """提取音频分块，支持断点续传"""
        target_sr = target_sr or self.config.sample_rate
        
        try:
            # 获取视频总时长
            total_duration = self._get_video_duration(audio_path)
            logger.info(f"视频总时长: {total_duration:.1f}秒 ({total_duration/3600:.1f}小时)")
            
            # 计算分块数量
            chunk_duration = min(max(self.config.chunk_duration, self.config.min_chunk_duration), 
                                self.config.max_chunk_duration)
            total_chunks = int(np.ceil(total_duration / chunk_duration))
            
            # 将秒转换为分钟显示
            chunk_duration_minutes = chunk_duration / 60
            logger.info(f"分块设置: 每个分块约{chunk_duration_minutes:.1f}分钟，共{total_chunks}个分块")
            
            # 加载检查点
            checkpoint_file = self._get_chunk_checkpoint_path(audio_path)
            checkpoint_data = self._load_chunk_checkpoint(checkpoint_file)
            
            # 初始化检查点数据
            if 'extracted_chunks' not in checkpoint_data:
                checkpoint_data['extracted_chunks'] = []
            
            if 'chunk_info' not in checkpoint_data:
                checkpoint_data['chunk_info'] = []
            
            extracted_chunks = checkpoint_data['extracted_chunks']
            chunk_info_list = checkpoint_data['chunk_info']
            
            # 重置内存日志计时器
            reset_memory_log_timer()
            
            # 分块处理
            for chunk_index in range(total_chunks):
                # 检查是否已处理
                if chunk_index in extracted_chunks and not force_recreate:
                    logger.info(f"跳过已处理分块: {chunk_index + 1}/{total_chunks}")
                    continue
                
                # 计算分块时间范围
                start_time = chunk_index * chunk_duration
                end_time = min((chunk_index + 1) * chunk_duration, total_duration)
                chunk_duration_actual = end_time - start_time
                
                # 跳过太短的分块（小于10秒）
                if chunk_duration_actual < 10:
                    logger.info(f"跳过过短分块: {chunk_index + 1} ({start_time//60}分-{end_time//60}分)")
                    extracted_chunks.append(chunk_index)
                    continue
                
                logger.info(f"提取分块 {chunk_index + 1}/{total_chunks}: {start_time//60}分-{end_time//60}分")
                
                try:
                    # 提取音频分块
                    chunk_path = self._extract_audio_chunk(
                        audio_path, start_time, end_time, target_sr, 
                        chunk_index, force_recreate
                    )
                    
                    if chunk_path:
                        # 更新检查点
                        extracted_chunks.append(chunk_index)
                        chunk_info = {
                            'index': chunk_index,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': chunk_duration_actual,
                            'path': chunk_path
                        }
                        
                        # 更新或添加chunk_info
                        found = False
                        for i, info in enumerate(chunk_info_list):
                            if info['index'] == chunk_index:
                                chunk_info_list[i] = chunk_info
                                found = True
                                break
                        
                        if not found:
                            chunk_info_list.append(chunk_info)
                        
                        # 保存检查点
                        checkpoint_data['extracted_chunks'] = extracted_chunks
                        checkpoint_data['chunk_info'] = chunk_info_list
                        checkpoint_data['total_duration'] = total_duration
                        checkpoint_data['sample_rate'] = target_sr
                        checkpoint_data['last_updated'] = datetime.now().isoformat()
                        
                        self._save_chunk_checkpoint(checkpoint_file, checkpoint_data)
                        
                        # 记录内存使用和进度
                        cumulative_duration = end_time
                        log_memory_usage(
                            f"分块{chunk_index + 1}提取完成", 
                            cumulative_duration, 
                            total_duration
                        )
                        
                        # 定期清理内存
                        if (chunk_index + 1) % 3 == 0:
                            cleanup_memory()
                    else:
                        logger.warning(f"分块 {chunk_index + 1} 提取失败")
                        
                except Exception as e:
                    logger.error(f"分块 {chunk_index + 1} 提取失败: {e}")
                    continue
            
            # 按开始时间排序chunk_info
            chunk_info_list.sort(key=lambda x: x['start_time'])
            
            logger.info(f"音频分块提取完成，共 {len(chunk_info_list)} 个分块")
            log_memory_usage("音频分块提取完成")
            
            return {
                'audio_path': audio_path,
                'total_duration': total_duration,
                'sample_rate': target_sr,
                'total_chunks': total_chunks,
                'chunks': chunk_info_list,
                'checkpoint_file': checkpoint_file
            }
            
        except Exception as e:
            logger.error(f"音频分块提取失败: {e}")
            raise
    
    def load_audio_chunk(self, chunk_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """加载单个音频分块并添加调试信息"""
        target_sr = target_sr or self.config.sample_rate
        
        try:
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"音频分块文件不存在: {chunk_path}")
            
            file_size = os.path.getsize(chunk_path)
            logger.info(f"加载音频分块: {chunk_path}, 文件大小: {file_size/1024:.1f}KB")
            
            if os.path.getsize(chunk_path) == 0:
                raise ValueError(f"音频分块文件为空: {chunk_path}")
            
            audio, sr = librosa.load(chunk_path, sr=target_sr, mono=True)
            
            # 调试信息
            if len(audio) > 0:
                audio_abs = np.abs(audio)
                audio_energy = np.mean(audio ** 2)
                audio_max = np.max(audio_abs)
                audio_min = np.min(audio)
                audio_std = np.std(audio)
                
                logger.info(f"音频统计 - 长度: {len(audio)}样本, 时长: {len(audio)/sr:.1f}s")
                logger.info(f"音频统计 - 能量: {audio_energy:.6f}, 最大振幅: {audio_max:.4f}, 最小振幅: {audio_min:.4f}, 标准差: {audio_std:.6f}")
                
                # 检查是否静音
                if audio_energy < 1e-6:
                    logger.warning("音频能量极低，可能为静音")
                if audio_max < 0.01:
                    logger.warning("音频振幅极小，可能为静音")
            
            return audio, sr
        except Exception as e:
            logger.error(f"加载音频分块失败 {chunk_path}: {e}")
            raise
     
    def _frames_to_segments_simple(self, is_speech: np.ndarray, hop_length: int, sample_rate: int):
        """简单的帧到时间段转换"""
        segments = []
        current_start = None
        
        for i, speech in enumerate(is_speech):
            frame_time = i * hop_length / sample_rate
            
            if speech and current_start is None:
                current_start = frame_time
            elif not speech and current_start is not None:
                segments.append((current_start, frame_time))
                current_start = None
        
        # 处理最后一段
        if current_start is not None:
            end_time = len(is_speech) * hop_length / sample_rate
            segments.append((current_start, end_time))
        
        return segments
    
    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float, Dict]]:
        """完整的VAD检测 - 包含特殊声音特征提取"""
        logger.info(f"开始VAD检测: 音频长度 {len(audio)} 样本, 时长 {len(audio)/sample_rate:.2f}秒")
        
        if len(audio) == 0:
            logger.warning("音频为空，跳过VAD检测")
            return []
        
        # 计算音频统计信息
        audio_energy = np.mean(audio ** 2)
        audio_abs = np.abs(audio)
        audio_max = np.max(audio_abs)
        
        logger.info(f"音频统计 - 能量: {audio_energy:.6f}, 最大振幅: {audio_max:.4f}")
        
        # 如果音频能量过低，直接返回
        if audio_energy < 1e-7:
            logger.warning("音频能量过低，跳过VAD检测")
            return []
        
        # 使用更敏感的帧检测
        frame_size = int(0.03 * sample_rate)  # 30ms
        hop_length = frame_size // 2
        
        # 确保音频长度足够
        if len(audio) < frame_size:
            logger.warning(f"音频长度({len(audio)})小于帧大小({frame_size})，跳过VAD检测")
            return []
        
        frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_length)
        
        # 计算每帧能量
        energies = np.mean(frames ** 2, axis=0)
        
        # 动态阈值：使用更低的百分位数
        energy_threshold_percentile = 20  # 使用20百分位
        energy_threshold = np.percentile(energies, energy_threshold_percentile)
        
        # 设置绝对最小值阈值
        energy_threshold = max(energy_threshold, 1e-6)  # 非常低的阈值
        
        logger.info(f"VAD阈值 - 百分位{energy_threshold_percentile}: {energy_threshold:.8f}, 平均能量: {np.mean(energies):.8f}")
        
        # 检测语音帧
        is_speech = energies > energy_threshold
        
        # 统计语音帧比例
        speech_ratio = np.sum(is_speech) / len(is_speech)
        logger.info(f"语音帧比例: {speech_ratio:.1%} ({np.sum(is_speech)}/{len(is_speech)}帧)")
        
        # 提取每帧的特征
        frame_features = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            features = self._extract_frame_features(frame, sample_rate)
            frame_features.append(features)
        
        # 转换为时间段（使用完整的特征提取）
        segments = self._frames_to_segments(is_speech, frame_features, frame_size, hop_length, sample_rate)
        
        logger.info(f"VAD检测完成: 发现 {len(segments)} 个语音段，总时长: {sum(end-start for start, end, _ in segments):.1f}s")
        return segments

    def _frames_to_segments_simple(self, is_speech: np.ndarray, hop_length: int, sample_rate: int):
        """简单的帧到时间段转换"""
        segments = []
        current_start = None
        
        for i, speech in enumerate(is_speech):
            frame_time = i * hop_length / sample_rate
            
            if speech and current_start is None:
                current_start = frame_time
            elif not speech and current_start is not None:
                segments.append((current_start, frame_time))
                current_start = None
        
        # 处理最后一段
        if current_start is not None:
            end_time = len(is_speech) * hop_length / sample_rate
            segments.append((current_start, end_time))
        
        return segments
    
    def _extract_frame_features(self, frame: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """提取帧特征 - 成人内容优化版"""
        # 能量特征
        energy = np.mean(frame ** 2)
        log_energy = np.log(energy + 1e-10)
        
        # 轻量级频谱特征（性能优化）
        stft = librosa.stft(frame, n_fft=256, hop_length=160)  # 减少FFT点数
        magnitude = np.abs(stft)
        
        # 频谱质心（简化计算）
        spectral_centroid = librosa.feature.spectral_centroid(
            S=magnitude, sr=sample_rate, n_fft=256
        )[0].mean()
        
        # 频谱带宽（简化计算）
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            S=magnitude, sr=sample_rate, n_fft=256
        )[0].mean()
        
        # 过零率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(frame)[0].mean()
        
        # 选择性计算MFCC（性能优化）
        if not self.config.skip_mfcc_calculation:
            mfccs = librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=8)  # 减少MFCC维度
            mfcc_mean = np.mean(mfccs, axis=1)
        else:
            mfcc_mean = np.zeros(8)
        
        # 重点检测特殊声音（成人内容优化）
        moan_features = self._detect_moan_features(frame, sample_rate)
        whisper_features = self._detect_whisper_features(frame, sample_rate, magnitude)
        
        # 选择性检测尖叫（性能优化）
        if self.config.detect_screams:
            scream_features = self._detect_scream_features(frame, sample_rate, magnitude)
        else:
            scream_features = {'is_scream': False}
        
        # 日语元音检测
        vowel_score = self._detect_japanese_vowels(frame, sample_rate) if self.config.vowel_detection else 0.0
        
        return {
            'energy': energy,
            'log_energy': log_energy,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'zero_crossing_rate': zero_crossing_rate,
            'mfcc_mean': mfcc_mean,
            'vowel_score': vowel_score,
            'moan_probability': moan_features['probability'],
            'is_moan': moan_features['is_moan'],
            'is_whisper': whisper_features['is_whisper'],
            'is_scream': scream_features['is_scream'],
            'is_japanese_phoneme': vowel_score > self.config.japanese_phoneme_threshold
        }
    
    def _detect_moan_features(self, frame: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """检测呻吟声特征 - 成人内容优化版"""
        try:
            # 应用呻吟声滤波器（优化性能）
            moan_filtered = signal.lfilter(*self.moan_filter, frame)  # 使用lfilter替代filtfilt提升性能
            
            # 计算低频能量
            moan_energy = np.mean(moan_filtered ** 2)
            total_energy = np.mean(frame ** 2)
            
            # 低频能量比（大幅降低阈值）
            low_freq_ratio = moan_energy / (total_energy + 1e-10)
            
            # 简化的节奏特征检测（性能优化）
            if len(frame) > 100:
                # 使用简化的自相关计算
                autocorr = np.correlate(moan_filtered[:500], moan_filtered[:500], mode='valid')
                if len(autocorr) > 0:
                    # 寻找主要峰值
                    peaks, _ = signal.find_peaks(autocorr[:200], height=0.02, distance=20)  # 大幅降低阈值
                    rhythm_regularity = min(len(peaks) / 3.0, 1.0)  # 简化节奏规律性计算
                else:
                    rhythm_regularity = 0.0
            else:
                rhythm_regularity = 0.0
            
            # 成人内容专用检测阈值（大幅提高灵敏度）
            is_moan = (low_freq_ratio > 0.08 or  # 大幅降低低频能量比阈值
                      (total_energy > 0.0003 and low_freq_ratio > 0.05) or  # 微弱声音检测
                      rhythm_regularity > 0.15)  # 降低节奏规律性阈值
            
            # 优化概率计算，提高对微弱声音的敏感性
            probability = min(low_freq_ratio * 5 + rhythm_regularity * 1.2 + total_energy * 100, 1.0)
            
            # 成人内容专用增强检测
            if total_energy > 0.0002 and low_freq_ratio > 0.03:  # 检测极微弱呻吟声
                is_moan = True
                probability = max(probability, 0.4)
            
            # 喘息声和快速呼吸检测
            if total_energy > 0.0005 and low_freq_ratio > 0.06 and rhythm_regularity > 0.1:
                is_moan = True
                probability = max(probability, 0.7)
            
            return {
                'is_moan': is_moan,
                'probability': probability,
                'low_freq_ratio': low_freq_ratio,
                'rhythm_regularity': rhythm_regularity
            }
        except:
            return {'is_moan': False, 'probability': 0.0}
    
    def _detect_whisper_features(self, frame: np.ndarray, sample_rate: int, magnitude: np.ndarray) -> Dict[str, Any]:
        """检测耳语特征 - 成人内容优化版"""
        # 耳语特征（成人内容专用优化）
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate, n_fft=256)[0].mean()
        
        # 简化的频谱滚降计算（性能优化）
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate, roll_percent=0.85)[0].mean()
        
        # 耳语通常能量较低但过零率较高
        energy = np.mean(frame ** 2)
        zcr = librosa.feature.zero_crossing_rate(frame)[0].mean()
        
        # 成人内容专用耳语检测阈值（大幅提高灵敏度）
        is_whisper = (spectral_centroid < 1500 and  # 大幅提高频谱质心阈值
                     spectral_rolloff < 5000 and    # 大幅提高频谱滚降阈值
                     energy < 0.02 and             # 大幅提高能量阈值
                     zcr > 0.02)                   # 大幅降低过零率阈值
        
        # 成人内容专用增强检测
        if energy < 0.008 and spectral_centroid < 1000:  # 微弱耳语检测
            is_whisper = True
        
        # 亲密对话场景检测
        if (energy < 0.015 and 
            spectral_centroid < 1200 and 
            zcr > 0.015 and 
            spectral_rolloff < 4000):
            is_whisper = True
        
        # 呼吸声检测（成人内容常见）
        if energy < 0.003 and spectral_centroid < 800 and zcr < 0.1:
            is_whisper = True
        
        return {'is_whisper': is_whisper}
    
    def _detect_scream_features(self, frame: np.ndarray, sample_rate: int, magnitude: np.ndarray) -> Dict[str, Any]:
        """检测尖叫特征"""
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)[0].mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sample_rate)[0].mean()
        energy = np.mean(frame ** 2)
        
        # 大幅降低尖叫检测阈值，提高成人内容检测灵敏度
        is_scream = (spectral_centroid > 1500 and  # 降低频谱质心阈值
                    spectral_bandwidth > 1000 and  # 降低频谱带宽阈值
                    energy > 0.01)                # 大幅降低能量阈值
        
        # 添加额外的尖叫检测条件
        if spectral_centroid > 1800 and energy > 0.005:
            is_scream = True
            
        return {'is_scream': is_scream}
    
    def _detect_japanese_vowels(self, frame: np.ndarray, sample_rate: int) -> float:
        """检测日语元音特征"""
        # 优化的日语元音检测（あ、い、う、え、お）
        try:
            # 计算频谱包络
            stft = librosa.stft(frame, n_fft=512)
            magnitude = np.abs(stft)
            
            # 日语元音通常在某些频率有特征峰
            # 优化检测方法，提高灵敏度
            freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=512)
            
            # 扩展日语元音特征频率范围，提高检测灵敏度
            vowel_ranges = [
                (200, 400),    # あ（扩展范围）
                (280, 450),    # い（扩展范围）
                (180, 350),    # う（扩展范围）
                (350, 550),    # え（扩展范围）
                (300, 500),    # お（扩展范围）
            ]
            
            vowel_score = 0.0
            for low, high in vowel_ranges:
                mask = (freq_bins >= low) & (freq_bins <= high)
                if np.any(mask):
                    # 使用最大能量而不是平均能量，提高对峰值特征的敏感性
                    range_energy = np.max(magnitude[mask, :])
                    total_energy = np.max(magnitude)
                    if total_energy > 0:
                        # 使用对数缩放提高低能量信号的检测
                        ratio = range_energy / total_energy
                        vowel_score = max(vowel_score, ratio * 1.5)  # 提高权重
            
            return min(vowel_score, 1.0)
        except:
            return 0.0
    
    def _decide_if_voice(self, features: Dict[str, Any]) -> bool:
        """判断是否为语音/声音"""
        is_voice = False
        
        # 规则1：基础能量阈值（大幅降低）
        if features['energy'] > self.config.energy_threshold:
            is_voice = True
        
        # 规则2：特殊声音检测（降低阈值）
        if features['is_moan'] or features['is_scream'] or features['is_whisper']:
            is_voice = True
        
        # 规则3：日语语音特征（降低阈值）
        if features['is_japanese_phoneme'] or features['vowel_score'] > 0.05:
            is_voice = True
        
        # 规则4：频谱特征（放宽范围）
        if (features['spectral_centroid'] > 50 and 
            features['spectral_centroid'] < 5000 and
            features['spectral_bandwidth'] > 200):
            is_voice = True
        
        # 规则5：过零率（适用于呼吸声等，降低阈值）
        if features['zero_crossing_rate'] < 0.2 and features['energy'] > 0.0005:
            is_voice = True
        
        # 规则6：呻吟声概率检测
        if features['moan_probability'] > 0.3:
            is_voice = True
            
        # 规则7：最低能量检测（确保极低能量声音也能被检测）
        if features['energy'] > 0.0001 and features['spectral_centroid'] > 30:
            is_voice = True
        
        return is_voice
    
    def _frames_to_segments(self, is_speech: List[bool], frame_features: List[Dict], 
                           frame_size: int, hop_length: int, sample_rate: int) -> List[Tuple[float, float, Dict]]:
        """将帧检测结果转换为时间段"""
        segments = []
        current_start = None
        current_features = []
        
        for i, speech in enumerate(is_speech):
            frame_time = i * hop_length / sample_rate
            
            if speech and current_start is None:
                current_start = frame_time
                current_features.append(frame_features[i])
            elif speech and current_start is not None:
                current_features.append(frame_features[i])
            elif not speech and current_start is not None:
                duration = frame_time - current_start
                
                if duration >= self.config.min_speech_duration:
                    # 聚合特征
                    segment_info = self._aggregate_features(current_features)
                    segments.append((current_start, frame_time, segment_info))
                
                current_start = None
                current_features = []
        
        # 处理最后一段
        if current_start is not None:
            end_time = len(is_speech) * hop_length / sample_rate
            duration = end_time - current_start
            if duration >= self.config.min_speech_duration:
                segment_info = self._aggregate_features(current_features)
                segments.append((current_start, end_time, segment_info))
        
        return segments
    
    def _aggregate_features(self, features_list: List[Dict]) -> Dict[str, Any]:
        """聚合多个帧的特征"""
        if not features_list:
            return {}
        
        aggregated = {
            'has_moans': any(f['is_moan'] for f in features_list),
            'has_whispers': any(f['is_whisper'] for f in features_list),
            'has_screams': any(f['is_scream'] for f in features_list),
            'has_japanese_phonemes': any(f['is_japanese_phoneme'] for f in features_list),
            'avg_energy': np.mean([f['energy'] for f in features_list]),
            'avg_moan_prob': np.mean([f['moan_probability'] for f in features_list]),
            'avg_vowel_score': np.mean([f['vowel_score'] for f in features_list]),
            'frame_count': len(features_list)
        }
        
        return aggregated
    
    def _merge_segments(self, segments: List[Tuple[float, float, Dict]]) -> List[Tuple[float, float, Dict]]:
        """合并相邻的语音段"""
        if not segments:
            return []
        
        merged = []
        current_start, current_end, current_info = segments[0]
        
        for start, end, info in segments[1:]:
            if start - current_end <= self.config.merge_gap:
                # 合并
                current_end = end
                # 合并信息
                for key in ['has_moans', 'has_whispers', 'has_screams', 'has_japanese_phonemes']:
                    current_info[key] = current_info[key] or info[key]
                current_info['avg_energy'] = (current_info['avg_energy'] + info['avg_energy']) / 2
                current_info['avg_moan_prob'] = (current_info['avg_moan_prob'] + info['avg_moan_prob']) / 2
                current_info['avg_vowel_score'] = (current_info['avg_vowel_score'] + info['avg_vowel_score']) / 2
                current_info['frame_count'] += info['frame_count']
            else:
                merged.append((current_start, current_end, current_info))
                current_start, current_end, current_info = start, end, info
        
        merged.append((current_start, current_end, current_info))
        return merged
    
    def _filter_segments(self, segments: List[Tuple[float, float, Dict]]) -> List[Tuple[float, float, Dict]]:
        """过滤和分割过长的段"""
        filtered_segments = []
        
        for start, end, info in segments:
            duration = end - start
            
            if duration > self.config.max_segment_duration:
                # 分割过长的段
                num_splits = int(np.ceil(duration / self.config.max_segment_duration))
                split_duration = duration / num_splits
                
                for i in range(num_splits):
                    split_start = start + i * split_duration
                    split_end = min(start + (i + 1) * split_duration, end)
                    
                    # 复制信息（可能不准确，但保持简单）
                    filtered_segments.append((split_start, split_end, info.copy()))
            else:
                filtered_segments.append((start, end, info))
        
        return filtered_segments
    
    def _apply_padding(self, segments: List[Tuple[float, float, Dict]], total_duration: float) -> List[Tuple[float, float, Dict]]:
        """应用时间填充"""
        padded_segments = []
        
        for start, end, info in segments:
            padded_start = max(0, start - self.config.padding)
            padded_end = min(total_duration, end + self.config.padding)
            
            # 确保填充后的段至少包含原始内容
            if padded_start <= start and padded_end >= end:
                padded_segments.append((padded_start, padded_end, info))
            else:
                padded_segments.append((start, end, info))
        
        return padded_segments
    
    def analyze_audio_with_chunks(self, audio_path: str, output_json: Optional[str] = None,
                            force_recreate: bool = False) -> Dict[str, Any]:
        """使用音频分块进行VAD分析（支持断点续传）"""
        logger.info(f"开始分块VAD分析: {audio_path}")
        
        # 提取音频分块（支持断点续传）
        chunk_info = self.extract_audio_chunks(audio_path, force_recreate=force_recreate)
        
        # 加载VAD检查点
        vad_checkpoint_file = self._get_vad_checkpoint_path(audio_path)
        vad_checkpoint = self._load_chunk_checkpoint(vad_checkpoint_file)
        
        # 重置内存日志计时器
        reset_memory_log_timer()
        
        all_segments = []
        processed_chunks = vad_checkpoint.get('processed_chunks', [])
        
        # 处理每个音频分块
        for chunk in chunk_info['chunks']:
            chunk_index = chunk['index']
            
            # 检查是否已处理
            if chunk_index in processed_chunks and not force_recreate:
                logger.info(f"跳过已处理VAD分块: {chunk_index + 1}/{len(chunk_info['chunks'])}")
                continue
            
            logger.info(f"处理VAD分块 {chunk_index + 1}/{len(chunk_info['chunks'])}: {chunk['start_time']//60}分-{chunk['end_time']//60}分")
            
            try:
                # 加载音频分块
                audio, sr = self.load_audio_chunk(chunk['path'])
                
                # 检测语音活动
                chunk_segments = self.detect_voice_activity(audio, sr)
                
                # 调整时间戳以匹配全局时间
                adjusted_segments = []
                for start, end, info in chunk_segments:
                    adjusted_start = chunk['start_time'] + start
                    adjusted_end = chunk['start_time'] + end
                    adjusted_segments.append((adjusted_start, adjusted_end, info))
                
                # 添加到总结果中
                all_segments.extend(adjusted_segments)
                
                # 更新检查点
                processed_chunks.append(chunk_index)
                vad_checkpoint['processed_chunks'] = processed_chunks
                vad_checkpoint['segments'] = all_segments
                vad_checkpoint['total_duration'] = chunk_info['total_duration']
                vad_checkpoint['last_updated'] = datetime.now().isoformat()
                
                self._save_chunk_checkpoint(vad_checkpoint_file, vad_checkpoint)
                
                # 记录进度
                cumulative_duration = chunk['end_time']
                log_memory_usage(
                    f"VAD分块{chunk_index + 1}处理完成", 
                    cumulative_duration, 
                    chunk_info['total_duration']
                )
                
                # 定期清理内存
                if (chunk_index + 1) % 3 == 0:
                    cleanup_memory()
                    
            except Exception as e:
                logger.error(f"VAD分块 {chunk_index + 1} 处理失败: {e}")
                continue
        
        # 后处理：合并、过滤、填充
        logger.info("开始后处理: 合并相邻语音段")
        all_segments = self._merge_segments(all_segments)
        logger.info(f"合并后语音段数量: {len(all_segments)}")
        
        logger.info("开始后处理: 过滤短语音段")
        all_segments = self._filter_segments(all_segments)
        logger.info(f"过滤后语音段数量: {len(all_segments)}")
        
        logger.info("开始后处理: 应用时间填充")
        all_segments = self._apply_padding(all_segments, chunk_info['total_duration'])
        
        # 统计信息
        speech_duration = sum(end - start for start, end, _ in all_segments)
        speech_ratio = speech_duration / chunk_info['total_duration'] if chunk_info['total_duration'] > 0 else 0
        
        # 特殊声音统计
        special_counts = {
            'moans': sum(1 for _, _, info in all_segments if info.get('has_moans', False)),
            'whispers': sum(1 for _, _, info in all_segments if info.get('has_whispers', False)),
            'screams': sum(1 for _, _, info in all_segments if info.get('has_screams', False)),
            'japanese_phonemes': sum(1 for _, _, info in all_segments if info.get('has_japanese_phonemes', False))
        }
        
        # 整理结果
        results = {
            'audio_path': audio_path,
            'total_duration': chunk_info['total_duration'],
            'speech_duration': speech_duration,
            'speech_ratio': speech_ratio,
            'segment_count': len(all_segments),
            'special_sounds': special_counts,
            'segments': [
                {
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'metadata': info
                }
                for start, end, info in all_segments
            ],
            'vad_config': self.config.__dict__,
            'analysis_time': datetime.now().isoformat(),
            'processing_mode': 'chunked',
            'chunk_info': chunk_info
        }
        
        # 保存结果
        if output_json:
            os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            logger.info(f"VAD结果已保存到: {output_json}")
        
        logger.info(f"VAD分析完成: 检测到 {len(all_segments)} 个语音段, 语音比例: {speech_ratio:.1%}")
        
        # 可选：处理完成后删除VAD检查点（保留音频分块文件）
        if os.path.exists(vad_checkpoint_file):
            os.remove(vad_checkpoint_file)
            logger.info(f"VAD处理完成，删除检查点文件: {vad_checkpoint_file}")
        
        return results
    
    def analyze_audio(self, audio_path: str, output_json: Optional[str] = None, 
                     force_recreate: bool = False) -> Dict[str, Any]:
        """分析音频并返回VAD结果（统一使用分块处理）"""
        return self.analyze_audio_with_chunks(audio_path, output_json, force_recreate)

class TranscriptionWithVAD:
    """集成VAD的Whisper转录系统（支持断点续传）"""
    
    def __init__(self, vad_config: Optional[VADConfig] = None, 
                 trans_config: Optional[TranscriptionConfig] = None):
        self.vad_config = vad_config or VADConfig()
        self.trans_config = trans_config or TranscriptionConfig()
        
        # 初始化组件
        self.vad = JapaneseAdultVAD(self.vad_config)
        
        # 懒加载Whisper模型（不在初始化时加载）
        self.model = None
        self.model_loaded = False
        
        # 创建临时目录
        os.makedirs(self.trans_config.temp_dir, exist_ok=True)
        
        # 断点状态
        self.checkpoint_state = {}
    
    def _lazy_load_whisper_model(self):
        """懒加载Whisper模型，只在需要时加载"""
        if self.model_loaded:
            return True
            
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper未安装，转录功能不可用")
            return False
        
        try:
            logger.info(f"懒加载Whisper模型: {self.trans_config.model_size}")
            log_memory_usage("模型加载前")
            
            self.model = whisper.load_model(
                self.trans_config.model_size,
                device=self.trans_config.device
            )
            self.model_loaded = True
            
            log_memory_usage("模型加载后")
            logger.info("Whisper模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"Whisper模型加载失败: {e}")
            self.model = None
            self.model_loaded = False
            return False
    
    def get_checkpoint_path(self, audio_path: str) -> str:
        """获取检查点文件路径"""
        audio_name = Path(audio_path).stem
        model_name = self.vad_config.model_name.replace('/', '_')
        
        # 创建隔离目录：temp/视频名_模型名
        isolation_dir = os.path.join(self.trans_config.temp_dir, f"{audio_name}_{model_name}")
        os.makedirs(isolation_dir, exist_ok=True)
        
        # 检查点文件直接放在隔离目录中
        checkpoint_name = f"{audio_name}_transcription_checkpoint.json"
        return os.path.join(isolation_dir, checkpoint_name)
    
    def load_checkpoint(self, audio_path: str) -> Optional[Dict]:
        """加载检查点"""
        checkpoint_path = self.get_checkpoint_path(audio_path)
        
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                logger.info(f"加载检查点: {checkpoint_path}")
                return checkpoint
            except Exception as e:
                logger.warning(f"加载检查点失败: {e}")
        
        return None
    
    def save_checkpoint(self, audio_path: str, state: Dict):
        """保存检查点"""
        checkpoint_path = self.get_checkpoint_path(audio_path)
        
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
            # 同时备份一份
            backup_path = checkpoint_path.replace('.json', f'_backup_{int(time.time())}.json')
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
            logger.debug(f"检查点已保存: {checkpoint_path}")
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def extract_audio_segment_from_chunk(self, audio_path: str, chunk_info: Dict, 
                                       start_time: float, end_time: float) -> Optional[str]:
        """从音频分块中提取音频段到临时文件"""
        try:
            audio_name = Path(audio_path).stem
            model_name = self.vad_config.model_name.replace('/', '_')
            
            # 创建临时目录
            temp_dir = os.path.join(self.trans_config.temp_dir, f"{audio_name}_{model_name}", "segments")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 查找包含该时间段的音频分块
            target_chunk = None
            for chunk in chunk_info['chunks']:
                if chunk['start_time'] <= start_time < chunk['end_time']:
                    target_chunk = chunk
                    break
            
            if not target_chunk:
                logger.warning(f"未找到包含时间 {start_time}s 的音频分块")
                return None
            
            # 加载音频分块
            audio, sr = self.vad.load_audio_chunk(target_chunk['path'])
            
            # 计算相对于分块的开始时间
            chunk_start_time = target_chunk['start_time']
            relative_start_time = start_time - chunk_start_time
            relative_end_time = min(end_time - chunk_start_time, target_chunk['duration'])
            
            # 计算样本索引
            start_sample = int(relative_start_time * sr)
            end_sample = int(relative_end_time * sr)
            
            if start_sample >= len(audio) or end_sample > len(audio):
                logger.warning(f"时间范围超出音频分块长度: {relative_start_time}-{relative_end_time}")
                return None
            
            # 提取段
            segment_audio = audio[start_sample:end_sample]
            
            # 保存到临时文件
            temp_filename = f"segment_{int(start_time)}_{int(end_time)}.wav"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            sf.write(temp_path, segment_audio, sr)
            return temp_path
            
        except Exception as e:
            logger.error(f"从音频分块提取音频段失败: {e}")
            return None
    
    def transcribe_segment(self, audio_path: str, segment_index: int, 
                          start_time: float, end_time: float, chunk_info: Optional[Dict] = None) -> Tuple[Dict[str, Any], float]:
        """转录单个音频段，返回结果和处理时间"""
        # 懒加载Whisper模型
        if not self._lazy_load_whisper_model():
            raise ValueError("Whisper模型加载失败")
        
        start_time_processing = time.time()
        
        # 提取音频段（优先从音频分块中提取）
        temp_audio_path = None
        if chunk_info:
            temp_audio_path = self.extract_audio_segment_from_chunk(audio_path, chunk_info, start_time, end_time)
        
        # 如果无法从分块提取，则使用原始方法（从完整音频提取）
        if not temp_audio_path:
            # 使用原始方法提取音频段
            temp_audio_path = self.extract_audio_segment(audio_path, start_time, end_time)
        
        if not temp_audio_path:
            result = {
                'index': segment_index,
                'start_time': start_time,
                'end_time': end_time,
                'success': False,
                'error': '提取音频段失败'
            }
            return result, 0.0
        
        try:
            # 转录
            result = self.model.transcribe(
                temp_audio_path,
                language=self.trans_config.language,
                task=self.trans_config.task,
                beam_size=self.trans_config.beam_size,
                best_of=self.trans_config.best_of,
                temperature=self.trans_config.temperature,
                compression_ratio_threshold=self.trans_config.compression_ratio_threshold,
                logprob_threshold=self.trans_config.logprob_threshold,
                no_speech_threshold=self.trans_config.no_speech_threshold,
                condition_on_previous_text=self.trans_config.condition_on_previous_text,
                initial_prompt=self.trans_config.initial_prompt,
                word_timestamps=self.trans_config.word_timestamps,
                prepend_punctuations=self.trans_config.prepend_punctuations,
                append_punctuations=self.trans_config.append_punctuations
            )
            
            # 调整时间戳
            for segment in result.get('segments', []):
                segment['start'] += start_time
                segment['end'] += start_time
            
            # 清理临时文件
            try:
                os.remove(temp_audio_path)
            except:
                pass
            
            processing_time = time.time() - start_time_processing
            
            return {
                'index': segment_index,
                'start_time': start_time,
                'end_time': end_time,
                'success': True,
                'result': result
            }, processing_time
            
        except Exception as e:
            logger.error(f"转录段 {segment_index} 失败: {e}")
            
            # 清理临时文件
            try:
                os.remove(temp_audio_path)
            except:
                pass
            
            processing_time = time.time() - start_time_processing
            
            return {
                'index': segment_index,
                'start_time': start_time,
                'end_time': end_time,
                'success': False,
                'error': str(e)
            }, processing_time
    
    def extract_audio_segment(self, audio_path: str, start_time: float, 
                            end_time: float, temp_dir: Optional[str] = None) -> Optional[str]:
        """提取音频段到临时文件（从完整音频）"""
        try:
            if temp_dir is None:
                # 按视频名+模型名创建隔离目录
                audio_name = Path(audio_path).stem
                model_name = self.vad_config.model_name.replace('/', '_')
                temp_dir = os.path.join(self.trans_config.temp_dir, f"{audio_name}_{model_name}")
            
            os.makedirs(temp_dir, exist_ok=True)
            
            # 加载完整音频
            audio, sr = self.vad.load_audio_chunk(audio_path)  # 注意：这里假设已经分块
            
            # 计算样本索引
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            if start_sample >= len(audio) or end_sample > len(audio):
                logger.warning(f"时间范围超出音频长度: {start_time}-{end_time}")
                return None
            
            # 提取段
            segment_audio = audio[start_sample:end_sample]
            
            # 保存到临时文件
            temp_filename = f"segment_{int(start_time)}_{int(end_time)}.wav"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            sf.write(temp_path, segment_audio, sr)
            return temp_path
            
        except Exception as e:
            logger.error(f"提取音频段失败: {e}")
            return None
    
    def transcribe_with_vad(self, audio_path: str, 
                           vad_results: Optional[Dict] = None,
                           force_redo: bool = False) -> Dict[str, Any]:
        """使用VAD进行转录（支持断点续传）"""
        logger.info(f"开始转录: {audio_path}")
        
        # 加载或执行VAD分析
        if vad_results is None:
            # 尝试从检查点加载VAD结果
            checkpoint = None if force_redo else self.load_checkpoint(audio_path)
            
            if checkpoint and 'vad_results' in checkpoint:
                vad_results = checkpoint['vad_results']
                logger.info("使用检查点中的VAD结果")
            else:
                # 执行VAD分析，按视频名+模型名隔离保存VAD结果
                audio_name = Path(audio_path).stem
                model_name = self.vad_config.model_name.replace('/', '_')
                
                # 创建隔离目录：temp/视频名_模型名/
                isolation_dir = os.path.join(self.trans_config.temp_dir, f"{audio_name}_{model_name}")
                os.makedirs(isolation_dir, exist_ok=True)
                
                vad_json_path = os.path.join(
                    isolation_dir,
                    f"{audio_name}_vad.json"
                )
                vad_results = self.vad.analyze_audio(audio_path, vad_json_path, force_redo)
        
        # 准备转录任务
        segments = vad_results.get('segments', [])
        chunk_info = vad_results.get('chunk_info', {})
        
        if not segments:
            logger.warning("未检测到语音段")
            return {
                'audio_path': audio_path,
                'success': False,
                'error': 'No speech segments detected',
                'vad_results': vad_results
            }
        
        # 加载检查点状态
        checkpoint_state = {}
        if not force_redo:
            checkpoint = self.load_checkpoint(audio_path)
            if checkpoint:
                checkpoint_state = checkpoint.get('transcription_state', {})
        
        # 初始化状态
        total_segments = len(segments)
        completed_segments = checkpoint_state.get('completed_segments', [])
        failed_segments = checkpoint_state.get('failed_segments', [])
        results = checkpoint_state.get('results', [])
        
        # 确定需要转录的段
        pending_segments = []
        for i, seg in enumerate(segments):
            if i not in completed_segments and i not in failed_segments:
                pending_segments.append((i, seg))
        
        logger.info(f"转录进度: {len(completed_segments)}/{total_segments} 完成, "
                   f"{len(failed_segments)} 失败, {len(pending_segments)} 待处理")
        
        # 转录待处理的段
        total_start_time = time.time()
        cumulative_time = 0.0
        
        for i, seg in pending_segments:
            start_time = seg['start']
            end_time = seg['end']
            
            # 转录（如果chunk_info可用，则使用分块提取）
            if chunk_info:
                segment_result, segment_time = self.transcribe_segment(
                    audio_path, i, start_time, end_time, chunk_info
                )
            else:
                segment_result, segment_time = self.transcribe_segment(
                    audio_path, i, start_time, end_time
                )
            
            # 更新累计时间
            cumulative_time += segment_time
            
            # 格式化时间显示
            segment_time_formatted = format_time(segment_time)
            cumulative_time_formatted = format_time(cumulative_time)
            progress_percent = (len(completed_segments) + len(failed_segments) + 1) / total_segments * 100
            
            # 更新状态
            if segment_result['success']:
                completed_segments.append(i)
                results.append(segment_result)
                logger.info(f"片段 {i+1}/{total_segments} - 耗时: {segment_time_formatted} - "
                           f"进度: {len(completed_segments) + len(failed_segments)}/{total_segments} ({progress_percent:.1f}%) - "
                           f"累计: {cumulative_time_formatted}")
            else:
                failed_segments.append(i)
                logger.error(f"片段 {i+1}/{total_segments} - 耗时: {segment_time_formatted} - "
                           f"进度: {len(completed_segments) + len(failed_segments)}/{total_segments} ({progress_percent:.1f}%) - "
                           f"累计: {cumulative_time_formatted} - 失败: {segment_result.get('error', 'Unknown error')}")
            
            # 定期保存检查点
            if len(completed_segments) % self.trans_config.save_checkpoint_interval == 0:
                self._save_transcription_checkpoint(
                    audio_path, vad_results, completed_segments, 
                    failed_segments, results
                )
        
        # 最终保存检查点
        self._save_transcription_checkpoint(
            audio_path, vad_results, completed_segments, 
            failed_segments, results
        )
        
        # 合并结果
        final_result = self._merge_transcription_results(results, vad_results)
        
        # 保存输出
        output_paths = self._save_output_files(audio_path, final_result)
        
        logger.info(f"转录完成: {audio_path}")
        
        return {
            'audio_path': audio_path,
            'success': True,
            'vad_results': vad_results,
            'transcription_results': final_result,
            'output_paths': output_paths,
            'statistics': {
                'total_segments': total_segments,
                'completed_segments': len(completed_segments),
                'failed_segments': len(failed_segments),
                'success_rate': len(completed_segments) / total_segments if total_segments > 0 else 0
            }
        }
    
    def _save_transcription_checkpoint(self, audio_path: str, vad_results: Dict,
                                      completed_segments: List[int], 
                                      failed_segments: List[int],
                                      results: List[Dict]):
        """保存转录检查点"""
        checkpoint_state = {
            'audio_path': audio_path,
            'timestamp': datetime.now().isoformat(),
            'vad_results': vad_results,
            'transcription_state': {
                'completed_segments': completed_segments,
                'failed_segments': failed_segments,
                'results': results
            }
        }
        
        self.save_checkpoint(audio_path, checkpoint_state)
    
    def _merge_transcription_results(self, segment_results: List[Dict], 
                                    vad_results: Dict) -> Dict[str, Any]:
        """合并所有段的转录结果，确保句子级别的时间戳"""
        all_segments = []
        all_text = []
        
        # 按时间排序
        segment_results.sort(key=lambda x: x['start_time'])
        
        for seg_result in segment_results:
            if not seg_result['success']:
                continue
            
            result_data = seg_result.get('result', {})
            text = result_data.get('text', '').strip()
            
            if text:
                all_text.append(text)
            
            # 添加段，确保保留句子级别的时间戳
            for seg in result_data.get('segments', []):
                # 确保每个segment都有正确的时间戳
                segment_copy = seg.copy()
                
                # 如果segment有words字段（单词级别时间戳），则保留
                if 'words' in segment_copy:
                    # 确保单词时间戳正确
                    for word in segment_copy['words']:
                        word['start'] = max(0, word.get('start', 0))
                        word['end'] = max(0, word.get('end', 0))
                
                # 确保segment时间戳正确
                segment_copy['start'] = max(0, segment_copy.get('start', 0))
                segment_copy['end'] = max(0, segment_copy.get('end', 0))
                
                all_segments.append(segment_copy)
        
        # 合并文本
        full_text = ' '.join(all_text)
        
        return {
            'vad_metadata': {
                'total_duration': vad_results.get('total_duration', 0),
                'speech_duration': vad_results.get('speech_duration', 0),
                'segment_count': vad_results.get('segment_count', 0),
                'special_sounds': vad_results.get('special_sounds', {})
            },
            'text': full_text,
            'segments': all_segments,
            'language': self.trans_config.language,
            'transcription_time': datetime.now().isoformat()
        }
    
    def _save_output_files(self, audio_path: str, result: Dict) -> Dict[str, str]:
        """保存输出文件"""
        audio_name = Path(audio_path).stem
        model_name = self.vad_config.model_name.replace('/', '_')
        
        # 创建隔离目录：temp/视频名_模型名
        isolation_dir = os.path.join(self.trans_config.temp_dir, f"{audio_name}_{model_name}")
        os.makedirs(isolation_dir, exist_ok=True)
        
        # 转录文本文件放在隔离目录下，命名为"transcription.txt"（与video-translation.py期望一致）
        transcription_path = os.path.join(isolation_dir, "transcription.txt")
        
        # 其他临时文件放在隔离目录下
        isolation_base_path = os.path.join(isolation_dir, audio_name)
        
        output_paths = {}
        
        if WHISPER_AVAILABLE:
            # 使用Whisper的writer保存各种格式
            for format_type in self.trans_config.output_formats:
                if format_type == 'srt':
                    # 对于SRT格式，我们生成转录文本文件而不是SRT文件
                    output_path = transcription_path
                    writer_dir = isolation_dir
                    
                    # 生成转录文本内容
                    transcription_text = self._generate_transcription_text(result)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(transcription_text)
                    
                    output_paths['txt'] = output_path
                    logger.info(f"保存转录文本格式: {output_path}")
                else:
                    # 其他格式放在隔离目录下
                    output_path = f"{isolation_base_path}.{format_type}"
                    writer_dir = isolation_dir
                    
                    try:
                        writer = get_writer(format_type, writer_dir)
                        writer(result, audio_path)
                        
                        output_paths[format_type] = output_path
                        logger.info(f"保存 {format_type.upper()} 格式: {output_path}")
                    except Exception as e:
                        logger.error(f"保存 {format_type} 格式失败: {e}")
        
        # 保存JSON结果（包含VAD元数据）到隔离目录
        json_path = f"{isolation_base_path}_full.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        output_paths['json_full'] = json_path
        
        return output_paths
    
    def _generate_transcription_text(self, result: Dict) -> str:
        """生成转录文本内容"""
        text = ""
        segments = result.get('segments', [])
        
        for i, segment in enumerate(segments, 1):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            segment_text = segment.get('text', '').strip()
            
            # 格式化时间戳
            start_str = self._format_timestamp(start_time)
            end_str = self._format_timestamp(end_time)
            
            text += f"[{start_str} - {end_str}] {segment_text}\n"
        
        return text
    
    def _format_timestamp(self, seconds: float) -> str:
        """格式化时间戳，参照JUQ-587-C.srt文件的正确格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        # 将秒数拆分为整数秒和小数秒（毫秒）
        int_seconds = int(secs)
        milliseconds = int((secs - int_seconds) * 1000)
        
        # 使用逗号分隔毫秒，符合SRT标准格式
        return f"{hours:02d}:{minutes:02d}:{int_seconds:02d},{milliseconds:03d}"

    def process_batch(self, audio_files: List[str], 
                     force_redo: bool = False) -> Dict[str, Any]:
        """批量处理音频文件"""
        results = {}
        
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                logger.warning(f"文件不存在: {audio_file}")
                continue
            
            try:
                logger.info(f"处理文件: {audio_file}")
                result = self.transcribe_with_vad(audio_file, force_redo=force_redo)
                results[audio_file] = result
                
                # 生成摘要日志
                if result.get('success'):
                    stats = result.get('statistics', {})
                    logger.info(f"完成: {audio_file} - "
                               f"成功率: {stats.get('success_rate', 0):.1%}")
            
            except Exception as e:
                logger.error(f"处理失败 {audio_file}: {e}")
                results[audio_file] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results

def main():
    """主函数：示例使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="日语成人视频转录工具（集成VAD）")
    parser.add_argument("input", help="输入音频/视频文件或目录")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper模型大小")
    parser.add_argument("--language", default="ja", help="语言代码")
    parser.add_argument("--force-redo", action="store_true", help="强制重新处理")
    parser.add_argument("--vad-only", action="store_true", help="仅执行VAD分析")
    parser.add_argument("--batch", action="store_true", help="批量处理目录")
    parser.add_argument("--cleanup", action="store_true", help="程序开始执行前清理临时文件")
    parser.add_argument("--chunk-duration", type=float, default=180.0, help="音频分块时长（秒）")
    parser.add_argument("--test-audio", action="store_true", help="测试音频分块")
    
    args = parser.parse_args()
    
    # 如果指定了--cleanup参数，在程序开始前清理临时文件
    if args.cleanup:
        import shutil
        # 只清理当前视频对应的临时目录（视频名+模型名）
        video_name = Path(args.input).stem
        model_name = args.model
        isolation_dir_name = f"{video_name}_{model_name}"
        
        temp_dir = Path(__file__).parent / "temp"
        target_dir = temp_dir / isolation_dir_name
        
        if target_dir.exists():
            logger.info(f"清理视频临时目录: {target_dir}")
            shutil.rmtree(target_dir)
        else:
            logger.info(f"视频临时目录不存在，无需清理: {target_dir}")
    
    # 配置VAD（针对成人视频优化）
    vad_config = VADConfig(
        threshold=0.2,  # 降低阈值
        min_speech_duration=0.1,  # 降低最小持续时间
        low_freq=50,  # 更低频率
        energy_threshold=0.0005,  # 大大降低能量阈值
        merge_gap=0.2,
        padding=0.1,
        max_segment_duration=30.0,
        japanese_phoneme_threshold=0.2,  # 降低阈值
        vowel_detection=True,
        model_name=args.model,
        chunk_duration=180.0
    )
    
    # 配置转录 - 使用项目目录下的temp
    trans_config = TranscriptionConfig(
        model_size=args.model,
        language=args.language
    )
    
    # 创建处理器
    processor = TranscriptionWithVAD(vad_config, trans_config)
    
    # 测试音频分块
    if args.test_audio:
        # 提取音频分块
        chunk_info = processor.vad.extract_audio_chunks(args.input)
        if chunk_info['chunks']:
            # 测试第一个分块
            test_chunk = chunk_info['chunks'][0]['path']
            print(f"测试音频分块: {test_chunk}")
            processor.vad.test_audio_chunk(test_chunk)
        return
    
    # 处理输入
    input_path = args.input
    
    if os.path.isdir(input_path) and args.batch:
        # 批量处理目录
        import glob
        
        audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', 
                           '*.mp4', '*.avi', '*.mkv', '*.mov']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(input_path, ext)))
        
        logger.info(f"找到 {len(audio_files)} 个音频/视频文件")
        
        results = processor.process_batch(audio_files, force_redo=args.force_redo)
        
        # 保存批量处理摘要到项目目录下的temp目录
        summary_path = os.path.join(trans_config.temp_dir, "batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'processed_files': list(results.keys()),
                'success_count': sum(1 for r in results.values() if r.get('success')),
                'failure_count': sum(1 for r in results.values() if not r.get('success')),
                'details': results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批量处理完成！摘要已保存到: {summary_path}")
        
    else:
        # 处理单个文件
        if args.vad_only:
            # 仅VAD分析，结果保存到项目目录下的temp目录
            vad_results = processor.vad.analyze_audio(
                input_path,
                os.path.join(trans_config.temp_dir, f"{Path(input_path).stem}_vad.json"),
                args.force_redo
            )
            logger.info(f"VAD分析完成: {input_path}")
            logger.info(f"检测到 {vad_results.get('segment_count', 0)} 个语音段")
            logger.info(f"语音比例: {vad_results.get('speech_ratio', 0):.1%}")
            logger.info(f"特殊声音统计: {vad_results.get('special_sounds', {})}")
            
            # 显示分块信息
            if 'chunk_info' in vad_results:
                chunk_info = vad_results['chunk_info']
                logger.info(f"音频分块: 共 {len(chunk_info.get('chunks', []))} 个分块，保存在: {chunk_info.get('checkpoint_file', '')}")
        else:
            # 完整转录
            result = processor.transcribe_with_vad(
                input_path, 
                force_redo=args.force_redo
            )
            
            if result.get('success'):
                logger.info(f"转录成功: {input_path}")
                logger.info(f"输出文件: {result.get('output_paths', {})}")
                
                # 显示分块信息
                vad_results = result.get('vad_results', {})
                if 'chunk_info' in vad_results:
                    chunk_info = vad_results['chunk_info']
                    logger.info(f"音频分块: 共 {len(chunk_info.get('chunks', []))} 个分块")
            else:
                logger.error(f"转录失败: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
