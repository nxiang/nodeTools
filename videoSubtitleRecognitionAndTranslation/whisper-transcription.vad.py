#!/usr/bin/env python3
"""
Whisper转录 + 日语成人向视频VAD检测
集成断点续传功能的完整解决方案
"""

import os
import json
import logging
import numpy as np
import torch
import torchaudio
from scipy import signal
import librosa
import soundfile as sf
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# 导入现有的whisper相关模块
try:
    import whisper
    from whisper.utils import get_writer
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("警告: whisper未安装，部分功能可能受限")

# 检查moviepy可用性（用于视频文件支持）
try:
    # 新版本moviepy (2.2.1+) 中VideoFileClip直接位于moviepy包中
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("警告: moviepy未安装，视频文件处理功能受限")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """针对日语成人视频的VAD配置"""
    # 基础参数
    sample_rate: int = 16000
    frame_duration: int = 30  # 毫秒
    threshold: float = 0.4  # 更低的阈值以适应成人内容
    model_name: str = "default"  # 模型名称，用于临时文件隔离
    
    # 成人视频特定参数
    min_speech_duration: float = 0.2  # 更短的最小持续时间
    min_silence_duration: float = 0.15
    
    # 频率范围（针对日语人声和特殊声音优化）
    low_freq: int = 70  # 更低的低频以检测喘息声
    high_freq: int = 4500  # 日语语音频率上限
    
    # 能量阈值
    energy_threshold: float = 0.005  # 更低的能量阈值
    
    # 特殊声音检测
    detect_moans: bool = True
    detect_whispers: bool = True
    detect_screams: bool = True
    moan_freq_range: Tuple[int, int] = (65, 280)  # 呻吟声频率范围
    
    # 后处理
    merge_gap: float = 0.5  # 合并间隙
    padding: float = 0.25  # 填充时间
    max_segment_duration: float = 180.0  # 最大段持续时间
    
    # 针对日语语音的特殊参数
    japanese_phoneme_threshold: float = 0.3
    vowel_detection: bool = True  # 日语元音检测
    
    # VAD模型选择
    use_webrtc: bool = False  # WebRTC VAD通常不适合成人内容
    use_silero: bool = False  # Silero VAD

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
    output_dir: str = "./transcription_output"
    
    # 临时文件目录
    temp_dir: str = "./temp"
    
    # 断点续传
    checkpoint_dir: str = "./temp/checkpoints"
    save_checkpoint_interval: int = 10  # 每10秒保存一次检查点

class JapaneseAdultVAD:
    """日语成人视频专用的VAD检测器"""
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._init_filters()
        
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
        
    def load_audio(self, audio_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """加载音频文件（支持视频文件）"""
        target_sr = target_sr or self.config.sample_rate
        
        try:
            # 检查文件扩展名
            file_ext = Path(audio_path).suffix.lower()
            
            # 音频文件
            if file_ext in ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'):
                logger.info(f"加载音频文件: {audio_path}")
                audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
                return audio, sr
            
            # 视频文件
            elif file_ext in ('.mkv', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm'):
                logger.info(f"检测到视频文件: {audio_path}")
                
                # 优先使用moviepy（如果可用）
                if MOVIEPY_AVAILABLE:
                    try:
                        logger.info("使用moviepy提取视频音频...")
                        video = VideoFileClip(audio_path)
                        
                        # 提取音频到临时文件
                        temp_audio_path = f"./temp_{int(time.time())}.wav"
                        video.audio.write_audiofile(temp_audio_path, fps=target_sr)
                        
                        # 加载音频
                        audio, sr = librosa.load(temp_audio_path, sr=target_sr, mono=True)
                        
                        # 清理临时文件
                        try:
                            os.remove(temp_audio_path)
                        except:
                            pass
                        
                        video.close()
                        logger.info(f"视频音频提取成功: 时长 {len(audio)/sr:.1f}s")
                        return audio, sr
                        
                    except Exception as e:
                        logger.warning(f"moviepy处理失败: {e}")
                
                # 后备方案：使用torchaudio
                try:
                    logger.info("使用torchaudio提取视频音频...")
                    audio, sr = torchaudio.load(audio_path)
                    audio = audio.numpy()
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=0)
                    
                    if sr != target_sr:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                        sr = target_sr
                    
                    logger.info(f"torchaudio音频提取成功: 时长 {len(audio)/sr:.1f}s")
                    return audio, sr
                    
                except Exception as e:
                    logger.warning(f"torchaudio处理失败: {e}")
                
                # 最后尝试使用librosa
                logger.info("使用librosa作为后备方案...")
                audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
                logger.info(f"librosa音频加载成功: 时长 {len(audio)/sr:.1f}s")
                return audio, sr
            
            else:
                # 未知文件类型，尝试多种方法
                logger.warning(f"未知文件类型: {file_ext}，尝试自动检测...")
                
                # 尝试librosa
                try:
                    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
                    logger.info(f"librosa自动检测成功: 时长 {len(audio)/sr:.1f}s")
                    return audio, sr
                except Exception as e:
                    logger.error(f"所有音频加载方法都失败了: {e}")
                    raise
            
        except Exception as e:
            logger.error(f"加载音频失败 {audio_path}: {e}")
            raise
    
    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float, Dict]]:
        """检测语音活动并返回时间段及元数据"""
        # 预处理：带通滤波
        audio_filtered = signal.filtfilt(*self.bandpass_filter, audio)
        
        # 计算帧大小
        frame_size = int(self.config.frame_duration * sample_rate / 1000)
        hop_length = frame_size // 2  # 50%重叠
        
        # 使用滑动窗口处理
        num_frames = (len(audio_filtered) - frame_size) // hop_length + 1
        frames = []
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_size
            frames.append(audio_filtered[start:end])
        
        # 检测每帧
        is_speech = []
        frame_features = []
        
        for i, frame in enumerate(frames):
            features = self._extract_frame_features(frame, sample_rate)
            frame_features.append(features)
            
            # 决策逻辑
            is_voice = self._decide_if_voice(features)
            is_speech.append(is_voice)
        
        # 转换为时间段
        segments = self._frames_to_segments(is_speech, frame_features, frame_size, hop_length, sample_rate)
        
        # 后处理
        segments = self._merge_segments(segments)
        segments = self._filter_segments(segments)
        segments = self._apply_padding(segments, len(audio)/sample_rate)
        
        return segments
    
    def _extract_frame_features(self, frame: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """提取帧特征"""
        # 能量特征
        energy = np.mean(frame ** 2)
        log_energy = np.log(energy + 1e-10)
        
        # 频谱特征
        stft = librosa.stft(frame, n_fft=512, hop_length=160)
        magnitude = np.abs(stft)
        
        # 频谱质心
        spectral_centroid = librosa.feature.spectral_centroid(
            S=magnitude, sr=sample_rate
        )[0].mean()
        
        # 频谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            S=magnitude, sr=sample_rate
        )[0].mean()
        
        # 过零率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(frame)[0].mean()
        
        # MFCC特征（针对语音）
        mfccs = librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # 特殊声音检测
        moan_features = self._detect_moan_features(frame, sample_rate)
        whisper_features = self._detect_whisper_features(frame, sample_rate, magnitude)
        scream_features = self._detect_scream_features(frame, sample_rate, magnitude)
        
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
        """检测呻吟声特征"""
        try:
            # 应用呻吟声滤波器
            moan_filtered = signal.filtfilt(*self.moan_filter, frame)
            
            # 计算低频能量
            moan_energy = np.mean(moan_filtered ** 2)
            total_energy = np.mean(frame ** 2)
            
            # 低频能量比
            low_freq_ratio = moan_energy / (total_energy + 1e-10)
            
            # 计算节奏特征（呻吟声通常有节奏）
            autocorr = np.correlate(moan_filtered, moan_filtered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # 寻找峰值
            peaks, _ = signal.find_peaks(autocorr[:1000], height=0.1)
            rhythm_regularity = len(peaks) / 10.0  # 简化指标
            
            is_moan = (low_freq_ratio > 0.25 and rhythm_regularity > 0.3)
            probability = min(low_freq_ratio * 2 + rhythm_regularity * 0.5, 1.0)
            
            return {
                'is_moan': is_moan,
                'probability': probability,
                'low_freq_ratio': low_freq_ratio,
                'rhythm_regularity': rhythm_regularity
            }
        except:
            return {'is_moan': False, 'probability': 0.0}
    
    def _detect_whisper_features(self, frame: np.ndarray, sample_rate: int, magnitude: np.ndarray) -> Dict[str, Any]:
        """检测耳语特征"""
        # 耳语通常频谱质心较低，高频能量较少
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)[0].mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate)[0].mean()
        
        # 耳语通常能量较低但过零率较高
        energy = np.mean(frame ** 2)
        zcr = librosa.feature.zero_crossing_rate(frame)[0].mean()
        
        is_whisper = (spectral_centroid < 1000 and 
                     spectral_rolloff < 4000 and 
                     energy < 0.01 and 
                     zcr > 0.05)
        
        return {'is_whisper': is_whisper}
    
    def _detect_scream_features(self, frame: np.ndarray, sample_rate: int, magnitude: np.ndarray) -> Dict[str, Any]:
        """检测尖叫特征"""
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)[0].mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sample_rate)[0].mean()
        energy = np.mean(frame ** 2)
        
        is_scream = (spectral_centroid > 2000 and 
                    spectral_bandwidth > 1500 and 
                    energy > 0.05)
        
        return {'is_scream': is_scream}
    
    def _detect_japanese_vowels(self, frame: np.ndarray, sample_rate: int) -> float:
        """检测日语元音特征"""
        # 简化的日语元音检测（あ、い、う、え、お）
        try:
            # 计算频谱包络
            stft = librosa.stft(frame, n_fft=512)
            magnitude = np.abs(stft)
            
            # 日语元音通常在某些频率有特征峰
            # 这里使用简化的检测方法
            freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=512)
            
            # 检查日语元音特征频率（简化版）
            vowel_ranges = [
                (250, 350),    # あ
                (300, 400),    # い
                (200, 300),    # う
                (400, 500),    # え
                (350, 450),    # お
            ]
            
            vowel_score = 0.0
            for low, high in vowel_ranges:
                mask = (freq_bins >= low) & (freq_bins <= high)
                if np.any(mask):
                    range_energy = np.mean(magnitude[mask, :])
                    total_energy = np.mean(magnitude)
                    if total_energy > 0:
                        vowel_score = max(vowel_score, range_energy / total_energy)
            
            return min(vowel_score, 1.0)
        except:
            return 0.0
    
    def _decide_if_voice(self, features: Dict[str, Any]) -> bool:
        """判断是否为语音/声音"""
        is_voice = False
        
        # 规则1：基础能量阈值
        if features['energy'] > self.config.energy_threshold:
            is_voice = True
        
        # 规则2：特殊声音检测
        if features['is_moan'] or features['is_scream']:
            is_voice = True
        
        # 规则3：日语语音特征
        if features['is_japanese_phoneme']:
            is_voice = True
        
        # 规则4：频谱特征
        if (features['spectral_centroid'] > 85 and 
            features['spectral_centroid'] < 4500 and
            features['spectral_bandwidth'] > 300):
            is_voice = True
        
        # 规则5：过零率（适用于呼吸声等）
        if features['zero_crossing_rate'] < 0.15 and features['energy'] > 0.002:
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
    
    def analyze_audio(self, audio_path: str, output_json: Optional[str] = None, 
                     chunk_duration: float = 10.0, stream_mode: bool = True) -> Dict[str, Any]:
        """分析音频并返回VAD结果（支持流式操作）"""
        logger.info(f"开始VAD分析: {audio_path}")
        
        try:
            if stream_mode:
                # 流式处理模式
                segments = self._analyze_audio_streaming(audio_path, chunk_duration)
                
                # 获取音频总时长 - 使用load_audio方法兼容视频文件
                audio, sr = self.load_audio(audio_path, self.config.sample_rate)
                total_duration = len(audio) / sr
            else:
                # 传统批量处理模式（保持向后兼容）
                audio, sr = self.load_audio(audio_path, self.config.sample_rate)
                total_duration = len(audio) / sr
                segments = self.detect_voice_activity(audio, sr)
            
            # 统计信息
            speech_duration = sum(end - start for start, end, _ in segments)
            speech_ratio = speech_duration / total_duration if total_duration > 0 else 0
            
            # 特殊声音统计
            special_counts = {
                'moans': sum(1 for _, _, info in segments if info.get('has_moans', False)),
                'whispers': sum(1 for _, _, info in segments if info.get('has_whispers', False)),
                'screams': sum(1 for _, _, info in segments if info.get('has_screams', False)),
                'japanese_phonemes': sum(1 for _, _, info in segments if info.get('has_japanese_phonemes', False))
            }
            
            # 整理结果
            results = {
                'audio_path': audio_path,
                'total_duration': total_duration,
                'speech_duration': speech_duration,
                'speech_ratio': speech_ratio,
                'segment_count': len(segments),
                'special_sounds': special_counts,
                'segments': [
                    {
                        'start': start,
                        'end': end,
                        'duration': end - start,
                        'metadata': info
                    }
                    for start, end, info in segments
                ],
                'vad_config': self.config.__dict__,
                'analysis_time': datetime.now().isoformat(),
                'processing_mode': 'streaming' if stream_mode else 'batch'
            }
            
            # 保存结果
            if output_json:
                os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"VAD结果已保存到: {output_json}")
            
            logger.info(f"VAD分析完成: 检测到 {len(segments)} 个语音段, 语音比例: {speech_ratio:.1%}")
            return results
            
        except Exception as e:
            logger.error(f"VAD分析失败: {e}")
            raise

    def _analyze_audio_streaming(self, audio_path: str, chunk_duration: float = 30.0, 
                               checkpoint_file: Optional[str] = None) -> List[Tuple[float, float, Dict]]:
        """流式分析音频/视频文件（支持断点续传）"""
        logger.info(f"开始流式VAD分析: {audio_path}, 块时长: {chunk_duration}s")
        
        # 创建临时目录结构：temp/视频名_模型名/ 用于检查点文件，temp/ 用于srt文件
        audio_name = Path(audio_path).stem
        model_name = self.config.model_name if hasattr(self.config, 'model_name') else 'default'
        
        # 主临时目录
        temp_dir = os.path.join(os.path.dirname(audio_path), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 隔离目录：按视频名+模型名创建子目录
        isolation_dir = os.path.join(temp_dir, f"{audio_name}_{model_name}")
        os.makedirs(isolation_dir, exist_ok=True)
        
        # 如果没有指定检查点文件，使用隔离目录中的默认路径
        if checkpoint_file is None:
            checkpoint_file = os.path.join(isolation_dir, f"{audio_name}_vad_checkpoint.json")
        
        # 尝试加载检查点
        checkpoint_data = self._load_vad_checkpoint(checkpoint_file)
        
        try:
            # 首先加载整个音频以获取信息（兼容视频文件）
            logger.info("加载音频以获取基本信息...")
            full_audio, sample_rate = self.load_audio(audio_path, self.config.sample_rate)
            total_duration = len(full_audio) / sample_rate
            
            logger.info(f"音频信息: 总时长 {total_duration:.1f}s, 采样率 {sample_rate}Hz")
            
            # 计算块大小
            chunk_size = int(chunk_duration * sample_rate)
            logger.info(f"块大小: {chunk_size} 样本, 约 {chunk_duration}s")
            
            # 初始化状态（从检查点恢复或新建）
            if checkpoint_data:
                logger.info(f"从检查点恢复: {checkpoint_file}")
                all_segments = checkpoint_data.get('all_segments', [])
                current_segment = checkpoint_data.get('current_segment')
                current_time_offset = checkpoint_data.get('current_time_offset', 0.0)
                chunk_count = checkpoint_data.get('chunk_count', 0)
                total_segments_detected = checkpoint_data.get('total_segments_detected', 0)
                current_sample = checkpoint_data.get('current_sample', 0)
                
                logger.info(f"恢复状态: 已处理 {chunk_count} 个块, 时间偏移 {current_time_offset:.1f}s, 已检测 {total_segments_detected} 个语音段")
            else:
                logger.info("未找到检查点，从头开始处理")
                all_segments = []
                current_segment = None
                current_time_offset = 0.0
                chunk_count = 0
                total_segments_detected = 0
                current_sample = 0
            
            # 手动分块处理（替代soundfile的流式读取）
            total_samples = len(full_audio)
            
            while current_sample < total_samples:
                chunk_count += 1
                
                # 读取音频块
                end_sample = min(current_sample + chunk_size, total_samples)
                audio_chunk = full_audio[current_sample:end_sample]
                
                if len(audio_chunk) == 0:
                    logger.info("音频处理完成")
                    break
                
                logger.info(f"处理第 {chunk_count} 个音频块, 当前时间偏移: {current_time_offset:.1f}s")
                
                # 处理当前音频块
                chunk_segments = self._process_audio_chunk(
                    audio_chunk, sample_rate, current_time_offset, current_segment
                )
                
                # 记录块处理结果
                if chunk_segments:
                    logger.info(f"块 {chunk_count}: 检测到 {len(chunk_segments)} 个语音段")
                    total_segments_detected += len(chunk_segments)
                else:
                    logger.debug(f"块 {chunk_count}: 未检测到语音活动")
                
                # 更新状态
                if chunk_segments:
                    # 处理跨块语音段
                    if current_segment is not None:
                        # 检查是否需要合并最后一个段
                        last_segment = chunk_segments[-1]
                        if (last_segment[0] - current_segment[1] <= self.config.merge_gap and
                            last_segment[2].get('has_moans') == current_segment[2].get('has_moans')):
                            # 合并段
                            merged_start = current_segment[0]
                            merged_end = last_segment[1]
                            merged_info = self._merge_segment_info(current_segment[2], last_segment[2])
                            
                            logger.info(f"跨块合并: 段 {current_segment[0]:.1f}s-{current_segment[1]:.1f}s 与段 {last_segment[0]:.1f}s-{last_segment[1]:.1f}s 合并")
                            
                            # 替换最后一个段
                            chunk_segments[-1] = (merged_start, merged_end, merged_info)
                            
                            # 移除已合并的当前段
                            if current_segment in all_segments:
                                all_segments.remove(current_segment)
                        
                        current_segment = None
                    
                    # 添加新段
                    all_segments.extend(chunk_segments)
                    
                    # 更新当前段状态
                    if chunk_segments:
                        current_segment = chunk_segments[-1]
                        logger.debug(f"当前活跃段: {current_segment[0]:.1f}s-{current_segment[1]:.1f}s")
                
                # 更新时间偏移和样本位置
                chunk_duration_actual = len(audio_chunk) / sample_rate
                current_time_offset += chunk_duration_actual
                current_sample += len(audio_chunk)
                
                # 保存检查点
                self._save_vad_checkpoint(checkpoint_file, {
                    'all_segments': all_segments,
                    'current_segment': current_segment,
                    'current_time_offset': current_time_offset,
                    'chunk_count': chunk_count,
                    'total_segments_detected': total_segments_detected,
                    'current_sample': current_sample,
                    'total_duration': total_duration,
                    'sample_rate': sample_rate,
                    'last_saved': datetime.now().isoformat()
                })
                logger.info(f"检查点已保存: {checkpoint_file}")
                
                # 进度日志
                progress = min(current_time_offset / total_duration, 1.0)
                
                logger.info(f"流式分析进度: {progress:.1%} ({current_time_offset:.1f}s/{total_duration:.1f}s), 已检测 {len(all_segments)} 个语音段")
            
            # 处理最后一个未完成的段
            if current_segment is not None:
                all_segments.append(current_segment)
                logger.info(f"处理最后一个未完成段: {current_segment[0]:.1f}s-{current_segment[1]:.1f}s")
            
            logger.info(f"流式处理完成: 共处理 {chunk_count} 个音频块, 累计检测 {total_segments_detected} 个语音段")
            
            # 后处理：合并、过滤、填充
            logger.info("开始后处理: 合并相邻语音段")
            all_segments = self._merge_segments(all_segments)
            logger.info(f"合并后语音段数量: {len(all_segments)}")
            
            logger.info("开始后处理: 过滤短语音段")
            all_segments = self._filter_segments(all_segments)
            logger.info(f"过滤后语音段数量: {len(all_segments)}")
            
            logger.info("开始后处理: 应用时间填充")
            all_segments = self._apply_padding(all_segments, total_duration)
            
            # 处理完成后删除检查点文件
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                logger.info(f"处理完成，删除检查点文件: {checkpoint_file}")
            
            logger.info(f"流式VAD分析完成: 处理了 {total_duration:.1f}s 音频, 检测到 {len(all_segments)} 个语音段")
            return all_segments
            
        except Exception as e:
            logger.error(f"流式分析失败: {e}")
            # 保存错误检查点以便恢复
            if checkpoint_file:
                self._save_vad_checkpoint(checkpoint_file, {
                    'all_segments': all_segments,
                    'current_segment': current_segment,
                    'current_time_offset': current_time_offset,
                    'chunk_count': chunk_count,
                    'total_segments_detected': total_segments_detected,
                    'current_sample': current_sample,
'total_duration': total_duration,
                    'sample_rate': sample_rate,
                    'error': str(e),
                    'last_saved': datetime.now().isoformat()
                })
                logger.error(f"错误检查点已保存: {checkpoint_file}")
            raise
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int, 
                           time_offset: float, current_segment: Optional[Tuple[float, float, Dict]]) -> List[Tuple[float, float, Dict]]:
        """处理单个音频块"""
        if len(audio_chunk) == 0:
            return []
        
        # 检测当前块的语音活动
        chunk_segments = self.detect_voice_activity(audio_chunk, sample_rate)
        
        # 调整时间戳以匹配全局时间
        adjusted_segments = []
        for start, end, info in chunk_segments:
            adjusted_start = time_offset + start
            adjusted_end = time_offset + end
            adjusted_segments.append((adjusted_start, adjusted_end, info))
        
        return adjusted_segments
    
    def _merge_segment_info(self, info1: Dict, info2: Dict) -> Dict:
        """合并两个段的元数据信息"""
        merged = {}
        
        # 布尔值：使用逻辑或
        for key in ['has_moans', 'has_whispers', 'has_screams', 'has_japanese_phonemes']:
            if key in info1 or key in info2:
                merged[key] = info1.get(key, False) or info2.get(key, False)
        
        # 数值：取平均值
        for key in ['avg_energy', 'avg_moan_prob', 'avg_vowel_score']:
            if key in info1 or key in info2:
                val1 = info1.get(key, 0)
                val2 = info2.get(key, 0)
                merged[key] = (val1 + val2) / 2
        
        # 计数：求和
        if 'frame_count' in info1 or 'frame_count' in info2:
            merged['frame_count'] = info1.get('frame_count', 0) + info2.get('frame_count', 0)
        
        return merged
    
    def _load_vad_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """加载VAD检查点文件"""
        if not os.path.exists(checkpoint_file):
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # 验证检查点数据的完整性
            required_fields = ['all_segments', 'current_time_offset', 'chunk_count', 'current_sample']
            if all(field in checkpoint_data for field in required_fields):
                logger.info(f"成功加载检查点: {checkpoint_file}")
                return checkpoint_data
            else:
                logger.warning(f"检查点文件不完整: {checkpoint_file}")
                return None
                
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None
    
    def _save_vad_checkpoint(self, checkpoint_file: str, checkpoint_data: Dict) -> bool:
        """保存VAD检查点文件"""
        try:
            # 确保临时目录存在
            temp_dir = os.path.dirname(checkpoint_file)
            os.makedirs(temp_dir, exist_ok=True)
            
            # 保存检查点数据
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"检查点保存成功: {checkpoint_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            return False
    
    def analyze_audio_with_checkpoint(self, audio_path: str, chunk_duration: float = 30.0, 
                                    checkpoint_file: Optional[str] = None) -> List[Tuple[float, float, Dict]]:
        """
        支持断点续传的音频分析
        
        Args:
            audio_path: 音频/视频文件路径
            chunk_duration: 每个处理块的时间长度（秒）
            checkpoint_file: 检查点文件路径，如果为None则自动生成
            
        Returns:
            语音段列表，每个段包含开始时间、结束时间和元数据
        """
        return self._analyze_audio_streaming(audio_path, chunk_duration, checkpoint_file)
    
    def resume_from_checkpoint(self, audio_path: str, checkpoint_file: str) -> List[Tuple[float, float, Dict]]:
        """
        从检查点恢复分析
        
        Args:
            audio_path: 音频/视频文件路径
            checkpoint_file: 检查点文件路径
            
        Returns:
            语音段列表
        """
        logger.info(f"从检查点恢复分析: {checkpoint_file}")
        
        # 验证检查点文件是否存在
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_file}")
        
        # 使用检查点文件进行分析
        return self.analyze_audio_with_checkpoint(audio_path, checkpoint_file=checkpoint_file)
    
    def cleanup_checkpoint(self, checkpoint_file: str) -> bool:
        """清理检查点文件"""
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                logger.info(f"检查点文件已清理: {checkpoint_file}")
                return True
            return False
        except Exception as e:
            logger.error(f"清理检查点失败: {e}")
            return False

class TranscriptionWithVAD:
    """集成VAD的Whisper转录系统（支持断点续传）"""
    
    def __init__(self, vad_config: Optional[VADConfig] = None, 
                 trans_config: Optional[TranscriptionConfig] = None):
        self.vad_config = vad_config or VADConfig()
        self.trans_config = trans_config or TranscriptionConfig()
        
        # 初始化组件
        self.vad = JapaneseAdultVAD(self.vad_config)
        
        # 初始化Whisper模型
        if WHISPER_AVAILABLE:
            logger.info(f"加载Whisper模型: {self.trans_config.model_size}")
            self.model = whisper.load_model(
                self.trans_config.model_size,
                device=self.trans_config.device
            )
        else:
            self.model = None
            logger.warning("Whisper未安装，转录功能不可用")
        
        # 创建输出目录和临时目录（不创建checkpoints目录）
        os.makedirs(self.trans_config.output_dir, exist_ok=True)
        os.makedirs(self.trans_config.temp_dir, exist_ok=True)
        # 注释掉checkpoint目录创建：os.makedirs(self.trans_config.checkpoint_dir, exist_ok=True)
        
        # 断点状态
        self.checkpoint_state = {}
    
    def get_checkpoint_path(self, audio_path: str) -> str:
        """获取检查点文件路径（不创建checkpoints目录）"""
        audio_name = Path(audio_path).stem
        model_name = self.vad_config.model_name if hasattr(self.vad_config, 'model_name') else 'default'
        
        # 创建隔离目录：temp/视频名_模型名（不创建checkpoints子目录）
        isolation_dir = os.path.join(self.trans_config.temp_dir, f"{audio_name}_{model_name}")
        os.makedirs(isolation_dir, exist_ok=True)
        
        # 检查点文件直接放在隔离目录中，不创建checkpoints子目录
        checkpoint_name = f"{audio_name}_checkpoint.json"
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
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            # 同时备份一份
            backup_path = checkpoint_path.replace('.json', f'_backup_{int(time.time())}.json')
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"检查点已保存: {checkpoint_path}")
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def extract_audio_segment(self, audio_path: str, start_time: float, 
                            end_time: float, temp_dir: Optional[str] = None) -> Optional[str]:
        """提取音频段到临时文件"""
        try:
            if temp_dir is None:
                # 按视频名+模型名创建隔离目录
                audio_name = Path(audio_path).stem
                model_name = self.vad_config.model_name if hasattr(self.vad_config, 'model_name') else 'default'
                temp_dir = os.path.join(self.trans_config.temp_dir, f"{audio_name}_{model_name}")
            
            os.makedirs(temp_dir, exist_ok=True)
            
            # 加载完整音频
            audio, sr = self.vad.load_audio(audio_path, self.vad_config.sample_rate)
            
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
    
    def transcribe_segment(self, audio_path: str, segment_index: int, 
                          start_time: float, end_time: float) -> Tuple[Dict[str, Any], float]:
        """转录单个音频段，返回结果和处理时间"""
        if not self.model:
            raise ValueError("Whisper模型未加载")
        
        start_time_processing = time.time()
        
        # 提取音频段
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
                model_name = self.vad_config.model_name if hasattr(self.vad_config, 'model_name') else 'default'
                
                # 创建隔离目录：temp/视频名_模型名/
                isolation_dir = os.path.join(self.trans_config.temp_dir, f"{audio_name}_{model_name}")
                os.makedirs(isolation_dir, exist_ok=True)
                
                vad_json_path = os.path.join(
                    isolation_dir,
                    f"{audio_name}_vad.json"
                )
                vad_results = self.vad.analyze_audio(audio_path, vad_json_path)
        
        # 准备转录任务
        segments = vad_results.get('segments', [])
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
            metadata = seg.get('metadata', {})
            
            # 转录
            segment_result, segment_time = self.transcribe_segment(audio_path, i, start_time, end_time)
            
            # 更新累计时间
            cumulative_time += segment_time
            
            # 格式化时间显示
            segment_time_formatted = format_time(segment_time)
            cumulative_time_formatted = format_time(cumulative_time)
            segment_duration = end_time - start_time
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
        model_name = self.vad_config.model_name if hasattr(self.vad_config, 'model_name') else 'default'
        
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
    
    args = parser.parse_args()
    
    # 配置VAD（针对成人视频优化）
    vad_config = VADConfig(
        threshold=0.35,
        min_speech_duration=0.18,
        low_freq=65,
        energy_threshold=0.004,
        merge_gap=0.6,
        padding=0.3,
        max_segment_duration=180.0,
        japanese_phoneme_threshold=0.25,
        vowel_detection=True,
        model_name=args.model  # 使用实际的Whisper模型名称
    )
    
    # 配置转录 - 所有目录都放在temp下
    trans_config = TranscriptionConfig(
        model_size=args.model,
        language=args.language,
        output_dir="./temp",  # 输出目录改为temp
        checkpoint_dir="./temp/checkpoints",  # 检查点目录放在temp下
        temp_dir="./temp"
    )
    
    # 创建处理器
    processor = TranscriptionWithVAD(vad_config, trans_config)
    
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
        
        # 保存批量处理摘要到temp目录
        summary_path = os.path.join("./temp", "batch_summary.json")
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
                # 仅VAD分析，结果保存到temp目录
                vad_results = processor.vad.analyze_audio(
                    input_path,
                    os.path.join("./temp", f"{Path(input_path).stem}_vad.json")
                )
                logger.info(f"VAD分析完成: {input_path}")
                logger.info(f"检测到 {vad_results.get('segment_count', 0)} 个语音段")
                logger.info(f"语音比例: {vad_results.get('speech_ratio', 0):.1%}")
                logger.info(f"特殊声音统计: {vad_results.get('special_sounds', {})}")
            else:
                # 完整转录
                result = processor.transcribe_with_vad(
                    input_path, 
                    force_redo=args.force_redo
                )
                
                if result.get('success'):
                    logger.info(f"转录成功: {input_path}")
                    logger.info(f"输出文件: {result.get('output_paths', {})}")
                else:
                    logger.error(f"转录失败: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
