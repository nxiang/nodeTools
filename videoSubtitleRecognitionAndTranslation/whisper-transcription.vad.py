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
import warnings
warnings.filterwarnings("ignore")

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

# 尝试导入音频增强库
try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False
    print("提示: 如需噪声抑制，请安装: pip install noisereduce")

try:
    import scipy.signal as signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: scipy未安装，音频增强功能受限")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("错误: 请安装 librosa: pip install librosa")

# ================== 语音增强类 ==================
class VoiceEnhancer:
    """专门针对弱人声的增强处理器"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        self.nyquist = sr // 2
        
    def adaptive_noise_reduction(self, audio: np.ndarray, noise_threshold_db=-40) -> np.ndarray:
        """自适应噪声抑制 - 特别针对低音量人声"""
        if not NOISE_REDUCE_AVAILABLE or len(audio) < self.sr * 0.5:
            return audio
            
        try:
            # 计算音频能量
            rms = np.sqrt(np.mean(audio**2))
            if rms < 1e-6:
                return audio
                
            # 如果音频太安静，使用更激进的降噪
            if rms < 0.01:
                # 寻找噪声片段（开头100ms通常为纯噪声）
                noise_sample_len = min(256, len(audio) // 4)
                if noise_sample_len > 10:
                    noise_clip = audio[:noise_sample_len]
                    reduced = nr.reduce_noise(
                        y=audio,
                        y_noise=noise_clip,
                        sr=self.sr,
                        prop_decrease=0.95,  # 激进降噪
                        n_fft=512,
                        win_length=256,
                        hop_length=128
                    )
                    return reduced
                    
            # 常规降噪
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.sr,
                prop_decrease=0.8,
                stationary=False,
                n_fft=512,
                win_length=256,
                hop_length=128
            )
            return reduced
            
        except Exception as e:
            logging.warning(f"降噪失败: {e}")
            return audio
    
    def spectral_enhancement(self, audio: np.ndarray, enhance_db=12) -> np.ndarray:
        """频谱增强 - 重点增强人声频率"""
        if not SCIPY_AVAILABLE or len(audio) < self.sr * 0.1:
            return audio
            
        try:
            # 设计人声增强滤波器
            # 人声主要频率范围：80Hz - 4000Hz
            lowcut = 80.0
            highcut = 4000.0
            
            # 带通滤波器
            sos = signal.butter(
                6, 
                [lowcut/self.nyquist, highcut/self.nyquist], 
                btype='band',
                output='sos'
            )
            
            filtered = signal.sosfiltfilt(sos, audio)
            
            # 动态增益控制
            envelope = np.abs(signal.hilbert(filtered))
            envelope_smooth = np.convolve(
                envelope, 
                np.ones(100)/100, 
                mode='same'
            )
            
            # 对低能量部分应用更大增益
            gain = np.ones_like(audio)
            mask = envelope_smooth < np.percentile(envelope_smooth, 50)
            gain[mask] = 2.0  # 低能量部分2倍增益
            gain[~mask] = 1.2  # 高能量部分1.2倍增益
            
            enhanced = filtered * gain
            
            # 限制动态范围，防止削波
            max_val = np.max(np.abs(enhanced))
            if max_val > 0:
                enhanced = enhanced / max_val * 0.95
                
            return enhanced
            
        except Exception as e:
            logging.warning(f"频谱增强失败: {e}")
            return audio
    
    def dynamic_range_compression(self, audio: np.ndarray, threshold=0.1, ratio=4.0) -> np.ndarray:
        """动态范围压缩 - 提升弱声音，抑制过强声音"""
        try:
            # 软膝压缩
            compressed = audio.copy()
            abs_audio = np.abs(audio)
            
            # 压缩曲线
            gain_reduction = np.where(
                abs_audio > threshold,
                1.0 - (1.0 - 1.0/ratio) * (abs_audio - threshold) / (1.0 - threshold),
                1.0
            )
            
            # 应用压缩
            compressed = compressed * gain_reduction
            
            # 确保不超过-1到1范围
            max_val = np.max(np.abs(compressed))
            if max_val > 0:
                compressed = compressed / max_val * 0.98
                
            return compressed
            
        except Exception as e:
            logging.warning(f"动态压缩失败: {e}")
            return audio
    
    def breath_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """呼吸声/气声增强 - 特别针对耳语"""
        if not SCIPY_AVAILABLE:
            return audio
            
        try:
            # 针对高频气声的增强 (2000-6000Hz)
            highcut1 = 2000.0
            highcut2 = 6000.0
            
            # 高频带通滤波器
            sos_high = signal.butter(
                4,
                [highcut1/self.nyquist, highcut2/self.nyquist],
                btype='band',
                output='sos'
            )
            
            high_freq = signal.sosfiltfilt(sos_high, audio)
            
            # 增强高频成分
            enhanced = audio + high_freq * 0.5
            
            # 限制动态范围
            max_val = np.max(np.abs(enhanced))
            if max_val > 0:
                enhanced = enhanced / max_val * 0.9
                
            return enhanced
            
        except Exception as e:
            logging.warning(f"呼吸声增强失败: {e}")
            return audio
    
    def full_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """完整的增强流程"""
        enhanced = audio.copy()
        
        # 1. 降噪
        enhanced = self.adaptive_noise_reduction(enhanced)
        
        # 2. 频谱增强
        enhanced = self.spectral_enhancement(enhanced)
        
        # 3. 动态压缩
        enhanced = self.dynamic_range_compression(enhanced)
        
        # 4. 呼吸声增强
        enhanced = self.breath_enhancement(enhanced)
        
        return enhanced

# ================== 时间格式化类 ==================
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

# ================== 性能追踪器 ==================
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

# ================== VAD处理器增强版 ==================
class WhisperVADProcessor:
    """Whisper专用VAD处理器，针对弱人声优化"""
    
    def __init__(self, device: str = None, ultra_sensitive: bool = False):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ultra_sensitive = ultra_sensitive  # 超敏感模式
        self.model = None
        self.available = False
        self.voice_enhancer = VoiceEnhancer()
        
        if not SILERO_VAD_AVAILABLE:
            logging.warning("Silero VAD不可用")
            return
        
        try:
            logging.info(f"正在加载Silero VAD模型 (设备: {self.device})...")
            
            # 使用torch.hub加载（最可靠的方式）
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            
            self.model = model.to(self.device)
            self.model.eval()
            
            # 获取工具函数
            (get_speech_timestamps, _, read_audio, _, _) = utils
            self.get_speech_timestamps_fn = get_speech_timestamps
            self.available = True
            
            logging.info("Silero VAD加载成功")
            
        except Exception as e:
            logging.error(f"加载Silero VAD失败: {e}")
            self.available = False
    
    def detect_weak_speech(self, audio: np.ndarray, sr: int = 16000, 
                          vad_threshold: float = 0.2,
                          min_speech_duration_ms: int = 200,
                          aggressive_mode: bool = True) -> List[Dict[str, float]]:
        """专门检测弱人声，针对成人内容中的轻声、耳语优化"""
        if not self.available or self.model is None:
            return []
        
        try:
            # 增强弱人声
            enhanced_audio = self.voice_enhancer.full_enhancement(audio)
            
            # 确保音频是单声道
            if len(enhanced_audio.shape) > 1:
                enhanced_audio = enhanced_audio.mean(axis=0) if enhanced_audio.shape[0] > 1 else enhanced_audio[0]
            
            # 确保音频是float32类型
            enhanced_audio = enhanced_audio.astype(np.float32)
            
            # 转换为PyTorch张量并移动到设备
            audio_tensor = torch.from_numpy(enhanced_audio).to(self.device)
            
            # 根据模式选择VAD参数
            if aggressive_mode:
                # 激进模式：检测更多可能的语音段
                vad_params = {
                    "threshold": vad_threshold,
                    "min_speech_duration_ms": min_speech_duration_ms,
                    "min_silence_duration_ms": 100,
                    "window_size_samples": 256,
                    "speech_pad_ms": 100,
                }
            else:
                # 保守模式
                vad_params = {
                    "threshold": max(vad_threshold, 0.3),
                    "min_speech_duration_ms": 300,
                    "min_silence_duration_ms": 200,
                    "window_size_samples": 512,
                    "speech_pad_ms": 100,
                }
            
            # 检测语音时间戳
            vad_start = time.time()
            
            timestamps = self.get_speech_timestamps_fn(
                audio_tensor, 
                self.model,
                threshold=vad_params["threshold"],
                min_speech_duration_ms=vad_params["min_speech_duration_ms"],
                min_silence_duration_ms=vad_params["min_silence_duration_ms"],
                window_size_samples=vad_params["window_size_samples"],
                speech_pad_ms=vad_params["speech_pad_ms"],
                return_seconds=True
            )
            
            vad_time = time.time() - vad_start
            
            if not timestamps:
                return []
            
            # 后处理：过滤可能的噪声段
            filtered_timestamps = []
            for ts in timestamps:
                # 提取原始音频段（使用原始音频进行能量分析）
                start_sample = int(ts['start'] * sr)
                end_sample = int(ts['end'] * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # 计算能量特征
                segment_energy = np.mean(np.abs(segment_audio))
                total_energy = np.mean(np.abs(audio))
                
                # 跳过能量过低的段（可能是VAD误检）
                if segment_energy < total_energy * 0.1 and segment_energy < 0.001:
                    logging.debug(f"过滤低能量段: {ts['start']:.2f}-{ts['end']:.2f}s, 能量比: {segment_energy/total_energy:.3f}")
                    continue
                
                # 计算过零率（人声通常较低）
                zero_crossings = np.sum(np.abs(np.diff(np.sign(segment_audio)))) / len(segment_audio)
                if zero_crossings > 0.4:  # 过零率太高，可能是噪声
                    logging.debug(f"过滤高过零率段: {ts['start']:.2f}-{ts['end']:.2f}s, 过零率: {zero_crossings:.3f}")
                    continue
                
                filtered_timestamps.append(ts)
            
            logging.debug(f"弱人声检测: 原始 {len(timestamps)} 个，过滤后 {len(filtered_timestamps)} 个，耗时 {vad_time:.3f}秒")
            
            return filtered_timestamps
            
        except Exception as e:
            logging.error(f"弱人声检测失败: {e}")
            return []
    
    def detect_ultra_sensitive_speech(self, audio: np.ndarray, sr: int = 16000) -> List[Dict[str, float]]:
        """超敏感语音检测 - 针对极弱人声"""
        if not self.available or self.model is None:
            return []
        
        try:
            # 超强增强
            enhanced_audio = self.voice_enhancer.full_enhancement(audio)
            
            # 多阶段检测
            timestamps_list = []
            
            # 第一阶段：常规检测
            audio_tensor = torch.from_numpy(enhanced_audio).to(self.device)
            timestamps = self.get_speech_timestamps_fn(
                audio_tensor,
                self.model,
                threshold=0.1,  # 极低阈值
                min_speech_duration_ms=50,  # 超短语音
                min_silence_duration_ms=30,
                window_size_samples=128,  # 小窗口
                speech_pad_ms=30,
                return_seconds=True
            )
            
            if timestamps:
                timestamps_list.extend(timestamps)
            
            # 第二阶段：基于能量的检测（针对极弱语音）
            # 计算能量包络
            frame_length = int(0.02 * sr)  # 20ms
            hop_length = frame_length // 2
            
            energy = []
            for i in range(0, len(enhanced_audio) - frame_length, hop_length):
                frame = enhanced_audio[i:i+frame_length]
                energy.append(np.mean(frame**2))
            
            if energy:
                energy = np.array(energy)
                energy_db = 10 * np.log10(energy + 1e-10)
                
                # 寻找可能的话音起始点
                threshold_db = np.percentile(energy_db, 30)  # 使用30%分位数
                potential_speech = energy_db > threshold_db
                
                # 合并相邻的语音段
                speech_segments = []
                in_speech = False
                start_idx = 0
                
                for i, is_speech in enumerate(potential_speech):
                    if is_speech and not in_speech:
                        start_idx = i
                        in_speech = True
                    elif not is_speech and in_speech:
                        duration_frames = i - start_idx
                        if duration_frames >= 5:  # 至少100ms
                            start_time = start_idx * hop_length / sr
                            end_time = i * hop_length / sr
                            speech_segments.append({
                                'start': start_time,
                                'end': end_time
                            })
                        in_speech = False
                
                if in_speech:
                    end_time = len(potential_speech) * hop_length / sr
                    speech_segments.append({
                        'start': start_idx * hop_length / sr,
                        'end': end_time
                    })
                
                timestamps_list.extend(speech_segments)
            
            # 去重和合并
            merged_timestamps = self._merge_timestamps(timestamps_list)
            
            logging.info(f"超敏感检测: 发现 {len(merged_timestamps)} 个语音段")
            return merged_timestamps
            
        except Exception as e:
            logging.error(f"超敏感检测失败: {e}")
            return []
    
    def _merge_timestamps(self, timestamps: List[Dict]) -> List[Dict]:
        """合并重叠的时间戳"""
        if not timestamps:
            return []
        
        # 按开始时间排序
        timestamps.sort(key=lambda x: x['start'])
        
        merged = []
        current = timestamps[0]
        
        for ts in timestamps[1:]:
            # 如果重叠或间隔很短（<0.1秒），合并
            if ts['start'] <= current['end'] + 0.1:
                current['end'] = max(current['end'], ts['end'])
            else:
                merged.append(current)
                current = ts
        
        merged.append(current)
        
        # 过滤太短的段
        filtered = [ts for ts in merged if ts['end'] - ts['start'] >= 0.15]
        
        return filtered
    
    def enhance_weak_voice(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """增强弱人声，特别针对耳语、轻声等"""
        try:
            import scipy.signal as signal
            
            # 1. 确保音频是float32类型
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # 2. 计算RMS能量
            rms = np.sqrt(np.mean(audio**2))
            if rms < 1e-6:
                return audio  # 静音，直接返回
            
            # 3. 标准化到-1到1范围（保持相对动态范围）
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            # 4. 多频段动态均衡 - 特别增强人声频率范围
            nyquist = sr // 2
            
            # 定义人声关键频率带
            frequency_bands = [
                (80, 300, 2.0),   # 低频共振区（男性声音基础频率）- 中等增强
                (300, 1000, 3.0), # 主要元音区 - 强增强
                (1000, 3000, 2.5), # 辅音和语音清晰度 - 强增强
                (3000, 5000, 1.5), # 高音和细节 - 轻微增强
            ]
            
            enhanced_bands = []
            for lowcut, highcut, gain in frequency_bands:
                if highcut < nyquist:
                    # 设计带通滤波器
                    b, a = signal.butter(
                        4, 
                        [lowcut/nyquist, highcut/nyquist], 
                        btype='band'
                    )
                    
                    # 应用滤波器
                    filtered = signal.filtfilt(b, a, audio)
                    
                    # 动态压缩：对弱信号部分增强更多
                    threshold = 0.05  # 低阈值，更多信号被增强
                    ratio = 3.0  # 压缩比
                    
                    # 计算压缩量
                    abs_filtered = np.abs(filtered)
                    gain_reduction = np.where(
                        abs_filtered > threshold,
                        1.0 / ratio,  # 超过阈值部分压缩
                        1.0 + (1.0 - abs_filtered/threshold) * (gain - 1.0)  # 弱信号线性增强
                    )
                    
                    # 应用压缩
                    compressed = filtered * gain_reduction
                    
                    enhanced_bands.append(compressed)
            
            # 5. 合并所有频段
            if enhanced_bands:
                # 加权合并，重点增强中频段
                weights = [0.7, 1.0, 0.9, 0.6]  # 对应上面的频段
                weighted_sum = np.zeros_like(audio)
                total_weight = 0
                
                for i, band in enumerate(enhanced_bands):
                    if i < len(weights):
                        weighted_sum += band * weights[i]
                        total_weight += weights[i]
                
                if total_weight > 0:
                    enhanced_audio = weighted_sum / total_weight
                else:
                    enhanced_audio = np.sum(enhanced_bands, axis=0) / len(enhanced_bands)
            else:
                enhanced_audio = audio
            
            # 6. 自适应噪声门
            # 计算短期能量
            frame_length = int(0.02 * sr)  # 20ms帧
            hop_length = frame_length // 2
            
            energy = []
            for i in range(0, len(enhanced_audio) - frame_length, hop_length):
                frame = enhanced_audio[i:i+frame_length]
                energy.append(np.mean(frame**2))
            
            if energy:
                energy = np.array(energy)
                energy_db = 10 * np.log10(energy + 1e-10)
                
                # 自适应阈值
                mean_db = np.mean(energy_db)
                std_db = np.std(energy_db)
                
                # 对于非常安静的内容，使用更低的阈值
                if mean_db < -40:
                    threshold_db = mean_db + std_db * 0.5
                else:
                    threshold_db = mean_db - std_db * 1.0
                
                # 应用噪声门（软阈值）
                threshold_linear = 10**(threshold_db/10)
                
                gated_audio = enhanced_audio.copy()
                for i in range(0, len(gated_audio) - frame_length, hop_length):
                    frame_start = i
                    frame_end = i + frame_length
                    frame = gated_audio[frame_start:frame_end]
                    
                    frame_energy = np.mean(frame**2)
                    
                    if frame_energy < threshold_linear * 0.1:
                        # 完全静音
                        gated_audio[frame_start:frame_end] *= 0.01
                    elif frame_energy < threshold_linear:
                        # 软过渡
                        attenuation = (frame_energy / threshold_linear) ** 0.5
                        gated_audio[frame_start:frame_end] *= attenuation
            
            # 7. 最终标准化，避免削波
            max_val = np.max(np.abs(gated_audio))
            if max_val > 0:
                gated_audio = gated_audio / max_val * 0.95  # 保留5%的headroom
            
            return gated_audio
            
        except Exception as e:
            logging.warning(f"弱人声增强失败: {e}")
            return audio

# ================== 语速/音调分析 ==================
class VoicePatternAnalyzer:
    """分析人声模式，优化检测"""
    
    @staticmethod
    def analyze_voice_pattern(audio: np.ndarray, sr: int = 16000) -> Dict:
        """分析音频的人声特征"""
        try:
            if not LIBROSA_AVAILABLE:
                return {"has_possible_speech": True}
            
            # 计算MFCC特征（人声特征）
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=13,
                hop_length=256
            )
            
            # 计算频谱质心（音调特征）
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=sr
            )
            
            # 计算过零率（语音/非语音区分）
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            
            # 分析结果
            return {
                'mfcc_mean': np.mean(mfcc, axis=1),
                'spectral_centroid_mean': np.mean(spectral_centroid),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'energy': np.mean(np.abs(audio)),
                'has_possible_speech': np.mean(mfcc[0:5]) < -5.0  # 简单判断
            }
            
        except Exception as e:
            logging.warning(f"声音模式分析失败: {e}")
            return {"has_possible_speech": True}

# ================== 音频处理器 ==================
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

# ================== 进度管理器 ==================
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

# ================== 弱人声转录器增强版 ==================
class WeakVoiceTranscriber:
    """弱人声转录器 - 专门针对成人内容中的轻声、耳语优化"""
    
    def __init__(self, input_file: str, model_name: str, language: str = "ja", 
                 overlap_seconds: float = 2.0, use_silero_vad: bool = True,
                 vad_threshold: float = 0.1,  # 更低阈值
                 min_speech_duration_ms: int = 100,  # 更短时长
                 aggressive_vad: bool = True,
                 whisper_temperature: float = 0.0,
                 whisper_beam_size: int = 5,
                 adult_mode: bool = True,
                 filter_hallucinations: bool = True,
                 ultra_sensitive: bool = False,  # 新增：超敏感模式
                 whisper_compression_ratio_threshold: float = 1.5,  # 新增：压缩比阈值
                 test_percentage: int = 0):  # 新增：测试模式百分比
        
        self.input_file = os.path.abspath(input_file)
        self.model_name = model_name
        self.language = language
        self.overlap_seconds = overlap_seconds
        self.use_silero_vad = use_silero_vad
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.aggressive_vad = aggressive_vad
        self.whisper_temperature = whisper_temperature
        self.whisper_beam_size = whisper_beam_size
        self.adult_mode = adult_mode
        self.filter_hallucinations = filter_hallucinations
        self.ultra_sensitive = ultra_sensitive  # 超敏感模式
        self.whisper_compression_ratio_threshold = whisper_compression_ratio_threshold
        self.test_percentage = test_percentage  # 测试模式百分比
        
        # 幻觉文本黑名单（增强版）
        self.hallucination_patterns = [
            r'ご視聴ありがとうございました[。.]*$',
            r'ご視聴ありがとうございます[。.]*$',
            r'これは日本語の音声です[。.]*$',
            r'音声です[。.]*$',
            r'以上です[。.]*$',
            r'音声を上です[。.]*$',
            r'^[\s、。]*$',  # 空白或只有标点
            r'^[あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん]{1,3}[、。]*$',  # 只有假名
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
        
        # 初始化VAD处理器（启用超敏感模式）
        if self.use_silero_vad and SILERO_VAD_AVAILABLE:
            self.performance.start_stage("弱人声VAD模型加载")
            self.vad_processor = WhisperVADProcessor(ultra_sensitive=self.ultra_sensitive)
            if self.vad_processor.available:
                logging.info("成功启用弱人声VAD，专注检测轻声、耳语")
                if self.ultra_sensitive:
                    logging.info("启用超敏感模式，将检测极弱人声")
                self.use_silero_vad = True
            else:
                logging.warning("VAD不可用，将回退到faster-whisper内置VAD")
                self.use_silero_vad = False
            self.performance.end_stage()
        else:
            logging.info("使用faster-whisper内置VAD")
        
        # 初始化语音分析器
        self.voice_analyzer = VoicePatternAnalyzer()
        
        logging.info(f"初始化完成: 输入={self.input_file}, 模型={model_name}, 语言={language}")
        logging.info(f"VAD模式: {'超敏感' if ultra_sensitive else '激进' if aggressive_vad else '保守'}")
        logging.info(f"VAD阈值: {vad_threshold}")
        logging.info(f"最小语音时长: {min_speech_duration_ms}ms")
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
        """获取转录参数 - 针对弱人声优化（增强版）"""
        options = {
            "language": self.language,
            "beam_size": self.whisper_beam_size,
            "best_of": 5,  # 增加候选数量
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 温度采样
            "patience": 1.5,  # 增加耐心值
            "condition_on_previous_text": True,  # 启用上下文
            "word_timestamps": False,
            "compression_ratio_threshold": self.whisper_compression_ratio_threshold,
            "no_speech_threshold": 0.3,  # 更低的无语音阈值
            # 移除不支持的参数 logprob_threshold
            "suppress_tokens": [-1],
            "without_timestamps": False,
            "max_initial_timestamp": 1.0,
            "length_penalty": 0.5,  # 长度惩罚
            "repetition_penalty": 1.2,  # 重复惩罚
        }
        
        # 根据是否使用外部VAD设置vad_filter
        if use_external_vad:
            options["vad_filter"] = False
        else:
            # 针对内置VAD的优化参数
            options["vad_filter"] = True
            options["vad_parameters"] = {
                "threshold": 0.1,
                "min_speech_duration_ms": 100,
                "max_speech_duration_s": 30,
                "min_silence_duration_ms": 50,
                "speech_pad_ms": 50,
            }
        
        # 针对日语成人内容优化提示词
        if self.language == "ja" and self.adult_mode:
            # 使用更具体的提示词
            options["initial_prompt"] = "自然な会話、ささやき声、ため息、息づかいなどを正確に書き起こしてください。"
        
        return options

    def _transcribe_weak_voice(self, audio: np.ndarray, chunk_start_time: float, 
                              chunk_id: int, sr: int = 16000) -> Dict:
        """转录弱人声 - 专门针对轻声、耳语优化（增强版）"""
        try:
            # 分析音频特征
            voice_pattern = self.voice_analyzer.analyze_voice_pattern(audio, sr)
            has_possible_speech = voice_pattern.get('has_possible_speech', True)
            
            if not has_possible_speech:
                logging.info(f"chunk {chunk_id}: 音频分析显示可能无人声")
                return {
                    "chunk_id": chunk_id,
                    "chunk_start": chunk_start_time,
                    "chunk_end": chunk_start_time + len(audio)/sr,
                    "segments": [],
                    "language": self.language,
                    "vad_type": "analyzed_no_speech",
                }
            
            # 使用弱人声VAD检测语音段
            vad_start = time.time()
            
            if self.ultra_sensitive:
                # 超敏感模式
                speech_timestamps = self.vad_processor.detect_ultra_sensitive_speech(audio, sr=sr)
            else:
                # 常规模式
                speech_timestamps = self.vad_processor.detect_weak_speech(
                    audio, sr=sr,
                    vad_threshold=self.vad_threshold,
                    min_speech_duration_ms=self.min_speech_duration_ms,
                    aggressive_mode=self.aggressive_vad
                )
            
            vad_time = time.time() - vad_start
            
            if not speech_timestamps:
                logging.info(f"chunk {chunk_id}: 未检测到语音段")
                return {
                    "chunk_id": chunk_id,
                    "chunk_start": chunk_start_time,
                    "chunk_end": chunk_start_time + len(audio)/sr,
                    "segments": [],
                    "language": self.language,
                    "vad_type": "weak_voice",
                    "vad_time": vad_time,
                    "speech_segments": 0,
                    "transcribed_segments": 0,
                }
            
            logging.info(f"chunk {chunk_id}: 检测到 {len(speech_timestamps)} 个语音段")
            
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
                
                # 跳过极短的段（但超敏感模式不跳过）
                if not self.ultra_sensitive and duration < 0.15:
                    continue
                
                # 对弱语音段进行额外增强
                try:
                    enhancer = VoiceEnhancer(sr)
                    segment_audio = enhancer.full_enhancement(segment_audio)
                except:
                    pass
                
                # 转录
                segment_start = time.time()
                try:
                    # 多次尝试转录
                    max_attempts = 3
                    best_result = None
                    best_confidence = -float('inf')
                    
                    for attempt in range(max_attempts):
                        segments, info = self.model.transcribe(
                            segment_audio,
                            **transcription_options
                        )
                        
                        for segment in segments:
                            confidence = getattr(segment, 'avg_logprob', 0.0)
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_result = segment
                        
                        if best_confidence > -1.0:  # 置信度阈值
                            break
                    
                    segment_time = time.time() - segment_start
                    
                    if best_result:
                        text = best_result.text.strip()
                        filtered_text = self._filter_hallucination_text(text, duration)
                        
                        if filtered_text:
                            segment_dict = {
                                "start": absolute_start + best_result.start,
                                "end": absolute_start + best_result.end,
                                "text": filtered_text,
                                "chunk_id": chunk_id,
                                "segment_index": i,
                                "duration": duration,
                                "original_text": text,
                                "confidence": best_confidence,
                                "no_speech_prob": getattr(best_result, 'no_speech_prob', 0.0),
                                "is_weak_voice": "true",
                            }
                            all_segments.append(segment_dict)
                            logging.debug(f"语音段 {i+1} 转录完成: {filtered_text[:30]}... (置信度: {best_confidence:.2f}, 耗时: {segment_time:.2f}s)")
                    
                except Exception as e:
                    logging.warning(f"语音段 {i+1} 转录失败: {e}")
                    continue
            
            # 后处理：合并相邻的相似片段
            filtered_segments = self._post_process_segments(all_segments)
            
            return {
                "chunk_id": chunk_id,
                "chunk_start": chunk_start_time,
                "chunk_end": chunk_start_time + len(audio)/sr,
                "segments": filtered_segments,
                "language": self.language,
                "vad_type": "ultra_sensitive" if self.ultra_sensitive else "weak_voice",
                "vad_time": vad_time,
                "speech_segments": len(speech_timestamps),
                "transcribed_segments": len(filtered_segments),
                "filtered_hallucinations": len(all_segments) - len(filtered_segments),
            }
            
        except Exception as e:
            logging.error(f"弱人声转录失败: {e}")
            return None
    
    def _filter_hallucination_text(self, text: str, duration: float) -> str:
        """过滤幻觉文本"""
        if not text or len(text.strip()) == 0:
            return ""
        
        # 幻觉模式匹配
        if self.filter_hallucinations:
            for pattern in self.hallucination_patterns:
                if re.match(pattern, text.strip()):
                    logging.debug(f"过滤幻觉文本: {text}")
                    return ""
        
        # 成人内容特定过滤
        if self.adult_mode:
            # 可以根据需要添加
            pass
        
        # 检查文本长度是否合理
        # 日语平均语速：4-7字符/秒
        chars_per_second = len(text) / duration if duration > 0 else 0
        
        # 如果语速异常慢或异常快，可能是幻觉
        if duration > 3.0 and chars_per_second < 0.5:  # 超过3秒但少于0.5字符/秒
            logging.debug(f"过滤语速异常慢文本: {text} ({chars_per_second:.1f}字符/秒)")
            return ""
        
        if chars_per_second > 12.0:  # 超过12字符/秒，异常快
            logging.debug(f"过滤语速异常快文本: {text} ({chars_per_second:.1f}字符/秒)")
            return ""
        
        return text.strip()
    
    def _post_process_segments(self, segments: List[Dict]) -> List[Dict]:
        """后处理：合并相邻的相似片段"""
        if not segments:
            return []
        
        # 按开始时间排序
        segments.sort(key=lambda x: x["start"])
        
        filtered = []
        last_segment = None
        
        for segment in segments:
            current_text = segment["text"]
            
            if not current_text:
                continue
            
            if not last_segment:
                filtered.append(segment)
                last_segment = segment
                continue
            
            # 检查时间重叠
            overlap = min(last_segment["end"], segment["end"]) - max(last_segment["start"], segment["start"])
            gap = segment["start"] - last_segment["end"]
            
            # 如果重叠或间隔很短，且文本相似，合并
            if (overlap > 0.1 or gap < 0.3) and self._texts_are_similar(last_segment["text"], current_text):
                # 合并
                last_segment["end"] = segment["end"]
                # 合并文本（选择更长的或置信度更高的）
                last_confidence = last_segment.get("confidence", 0)
                current_confidence = segment.get("confidence", 0)
                
                if current_confidence > last_confidence:
                    last_segment["text"] = current_text
                    last_segment["confidence"] = current_confidence
                
                continue
            
            filtered.append(segment)
            last_segment = segment
        
        return filtered
    
    def _texts_are_similar(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """判断两个文本是否相似"""
        import difflib
        
        if text1 == text2:
            return True
        
        # 去除标点
        clean1 = re.sub(r'[、。,!?]', '', text1)
        clean2 = re.sub(r'[、。,!?]', '', text2)
        
        # 使用difflib计算相似度
        similarity = difflib.SequenceMatcher(None, clean1, clean2).ratio()
        
        return similarity > threshold
    
    def _transcribe_with_builtin_vad(self, audio: np.ndarray, chunk_start_time: float,
                                    chunk_id: int, sr: int = 16000) -> Dict:
        """使用faster-whisper内置VAD进行转录"""
        try:
            # 转录参数
            transcription_options = self._get_transcription_options(use_external_vad=False)
            
            # 内置VAD参数 - 针对弱人声优化
            vad_params = {
                "threshold": 0.1,  # 低阈值
                "min_speech_duration_ms": 100,
                "max_speech_duration_s": 30,
                "min_silence_duration_ms": 100,
                "speech_pad_ms": 50,
            }
            
            # 执行转录
            segments, info = self.model.transcribe(
                audio,
                vad_parameters=vad_params,
                **transcription_options
            )
            
            # 处理结果
            chunk_segments = []
            for segment in segments:
                text = segment.text.strip()
                duration = segment.end - segment.start
                
                # 应用抗幻觉过滤
                filtered_text = self._filter_hallucination_text(text, duration)
                
                if filtered_text:
                    segment_dict = {
                        "start": chunk_start_time + segment.start,
                        "end": chunk_start_time + segment.end,
                        "text": filtered_text,
                        "chunk_id": chunk_id,
                        "original_text": text,
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
            logging.error(f"使用内置VAD转录失败: {e}")
            return None
    
    def transcribe_chunk(self, chunk_id: int, chunk_path: str, chunk_start_time: float) -> Dict:
        """转录单个chunk"""
        try:
            # 检查chunk文件
            if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
                logging.error(f"chunk文件无效: {chunk_path}")
                return None
            
            # 加载chunk音频
            try:
                audio, sr = librosa.load(chunk_path, sr=16000, mono=True)
                
                # 检查音频数据
                if len(audio) == 0 or np.max(np.abs(audio)) < 0.0001:
                    logging.warning(f"chunk {chunk_id} 音频数据过小")
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
                result = self._transcribe_weak_voice(audio, chunk_start_time, chunk_id, sr)
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
            
            # 测试模式：限制处理的chunk数量
            if self.test_percentage > 0:
                original_count = len(chunks)
                max_chunks = max(1, int(original_count * self.test_percentage / 100))
                chunks = chunks[:max_chunks]
                print(f"\n测试模式：仅处理前 {max_chunks}/{original_count} 个chunks ({self.test_percentage}%)")
                logging.info(f"测试模式：限制为 {max_chunks}/{original_count} chunks")
            
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
            
            total_hallucinations_filtered = 0
            total_weak_voice_segments = 0
            
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
                    
                    # 统计弱人声片段
                    segments = result.get("segments", [])
                    weak_segments = sum(1 for seg in segments if seg.get("is_weak_voice", False))
                    total_weak_voice_segments += weak_segments
                    
                    logging.info(f"完成chunk {chunk_id}，VAD: {vad_type}, 语音段: {speech_segments}, 转录段: {transcribed_segments}, 弱人声: {weak_segments}, 过滤幻觉: {filtered_hallucinations}, 耗时: {TimeFormatter.format_seconds(chunk_time)}")
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
            print(f"弱人声转录完成!")
            print("="*70)
            
            # 测试模式信息
            if self.test_percentage > 0:
                print(f"测试模式: 是 ({self.test_percentage}%)")
                print(f"处理chunks数: {final_result.get('info', {}).get('total_chunks_processed', 0)}")
            
            print(f"音频时长: {TimeFormatter.format_seconds(audio_duration)}")
            print(f"总处理时间: {TimeFormatter.format_seconds(total_time)}")
            
            if audio_duration > 0:
                real_time_factor = total_time / audio_duration
                print(f"实时因子: {real_time_factor:.2f}x")
            
            print(f"\n弱人声检测统计:")
            print(f"  过滤幻觉文本总数: {total_hallucinations_filtered}")
            print(f"  检测到弱人声片段: {total_weak_voice_segments}")
            print(f"  最终有效片段数: {final_result.get('info', {}).get('total_segments', 0)}")
            print(f"  VAD类型: {'弱人声专用VAD' if self.use_silero_vad else '内置VAD'}")
            print(f"  VAD模式: {'超敏感' if self.ultra_sensitive else '激进' if self.aggressive_vad else '保守'}")
            print(f"  VAD阈值: {self.vad_threshold}")
            print(f"  最小语音时长: {self.min_speech_duration_ms}ms")
            
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
        total_chunks_processed = 0
        
        # 按chunk_id排序
        for chunk_id in sorted(map(int, chunk_results.keys())):
            result = chunk_results[str(chunk_id)]
            if result and "segments" in result:
                all_segments.extend(result["segments"])
                total_chunks_processed += 1
        
        # 按开始时间排序
        all_segments.sort(key=lambda x: x["start"])
        
        # 最终过滤：移除重复片段
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
                "total_chunks_processed": total_chunks_processed,
                "test_percentage": self.test_percentage,
                "vad_type": "weak_voice" if self.use_silero_vad else "builtin",
                "vad_threshold": self.vad_threshold,
                "min_speech_duration_ms": self.min_speech_duration_ms,
                "aggressive_vad": self.aggressive_vad,
                "ultra_sensitive": self.ultra_sensitive,
                "whisper_temperature": self.whisper_temperature,
                "whisper_beam_size": self.whisper_beam_size,
                "whisper_compression_ratio_threshold": self.whisper_compression_ratio_threshold,
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
        
        # TXT格式
        txt_path = os.path.join(self.temp_dir, "transcription.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            # 写入文件头信息
            f.write(f"视频: {result.get('info', {}).get('audio_file', 'Unknown')}\n")
            f.write(f"模型: {result.get('info', {}).get('model', 'Unknown')}\n")
            f.write(f"语言: {result.get('language', 'Unknown')}\n")
            f.write(f"VAD类型: {result.get('info', {}).get('vad_type', 'Unknown')}\n")
            f.write(f"超敏感模式: {'启用' if result.get('info', {}).get('ultra_sensitive', False) else '禁用'}\n")
            
            # 测试模式信息
            if self.test_percentage > 0:
                f.write(f"测试模式: 是 ({self.test_percentage}%)\n")
                f.write(f"处理chunks数: {result.get('info', {}).get('total_chunks_processed', 0)}\n")
            
            f.write(f"总片段数: {result.get('info', {}).get('total_segments', 0)}\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入带时间戳的文本内容
            segments = result.get("segments", [])
            for segment in segments:
                start_time = TimeFormatter.format_timestamp(segment["start"])
                end_time = TimeFormatter.format_timestamp(segment["end"])
                text = segment.get("text", "").strip()
                
                if text:
                    weak_mark = "[弱]" if segment.get("is_weak_voice", False) else ""
                    f.write(f"[{start_time} - {end_time}] {weak_mark}{text}\n")
        
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

# ================== 主函数 ==================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="日语弱人声转录工具 - 专为轻声、耳语优化")
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
    parser.add_argument("--vad-threshold", type=float, default=0.1,
                       help="VAD检测阈值 (0.05-0.2，越低越敏感) (默认: 0.1)")
    parser.add_argument("--min-speech-duration", type=int, default=100,
                       help="最小语音时长（毫秒）(默认: 100)")
    parser.add_argument("--conservative-vad", action="store_true",
                       help="使用保守VAD模式（默认: 激进模式）")
    parser.add_argument("--ultra-sensitive", action="store_true",
                       help="启用超敏感模式，检测极弱人声")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Whisper温度参数 (默认: 0.0，确定性采样)")
    parser.add_argument("--beam-size", type=int, default=5,
                       help="Whisper束搜索大小 (默认: 5)")
    parser.add_argument("--compression-ratio", type=float, default=1.5,
                       help="压缩比阈值 (默认: 1.5)")
    parser.add_argument("--no-filter-hallucinations", action="store_true",
                       help="禁用幻觉文本过滤")
    parser.add_argument("--test", type=int, default=0,
                       help="测试模式：仅转录前百分之N的音频 (默认: 0=禁用，10=转录前10%)")
    
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
    required_packages = {
        "torch": "pip install torch",
        "librosa": "pip install librosa",
    }
    
    missing_packages = []
    for package, install_cmd in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(f"  {install_cmd}")
    
    if missing_packages:
        print("请安装以下依赖库:")
        for cmd in missing_packages:
            print(cmd)
        sys.exit(1)
    
    # 检查ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
    except FileNotFoundError:
        print("警告: 未找到ffmpeg，音频分块功能可能受限")
        print("请安装ffmpeg: https://ffmpeg.org/download.html")
    
    # 创建转录器并开始处理
    use_silero_vad = not args.no_silero_vad
    aggressive_vad = not args.conservative_vad
    filter_hallucinations = not args.no_filter_hallucinations
    
    transcriber = WeakVoiceTranscriber(
        args.input_file, 
        args.model, 
        args.language, 
        args.overlap,
        use_silero_vad,
        args.vad_threshold,
        args.min_speech_duration,
        aggressive_vad,
        args.temperature,
        args.beam_size,
        adult_mode=True,
        filter_hallucinations=filter_hallucinations,
        ultra_sensitive=args.ultra_sensitive,
        whisper_compression_ratio_threshold=args.compression_ratio,
        test_percentage=args.test
    )
    
    print("\n" + "="*70)
    print(f"开始弱人声转录: {args.input_file}")
    print(f"使用模型: {args.model}")
    print(f"使用VAD: {'弱人声专用VAD' if use_silero_vad else '内置VAD'}")
    print(f"VAD模式: {'超敏感' if args.ultra_sensitive else '激进' if aggressive_vad else '保守'}")
    print(f"VAD阈值: {args.vad_threshold} (越低越敏感)")
    print(f"最小语音时长: {args.min_speech_duration}ms")
    print(f"Whisper温度: {args.temperature}")
    print(f"抗幻觉模式: {'启用' if filter_hallucinations else '禁用'}")
    print(f"重叠时长: {args.overlap}秒")
    
    # 测试模式信息
    if args.test > 0:
        print(f"测试模式: 是 ({args.test}%)")
    
    print("="*70 + "\n")
    
    success = transcriber.process()
    
    if success:
        print("弱人声转录任务完成!")
        sys.exit(0)
    else:
        print("任务失败，请检查日志文件")
        sys.exit(1)

if __name__ == "__main__":
    main()
