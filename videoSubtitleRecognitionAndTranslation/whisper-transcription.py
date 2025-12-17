import argparse
import os
import sys
import json
import time
import subprocess
import hashlib
import re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import wave
import contextlib

try:
    import torch
except ImportError:
    torch = None

# 尝试导入 faster-whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("警告: 未安装faster-whisper，将使用标准whisper库")
    print("建议安装: pip install faster-whisper")

# 转录结果过滤器类
class TranscriptionFilter:
    """转录结果过滤器，用于过滤不合理的字幕"""
    
    def __init__(self, 
                 max_syllable_repeat: int = 4,
                 min_speech_rate: float = 0.3,
                 max_speech_rate: float = 20.0,
                 filter_patterns: list = None,
                 min_duration: float = 0.3,
                 max_duration: float = 30.0,
                 confidence_threshold: float = -1.0):
        """
        初始化过滤器（针对日语成人视频优化）
        """
        self.max_syllable_repeat = max_syllable_repeat
        self.min_speech_rate = min_speech_rate
        self.max_speech_rate = max_speech_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.confidence_threshold = confidence_threshold
        
        # 日语成人视频特定的过滤模式
        self.filter_patterns = filter_patterns or [
            r"ご視聴ありがとうございました",
            r"お疲れ様でした",
            r"終わり",
            r"^[あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんぁぃぅぇぉゃゅょっ\-ー、。，．,.\s]+$",
        ]
        
        # 编译正则表达式
        self.compiled_patterns = [re.compile(pattern) for pattern in self.filter_patterns]
        
        # 日语中允许更多的重复字符（如"あああ"、"ううう"）
        self.repeat_pattern = re.compile(r'(.)\1{' + str(max_syllable_repeat) + r',}')
    
    def calculate_speech_rate(self, text: str, duration: float) -> float:
        """
        计算语速（字符/秒）
        """
        if duration <= 0:
            return 0
        
        # 去除空格和标点，计算实际字符数
        cleaned_text = re.sub(r'[、。，．,.\s]', '', text)
        char_count = len(cleaned_text)
        
        return char_count / duration
    
    def is_excessive_repetition(self, text: str) -> bool:
        """
        检查是否有过多的重复字符
        """
        return bool(self.repeat_pattern.search(text))
    
    def is_filtered_pattern(self, text: str) -> bool:
        """
        检查是否匹配要过滤的模式
        """
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        return False
    
    def should_keep_segment(self, segment: dict, context_segments: list = None) -> bool:
        """
        判断是否应该保留这个字幕片段（针对日语成人视频优化）
        """
        text = segment.get('text', '').strip()
        
        # 如果文本为空，不保留
        if not text:
            return False
        
        # 日语常见短回应词列表
        japanese_short_words = ["あ", "う", "え", "お", "ん", "は", "へ", "も", "や", "ら", "れ", "ろ"]
        japanese_responses = ["はい", "うん", "ええ", "ああ", "いいえ", "はあ", "へえ", "うう", "んん"]
        
        # 检查是否为日语短词
        is_short_japanese = text in japanese_short_words or text in japanese_responses
        
        # 1. 检查置信度（如果可用）
        confidence = segment.get('confidence', 0)
        if confidence < self.confidence_threshold and not is_short_japanese:
            print(f"置信度过低({confidence:.2f}): {text[:30]}...")
            return False
        
        # 2. 检查是否匹配过滤模式（对短日语词更宽容）
        if self.is_filtered_pattern(text) and not is_short_japanese:
            print(f"过滤模式匹配: {text[:50]}...")
            return False
        
        # 3. 检查是否有过多的重复字符（对日语更宽容）
        if self.is_excessive_repetition(text) and not is_short_japanese:
            # 日语中允许一定重复（如"あああ"），但检查是否过度
            char = text[0]
            repeat_count = len(text)
            # 日语中允许重复3-4次，超过5次可能不合理
            if repeat_count > 5:
                print(f"过多重复字符({repeat_count}次): {text[:50]}...")
                return False
        
        # 4. 计算语速并检查是否在合理范围内（对日语成人视频更宽松）
        duration = segment.get('end', 0) - segment.get('start', 0)
        speech_rate = self.calculate_speech_rate(text, duration)
        
        # 日语成人视频中语速可能较慢，放宽下限
        if speech_rate < self.min_speech_rate and duration > 3.0 and not is_short_japanese:
            print(f"语速过慢({speech_rate:.1f}字符/秒, {duration:.1f}秒): {text[:30]}...")
            return False
        
        if speech_rate > self.max_speech_rate and not is_short_japanese:
            print(f"语速过快({speech_rate:.1f}字符/秒): {text[:30]}...")
            return False
        
        # 5. 检查片段时长（对短回应词更宽容）
        if duration < self.min_duration and not is_short_japanese:
            print(f"片段过短({duration:.1f}秒): {text[:30]}...")
            return False
        
        if duration > self.max_duration:
            print(f"片段过长({duration:.1f}秒): {text[:30]}...")
            return False
        
        # 6. 上下文检查：对于日语短回应词，允许更多重复
        if context_segments:
            recent_texts = [s.get('text', '').strip() for s in context_segments[-5:] if s != segment]
            
            # 如果是短日语词，允许更多重复
            if is_short_japanese:
                # 短词允许在较近的上下文中重复出现
                if text in recent_texts[-2:]:
                    # 如果在最近2个片段中重复，可能过滤
                    pass
            else:
                # 非短词使用正常检查
                if text in recent_texts:
                    print(f"重复文本: {text[:30]}...")
                    return False
        
        return True
    
    def filter_segments(self, segments: list, window_size: int = 5) -> list:
        """
        过滤字幕片段
        """
        filtered_segments = []
        total_before = len(segments)
        
        for i, segment in enumerate(segments):
            # 获取上下文窗口
            start_idx = max(0, i - window_size)
            end_idx = min(len(segments), i + window_size + 1)
            context_segments = segments[start_idx:end_idx]
            
            if self.should_keep_segment(segment, context_segments):
                filtered_segments.append(segment)
        
        total_after = len(filtered_segments)
        print(f"过滤完成: 从 {total_before} 个片段过滤到 {total_after} 个片段")
        print(f"移除了 {total_before - total_after} 个不合理片段")
        
        return filtered_segments

# 虚拟环境管理类
class VirtualEnvironmentManager:
    """管理虚拟环境的激活和退出"""
    
    def __init__(self, venv_path=None):
        """
        初始化虚拟环境管理器
        """
        if venv_path is None:
            # 自动检测虚拟环境路径
            script_dir = Path(__file__).parent
            possible_venv_paths = [
                script_dir / "whisperx_env",
                script_dir / "venv",
                script_dir / "env"
            ]
            
            for path in possible_venv_paths:
                if path.exists():
                    self.venv_path = path
                    break
            else:
                self.venv_path = None
        else:
            self.venv_path = Path(venv_path)
        
        self.original_sys_path = sys.path.copy()
        self.original_os_environ = os.environ.copy()
        self.is_activated = False
    
    def activate(self):
        """激活虚拟环境"""
        if self.venv_path is None:
            print("警告: 未找到虚拟环境，使用系统Python环境")
            return True
        
        try:
            # 获取虚拟环境的Python路径
            if os.name == 'nt':  # Windows
                python_exe = self.venv_path / "Scripts" / "python.exe"
                site_packages = self.venv_path / "Lib" / "site-packages"
            else:  # Unix/Linux
                python_exe = self.venv_path / "bin" / "python"
                site_packages = self.venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            
            if not python_exe.exists():
                print(f"警告: 虚拟环境Python可执行文件不存在: {python_exe}")
                return False
            
            # 添加虚拟环境的site-packages到sys.path
            if site_packages.exists():
                sys.path.insert(0, str(site_packages))
            
            # 设置环境变量
            os.environ['VIRTUAL_ENV'] = str(self.venv_path)
            
            # 更新PATH环境变量
            if os.name == 'nt':  # Windows
                venv_bin = self.venv_path / "Scripts"
            else:  # Unix/Linux
                venv_bin = self.venv_path / "bin"
            
            if venv_bin.exists():
                os.environ['PATH'] = str(venv_bin) + os.pathsep + os.environ['PATH']
            
            self.is_activated = True
            print(f"虚拟环境已激活: {self.venv_path}")
            return True
            
        except Exception as e:
            print(f"激活虚拟环境失败: {e}")
            return False
    
    def deactivate(self):
        """退出虚拟环境"""
        if not self.is_activated:
            return
        
        try:
            # 恢复原始sys.path
            sys.path[:] = self.original_sys_path
            
            # 恢复原始环境变量
            os.environ.clear()
            os.environ.update(self.original_os_environ)
            
            self.is_activated = False
            print("虚拟环境已退出")
            
        except Exception as e:
            print(f"退出虚拟环境失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.activate()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.deactivate()

# 延迟导入whisper，确保在虚拟环境激活后导入
def import_whisper():
    """在虚拟环境激活后导入whisper模块"""
    try:
        import whisper
        return whisper
    except ImportError as e:
        print(f"导入whisper模块失败: {e}")
        print("请确保虚拟环境中已安装whisper模块")
        return None

def format_timedelta(seconds):
    """将秒数格式化为时:分:秒格式"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_audio_duration(audio_path):
    """获取音频文件的时长（秒）"""
    try:
        with contextlib.closing(wave.open(str(audio_path), 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except:
        return 0

class SegmentTranscriber:
    """处理音频分段的转录"""
    
    def __init__(self, temp_base, model, language, model_size="base", preprocess_audio=True):
        self.temp_base = temp_base
        self.model = model  # 现在model可能为None，将在需要时懒加载
        self.language = language
        self.model_size = model_size  # 存储模型大小参数
        self.preprocess_audio = preprocess_audio  # 是否预处理音频，默认启用
        self.segments_dir = temp_base / "segments"
        self.segments_dir.mkdir(exist_ok=True)
        
        # 延迟导入的whisper模块
        self.whisper_module = None
        
        # 懒加载状态
        self.model_loaded = model is not None
        
        # 使用faster-whisper的标记
        self.use_faster_whisper = FASTER_WHISPER_AVAILABLE
    
    def _extract_limited_segments(self, video_path, segment_duration=600, max_segments=1):
        """提取指定数量的音频片段（用于测试模式）"""
        print(f"提取前 {max_segments} 个音频片段...")
        
        try:
            # 先获取视频总时长
            duration_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            
            result = subprocess.run(duration_cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                video_duration = float(result.stdout.strip())
            else:
                print("无法获取视频时长，使用默认值")
                video_duration = 3600  # 默认1小时
            
            # 计算最大片段数（不超过实际可能的片段数）
            max_possible_segments = int(video_duration // segment_duration) + 1
            num_segments = min(max_segments, max_possible_segments)
            
            segment_files = []
            
            # 逐个提取片段
            for i in range(num_segments):
                start_time = i * segment_duration
                segment_file = self.segments_dir / f"segment_{i:03d}.wav"
                
                # 如果文件已存在且大小合理，跳过
                if segment_file.exists() and segment_file.stat().st_size > 1000:
                    segment_files.append(segment_file)
                    print(f"片段 {i+1}/{num_segments} 已存在")
                    continue
                
                # 提取当前片段
                cmd = [
                    'ffmpeg', '-ss', str(start_time), '-i', str(video_path),
                    '-t', str(min(segment_duration, video_duration - start_time)),
                    '-ac', '1', '-ar', '16000',  # 单声道，16kHz采样率
                    '-acodec', 'pcm_s16le',      # PCM编码
                    '-y',  # 覆盖已存在文件
                    str(segment_file)
                ]
                
                print(f"提取片段 {i+1}/{num_segments} (开始时间: {start_time}s)")
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(timeout=300)  # 5分钟超时
                
                if process.returncode == 0:
                    segment_files.append(segment_file)
                else:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    print(f"片段 {i+1} 提取失败: {error_msg}")
                    # 创建空文件标记失败
                    segment_file.touch()
            
            print(f"提取完成，共 {len(segment_files)} 个片段")
            return segment_files
            
        except Exception as e:
            print(f"提取指定片段出错: {e}")
            return False
    
    def _preprocess_audio(self, audio_file):
        """对音频进行预处理，增强轻声语音"""
        try:
            # 创建预处理后的音频文件路径
            processed_file = audio_file.with_stem(f"{audio_file.stem}_processed")
            
            # 使用ffmpeg增强音频（针对轻声语音优化）
            cmd = [
                'ffmpeg', '-i', str(audio_file),
                '-af',
                # 组合多个音频滤波器增强轻声
                # 1. 动态范围压缩：增强轻声，限制大声
                'acompressor=threshold=0.02:ratio=20:attack=5:release=100,' +
                # 2. 音量标准化
                'loudnorm=I=-16:TP=-1.5:LRA=11,' +
                # 3. 高频增强（日语语音特征）
                'equalizer=f=3000:width_type=h:width=2000:g=3,' +
                # 4. 低频削减（减少背景噪声）
                'highpass=f=80,' +
                # 5. 使用compand替代limiter进行限幅防止失真
                'compand=attacks=0.3:decays=0.8:points=-80/-80|-12/-12|0/-3',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(processed_file)
            ]
            
            # 执行预处理
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=30)
            
            if result.returncode == 0 and processed_file.exists():
                print(f"音频预处理完成: {audio_file.name}")
                return processed_file
            else:
                print(f"音频预处理失败，使用原始音频: {result.stderr}")
                return audio_file
                
        except Exception as e:
            print(f"音频预处理出错: {e}")
            return audio_file
    
    def split_audio(self, audio_file, segment_duration=600, max_segments=None):
        """将音频分割成多个片段（默认10分钟一个片段）"""
        print(f"将音频分割成 {segment_duration} 秒的片段...")
        
        # 创建片段文件
        cmd = [
            'ffmpeg', '-i', str(audio_file),
            '-f', 'segment',
            '-segment_time', str(segment_duration),
            '-c', 'copy',
            '-reset_timestamps', '1',
            str(self.segments_dir / "segment_%03d.wav")
        ]
        
        # 修复编码问题：使用UTF-8编码处理输出
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print(f"音频分割失败: {result.stderr}")
            return False
        
        # 获取所有片段文件
        segment_files = sorted(list(self.segments_dir.glob("segment_*.wav")))
        
        # 应用测试模式限制
        if max_segments is not None and len(segment_files) > max_segments:
            segment_files = segment_files[:max_segments]
            print(f"测试模式: 只保留前 {max_segments} 个片段")
        
        return segment_files
    
    def split_audio_from_video(self, video_path, segment_duration=600, max_segments=None):
        """直接从视频文件分割音频，避免生成巨大的中间文件"""
        print(f"直接从视频分割音频成 {segment_duration} 秒的片段...")
        
        try:
            # 如果指定了最大片段数，使用不同的分割策略
            if max_segments is not None:
                print(f"测试模式: 只分割前 {max_segments} 个片段")
                
                # 方法3: 逐个提取指定数量的片段
                return self._extract_limited_segments(video_path, segment_duration, max_segments)
            
            # 方法1: 使用ffmpeg的segment功能直接分割视频音频
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-ac', '1', '-ar', '16000',  # 单声道，16kHz采样率
                '-acodec', 'pcm_s16le',      # PCM编码
                '-f', 'segment',
                '-segment_time', str(segment_duration),
                '-reset_timestamps', '1',
                str(self.segments_dir / "segment_%03d.wav")
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            # 使用Popen进行流式处理
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=1800)  # 30分钟超时
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                print(f"直接视频分割失败: {error_msg}")
                
                # 方法2: 回退到逐段提取
                return self._split_audio_sequential(video_path, segment_duration, max_segments)
            
            # 获取所有片段文件
            segment_files = sorted(list(self.segments_dir.glob("segment_*.wav")))
            print(f"直接视频分割完成，共 {len(segment_files)} 个片段")
            return segment_files
            
        except subprocess.TimeoutExpired:
            print("直接视频分割超时，尝试逐段提取...")
            return self._split_audio_sequential(video_path, segment_duration, max_segments)
        except Exception as e:
            print(f"直接视频分割出错: {e}")
            return self._split_audio_sequential(video_path, segment_duration, max_segments)
    
    def _split_audio_sequential(self, video_path, segment_duration=600, max_segments=None):
        """逐段提取音频，内存占用更小但速度较慢"""
        print(f"使用逐段提取方式分割音频...")
        
        try:
            # 先获取视频总时长
            duration_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            
            result = subprocess.run(duration_cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                video_duration = float(result.stdout.strip())
            else:
                print("无法获取视频时长，使用默认值")
                video_duration = 3600  # 默认1小时
            
            # 计算片段数量，考虑max_segments限制
            total_segments = int(video_duration // segment_duration) + 1
            if max_segments is not None:
                num_segments = min(max_segments, total_segments)
                print(f"测试模式: 只提取前 {num_segments} 个片段")
            else:
                num_segments = total_segments
            
            segment_files = []
            
            # 逐段提取音频
            for i in range(num_segments):
                start_time = i * segment_duration
                segment_file = self.segments_dir / f"segment_{i:03d}.wav"
                
                # 如果文件已存在且大小合理，跳过
                if segment_file.exists() and segment_file.stat().st_size > 1000:
                    segment_files.append(segment_file)
                    continue
                
                # 提取当前片段
                cmd = [
                    'ffmpeg', '-ss', str(start_time), '-i', str(video_path),
                    '-t', str(min(segment_duration, video_duration - start_time)),
                    '-ac', '1', '-ar', '16000',  # 单声道，16kHz采样率
                    '-acodec', 'pcm_s16le',      # PCM编码
                    '-y',  # 覆盖已存在文件
                    str(segment_file)
                ]
                
                print(f"提取片段 {i+1}/{num_segments} (开始时间: {start_time}s)")
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(timeout=300)  # 5分钟超时
                
                if process.returncode == 0:
                    segment_files.append(segment_file)
                else:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    print(f"片段 {i+1} 提取失败: {error_msg}")
                    # 创建空文件标记失败
                    segment_file.touch()
            
            print(f"逐段提取完成，共 {len(segment_files)} 个片段")
            return segment_files
            
        except Exception as e:
            print(f"逐段提取出错: {e}")
            return False
    
    def _lazy_load_whisper_model(self):
        """懒加载Whisper模型"""
        if self.model is not None:
            return True
        
        print("懒加载Whisper模型...")
        model_start = time.time()
        
        try:
            # 优先使用faster-whisper
            if self.use_faster_whisper:
                print(f"使用faster-whisper加载模型: {self.model_size}")
                
                # 设置设备
                device = "cuda" if torch and hasattr(torch, 'cuda') and torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                # 加载faster-whisper模型
                self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
                
                print(f"faster-whisper模型加载完成，使用设备: {device}")
            else:
                # 使用标准whisper库
                print(f"使用标准whisper加载模型: {self.model_size}")
                
                # 延迟导入whisper模块
                if self.whisper_module is None:
                    self.whisper_module = import_whisper()
                    if self.whisper_module is None:
                        print("错误: 无法导入whisper模块")
                        return False
                
                # 使用正确的模型大小进行加载
                self.model = self.whisper_module.load_model(self.model_size)
            
            # 记录内存使用情况
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                print(f"GPU内存使用: 已分配 {memory_allocated:.2f}GB, 已保留 {memory_reserved:.2f}GB")
            
            model_time = time.time() - model_start
            print(f"模型加载完成，耗时: {format_timedelta(model_time)}")
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None
            self.model_loaded = False
            return False

    def transcribe_segment(self, segment_file, segment_index, start_time=0):
        """转录单个音频片段（针对日语成人视频优化）"""
        print(f"转录片段 {segment_index}: {segment_file.name}")
        
        try:
            # 检查文件大小
            file_size = segment_file.stat().st_size
            if file_size < 1000:  # 小于1KB的文件可能有问题
                print(f"警告: 片段文件过小 ({file_size} bytes)，可能为空或损坏")
                return []
            
            # 音频预处理（如果启用）
            processed_file = segment_file
            if self.preprocess_audio:
                processed_file = self._preprocess_audio(segment_file)
            
            # 懒加载Whisper模型
            if not self._lazy_load_whisper_model():
                print("错误: Whisper模型加载失败")
                return []
            
            if self.use_faster_whisper:
                # 针对日语成人视频优化的faster-whisper参数
                segments, info = self.model.transcribe(
                    str(processed_file),
                    language=self.language,
                    
                    # 解码参数优化
                    beam_size=10,  # 增大束搜索宽度，提高识别准确性
                    best_of=10,    # 增加候选数量
                    
                    # 温度参数优化 - 使用更温和的温度范围
                    temperature=(0.0, 0.1, 0.2, 0.3, 0.4),  # 更多温度点，适应不同语音
                    
                    # 语音检测参数优化（即使不使用VAD）
                    compression_ratio_threshold=2.8,  # 提高压缩比阈值，不过滤轻声语音
                    no_speech_threshold=0.5,          # 降低无语音阈值，更容易检测轻声
                    
                    # 语言模型参数优化
                    condition_on_previous_text=False,  # 不依赖上文，独立识别每段
                    suppress_blank=False,             # 不抑制空白，避免过滤轻声
                    
                    # 时间戳参数优化
                    word_timestamps=True,             # 启用单词级时间戳，更好处理断续语音
                    prepend_punctuations="\\\"\\\'¿([{-",  # 标点设置
                    append_punctuations="\"\'.。,，!！?？:："')]}、',
                    
                    # 针对日语的特殊优化
                    initial_prompt="こんにちは、はい、いいえ、ああ、うん、ええ、はいはい、あっ",  # 日语常见开头词提示
                    
                    # 重复惩罚
                    repetition_penalty=1.1,           # 轻微惩罚重复，防止过度重复
                    
                    # 长度惩罚
                    length_penalty=0.8,              # 降低长度惩罚，接受更短片段
                    
                    # 不使用VAD（根据您的要求）
                    vad_filter=False,
                    
                    # 启用更多选项
                    max_initial_timestamp=1.0,        # 允许更早的时间戳
                    max_new_tokens=100,               # 增加最大新token数
                    
                    # 采样策略
                    patience=2.0,                     # 增加耐心值，更全面搜索
                )
                
                # 转换faster-whisper的segments格式为标准格式
                result_segments = []
                for segment in segments:
                    # 验证时间戳
                    if segment.end <= segment.start:
                        print(f"警告: 无效时间戳 {segment.start}-{segment.end}，跳过")
                        continue
                    
                    seg_duration = segment.end - segment.start
                    if seg_duration < 0.1 or seg_duration > 30:  # 过短或过长的片段
                        print(f"警告: 异常片段时长 {seg_duration:.2f}s，跳过")
                        continue
                    
                    # 检查片段是否合理
                    text = segment.text.strip()
                    if not text:
                        continue
                        
                    # 对日语成人视频的特殊处理：保留短但有意义的片段
                    # 日语中短的回应词（如"あっ"、"うん"、"はい"）可能是重要对话
                    japanese_short_responses = ["あ", "う", "え", "お", "ん", "はい", "うん", "ええ", "あっ", "はっ"]
                    
                    # 如果文本长度短但可能是日语回应词，保留
                    is_short_response = len(text) <= 3 and any(text == resp or text.startswith(resp) for resp in japanese_short_responses)
                    
                    if len(text) < 2 and not is_short_response:
                        continue
                    
                    # 获取置信度信息
                    avg_logprob = getattr(segment, 'avg_logprob', 0)
                    no_speech_prob = getattr(segment, 'no_speech_prob', 0)
                    
                    # 对于日语成人视频，即使no_speech_prob较高也可能包含重要语音
                    # 调整置信度计算，考虑日语语音特点
                    adjusted_confidence = avg_logprob
                    if no_speech_prob > 0.3:
                        # 如果无语音概率较高，但文本是日语常见词，适当提高置信度
                        if any(word in text for word in ["あ", "う", "ん", "はい", "いいえ", "ああ"]):
                            adjusted_confidence += 0.2
                    
                    # 计算语速调整置信度（日语中语速变化大）
                    char_count = len(re.sub(r'[、。，．,.\s]', '', text))
                    speech_rate = char_count / seg_duration if seg_duration > 0 else 0
                    
                    # 日语正常语速范围：2-15字符/秒，成人视频可能更慢
                    if 1.0 <= speech_rate <= 20.0:
                        # 在合理语速范围内，维持置信度
                        pass
                    elif speech_rate < 1.0:
                        # 语速过慢，可能是呻吟声或轻声，适当降低置信度但不过滤
                        adjusted_confidence -= 0.1
                    elif speech_rate > 20.0:
                        # 语速过快，可能是噪声，降低置信度
                        adjusted_confidence -= 0.3
                    
                    result_segments.append({
                        'start': segment.start + start_time,
                        'end': segment.end + start_time,
                        'text': text,
                        'confidence': adjusted_confidence,
                        'no_speech_prob': no_speech_prob,
                        'speech_rate': speech_rate
                    })
                
                return result_segments
            else:
                # 使用标准whisper进行转录
                import whisper
                
                # 加载音频
                audio = self.whisper_module.load_audio(str(processed_file))
                
                # 转录当前片段，针对日语成人视频优化参数
                result = self.model.transcribe(
                    audio,
                    language=self.language,
                    task="transcribe",
                    fp16=False,
                    temperature=(0.0, 0.1, 0.2, 0.3, 0.4),
                    compression_ratio_threshold=2.8,
                    logprob_threshold=-0.8,
                    no_speech_threshold=0.5,
                    condition_on_previous_text=False,
                    best_of=10,
                    beam_size=10,
                    patience=2.0,
                    length_penalty=0.8,
                    suppress_tokens=[-1],
                    initial_prompt="こんにちは、はい、いいえ、ああ、うん、ええ、はいはい、あっ",
                    word_timestamps=True,
                    prepend_punctuations="\\\"\\\'¿([{-",
                    append_punctuations="\"\'.。,，!！?？:："')]}、',
                    repetition_penalty=1.1
                )
                
                # 调整时间戳，加上片段起始时间
                result_segments = []
                for segment in result['segments']:
                    # 验证时间戳
                    if segment['end'] <= segment['start']:
                        print(f"警告: 无效时间戳 {segment['start']}-{segment['end']}，跳过")
                        continue
                    
                    seg_duration = segment['end'] - segment['start']
                    if seg_duration < 0.1 or seg_duration > 30:  # 过短或过长的片段
                        print(f"警告: 异常片段时长 {seg_duration:.2f}s，跳过")
                        continue
                    
                    text = segment['text'].strip()
                    if not text:
                        continue
                        
                    # 日语短回应词处理
                    japanese_short_responses = ["あ", "う", "え", "お", "ん", "はい", "うん", "ええ", "あっ", "はっ"]
                    is_short_response = len(text) <= 3 and any(text == resp or text.startswith(resp) for resp in japanese_short_responses)
                    
                    if len(text) < 2 and not is_short_response:
                        continue
                    
                    segment['start'] += start_time
                    segment['end'] += start_time
                    segment['confidence'] = segment.get('avg_logprob', 0)
                    result_segments.append(segment)
                
                return result_segments
        except Exception as e:
            print(f"片段 {segment_index} 转录失败: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # 清理预处理文件
            if self.preprocess_audio and processed_file != segment_file and processed_file.exists():
                try:
                    processed_file.unlink()
                except:
                    pass
    
class WhisperTranscriber:
    def __init__(self, video_path, model_size="base", language="ja", segment_duration=60, 
                 cleanup=False, test_percentage=0, filter_transcription=True, preprocess_audio=True):
        """
        初始化转录器
        
        Args:
            video_path: 视频文件路径
            model_size: Whisper模型大小
            language: 音频语言代码 (默认: ja - 日语)
            segment_duration: 音频分段时长（秒），默认60秒
            cleanup: 是否在程序开始前清理临时文件，默认False
            test_percentage: 测试模式，仅转录前百分之N的音频
            filter_transcription: 是否启用智能过滤
            preprocess_audio: 是否预处理音频以增强轻声语音识别，默认启用
        """
        self.video_path = Path(video_path)
        self.model_size = model_size
        self.language = language
        self.segment_duration = segment_duration
        self.cleanup = cleanup
        self.test_percentage = test_percentage
        self.filter_transcription = filter_transcription
        self.preprocess_audio = preprocess_audio
        
        # 基础信息
        self.video_name = self.video_path.stem
        self.video_hash = hashlib.md5(str(self.video_path).encode()).hexdigest()[:8]
        self.temp_base = Path("temp") / f"{self.video_name}_{self.video_hash}_{self.model_size}"
        self.temp_base.mkdir(parents=True, exist_ok=True)
        
        # 状态文件路径
        self.state_file = self.temp_base / "transcription_state.json"
        self.audio_file = self.temp_base / "extracted_audio.wav"
        self.output_file = self.temp_base / "transcription.txt"
        self.progress_file = self.temp_base / "progress.txt"
        
        # 时间戳记录
        self.timestamps = {
            "start_time": None,
            "audio_extraction_time": None,
            "model_loading_time": None,
            "transcription_time": None,
            "total_time": None
        }
        
        # 加载状态
        self.state = self._load_state()
        self.segment_transcriber = None
        
        # whisper模块将在需要时延迟导入
        self.whisper_module = None
        
        # 模型将在需要时懒加载
        self.model = None
    
    def _cleanup_gpu_cache(self):
        """清理GPU缓存（如果有的话）"""
        if torch and hasattr(torch.cuda, 'empty_cache'):
            try:
                torch.cuda.empty_cache()
                print("GPU缓存已清理")
            except:
                pass
    
    def _load_state(self):
        """加载断点续传状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    
                    # 检查并修复状态数据
                    total_segments = state.get("total_segments", 0)
                    processed_segments = state.get("processed_segments", 0)
                    current_segment = state.get("current_segment", 0)
                    
                    # 确保索引在有效范围内
                    if processed_segments > total_segments:
                        print(f"修复状态数据: processed_segments({processed_segments}) > total_segments({total_segments})")
                        state["processed_segments"] = 0
                        
                    if current_segment > total_segments:
                        print(f"修复状态数据: current_segment({current_segment}) > total_segments({total_segments})")
                        state["current_segment"] = 0
                    
                    # 检查片段文件是否存在
                    segment_files = state.get("segment_files", [])
                    if segment_files and total_segments > 0:
                        existing_files = [f for f in segment_files if Path(f).exists()]
                        if len(existing_files) != len(segment_files):
                            print(f"警告: {len(segment_files) - len(existing_files)} 个片段文件丢失")
                            state["segment_files"] = existing_files
                            state["total_segments"] = len(existing_files)
                            # 重新调整索引
                            if state["processed_segments"] > len(existing_files):
                                state["processed_segments"] = 0
                            if state["current_segment"] > len(existing_files):
                                state["current_segment"] = 0
                    
                    print(f"发现之前的状态: {state.get('processed_segments', 0)}/{state.get('total_segments', 0)} 个片段")
                    return state
            except Exception as e:
                print(f"加载状态文件失败: {e}")
        
        # 初始状态
        return {
            "video_path": str(self.video_path),
            "model_size": self.model_size,
            "language": self.language,
            "audio_extracted": False,
            "audio_duration": 0,
            "segment_files": [],
            "processed_segments": 0,
            "total_segments": 0,
            "segments": [],
            "current_segment": 0
        }
    
    def _save_state(self):
        """保存当前状态"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存状态失败: {e}")
    
    def _save_progress(self, text):
        """保存当前进度到文件"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                f.write(text)
        except:
            pass

    def _extract_audio(self):
        """从视频中提取音频（优化断点续传）"""
        print(f"开始提取音频...")
        
        # 首先检查是否有有效的片段文件
        segments_dir = self.temp_base / "segments"
        
        # 1. 检查是否有片段文件（优先于音频文件检查）
        if segments_dir.exists():
            segment_files = sorted(list(segments_dir.glob("segment_*.wav")))
            if segment_files:
                # 应用测试模式限制
                if self.test_percentage > 0:
                    original_total = len(segment_files)
                    max_segments = max(1, min(original_total, int(original_total * self.test_percentage / 100)))
                    segment_files = segment_files[:max_segments]
                    print(f"测试模式: 仅处理前 {self.test_percentage}% 的音频 ({len(segment_files)}/{original_total} 个片段)")
                
                # 检查前几个片段文件
                valid_count = 0
                for i, seg_file in enumerate(segment_files[:5]):
                    if seg_file.exists() and seg_file.stat().st_size > 1000:
                        valid_count += 1
                
                if valid_count >= 3:  # 至少有3个有效片段文件
                    print(f"发现已存在的音频片段 (共 {len(segment_files)} 个片段)")
                    self.state["segment_files"] = [str(f) for f in segment_files]
                    self.state["total_segments"] = len(segment_files)
                    self.state["segment_duration"] = self.segment_duration
                    self.state["audio_extracted"] = True
                    
                    # 获取视频总时长
                    duration_cmd = [
                        'ffprobe', '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        str(self.video_path)
                    ]
                    result = subprocess.run(duration_cmd, capture_output=True, text=True, encoding='utf-8')
                    if result.returncode == 0:
                        video_duration = float(result.stdout.strip())
                        self.state["audio_duration"] = video_duration
                        print(f"视频总时长: {format_timedelta(video_duration)}")
                    
                    self._save_state()
                    return True
        
        # 2. 检查状态文件中的片段信息
        if self.state.get("segment_files") and self.state.get("audio_extracted"):
            # 应用测试模式限制
            segment_files = self.state["segment_files"]
            if self.test_percentage > 0:
                original_total = len(segment_files)
                max_segments = max(1, min(original_total, int(original_total * self.test_percentage / 100)))
                segment_files = segment_files[:max_segments]
                print(f"测试模式: 仅处理前 {self.test_percentage}% 的音频 ({len(segment_files)}/{original_total} 个片段)")
            
            if segment_files:
                first_segment = Path(segment_files[0])
                if first_segment.exists() and first_segment.stat().st_size > 1000:
                    print(f"使用已提取的音频片段 (共 {len(segment_files)} 个片段)")
                    self.state["segment_files"] = segment_files
                    self.state["total_segments"] = len(segment_files)
                    self._save_state()
                    return True
        
        # 3. 如果没有找到有效的片段文件，执行音频提取
        print("未找到有效的音频片段，开始提取音频...")
        start_time = time.time()
        
        try:
            # 方法1: 直接使用ffmpeg分割音频，避免生成巨大的中间文件
            print("使用内存优化方案提取音频...")
            
            # 先获取视频总时长
            duration_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(self.video_path)
            ]
            
            result = subprocess.run(duration_cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                video_duration = float(result.stdout.strip())
                print(f"视频总时长: {format_timedelta(video_duration)}")
            else:
                print("无法获取视频时长，使用默认值")
                video_duration = 3600  # 默认1小时
            
            # 创建分段器并直接分割视频音频
            self.segment_transcriber = SegmentTranscriber(
                self.temp_base, 
                None,  # 模型将在需要时懒加载
                self.language,
                self.model_size,
                self.preprocess_audio
            )
            
            # 计算测试模式需要的最大片段数
            max_segments = None
            if self.test_percentage > 0:
                # 根据视频时长和segment_duration计算总片段数
                total_segments_estimate = int(video_duration // self.segment_duration) + 1
                max_segments = max(1, min(total_segments_estimate, int(total_segments_estimate * self.test_percentage / 100)))
                print(f"测试模式: 仅处理前 {self.test_percentage}% 的音频 (大约 {max_segments} 个片段)")
            
            # 直接分割视频，避免生成extracted_audio.wav
            segment_files = self.segment_transcriber.split_audio_from_video(
                self.video_path, self.segment_duration, max_segments
            )
            
            if not segment_files:
                print("直接分割视频失败，回退到传统方法...")
                # 回退到传统方法
                return self._extract_audio_fallback()
            
            # 应用测试模式限制（如果split_audio_from_video没有应用）
            if self.test_percentage > 0 and max_segments and len(segment_files) > max_segments:
                segment_files = segment_files[:max_segments]
            
            # 保存片段信息
            self.state["segment_files"] = [str(f) for f in segment_files]
            self.state["total_segments"] = len(segment_files)
            self.state["segment_duration"] = self.segment_duration
            self.state["audio_duration"] = video_duration
            self.state["audio_extracted"] = True
            self.state["processed_segments"] = 0
            self.state["current_segment"] = 0
            self.state["segments"] = []
            
            self._save_state()
            
            audio_time = time.time() - start_time
            self.timestamps["audio_extraction_time"] = audio_time
            print(f"音频提取完成，共 {self.state['total_segments']} 个片段，时长: {format_timedelta(video_duration)}，耗时: {format_timedelta(audio_time)}")
            
            return True
            
        except Exception as e:
            print(f"内存优化音频提取出错: {e}")
            print("回退到传统方法...")
            return self._extract_audio_fallback()
    

    def _extract_audio_fallback(self):
        """传统音频提取方法（回退方案）"""
        print("使用传统音频提取方法...")
        start_time = time.time()
        
        try:
            # 使用ffmpeg提取音频（传统方法）
            cmd = [
                'ffmpeg', '-i', str(self.video_path),
                '-ac', '1', '-ar', '16000',  # 单声道，16kHz采样率
                '-acodec', 'pcm_s16le',      # PCM编码
                '-y',  # 覆盖已存在文件
                str(self.audio_file)
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            # 使用Popen进行流式处理，避免内存溢出
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=3600)  # 设置1小时超时
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                print(f"音频提取失败: {error_msg}")
                return False
            
            # 获取音频时长
            audio_duration = get_audio_duration(self.audio_file)
            self.state["audio_duration"] = audio_duration
            self.state["audio_extracted"] = True
            
            # 清除旧的片段文件（如果有）
            segments_dir = self.temp_base / "segments"
            if segments_dir.exists():
                for file in segments_dir.glob("*.wav"):
                    file.unlink()
            
            # 重置片段状态
            self.state["segment_files"] = []
            self.state["processed_segments"] = 0
            self.state["total_segments"] = 0
            self.state["current_segment"] = 0
            self.state["segments"] = []
            
            self._save_state()
            
            audio_time = time.time() - start_time
            self.timestamps["audio_extraction_time"] = audio_time
            print(f"音频提取完成，时长: {format_timedelta(audio_duration)}，耗时: {format_timedelta(audio_time)}")
            
            # 分割音频
            print("分割音频...")
            self.segment_transcriber = SegmentTranscriber(
                self.temp_base, 
                None,  # 模型将在需要时懒加载
                self.language,
                self.model_size,
                self.preprocess_audio
            )
            
            # 计算测试模式需要的最大片段数
            max_segments = None
            if self.test_percentage > 0:
                total_segments_estimate = int(audio_duration // self.segment_duration) + 1
                max_segments = max(1, min(total_segments_estimate, int(total_segments_estimate * self.test_percentage / 100)))
                print(f"测试模式: 仅处理前 {self.test_percentage}% 的音频 (大约 {max_segments} 个片段)")
            
            segment_files = self.segment_transcriber.split_audio(self.audio_file, self.segment_duration)
            
            # 应用测试模式限制
            if max_segments is not None and len(segment_files) > max_segments:
                segment_files = segment_files[:max_segments]
            
            if not segment_files:
                return False
            
            # 保存片段信息，包括当前的segment_duration
            self.state["segment_files"] = [str(f) for f in segment_files]
            self.state["total_segments"] = len(segment_files)
            self.state["segment_duration"] = self.segment_duration  # 保存当前参数值
            # 确保索引在有效范围内
            if self.state["current_segment"] >= len(segment_files):
                self.state["current_segment"] = 0
            if self.state["processed_segments"] >= len(segment_files):
                self.state["processed_segments"] = 0
            self._save_state()
            
            print(f"音频分割完成，共 {self.state['total_segments']} 个片段")
            return True
            
        except subprocess.TimeoutExpired:
            print("音频提取超时，可能需要更多时间处理长视频")
            return False
        except Exception as e:
            print(f"传统音频提取出错: {e}")
            return False

    def _prepare_segments(self):
        """准备音频片段（简化断点续传逻辑）"""
        # 检查是否已经有有效的分段器
        if self.segment_transcriber is not None:
            return True
        
        # 检查是否需要重新分割音频（如果segment_duration参数改变）
        if self.state.get("segment_duration") != self.segment_duration:
            print(f"检测到segment_duration参数改变，从{self.state.get('segment_duration', '未知')}秒改为{self.segment_duration}秒，重新分割音频...")
            return False  # 返回False让上层重新处理
        
        # 如果有片段信息，初始化分段器
        if (self.state["total_segments"] > 0 and 
            self.state["segment_files"] and
            self.state["processed_segments"] < self.state["total_segments"]):
            
            print(f"使用现有的 {self.state['total_segments']} 个音频片段")
            
            # 初始化分段器
            self.segment_transcriber = SegmentTranscriber(
                self.temp_base, 
                None,  # 模型将在需要时懒加载
                self.language,
                self.model_size,
                self.preprocess_audio
            )
            
            return True
        
        return False
    
    def _transcribe_segments(self):
        """转录所有音频片段（立即开始转录）"""
        print("开始转录音频片段...")
        transcribe_start = time.time()
        
        total_segments = self.state["total_segments"]
        start_from = self.state["processed_segments"]
        
        # 确保索引在有效范围内
        if start_from >= total_segments:
            print("所有片段已处理完成")
            return True
        
        # 如果start_from为0，说明是新的转录任务
        if start_from == 0:
            print(f"开始处理所有 {total_segments} 个片段")
        else:
            print(f"从第 {start_from + 1} 个片段开始，共 {total_segments} 个片段")
        
        failed_segments = []  # 记录失败的片段
        
        for i in range(start_from, total_segments):
            segment_file = Path(self.state["segment_files"][i])
            segment_index = i + 1
            
            if not segment_file.exists():
                print(f"片段文件不存在: {segment_file}")
                failed_segments.append(segment_index)
                continue
            
            # 检查文件大小
            file_size = segment_file.stat().st_size
            if file_size < 1000:  # 小于1KB的文件可能有问题
                print(f"警告: 片段文件过小 ({file_size} bytes)，可能为空或损坏")
                failed_segments.append(segment_index)
                continue
            
            print(f"\n处理片段 {segment_index}/{total_segments}...")
            segment_start = time.time()
            
            # 每处理3个片段清理一次GPU缓存（如果有的话）
            if i > 0 and i % 3 == 0:
                self._cleanup_gpu_cache()
            
            # 计算片段在原始音频中的起始时间
            segment_start_time = i * self.segment_duration
            
            segment_results = self.segment_transcriber.transcribe_segment(
                segment_file, 
                segment_index,
                segment_start_time
            )
            
            if segment_results:
                # 保存片段结果
                self.state["segments"].extend(segment_results)
                # 更新已处理片段数为下一个要处理的片段索引
                self.state["processed_segments"] = segment_index
                self.state["current_segment"] = i
                self._save_state()
                
                # 更新进度文件
                if segment_results:
                    segment_text = "\n".join([f"[{format_timedelta(s['start'])}] {s['text'].strip()}" 
                                            for s in segment_results])
                    self._save_progress(segment_text)
                
                segment_time = time.time() - segment_start
                cumulative_time = time.time() - transcribe_start
                print(f"片段 {segment_index} 完成，耗时: {format_timedelta(segment_time)}，累计耗时: {format_timedelta(cumulative_time)}")
                print(f"  转录到 {len(segment_results)} 个有效片段")
            else:
                print(f"片段 {segment_index} 转录失败：空结果")
                failed_segments.append(segment_index)
            
            # 更新总转录时间
            self.timestamps["transcription_time"] = time.time() - transcribe_start
        
        # 检查是否所有片段都失败了
        if len(failed_segments) == total_segments - start_from:
            print("所有片段转录失败，转录过程终止")
            return False
        
        # 如果有失败的片段，但至少成功了一些，则继续
        if failed_segments:
            print(f"警告: 有 {len(failed_segments)} 个片段转录失败: {failed_segments}")
            print(f"成功转录了 {len(self.state['segments'])}/{total_segments} 个片段")
        
        return True
    
    def _save_final_transcription(self):
        """保存最终的转录结果（带过滤功能）"""
        # 按时间戳排序所有片段
        all_segments = sorted(self.state["segments"], key=lambda x: x['start'])
        
        # 如果启用了过滤，使用过滤器
        if self.filter_transcription:
            print("\n启用智能过滤功能...")
            
            # 创建过滤器（针对日语成人视频优化）
            filter_processor = TranscriptionFilter(
                max_syllable_repeat=4,  # 日语中允许更多重复（如"あああ"）
                min_speech_rate=0.3,    # 降低最低语速阈值，接受更慢的语音
                max_speech_rate=20.0,   # 提高最高语速阈值
                min_duration=0.3,       # 降低最小片段时长，接受更短语音
                max_duration=30.0,
                confidence_threshold=-1.0,  # 降低置信度阈值，接受更多可能语音
                filter_patterns=[
                    r"ご視聴ありがとうございました",
                    r"お疲れ様でした",
                    r"終わり",
                    r"^[あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんぁぃぅぇぉゃゅょっ\-ー、。，．,.\s]+$"
                ]
            )
            
            # 过滤不合理片段
            filtered_segments = filter_processor.filter_segments(all_segments)
        else:
            filtered_segments = all_segments
        
        # 去重逻辑：合并相邻的重复字幕
        deduplicated_segments = []
        last_text = ""
        last_segment = None
        
        for segment in filtered_segments:
            current_text = segment['text'].strip()
            
            # 跳过空文本
            if not current_text:
                continue
                
            # 如果当前文本与上一个相同，且时间间隔小于5秒，合并时间范围
            if (current_text == last_text and last_segment and 
                segment['start'] - last_segment['end'] < 5.0):
                # 扩展上一个片段的结束时间
                last_segment['end'] = segment['end']
            else:
                # 添加新片段
                deduplicated_segments.append(segment)
                last_text = current_text
                last_segment = segment
        
        # 统计信息
        original_count = len(all_segments)
        filtered_count = len(filtered_segments)
        deduplicated_count = len(deduplicated_segments)
        
        print(f"\n统计信息:")
        print(f"  原始片段数: {original_count}")
        if self.filter_transcription:
            print(f"  过滤后片段数: {filtered_count}")
        print(f"  去重后片段数: {deduplicated_count}")
        if self.filter_transcription:
            print(f"  总共移除了 {original_count - deduplicated_count} 个不合理或重复片段")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"视频: {self.video_path.name}\n")
            f.write(f"模型: {self.model_size}\n")
            f.write(f"语言: {self.language}\n")
            f.write(f"音频预处理: {'启用' if self.preprocess_audio else '禁用'}\n")
            f.write(f"转录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"音频时长: {format_timedelta(self.state['audio_duration'])}\n")
            f.write(f"原始片段数: {original_count}\n")
            if self.filter_transcription:
                f.write(f"过滤后片段数: {filtered_count}\n")
            f.write(f"去重后片段数: {deduplicated_count}\n")
            f.write("=" * 60 + "\n\n")
            
            for segment in deduplicated_segments:
                start = format_timedelta(segment['start'])
                end = format_timedelta(segment['end'])
                text = segment['text'].strip()
                f.write(f"[{start} - {end}] {text}\n")
    
    def _cleanup(self):
        """清理临时文件"""
        try:
            # 删除状态文件
            if self.state_file.exists():
                self.state_file.unlink()
            
            # 删除进度文件
            if self.progress_file.exists():
                self.progress_file.unlink()
            
            # 删除音频文件
            if self.audio_file.exists():
                self.audio_file.unlink()
            
            # 删除片段目录
            segments_dir = self.temp_base / "segments"
            if segments_dir.exists():
                for file in segments_dir.glob("*.wav"):
                    file.unlink()
                segments_dir.rmdir()
            
            # 如果临时基础目录为空，也删除它
            if self.temp_base.exists() and not any(self.temp_base.iterdir()):
                self.temp_base.rmdir()
            
            print("临时文件已清理")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")
    
    def _lazy_load_whisper_model(self):
        """懒加载Whisper模型"""
        if self.model is not None:
            return True
        
        print("懒加载Whisper模型...")
        model_start = time.time()
        
        try:
            # 延迟导入whisper模块
            if self.whisper_module is None:
                self.whisper_module = import_whisper()
                if self.whisper_module is None:
                    print("错误: 无法导入whisper模块")
                    return False
            
            self.model = self.whisper_module.load_model(self.model_size)
            
            # 记录内存使用情况
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                print(f"GPU内存使用: 已分配 {memory_allocated:.2f}GB, 已保留 {memory_reserved:.2f}GB")
            
            model_time = time.time() - model_start
            print(f"模型加载完成，耗时: {format_timedelta(model_time)}")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None
            return False

    def transcribe(self):
        """执行转录过程"""
        print(f"开始转录视频: {self.video_path.name}")
        print(f"使用模型: {self.model_size}, 语言: {self.language}")
        print(f"智能过滤: {'启用' if self.filter_transcription else '禁用'}")
        print(f"音频预处理: {'启用' if self.preprocess_audio else '禁用'}")
        
        # 显示使用的库信息
        if FASTER_WHISPER_AVAILABLE:
            print("使用库: faster-whisper (高性能版本)")
        else:
            print("使用库: 标准whisper")
            print("提示: 安装faster-whisper可显著提高转录速度")
            print("      pip install faster-whisper")
        
        print(f"临时文件目录: {self.temp_base}")
        
        # 如果指定了cleanup，先清理临时文件
        if self.cleanup:
            print("清理临时文件...")
            self._cleanup()
            # 重新创建临时目录
            self.temp_base.mkdir(parents=True, exist_ok=True)
            print("临时文件已清理，开始转录")
        
        print("-" * 50)
        
        self.timestamps["start_time"] = time.time()
        
        try:
            # 阶段1: 提取音频
            print("阶段1: 检查音频文件...")
            if not self._extract_audio():
                print("音频提取失败")
                return False
            
            # 阶段2: 准备音频片段
            print("阶段2: 准备音频片段...")
            if not self._prepare_segments():
                # 如果准备失败，尝试重新提取
                print("重新准备音频片段...")
                # 清除状态，重新开始
                self.state["processed_segments"] = 0
                self.state["current_segment"] = 0
                self.state["segments"] = []
                self._save_state()
                
                if not self._extract_audio():
                    return False
                if not self._prepare_segments():
                    return False
            
            # 阶段3: 转录所有片段
            print("阶段3: 开始转录音频片段...")
            transcription_success = self._transcribe_segments()
            
            # 阶段4: 保存结果
            print("阶段4: 保存转录结果...")
            if self.state["segments"]:
                self._save_final_transcription()
                if transcription_success:
                    print(f"\n转录完成！结果保存在: {self.output_file}")
                else:
                    print(f"\n转录部分完成！已转录 {len(self.state['segments'])}/{self.state['total_segments']} 个片段")
                    print(f"部分转录结果已保存: {self.output_file}")
            else:
                # 没有任何片段转录成功
                print("\n转录失败！没有成功转录任何片段")
            
            # 即使部分失败，只要成功转录了内容就返回True
            if self.state["segments"]:
                transcription_success = True
            
            # 阶段5: 计算总时间
            self.timestamps["total_time"] = time.time() - self.timestamps["start_time"]
            
            # 打印时间统计
            self._print_timestamps()
            
            print(f"\n转录完成！结果保存在: {self.output_file}")
            print(f"临时文件保留在: {self.temp_base}")
            return True
            
        except KeyboardInterrupt:
            print("\n转录被用户中断")
            self._save_state()
            
            # 不保存部分转录结果
            if self.state["segments"]:
                print(f"已转录 {len(self.state['segments'])}/{self.state['total_segments']} 个片段，但未生成最终转录文件")
            
            return False
        except Exception as e:
            print(f"转录过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_timestamps(self):
        """打印各阶段时间戳"""
        print("\n" + "=" * 50)
        print("时间统计:")
        print("-" * 50)
        
        if self.timestamps["audio_extraction_time"]:
            print(f"音频提取: {format_timedelta(self.timestamps['audio_extraction_time'])}")
        
        if self.timestamps["model_loading_time"]:
            print(f"模型加载: {format_timedelta(self.timestamps['model_loading_time'])}")
        
        if self.timestamps["transcription_time"]:
            print(f"语音转录: {format_timedelta(self.timestamps['transcription_time'])}")
        
        if self.timestamps["total_time"]:
            print(f"总耗时: {format_timedelta(self.timestamps['total_time'])}")
        
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Whisper语音转录程序（针对日语成人视频优化）")
    parser.add_argument("video_path", help="视频文件路径")
    parser.add_argument("--model", "-m", default="base", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo"],
                       help="Whisper模型大小 (默认: base)")
    parser.add_argument("--language", "-l", default="ja", 
                       help="音频语言代码 (默认: ja - 日语)")
    parser.add_argument("--segment-duration", "-s", type=int, default=180,
                       help="音频分段时长（秒），默认180秒（3分钟）")
    parser.add_argument("--venv-path", help="虚拟环境路径（可选，默认自动检测）")
    parser.add_argument("--cleanup", action="store_true",
                       help="在程序开始前清理临时文件（默认不清理）")
    parser.add_argument("--test", type=int, default=0,
                       help="测试模式，仅转录前百分之N的音频 (默认: 0 - 转录全部音频)")
    parser.add_argument("--no-filter", action="store_true",
                       help="禁用智能过滤，保留所有字幕（默认启用过滤）")
    parser.add_argument("--no-preprocess-audio", action="store_true",
                       help="禁用音频预处理（默认启用）")
    parser.add_argument("--min-confidence", type=float, default=-1.0,
                       help="最低置信度阈值（默认: -1.0，针对日语成人视频优化）")
    
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    if not Path(args.video_path).exists():
        print(f"错误: 视频文件不存在: {args.video_path}")
        sys.exit(1)
    
    # 使用虚拟环境管理器
    venv_manager = VirtualEnvironmentManager(args.venv_path)
    
    with venv_manager:
        # 创建转录器并执行
        transcriber = WhisperTranscriber(
            video_path=args.video_path,
            model_size=args.model,
            language=args.language,
            segment_duration=args.segment_duration,
            cleanup=args.cleanup,
            test_percentage=args.test,
            filter_transcription=not args.no_filter,
            preprocess_audio=not args.no_preprocess_audio  # 默认启用，除非使用--no-preprocess-audio
        )
        
        success = transcriber.transcribe()
        
        if not success:
            # 检查是否有部分转录结果
            if transcriber.state["segments"]:
                print("转录过程部分完成，已保存可用内容")
                sys.exit(0)  # 返回0表示部分成功
            else:
                print("转录过程出现错误或中断")
                sys.exit(1)

if __name__ == "__main__":
    main()
