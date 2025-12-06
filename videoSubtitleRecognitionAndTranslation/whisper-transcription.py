import argparse
import os
import sys
import json
import time
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import wave
import contextlib
from dotenv import load_dotenv
import send2trash  # 新增导入，用于将文件移动到回收站
load_dotenv()

# 虚拟环境管理
class VirtualEnvironmentManager:
    """管理虚拟环境的自动激活和退出"""
    
    def __init__(self, venv_path=None):
        # 使用绝对路径，默认在当前脚本目录下
        if venv_path is None:
            script_dir = Path(__file__).parent
            self.venv_path = script_dir / "whisperx_env"
        else:
            self.venv_path = Path(venv_path).resolve()
        
        self.original_path = os.environ.get('PATH', '')
        self.original_pythonpath = os.environ.get('PYTHONPATH', '')
        self.is_activated = False
    
    def activate(self):
        """激活虚拟环境"""
        if not self.venv_path.exists():
            print(f"错误: 虚拟环境目录不存在: {self.venv_path}")
            return False
        
        # 获取虚拟环境的Python路径和Scripts路径
        python_exe = self.venv_path / "Scripts" / "python.exe"
        scripts_path = self.venv_path / "Scripts"
        
        if not python_exe.exists():
            print(f"错误: 虚拟环境Python可执行文件不存在: {python_exe}")
            return False
        
        # 设置环境变量
        os.environ['PATH'] = str(scripts_path) + os.pathsep + self.original_path
        os.environ['VIRTUAL_ENV'] = str(self.venv_path)
        
        # 更新Python路径
        sys.executable = str(python_exe)
        sys.prefix = str(self.venv_path)
        
        self.is_activated = True
        print(f"✓ 已激活虚拟环境: {self.venv_path}")
        return True
    
    def deactivate(self):
        """退出虚拟环境"""
        if self.is_activated:
            # 恢复原始环境变量
            os.environ['PATH'] = self.original_path
            if 'VIRTUAL_ENV' in os.environ:
                del os.environ['VIRTUAL_ENV']
            
            # 恢复Python路径（需要重新启动程序才能完全恢复）
            print("✓ 已退出虚拟环境")
            self.is_activated = False
    
    def __enter__(self):
        """上下文管理器入口"""
        self.activate()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.deactivate()

# 注意：移除了原 'whisper' 导入，改为导入 'whisperx'
try:
    import whisperx
    import torch
except ImportError:
    print("检测到缺少WhisperX依赖，尝试自动激活虚拟环境...")
    
    # 创建虚拟环境管理器
    venv_manager = VirtualEnvironmentManager("whisperx_env")
    
    if venv_manager.activate():
        # 重新尝试导入
        try:
            import whisperx
            import torch
            print("✓ 成功导入WhisperX和PyTorch")
        except ImportError as e:
            print(f"错误: 即使在虚拟环境中也无法导入依赖库: {e}")
            print("请确保虚拟环境已正确配置，或手动安装依赖:")
            print("pip install \"torch<2.6\" \"torchaudio<2.6\" \"pyannote.audio<3.0\" whisperx")
            sys.exit(1)
    else:
        print("错误: 无法激活虚拟环境，请手动安装依赖:")
        print("pip install \"torch<2.6\" \"torchaudio<2.6\" \"pyannote.audio<3.0\" whisperx")
        sys.exit(1)

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
    """处理音频分段的转录 (WhisperX版本)"""
    
    def __init__(self, temp_base, model, align_model, align_metadata, language, device, compute_type, batch_size, vad_model=None):
        self.temp_base = temp_base
        self.model = model
        self.align_model = align_model
        self.align_metadata = align_metadata
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.vad_model = vad_model 
        self.segments_dir = temp_base / "segments"
        self.segments_dir.mkdir(exist_ok=True)
    
    def split_audio(self, audio_file, segment_duration=600):
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
        return segment_files
    
    def _split_sentences(self, text, start_time, end_time):
        """将长文本按句子边界分割成多个片段"""
        # 日语句子分隔符（增加更多分隔符）
        sentence_endings = ['。', '！', '？', '!', '?', '…', '…', '、', '，', ',', ';', '；', '：', ':', '\n', '\r\n']
        
        sentences = []
        current_sentence = ""
        current_start = start_time
        
        # 按字符遍历文本
        for i, char in enumerate(text):
            current_sentence += char
            
            # 检查是否到达句子结束（降低阈值到30个字符）
            if char in sentence_endings or i == len(text) - 1 or len(current_sentence) >= 30:
                if current_sentence.strip():
                    # 计算当前句子的时间戳（按比例分配）
                    sentence_duration = (end_time - start_time) * (len(current_sentence) / len(text))
                    sentence_end = current_start + sentence_duration
                    
                    sentences.append({
                        'text': current_sentence.strip(),
                        'start': current_start,
                        'end': sentence_end
                    })
                    
                    # 更新下一个句子的开始时间
                    current_start = sentence_end
                    current_sentence = ""
        
        # 如果没有找到句子边界，按更短的长度分割（降低到20个字符）
        if not sentences and len(text) > 20:
            max_length = 20  # 每段最多20个字符（降低阈值）
            total_duration = end_time - start_time
            segment_count = (len(text) + max_length - 1) // max_length
            
            for i in range(segment_count):
                seg_start = i * max_length
                seg_end = min((i + 1) * max_length, len(text))
                seg_text = text[seg_start:seg_end].strip()
                
                if seg_text:
                    seg_duration = total_duration * (len(seg_text) / len(text))
                    seg_start_time = start_time + (total_duration * seg_start / len(text))
                    seg_end_time = seg_start_time + seg_duration
                    
                    sentences.append({
                        'text': seg_text,
                        'start': seg_start_time,
                        'end': seg_end_time
                    })
        
        return sentences if sentences else [{'text': text, 'start': start_time, 'end': end_time}]

    def transcribe_segment(self, segment_file, segment_index, start_time=0):
        """使用WhisperX转录单个音频片段 (集成VAD版本)"""
        print(f"转录片段 {segment_index}: {segment_file.name}")
        
        try:
            # 1. 加载音频
            audio = whisperx.load_audio(str(segment_file))
            
            # 2. 执行VAD检测（如果VAD模型可用）
            vad_segments = None
            if self.vad_model is not None:
                try:
                    # pyannote.audio的Pipeline需要文件路径而不是音频数据
                    vad_result = self.vad_model(str(segment_file))
                    
                    # 调试：打印VAD结果的完整结构
                    print(f"   VAD结果类型: {type(vad_result)}")
                    print(f"   VAD结果内容: {vad_result}")
                    
                    # 提取VAD段
                    if hasattr(vad_result, 'get_timeline'):
                        # pyannote.core.Annotation对象
                        vad_segments = list(vad_result.get_timeline())
                    elif isinstance(vad_result, dict) and 'segments' in vad_result:
                        # 字典格式，包含segments字段
                        vad_segments = vad_result['segments']
                    elif isinstance(vad_result, (list, tuple)):
                        # 直接是列表/元组格式
                        vad_segments = vad_result
                    else:
                        # 未知格式，直接使用
                        vad_segments = vad_result
                    
                    print(f"   VAD检测到 {len(vad_segments)} 个语音段")
                    # 调试：打印VAD段的结构
                    if len(vad_segments) > 0:
                        print(f"   第一个VAD段类型: {type(vad_segments[0])}, 值: {vad_segments[0]}")
                except Exception as e:
                    print(f"   VAD检测失败: {e}，将继续不使用VAD")
                    vad_segments = None
            
            # 3. 使用WhisperX转录，使用支持的参数
            result = self.model.transcribe(
                audio,
                language=self.language,
                task="transcribe",
                batch_size=self.batch_size
            )
            
            # 4. 【可选但推荐】如果VAD检测成功，过滤掉非语音段的转录结果
            #    这能有效减少"幻听"，是你需要的核心功能
            if vad_segments is not None and len(vad_segments) > 0:
                filtered_segments = []
                original_segment_count = len(result['segments'])
                
                # 调试：打印VAD段的结构
                print(f"   VAD段类型: {type(vad_segments)}")
                if len(vad_segments) > 0:
                    print(f"   第一个VAD段类型: {type(vad_segments[0])}, 值: {vad_segments[0]}")
                
                for whisper_segment in result['segments']:
                    seg_start = whisper_segment['start']
                    seg_end = whisper_segment['end']
                    
                    # 检查VAD段的结构并正确访问属性
                    is_speech = False
                    for vad_segment in vad_segments:
                        # 调试：打印VAD段的结构
                        if hasattr(vad_segment, 'start') and hasattr(vad_segment, 'end'):
                            # 标准pyannote.audio Segment对象
                            vad_start = vad_segment.start
                            vad_end = vad_segment.end
                        elif isinstance(vad_segment, dict) and 'start' in vad_segment and 'end' in vad_segment:
                            # 字典格式的VAD段
                            vad_start = vad_segment['start']
                            vad_end = vad_segment['end']
                        elif isinstance(vad_segment, (list, tuple)) and len(vad_segment) >= 2:
                            # 列表/元组格式的VAD段
                            vad_start = vad_segment[0]
                            vad_end = vad_segment[1]
                        else:
                            # 未知格式，跳过
                            continue
                            
                        # 检查时间重叠，使用更宽松的重叠条件
                        if (vad_start <= seg_start <= vad_end or 
                            vad_start <= seg_end <= vad_end or
                            seg_start <= vad_start <= seg_end or
                            (vad_start <= seg_start and seg_end <= vad_end) or
                            (seg_start <= vad_start and vad_end <= seg_end)):
                            is_speech = True
                            break
                    
                    if is_speech:
                        filtered_segments.append(whisper_segment)
                
                # 用过滤后的片段替换原结果
                result['segments'] = filtered_segments
                print(f"   经过VAD过滤，保留 {len(filtered_segments)} 个语音段（原 {original_segment_count} 个）")
            
            # 5. 进行语音对齐（原有逻辑保持不变）
            if self.align_model is not None:
                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False
                )
            
            # 6. 调整时间戳（原有逻辑保持不变）
            for segment in result['segments']:
                segment['start'] += start_time
                segment['end'] += start_time
            
            # 7. 句子级别分割 - 降低分割阈值，使句子划分更灵敏
            final_segments = []
            for segment in result['segments']:
                text = segment['text'].strip()
                if len(text) > 20:  # 降低阈值到20个字符，使句子划分更灵敏
                    sentences = self._split_sentences(text, segment['start'], segment['end'])
                    final_segments.extend(sentences)
                else:
                    final_segments.append(segment)
            
            print(f"   句子分割: 从 {len(result['segments'])} 个段分割为 {len(final_segments)} 个句子")
            
            return final_segments
            
        except Exception as e:
            print(f"片段 {segment_index} 转录失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
class WhisperXTranscriber:
    def __init__(self, video_path, model_size="base", language="ja", segment_duration=180,
                 batch_size=4, compute_type="int8", device="cpu", cleanup=False):
        """初始化转录器"""
        self.video_path = Path(video_path)
        self.model_size = model_size
        self.language = language
        self.segment_duration = segment_duration
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.device = device
        self.cleanup = cleanup  # 是否清理临时文件
        
        # 生成临时文件目录（基于视频文件哈希）
        video_hash = hashlib.md5(str(self.video_path).encode()).hexdigest()[:8]
        self.temp_base = Path(f"temp_{video_hash}")
        self.temp_base.mkdir(exist_ok=True)
        
        # 临时文件路径
        self.audio_file = self.temp_base / "audio.wav"
        self.state_file = self.temp_base / "transcription_state.json"
        self.progress_file = self.temp_base / "progress.txt"
        
        # 输出文件
        self.output_file = self.video_path.with_suffix(".txt")
        
        # 状态和计时器
        self.state = {
            "segments": [],
            "total_segments": 0,
            "audio_duration": 0,
            "completed_segments": set()
        }
        self.timestamps = {
            "start_time": 0,
            "audio_extraction_time": 0,
            "model_loading_time": 0,
            "transcription_time": 0,
            "total_time": 0
        }
        
        # WhisperX模型
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.vad_model = None
        
        # 加载状态
        self.state = self._load_state()
        self.segment_transcriber = None
    
    def _load_state(self):
        """加载断点续传状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
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
        """从视频中提取音频（与原代码一致）"""
        print(f"开始提取音频...")
        
        # 如果音频已提取且文件存在，直接使用
        if self.state["audio_extracted"] and self.audio_file.exists():
            audio_duration = get_audio_duration(self.audio_file)
            if audio_duration > 0:
                print(f"使用已提取的音频 (时长: {format_timedelta(audio_duration)})")
                self.state["audio_duration"] = audio_duration
                return True
            else:
                print("音频文件损坏，重新提取...")
        
        start_time = time.time()
        
        try:
            # 使用ffmpeg提取音频
            cmd = [
                'ffmpeg', '-i', str(self.video_path),
                '-ac', '1', '-ar', '16000',  # 单声道，16kHz采样率
                '-acodec', 'pcm_s16le',      # PCM编码
                '-y',  # 覆盖已存在文件
                str(self.audio_file)
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                print(f"音频提取失败: {result.stderr}")
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
            
            return True
            
        except Exception as e:
            print(f"音频提取出错: {e}")
            return False
    
    def _load_models(self):
        """加载 WhisperX 模型和对齐模型"""
        print("加载 WhisperX 模型...")
        model_start = time.time()
        
        try:
            # 首先尝试正常加载
            self.model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                language=self.language
            )
            
            # 加载对齐模型（用于获得精确的词级时间戳）
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=self.language,
                device=self.device,
                model_name="jonatasgrosman/wav2vec2-large-xlsr-53-japanese"  # 日语对齐模型
            )
            
            self.timestamps["model_loading_time"] = time.time() - model_start
            print(f"模型加载完成，耗时: {format_timedelta(self.timestamps['model_loading_time'])}")
            
            print("加载 VAD (语音活动检测) 模型...")
            try:
                # 使用安全加载方式加载VAD模型
                import torch
                import omegaconf
                
                # 允许omegaconf.listconfig.ListConfig类型
                torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
                
                # 使用安全上下文加载VAD模型，调整VAD参数使其更灵敏
                with torch.serialization.safe_globals([omegaconf.listconfig.ListConfig]):
                    self.vad_model = whisperx.vad.load_vad_model(
                        device=self.device,
                        vad_onset=0.3,   # 降低语音开始阈值，使其更灵敏 (0.3)
                        vad_offset=0.3   # 降低语音结束阈值，使其更灵敏 (0.3)
                    )
                print("VAD模型加载成功。")
            except Exception as e:
                print(f"警告：VAD模型加载失败，将继续不使用VAD。错误: {e}")
                self.vad_model = None
            
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            
            # 如果是权重加载错误，使用安全加载方式
            if "weights_only" in str(e) or "UnpicklingError" in str(e) or "omegaconf.listconfig.ListConfig" in str(e):
                print("检测到PyTorch权重加载问题，尝试使用安全加载方式...")
                return self._load_models_safe()
            
            import traceback
            traceback.print_exc()
            return False

    def _load_models_safe(self):
        """安全加载方式：解决PyTorch weights_only问题"""
        try:
            import torch
            import omegaconf
            
            print("使用安全加载方式加载WhisperX模型...")
            
            # 允许omegaconf.listconfig.ListConfig类型
            torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
            
            # 使用安全上下文加载模型
            with torch.serialization.safe_globals([omegaconf.listconfig.ListConfig]):
                # 加载主转录模型
                self.model = whisperx.load_model(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    language=self.language
                )
                
                # 加载对齐模型
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=self.language,
                    device=self.device,
                    model_name="jonatasgrosman/wav2vec2-large-xlsr-53-japanese"  # 日语对齐模型
                )
            
            print("加载 VAD (语音活动检测) 模型...")
            try:
                # 尝试不同的VAD模型加载方式
                # 方式1：检查WhisperX是否内置VAD功能
                if hasattr(whisperx, 'load_vad_model'):
                    # 使用安全上下文加载VAD模型
                    with torch.serialization.safe_globals([omegaconf.listconfig.ListConfig]):
                        self.vad_model = whisperx.load_vad_model(
                            device=self.device,
                            vad_onset=0.5,
                            vad_offset=0.5
                        )
                else:
                    # 方式2：尝试使用pyannote.audio的VAD功能
                    try:
                        from pyannote.audio import Pipeline
                        self.vad_model = Pipeline.from_pretrained(
                            "pyannote/voice-activity-detection",
                            use_auth_token=None
                        ).to(self.device)
                    except ImportError:
                        print("警告：无法加载VAD模型，pyannote.audio不可用")
                        self.vad_model = None
                
                if self.vad_model is not None:
                    print("VAD模型加载成功。")
                else:
                    print("警告：VAD模型加载失败，将继续不使用VAD")
            except Exception as e:
                print(f"警告：VAD模型加载失败，将继续不使用VAD。错误: {e}")
                self.vad_model = None

            print("WhisperX模型安全加载成功")
            
            return True
            
        except Exception as e:
            print(f"安全加载方式失败: {e}")
            
            # 如果安全加载也失败，尝试使用torch.load的weights_only=False
            print("尝试使用weights_only=False加载...")
            return self._load_models_unsafe()
    
    def _load_models_unsafe(self):
        """不安全加载方式：使用weights_only=False"""
        try:
            import torch
            
            print("使用weights_only=False加载WhisperX模型...")
            
            # 临时修改torch.load的默认行为
            original_load = torch.load
            
            def custom_load(f, map_location=None, **kwargs):
                # 强制设置weights_only=False
                kwargs['weights_only'] = False
                return original_load(f, map_location=map_location, **kwargs)
            
            # 临时替换torch.load函数
            torch.load = custom_load
            
            try:
                # 加载主转录模型
                self.model = whisperx.load_model(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    language=self.language
                )
                
                # 加载对齐模型
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=self.language,
                    device=self.device,
                    model_name="jonatasgrosman/wav2vec2-large-xlsr-53-japanese"  # 日语对齐模型
                )
                
                print("加载 VAD (语音活动检测) 模型...")
                try:
                    # 尝试不同的VAD模型加载方式
                    # 方式1：检查WhisperX是否内置VAD功能
                    if hasattr(whisperx, 'load_vad_model'):
                        # 在不安全加载方式中，VAD模型也会自动使用weights_only=False
                        self.vad_model = whisperx.load_vad_model(
                            device=self.device,
                            vad_onset=0.5,
                            vad_offset=0.5
                        )
                    else:
                        # 方式2：尝试使用pyannote.audio的VAD功能
                        try:
                            from pyannote.audio import Pipeline
                            self.vad_model = Pipeline.from_pretrained(
                                "pyannote/voice-activity-detection",
                                use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")  # 从环境变量获取
                            )
                            if self.vad_model is not None:
                                # 将设备字符串转换为torch.device对象
                                device = torch.device(self.device)
                                self.vad_model = self.vad_model.to(device)
                        except ImportError:
                            print("警告：无法加载VAD模型，pyannote.audio不可用")
                            self.vad_model = None
                        except Exception as e:
                            print(f"警告：pyannote VAD模型加载失败: {e}")
                            print("请确保已访问以下页面接受使用条款:")
                            print("1. https://huggingface.co/pyannote/voice-activity-detection")
                            print("2. https://huggingface.co/pyannote/segmentation")
                            self.vad_model = None
                    
                    if self.vad_model is not None:
                        print("VAD模型加载成功。")
                    else:
                        print("警告：VAD模型加载失败，将继续不使用VAD")
                except Exception as e:
                    print(f"警告：VAD模型加载失败，将继续不使用VAD。错误: {e}")
                    self.vad_model = None
                
                print("WhisperX模型加载成功（使用weights_only=False）")
                return True
                
            finally:
                # 恢复原始torch.load函数
                torch.load = original_load
                
        except Exception as e:
            print(f"不安全加载方式也失败: {e}")
            print("所有WhisperX加载方式都失败了，请检查PyTorch和WhisperX版本兼容性")
            return False
    
    def _prepare_segments(self):
        """准备音频片段（与原代码逻辑一致，但使用新的SegmentTranscriber）"""
        # 检查是否需要重新分割音频
        if self.state.get("segment_duration") != self.segment_duration:
            print(f"检测到segment_duration参数改变，从{self.state.get('segment_duration', '未知')}秒改为{self.segment_duration}秒，重新分割音频...")
            self.state["segment_files"] = []
            self.state["total_segments"] = 0
            self.state["processed_segments"] = 0
            self.state["current_segment"] = 0
            self.state["segments"] = []
        
        # 如果已经有片段信息且文件都存在，直接使用
        if (self.state["total_segments"] > 0 and 
            self.state["segment_files"] and
            all(Path(f).exists() for f in self.state["segment_files"])):
            
            print(f"使用现有的 {self.state['total_segments']} 个音频片段")
            self.segment_transcriber = SegmentTranscriber(
                self.temp_base, 
                self.model,
                self.align_model,
                self.align_metadata,
                self.language,
                self.device,
                self.compute_type,
                self.batch_size,
                self.vad_model
            )
            return True
        
        # 创建新的分段器并分割音频
        self.segment_transcriber = SegmentTranscriber(
            self.temp_base, 
            self.model,
            self.align_model,
            self.align_metadata,
            self.language,
            self.device,
            self.compute_type,
            self.batch_size,
            self.vad_model
        )
        
        segment_files = self.segment_transcriber.split_audio(self.audio_file, self.segment_duration)
        
        if not segment_files:
            return False
        
        # 保存片段信息
        self.state["segment_files"] = [str(f) for f in segment_files]
        self.state["total_segments"] = len(segment_files)
        self.state["segment_duration"] = self.segment_duration
        self.state["current_segment"] = self.state["processed_segments"]
        self._save_state()
        
        print(f"音频分割完成，共 {self.state['total_segments']} 个片段")
        return True
    
    def _transcribe_segments(self):
        """转录所有音频片段（断点续传逻辑不变）"""
        print("开始转录音频片段...")
        transcribe_start = time.time()
        
        total_segments = self.state["total_segments"]
        start_from = self.state["processed_segments"]
        
        if start_from >= total_segments:
            print("所有片段已处理完成")
            return True
        
        print(f"从第 {start_from + 1} 个片段开始，共 {total_segments} 个片段")
        
        # 计算每个片段的时长（用于时间戳调整）
        segment_duration = self.state["audio_duration"] / total_segments
        
        for i in range(start_from, total_segments):
            segment_file = Path(self.state["segment_files"][i])
            segment_index = i + 1
            
            if not segment_file.exists():
                print(f"片段文件不存在: {segment_file}")
                continue
            
            # 计算片段在原始音频中的起始时间
            segment_start_time = i * segment_duration
            
            # 转录当前片段
            print(f"\n处理片段 {segment_index}/{total_segments}...")
            segment_start = time.time()
            
            segment_results = self.segment_transcriber.transcribe_segment(
                segment_file, 
                segment_index,
                segment_start_time
            )
            
            if segment_results is not None:
                # 保存片段结果
                self.state["segments"].extend(segment_results)
                self.state["processed_segments"] = segment_index
                self.state["current_segment"] = i
                self._save_state()
                
                # 更新进度文件
                segment_text = "\n".join([f"[{format_timedelta(s['start'])}] {s['text'].strip()}" 
                                         for s in segment_results])
                self._save_progress(segment_text)
                
                segment_time = time.time() - segment_start
                cumulative_time = time.time() - self.timestamps["start_time"]
                if len(segment_results) > 0:
                    print(f"片段 {segment_index} 完成，耗时: {format_timedelta(segment_time)}，累计耗时: {format_timedelta(cumulative_time)}")
                else:
                    print(f"片段 {segment_index} 无语音内容，跳过，累计耗时: {format_timedelta(cumulative_time)}")
            else:
                print(f"片段 {segment_index} 转录失败")
                return False
            
            # 更新总转录时间
            self.timestamps["transcription_time"] = time.time() - transcribe_start
        
        return True
    
    def _save_final_transcription(self):
        """保存最终的转录结果（格式与原代码一致）"""
        # 按时间戳排序所有片段
        all_segments = sorted(self.state["segments"], key=lambda x: x['start'])
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"视频: {self.video_path.name}\n")
            f.write(f"模型: {self.model_size} (WhisperX, CPU模式)\n")
            f.write(f"语言: {self.language}\n")
            f.write(f"转录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"音频时长: {format_timedelta(self.state['audio_duration'])}\n")
            f.write("=" * 60 + "\n\n")
            
            for segment in all_segments:
                start = format_timedelta(segment['start'])
                end = format_timedelta(segment['end'])
                text = segment['text'].strip()
                f.write(f"[{start} - {end}] {text}\n")
    
    def _cleanup(self):
        """清理临时文件（修改为移动到回收站）"""
        try:
            # 使用send2trash将文件移动到回收站而不是直接删除
            if self.state_file.exists():
                send2trash.send2trash(str(self.state_file))
            if self.progress_file.exists():
                send2trash.send2trash(str(self.progress_file))
            if self.audio_file.exists():
                send2trash.send2trash(str(self.audio_file))
            
            segments_dir = self.temp_base / "segments"
            if segments_dir.exists():
                # 先将所有音频片段文件移动到回收站
                for file in segments_dir.glob("*.wav"):
                    send2trash.send2trash(str(file))
                # 然后移动空目录到回收站
                send2trash.send2trash(str(segments_dir))
            
            # 如果临时基础目录为空，也移动到回收站
            if self.temp_base.exists() and not any(self.temp_base.iterdir()):
                send2trash.send2trash(str(self.temp_base))
            
            print("临时文件已移动到回收站")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")
            # 如果send2trash失败，回退到原删除逻辑
            try:
                if self.state_file.exists():
                    self.state_file.unlink()
                if self.progress_file.exists():
                    self.progress_file.unlink()
                if self.audio_file.exists():
                    self.audio_file.unlink()
                if segments_dir.exists():
                    for file in segments_dir.glob("*.wav"):
                        file.unlink()
                    segments_dir.rmdir()
                print("临时文件已直接删除（回收站操作失败）")
            except Exception as e2:
                print(f"直接删除也失败: {e2}")
    
    def transcribe(self):
        """执行转录过程（主流程框架不变，内部更换为WhisperX）"""
        print(f"开始转录视频: {self.video_path.name}")
        print(f"使用模型: {self.model_size} (WhisperX), 语言: {self.language}")
        print(f"设备: {self.device}, 计算类型: {self.compute_type}, 批处理大小: {self.batch_size}")
        print(f"临时文件目录: {self.temp_base}")
        print("-" * 50)
        
        self.timestamps["start_time"] = time.time()
        
        try:
            # 阶段1: 提取音频
            if not self._extract_audio():
                return False
            
            # 阶段2: 加载WhisperX模型
            if not self._load_models():
                return False
            
            # 阶段3: 准备音频片段
            if not self._prepare_segments():
                return False
            
            # 阶段4: 转录所有片段（断点续传）
            transcription_success = self._transcribe_segments()
            
            # 阶段5: 保存结果（只有在所有片段都成功时才保存）
            if transcription_success and self.state["segments"]:
                self._save_final_transcription()
                print(f"\n转录完成！结果保存在: {self.output_file}")
            elif self.state["segments"]:
                # 部分片段失败，不保存结果文件，但保留状态用于断点续传
                print(f"\n转录部分完成，已处理 {len(self.state['segments'])} 个句子")
                print("注意：部分片段转录失败，未生成最终结果文件")
                print(f"下次运行将从失败片段继续，状态文件: {self.state_file}")
                return False
            else:
                # 没有任何成功转录的内容
                print("\n转录失败，未生成任何结果")
                return False
            
            if not transcription_success:
                return False
            
            # 阶段6: 清理临时文件（默认不清理，仅在指定--cleanup时清理）
            if self.cleanup:
                self._cleanup()
            else:
                print(f"临时文件保留在: {self.temp_base}")
            
            # 计算总时间
            self.timestamps["total_time"] = time.time() - self.timestamps["start_time"]
            self._print_timestamps()
            
            print(f"\n转录完成！结果保存在: {self.output_file}")
            return True
            
        except KeyboardInterrupt:
            print("\n转录被用户中断，已保存当前状态")
            self._save_state()
            print(f"下次运行将从当前片段继续，状态文件: {self.state_file}")
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
    # 使用虚拟环境管理器（使用绝对路径）
    script_dir = Path(__file__).parent
    venv_path = script_dir / "whisperx_env"
    venv_manager = VirtualEnvironmentManager(str(venv_path))
    
    # 检查是否需要激活虚拟环境
    try:
        import whisperx
        import torch
        print("✓ 当前环境已包含WhisperX依赖")
    except ImportError:
        print("检测到缺少WhisperX依赖，自动激活虚拟环境...")
        if not venv_manager.activate():
            print("错误: 无法激活虚拟环境，请确保虚拟环境已正确配置")
            print("手动安装命令: pip install \"torch<2.6\" \"torchaudio<2.6\" \"pyannote.audio<3.0\" whisperx")
            sys.exit(1)
    
    try:
        parser = argparse.ArgumentParser(description="WhisperX语音转录程序（支持断点续传，CPU优化）")
        parser.add_argument("video_path", help="视频文件路径")
        parser.add_argument("--model", "-m", default="base", 
                           choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo"],
                           help="Whisper模型大小 (默认: base)")
        parser.add_argument("--language", "-l", default="ja", 
                           help="音频语言代码 (默认: ja - 日语)")
        parser.add_argument("--segment-duration", "-s", type=int, default=180,
                           help="音频分段时长（秒），默认180秒（用于断点续传）")
        parser.add_argument("--batch-size", "-b", type=int, default=4,
                           help="批处理大小，CPU上建议较小值（默认: 4）")
        parser.add_argument("--compute-type", "-c", default="int8", choices=["float32", "int8"],
                           help="计算类型，CPU建议使用 int8 以节省内存（默认: int8）")
        parser.add_argument("--cleanup", action="store_true",
                           help="在程序开始前清理临时文件（默认不清理）")
        
        args = parser.parse_args()
        
        # 检查视频文件是否存在
        if not Path(args.video_path).exists():
            print(f"错误: 视频文件不存在: {args.video_path}")
            sys.exit(1)
        
        # 如果指定了--cleanup，先清理临时文件
        if args.cleanup:
            print("清理临时文件...")
            video_hash = hashlib.md5(str(args.video_path).encode()).hexdigest()[:8]
            temp_base = Path(f"temp_{video_hash}")
            if temp_base.exists():
                import shutil
                shutil.rmtree(temp_base)
                print(f"已清理临时目录: {temp_base}")
            else:
                print("未找到对应的临时目录，无需清理")
        
        # 创建转录器并执行（强制使用CPU）
        transcriber = WhisperXTranscriber(
            video_path=args.video_path,
            model_size=args.model,
            language=args.language,
            segment_duration=args.segment_duration,
            batch_size=args.batch_size,
            compute_type=args.compute_type,
            device="cpu",  # 强制使用CPU
            cleanup=args.cleanup  # 传递清理参数
        )
        
        success = transcriber.transcribe()
        
        if not success:
            if transcriber.state["segments"]:
                print("转录过程部分完成，已保存可用内容")
                sys.exit(0)
            else:
                print("转录过程出现错误或中断")
                sys.exit(1)
    
    finally:
        # 确保在程序结束时退出虚拟环境
        venv_manager.deactivate()

if __name__ == "__main__":
    main()
