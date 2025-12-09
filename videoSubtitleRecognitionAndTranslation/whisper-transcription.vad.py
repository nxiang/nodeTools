import os
import json
import hashlib
import time
import logging
import argparse
import shutil
import subprocess
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import whisper
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_timedelta(seconds):
    """将秒数格式化为时:分:秒.毫秒格式，用于SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_simple_timedelta(seconds):
    """简化版时间格式，用于文本输出"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TranscriptionStatus(Enum):
    """转录状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SegmentInfo:
    """音频片段信息"""
    index: int
    start_time: float
    end_time: float
    file_path: str
    status: TranscriptionStatus = TranscriptionStatus.PENDING
    transcription: Optional[str] = None
    segments: Optional[List[dict]] = None  # Whisper返回的详细分段信息（包含词级时间戳）
    error: Optional[str] = None
    processed_at: Optional[datetime] = None
    processing_time: Optional[float] = None  # 处理耗时（秒）
    
    def to_dict(self):
        return {
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "file_path": self.file_path,
            "status": self.status.value,
            "transcription": self.transcription,
            "segments": self.segments,  # 保存详细的分段信息
            "error": self.error,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "processing_time": self.processing_time
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            index=data["index"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            file_path=data["file_path"],
            status=TranscriptionStatus(data["status"]),
            transcription=data.get("transcription"),
            segments=data.get("segments"),  # 加载详细的分段信息
            error=data.get("error"),
            processed_at=datetime.fromisoformat(data["processed_at"]) if data.get("processed_at") else None,
            processing_time=data.get("processing_time")
        )


@dataclass
class TranscriptionState:
    """转录状态管理器"""
    audio_file: str
    original_file: str
    temp_dir: str
    segments: List[SegmentInfo] = field(default_factory=list)
    total_segments: int = 0
    completed_segments: int = 0
    failed_segments: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    model_name: str = "base"
    language: str = "ja"  # 默认日语
    segment_duration: int = 600
    
    def to_dict(self):
        return {
            "audio_file": self.audio_file,
            "original_file": self.original_file,
            "temp_dir": self.temp_dir,
            "segments": [segment.to_dict() for segment in self.segments],
            "total_segments": self.total_segments,
            "completed_segments": self.completed_segments,
            "failed_segments": self.failed_segments,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "model_name": self.model_name,
            "language": self.language,
            "segment_duration": self.segment_duration
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        state = cls(
            audio_file=data["audio_file"],
            original_file=data["original_file"],
            temp_dir=data["temp_dir"],
            total_segments=data["total_segments"],
            completed_segments=data["completed_segments"],
            failed_segments=data["failed_segments"],
            model_name=data.get("model_name", "base"),
            language=data.get("language", "ja"),
            segment_duration=data.get("segment_duration", 600)
        )
        
        if data.get("start_time"):
            state.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            state.end_time = datetime.fromisoformat(data["end_time"])
        
        state.segments = [SegmentInfo.from_dict(seg) for seg in data.get("segments", [])]
        return state


class VideoTranscriber:
    """支持断点续传的视频/音频转录器"""
    
    def __init__(self, 
                 model_name: str = "base"):
        """
        初始化转录器
        
        Args:
            model_name: Whisper模型名称
        """
        # 固定目录
        self.base_dir = Path(".")
        self.output_dir = self.base_dir / "temp"
        self.temp_base_dir = self.base_dir / "temp"
        
        self.model_name = model_name
        self.model = None
        
        # 时间戳记录
        self.timestamps = {
            "start_time": None,
            "audio_extraction_time": None,
            "model_loading_time": None,
            "transcription_time": None,
            "total_time": None
        }
        
        # 创建目录
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_base_dir.mkdir(exist_ok=True, parents=True)
    
    def get_temp_dir(self, video_file: str) -> Path:
        """
        获取临时目录
        
        Args:
            video_file: 视频文件路径
            
        Returns:
            临时目录路径
        """
        # 获取文件名（不带扩展名）
        video_name = Path(video_file).stem
        
        # 创建目录名：视频名_模型名（不需要哈希）
        temp_dir = self.temp_base_dir / f"{video_name}_{self.model_name}"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        return temp_dir
    
    def cleanup_temp_files(self, video_file: str):
        """
        清理特定视频的临时文件
        
        Args:
            video_file: 视频文件路径
        """
        temp_dir = self.get_temp_dir(video_file)
        if temp_dir.exists():
            logger.info(f"清理临时目录: {temp_dir}")
            shutil.rmtree(temp_dir)
    
    def extract_audio_from_video(self, video_file: str, temp_dir: Path) -> str:
        """
        从视频文件中提取音频
        
        Args:
            video_file: 视频文件路径
            temp_dir: 临时目录
            
        Returns:
            提取的音频文件路径
        """
        audio_file = temp_dir / "extracted_audio.wav"
        
        # 检查是否已存在提取的音频
        if audio_file.exists():
            logger.info(f"已存在提取的音频文件: {audio_file}")
            return str(audio_file)
        
        logger.info(f"从视频中提取音频: {video_file}")
        
        try:
            # 使用ffmpeg提取音频，确保使用正确的编码
            cmd = [
                'ffmpeg',
                '-i', video_file,
                '-vn',  # 忽略视频流
                '-ac', '1',  # 单声道（Whisper处理单声道更好）
                '-ar', '16000',  # 设置采样率为16000Hz（Whisper推荐）
                '-acodec', 'pcm_s16le',  # 使用PCM编码
                '-f', 'wav',  # 输出为wav格式
                '-y',  # 覆盖已存在文件
                str(audio_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                raise Exception(f"FFmpeg提取音频失败: {result.stderr}")
            
            # 验证提取的音频文件
            if not audio_file.exists() or audio_file.stat().st_size == 0:
                raise Exception(f"提取的音频文件为空: {audio_file}")
            
            logger.info(f"音频提取完成: {audio_file}")
            return str(audio_file)
            
        except FileNotFoundError:
            logger.error("未找到ffmpeg，请先安装ffmpeg")
            raise Exception("未找到ffmpeg，请先安装ffmpeg")
        except Exception as e:
            logger.error(f"提取音频失败: {e}")
            raise
    
    def _get_state_file_path(self, video_file: str) -> Path:
        """获取状态文件路径"""
        # 状态文件放在临时目录下的states子目录中
        temp_dir = self.get_temp_dir(video_file)
        state_dir = temp_dir / "states"
        state_dir.mkdir(exist_ok=True)
        return state_dir / "state.json"
    
    def load_state(self, video_file: str) -> Optional[TranscriptionState]:
        """加载转录状态"""
        state_file = self._get_state_file_path(video_file)
        
        if not state_file.exists():
            return None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            state = TranscriptionState.from_dict(state_data)
            
            # 检查临时目录是否存在，如果不存在则状态无效
            temp_dir = Path(state.temp_dir)
            if not temp_dir.exists():
                logger.warning(f"临时目录不存在: {temp_dir}")
                return None
                
            return state
        except Exception as e:
            logger.warning(f"加载状态文件失败: {e}")
            return None
    
    def save_state(self, state: TranscriptionState):
        """保存转录状态"""
        state_file = self._get_state_file_path(state.original_file)
        
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存状态文件失败: {e}")
    
    def prepare_segments(self, video_file: str, segment_duration: int = 600) -> TranscriptionState:
        """
        准备音频分段
        
        Args:
            video_file: 视频文件路径
            segment_duration: 分段时长（秒），默认10分钟
            
        Returns:
            转录状态对象
        """
        try:
            from pydub import AudioSegment
        except ImportError:
            logger.error("请安装pydub: pip install pydub")
            raise
        
        # 获取临时目录
        temp_dir = self.get_temp_dir(video_file)
        
        # 提取音频
        audio_file = self.extract_audio_from_video(video_file, temp_dir)
        
        # 创建分段子目录
        segments_dir = temp_dir / "segments"
        segments_dir.mkdir(exist_ok=True)
        
        # 使用pydub加载音频
        logger.info(f"加载音频文件: {audio_file}")
        try:
            audio = AudioSegment.from_wav(audio_file)
        except:
            # 如果不是wav格式，尝试通用加载
            audio = AudioSegment.from_file(audio_file)
            
        duration_ms = len(audio)
        duration_sec = duration_ms / 1000
        
        # 计算分段数
        segment_count = int(duration_sec // segment_duration) + (1 if duration_sec % segment_duration > 0 else 0)
        
        # 创建状态对象
        state = TranscriptionState(
            audio_file=audio_file,
            original_file=video_file,
            temp_dir=str(temp_dir),
            total_segments=segment_count,
            model_name=self.model_name,
            segment_duration=segment_duration,
            start_time=datetime.now()
        )
        
        # 分割音频并创建分段信息
        for i in range(segment_count):
            start_ms = i * segment_duration * 1000
            end_ms = min((i + 1) * segment_duration * 1000, duration_ms)
            
            segment = audio[start_ms:end_ms]
            segment_file = segments_dir / f"segment_{i:04d}.wav"
            
            # 导出为wav格式
            segment.export(str(segment_file), format="wav")
            
            # 验证保存的文件
            if not segment_file.exists() or segment_file.stat().st_size == 0:
                logger.error(f"片段 {i} 保存失败: {segment_file}")
                continue
            
            # 添加分段信息
            segment_info = SegmentInfo(
                index=i,
                start_time=start_ms / 1000,
                end_time=end_ms / 1000,
                file_path=str(segment_file)
            )
            state.segments.append(segment_info)
        
        # 如果实际创建的片段数少于预期，更新总数
        state.total_segments = len(state.segments)
        
        # 保存初始状态
        self.save_state(state)
        logger.info(f"音频分段完成，共{state.total_segments}段，临时文件保存在: {temp_dir}")
        return state
    
    def load_model(self):
        """加载Whisper模型"""
        if self.model is None:
            logger.info(f"加载Whisper模型: {self.model_name}")
            try:
                # 尝试加载模型
                self.model = whisper.load_model(self.model_name)
                logger.info(f"模型加载成功: {self.model_name}")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                # 如果模型名称包含"turbo"，尝试使用可能的后缀
                if "turbo" in self.model_name.lower():
                    logger.info("尝试加载不带turbo后缀的模型...")
                    try:
                        base_model_name = self.model_name.replace("-turbo", "").replace("_turbo", "")
                        self.model = whisper.load_model(base_model_name)
                        logger.info(f"备用模型加载成功: {base_model_name}")
                    except Exception as e2:
                        logger.error(f"备用模型也加载失败: {e2}")
                        raise
                else:
                    raise
    
    def check_audio_file(self, audio_path: str) -> Tuple[bool, str]:
        """
        检查音频文件是否有效
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            # 检查文件是否存在且大小大于0
            if not os.path.exists(audio_path):
                return False, "文件不存在"
            
            if os.path.getsize(audio_path) == 0:
                return False, "文件大小为0"
            
            # 尝试直接加载音频数据
            import whisper
            audio_data = whisper.load_audio(audio_path)
            
            # 检查是否有音频数据
            if len(audio_data) == 0:
                return False, "没有音频数据"
            
            # 检查音频是否为静音
            if np.max(np.abs(audio_data)) < 0.001:
                return False, "音频可能是静音"
            
            return True, f"音频有效，长度: {len(audio_data)/16000:.2f}秒"
            
        except Exception as e:
            return False, f"检查音频文件失败: {str(e)}"
    
    def transcribe_segment(self, segment_info: SegmentInfo, state: TranscriptionState) -> SegmentInfo:
        """
        转录单个音频片段
        
        Args:
            segment_info: 音频片段信息
            state: 转录状态
            
        Returns:
            更新后的片段信息
        """
        import time
        
        try:
            # 如果已经完成，直接返回
            if segment_info.status == TranscriptionStatus.COMPLETED:
                return segment_info
            
            # 标记为处理中
            segment_info.status = TranscriptionStatus.PROCESSING
            self.save_state(state)
            
            # 记录片段开始时间
            segment_start_time = time.time()
            
            # 加载模型（如果未加载）
            if self.model is None:
                self.load_model()
            
            # 检查音频文件是否有效
            is_valid, check_msg = self.check_audio_file(segment_info.file_path)
            if not is_valid:
                logger.warning(f"音频片段 {segment_info.index} 无效: {check_msg}")
                segment_info.status = TranscriptionStatus.FAILED
                segment_info.error = f"音频无效: {check_msg}"
                return segment_info
            
            logger.info(f"开始转录片段 {segment_info.index + 1}/{state.total_segments}")
            logger.info(f"音频检查结果: {check_msg}")
            
            # 转录音频片段（启用词级时间戳）
            result = self.model.transcribe(
                segment_info.file_path,
                language=state.language,
                fp16=False,
                verbose=False,
                condition_on_previous_text=False,
                word_timestamps=True  # 启用词级时间戳
            )
            
            # 计算片段转录耗时
            segment_time = time.time() - segment_start_time
            
            # 更新片段信息
            segment_info.transcription = result["text"]
            segment_info.segments = result.get("segments", [])  # 保存详细的分段信息（包含词级时间戳）
            segment_info.status = TranscriptionStatus.COMPLETED
            segment_info.processed_at = datetime.now()
            segment_info.processing_time = segment_time  # 记录片段处理时间
            
            logger.info(f"完成转录片段 {segment_info.index + 1}/{state.total_segments}，耗时: {segment_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"转录片段 {segment_info.index} 失败: {e}")
            logger.exception(e)  # 打印完整的堆栈跟踪
            
            segment_info.status = TranscriptionStatus.FAILED
            segment_info.error = str(e)
            
            # 如果是张量形状错误，尝试特殊处理
            if "cannot reshape tensor of 0 elements" in str(e):
                logger.info(f"检测到张量形状错误，尝试重新生成音频片段 {segment_info.index}")
                try:
                    # 尝试重新生成音频文件
                    self._regenerate_audio_segment(segment_info, state)
                    
                    # 重新转录
                    result = self.model.transcribe(
                        segment_info.file_path,
                        language=state.language,
                        fp16=False,
                        verbose=False,
                        condition_on_previous_text=False
                    )
                    
                    segment_info.transcription = result["text"]
                    segment_info.status = TranscriptionStatus.COMPLETED
                    segment_info.processed_at = datetime.now()
                    logger.info(f"重新转录成功完成片段 {segment_info.index + 1}/{state.total_segments}")
                except Exception as e2:
                    logger.error(f"重新转录也失败: {e2}")
        
        return segment_info
    
    def _regenerate_audio_segment(self, segment_info: SegmentInfo, state: TranscriptionState):
        """重新生成音频片段"""
        try:
            from pydub import AudioSegment
            
            # 加载原始音频
            audio = AudioSegment.from_wav(state.audio_file)
            
            # 计算片段位置
            start_ms = int(segment_info.start_time * 1000)
            end_ms = int(segment_info.end_time * 1000)
            
            # 提取片段
            segment = audio[start_ms:end_ms]
            
            # 重新导出
            segment.export(segment_info.file_path, format="wav")
            
            logger.info(f"重新生成音频片段 {segment_info.index}")
            
        except Exception as e:
            logger.error(f"重新生成音频片段失败: {e}")
            raise
    
    def resume_transcription(self, video_file: str) -> TranscriptionState:
        """
        恢复转录（断点续传）
        
        Args:
            video_file: 视频文件路径
            
        Returns:
            转录状态对象
        """
        # 加载已有状态
        state = self.load_state(video_file)
        if not state:
            raise ValueError("未找到转录状态，请先开始新的转录")
        
        logger.info(f"恢复转录: {video_file}")
        logger.info(f"临时目录: {state.temp_dir}")
        logger.info(f"进度: {state.completed_segments}/{state.total_segments}")
        
        # 加载模型
        if self.model is None:
            self.load_model()
        
        # 统计待处理片段
        pending_segments = [s for s in state.segments if s.status in [
            TranscriptionStatus.PENDING, 
            TranscriptionStatus.FAILED
        ]]
        
        if not pending_segments:
            logger.info("所有片段已完成转录")
            return state
        
        # 顺序处理待转录片段
        return self._process_segments_sequential(state, pending_segments)
    
    def transcribe(self, 
                  video_file: str, 
                  segment_duration: int = 600,
                  language: str = "ja",  # 默认日语
                  resume: bool = True,
                  cleanup: bool = False) -> TranscriptionState:
        """
        转录视频/音频文件
        
        Args:
            video_file: 视频/音频文件路径
            segment_duration: 分段时长（秒）
            language: 语言代码（默认日语）
            resume: 是否尝试恢复之前的转录
            cleanup: 是否清理临时文件
            
        Returns:
            转录状态对象
        """
        import time
        
        # 记录开始时间
        self.timestamps["start_time"] = time.time()
        total_start_time = self.timestamps["start_time"]
        
        # 检查文件是否存在
        if not Path(video_file).exists():
            raise FileNotFoundError(f"文件不存在: {video_file}")
        
        # 清理临时文件
        if cleanup:
            self.cleanup_temp_files(video_file)
        
        # 尝试恢复之前的转录
        if resume:
            existing_state = self.load_state(video_file)
            if existing_state:
                logger.info("检测到未完成的转录任务，尝试恢复...")
                return self.resume_transcription(video_file)
        
        # 开始新的转录
        logger.info(f"开始新的转录任务: {video_file}")
        
        # 准备音频分段
        audio_prep_start = time.time()
        state = self.prepare_segments(video_file, segment_duration)
        state.language = language
        audio_prep_time = time.time() - audio_prep_start
        self.timestamps["audio_extraction_time"] = audio_prep_time
        
        # 打印音频准备耗时
        print(f"音频提取与分段耗时: {format_simple_timedelta(audio_prep_time)}")
        print(f"累计耗时: {format_simple_timedelta(time.time() - total_start_time)}")
        
        # 加载模型
        model_load_start = time.time()
        self.load_model()
        model_load_time = time.time() - model_load_start
        self.timestamps["model_loading_time"] = model_load_time
        
        # 打印模型加载耗时
        print(f"模型加载耗时: {format_simple_timedelta(model_load_time)}")
        print(f"累计耗时: {format_simple_timedelta(time.time() - total_start_time)}")
        
        # 获取所有待处理片段
        pending_segments = [s for s in state.segments if s.status == TranscriptionStatus.PENDING]
        
        if not pending_segments:
            logger.warning("没有需要转录的片段")
            return state
        
        # 顺序处理所有片段
        transcription_start = time.time()
        result = self._process_segments_sequential(state, pending_segments)
        transcription_time = time.time() - transcription_start
        self.timestamps["transcription_time"] = transcription_time
        
        # 打印转录耗时
        print(f"转录处理耗时: {format_simple_timedelta(transcription_time)}")
        
        # 计算总耗时
        total_time = time.time() - total_start_time
        self.timestamps["total_time"] = total_time
        
        # 打印总耗时
        print(f"总耗时: {format_simple_timedelta(total_time)}")
        
        return result
    
    def _process_segments_sequential(self, 
                                   state: TranscriptionState, 
                                   segments_to_process: List[SegmentInfo]) -> TranscriptionState:
        """顺序处理音频片段"""
        import time
        
        try:
            # 按索引排序
            segments_to_process.sort(key=lambda x: x.index)
            
            # 记录转录开始时间
            transcription_start_time = time.time()
            
            # 顺序处理每个片段
            for i, segment in enumerate(segments_to_process):
                try:
                    # 计算当前累计耗时
                    current_cumulative_time = time.time() - transcription_start_time
                    logger.info(f"处理片段 {segment.index + 1}/{state.total_segments}，累计耗时: {format_simple_timedelta(current_cumulative_time)}")
                    
                    updated_segment = self.transcribe_segment(segment, state)
                    
                    # 更新状态
                    idx = segment.index
                    state.segments[idx] = updated_segment
                    
                    # 更新统计
                    if updated_segment.status == TranscriptionStatus.COMPLETED:
                        state.completed_segments += 1
                    elif updated_segment.status == TranscriptionStatus.FAILED:
                        state.failed_segments += 1
                    
                    # 保存状态
                    self.save_state(state)
                    
                    # 计算当前进度和预计剩余时间
                    completed_count = state.completed_segments + state.failed_segments
                    total_count = state.total_segments
                    progress_percentage = (completed_count / total_count * 100) if total_count > 0 else 0
                    
                    # 计算平均处理时间
                    completed_segments_with_time = [s for s in state.segments if s.processing_time is not None]
                    if completed_segments_with_time:
                        avg_time = sum(s.processing_time for s in completed_segments_with_time) / len(completed_segments_with_time)
                        remaining_segments = total_count - completed_count
                        estimated_remaining_time = avg_time * remaining_segments
                        logger.info(f"进度: {completed_count}/{total_count} ({progress_percentage:.1f}%)，预计剩余时间: {format_simple_timedelta(estimated_remaining_time)}")
                    else:
                        logger.info(f"进度: {completed_count}/{total_count} ({progress_percentage:.1f}%)")
                    
                except KeyboardInterrupt:
                    logger.info("转录被用户中断，保存当前进度...")
                    self.save_state(state)
                    raise
                except Exception as e:
                    logger.error(f"处理片段 {segment.index} 时出错: {e}")
                    segment.status = TranscriptionStatus.FAILED
                    segment.error = str(e)
                    state.failed_segments += 1
                    self.save_state(state)
            
            # 转录完成
            state.end_time = datetime.now()
            
            # 保存最终状态
            self.save_state(state)
            
            # 合并转录结果
            if state.completed_segments > 0:
                self._merge_transcriptions(state)
            
            # 输出摘要
            self._print_summary(state)
            
            return state
            
        except KeyboardInterrupt:
            logger.info("转录被用户中断，保存当前进度...")
            self.save_state(state)
            raise
            
        except Exception as e:
            logger.error(f"转录过程中发生错误: {e}")
            self.save_state(state)
            raise
    
    def _merge_transcriptions(self, state: TranscriptionState):
        """合并所有片段的转录结果"""
        # 按时间顺序排序片段
        sorted_segments = sorted(state.segments, key=lambda x: x.index)
        
        # 生成纯文本格式（保留原有格式）
        transcriptions = []
        for segment in sorted_segments:
            if segment.transcription:
                # 添加时间戳标记
                time_mark = f"[{segment.start_time:.1f}s-{segment.end_time:.1f}s]"
                transcriptions.append(f"{time_mark} {segment.transcription}")
        
        full_text = "\n".join(transcriptions)
        
        # 保存纯文本格式
        txt_file = self.output_dir / f"{Path(state.original_file).stem}_transcription.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        logger.info(f"纯文本转录已保存至: {txt_file}")
        
        # 生成SRT格式字幕（标准字幕格式）
        srt_entries = []
        entry_index = 1
        
        # 方法1：片段级别SRT（原有逻辑）
        for segment in sorted_segments:
            if segment.transcription:
                # 转换为SRT时间格式
                start_time_str = format_timedelta(segment.start_time)
                end_time_str = format_timedelta(segment.end_time)
                
                # 创建SRT条目
                srt_entry = f"{entry_index}\n{start_time_str} --> {end_time_str}\n{segment.transcription}\n"
                srt_entries.append(srt_entry)
                entry_index += 1
        
        srt_content = "\n".join(srt_entries)
        srt_file = self.output_dir / f"{Path(state.original_file).stem}_transcription.srt"
        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        logger.info(f"片段级别SRT字幕已保存至: {srt_file}")
        
        # 方法2：句子级别SRT（如果可用，基于Whisper的句子分段）
        sentence_srt_entries = []
        sentence_entry_index = 1
        
        for segment in sorted_segments:
            if segment.segments:  # 如果有Whisper的详细分段信息
                for whisper_segment in segment.segments:
                    # 计算绝对时间（片段起始时间 + 句子相对时间）
                    sentence_start = segment.start_time + whisper_segment['start']
                    sentence_end = segment.start_time + whisper_segment['end']
                    
                    # 转换为SRT时间格式
                    sentence_start_str = format_timedelta(sentence_start)
                    sentence_end_str = format_timedelta(sentence_end)
                    
                    # 创建句子级别SRT条目
                    sentence_srt_entry = f"{sentence_entry_index}\n{sentence_start_str} --> {sentence_end_str}\n{whisper_segment['text'].strip()}\n"
                    sentence_srt_entries.append(sentence_srt_entry)
                    sentence_entry_index += 1
        
        if sentence_srt_entries:  # 如果有句子级别时间戳，则生成句子级别SRT
            sentence_srt_content = "\n".join(sentence_srt_entries)
            sentence_srt_file = self.output_dir / f"{Path(state.original_file).stem}_sentence_level.srt"
            with open(sentence_srt_file, 'w', encoding='utf-8') as f:
                f.write(sentence_srt_content)
            
            logger.info(f"句子级别SRT字幕已保存至: {sentence_srt_file}")
        else:
            logger.info("未检测到句子级别时间戳信息，仅生成片段级别SRT字幕")
        
        # 同时保存为JSON格式（包含更多信息）
        json_output = {
            "original_file": state.original_file,
            "audio_file": state.audio_file,
            "model": state.model_name,
            "language": state.language,
            "total_duration": state.segments[-1].end_time if state.segments else 0,
            "segments": [seg.to_dict() for seg in sorted_segments],
            "full_text": full_text,
            "srt_content": srt_content,
            "completed_at": datetime.now().isoformat()
        }
        
        json_file = self.output_dir / f"{Path(state.original_file).stem}_transcription.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
    
    def _print_summary(self, state: TranscriptionState):
        """打印转录摘要"""
        print("\n" + "="*60)
        print("转录摘要")
        print("="*60)
        print(f"原始文件: {state.original_file}")
        print(f"音频文件: {state.audio_file}")
        print(f"使用模型: {state.model_name}")
        print(f"语言: {state.language}")
        print(f"临时目录: {state.temp_dir}")
        print(f"总片段数: {state.total_segments}")
        print(f"成功转录: {state.completed_segments}")
        print(f"失败片段: {state.failed_segments}")
        
        # 显示详细耗时统计
        if self.timestamps["total_time"]:
            print("\n耗时统计:")
            print(f"  音频提取与分段: {format_simple_timedelta(self.timestamps['audio_extraction_time'])}")
            print(f"  模型加载: {format_simple_timedelta(self.timestamps['model_loading_time'])}")
            print(f"  转录处理: {format_simple_timedelta(self.timestamps['transcription_time'])}")
            print(f"  总耗时: {format_simple_timedelta(self.timestamps['total_time'])}")
            
            # 计算各阶段占比
            total = self.timestamps["total_time"]
            if total > 0:
                print(f"\n各阶段耗时占比:")
                if self.timestamps['audio_extraction_time']:
                    audio_percent = (self.timestamps['audio_extraction_time'] / total) * 100
                    print(f"  音频提取与分段: {audio_percent:.1f}%")
                if self.timestamps['model_loading_time']:
                    model_percent = (self.timestamps['model_loading_time'] / total) * 100
                    print(f"  模型加载: {model_percent:.1f}%")
                if self.timestamps['transcription_time']:
                    trans_percent = (self.timestamps['transcription_time'] / total) * 100
                    print(f"  转录处理: {trans_percent:.1f}%")
        elif state.start_time and state.end_time:
            duration = (state.end_time - state.start_time).total_seconds()
            print(f"总耗时: {duration:.1f}秒")
        
        # 片段处理时间统计
        completed_segments_with_time = [s for s in state.segments if s.processing_time is not None]
        if completed_segments_with_time:
            total_processing_time = sum(s.processing_time for s in completed_segments_with_time)
            avg_processing_time = total_processing_time / len(completed_segments_with_time)
            max_processing_time = max(s.processing_time for s in completed_segments_with_time)
            min_processing_time = min(s.processing_time for s in completed_segments_with_time)
            
            print("\n片段处理时间统计:")
            print(f"  总处理时间: {format_simple_timedelta(total_processing_time)}")
            print(f"  平均处理时间: {format_simple_timedelta(avg_processing_time)}")
            print(f"  最长处理时间: {format_simple_timedelta(max_processing_time)}")
            print(f"  最短处理时间: {format_simple_timedelta(min_processing_time)}")
        
        if state.completed_segments == state.total_segments:
            print("状态: 已完成 ✓")
        else:
            print("状态: 部分完成（可恢复）")
        
        # 临时文件说明
        print(f"\n临时文件保存在: {state.temp_dir}")
        print("注意: 临时文件默认不会自动清理，可以用于恢复转录")
        print("如需清理临时文件，可以使用 --cleanup 参数")
        print("="*60)
    
    def get_progress(self, video_file: str) -> Dict:
        """获取转录进度"""
        state = self.load_state(video_file)
        if not state:
            return {"error": "未找到转录任务"}
        
        completed = len([s for s in state.segments if s.status == TranscriptionStatus.COMPLETED])
        total = len(state.segments)
        
        return {
            "original_file": state.original_file,
            "audio_file": state.audio_file,
            "temp_dir": state.temp_dir,
            "completed": completed,
            "total": total,
            "progress": f"{completed}/{total}",
            "percentage": (completed / total * 100) if total > 0 else 0,
            "status": "completed" if completed == total else "in_progress"
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="日语视频转录工具（支持断点续传）")
    
    parser.add_argument("video_file", help="视频文件路径")
    parser.add_argument("--model", "-m", default="base", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v3-turbo"],
                       help="Whisper模型大小（默认：base）")
    parser.add_argument("--segment-duration", "-d", type=int, default=180,
                       help="分段时长（秒，默认：180）")
    parser.add_argument("--language", "-l", default="ja",
                       help="语言代码（默认：ja）")
    parser.add_argument("--cleanup", action="store_true",
                       help="清理临时文件（程序运行前清理）")
    
    return parser.parse_args()


def main():
    """命令行入口"""
    args = parse_args()
    
    # 创建转录器
    transcriber = VideoTranscriber(model_name=args.model)
    
    try:
        # 开始转录
        state = transcriber.transcribe(
            video_file=args.video_file,
            segment_duration=args.segment_duration,
            language=args.language,
            cleanup=args.cleanup
        )
        
        # 检查进度
        progress = transcriber.get_progress(args.video_file)
        if "error" not in progress:
            print(f"\n当前进度: {progress['progress']} ({progress['percentage']:.1f}%)")
            print(f"临时文件目录: {progress['temp_dir']}")
        
    except KeyboardInterrupt:
        print("\n转录被中断，下次运行会自动恢复进度")
    except Exception as e:
        print(f"转录失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
