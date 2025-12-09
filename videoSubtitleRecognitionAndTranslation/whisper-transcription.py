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

# 虚拟环境管理类
class VirtualEnvironmentManager:
    """管理虚拟环境的激活和退出"""
    
    def __init__(self, venv_path=None):
        """
        初始化虚拟环境管理器
        
        Args:
            venv_path: 虚拟环境路径，如果为None则自动检测
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
    
    def __init__(self, temp_base, model, language):
        self.temp_base = temp_base
        self.model = model
        self.language = language
        self.segments_dir = temp_base / "segments"
        self.segments_dir.mkdir(exist_ok=True)
        
        # 延迟导入的whisper模块
        self.whisper_module = None
    
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
    
    def transcribe_segment(self, segment_file, segment_index, start_time=0):
        """转录单个音频片段"""
        print(f"转录片段 {segment_index}: {segment_file.name}")
        
        try:
            # 延迟导入whisper模块
            if self.whisper_module is None:
                self.whisper_module = import_whisper()
                if self.whisper_module is None:
                    print("错误: 无法导入whisper模块")
                    return []
            
            # 加载音频
            audio = self.whisper_module.load_audio(str(segment_file))
            
            # 转录当前片段
            result = self.model.transcribe(
                audio,
                language=self.language,
                task="transcribe",
                fp16=False,
                condition_on_previous_text=True  # 保持上下文连贯性
            )
            
            # 调整时间戳，加上片段起始时间
            for segment in result['segments']:
                segment['start'] += start_time
                segment['end'] += start_time
            
            return result['segments']
        except Exception as e:
            print(f"片段 {segment_index} 转录失败: {e}")
            return []

class WhisperTranscriber:
    def __init__(self, video_path, model_size="base", language="ja", segment_duration=60, cleanup=False):
        """
        初始化转录器
        
        Args:
            video_path: 视频文件路径
            model_size: Whisper模型大小 (tiny, base, small, medium, large, large-v1, large-v2, large-v3, large-v3-turbo, turbo)
            language: 音频语言代码 (默认: ja - 日语)
            segment_duration: 音频分段时长（秒），默认60秒
            cleanup: 是否在程序开始前清理临时文件，默认False
        """
        self.video_path = Path(video_path)
        self.model_size = model_size
        self.language = language
        self.segment_duration = segment_duration
        self.cleanup = cleanup
        
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
        """从视频中提取音频"""
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
            # 修复编码问题：使用UTF-8编码处理输出
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
    
    def _prepare_segments(self):
        """准备音频片段"""
        # 检查是否需要重新分割音频（如果segment_duration参数改变）
        if self.state.get("segment_duration") != self.segment_duration:
            print(f"检测到segment_duration参数改变，从{self.state.get('segment_duration', '未知')}秒改为{self.segment_duration}秒，重新分割音频...")
            # 清除旧的片段信息，强制重新分割
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
            # 即使使用现有片段，也需要初始化分段器
            self.segment_transcriber = SegmentTranscriber(
                self.temp_base, 
                self.model, 
                self.language
            )
            return True
        
        # 创建分段器并分割音频
        self.segment_transcriber = SegmentTranscriber(
            self.temp_base, 
            self.model, 
            self.language
        )
        
        segment_files = self.segment_transcriber.split_audio(self.audio_file, self.segment_duration)
        
        if not segment_files:
            return False
        
        # 保存片段信息，包括当前的segment_duration
        self.state["segment_files"] = [str(f) for f in segment_files]
        self.state["total_segments"] = len(segment_files)
        self.state["segment_duration"] = self.segment_duration  # 保存当前参数值
        self.state["current_segment"] = self.state["processed_segments"]
        self._save_state()
        
        print(f"将音频分割成 {self.segment_duration} 秒的片段...")
        print(f"音频分割完成，共 {self.state['total_segments']} 个片段")
        return True
    
    def _transcribe_segments(self):
        """转录所有音频片段"""
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
            
            if segment_results:
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
                cumulative_time = time.time() - transcribe_start
                print(f"片段 {segment_index} 完成，耗时: {format_timedelta(segment_time)}，累计耗时: {format_timedelta(cumulative_time)}")
            else:
                print(f"片段 {segment_index} 转录失败")
                return False
            
            # 更新总转录时间
            self.timestamps["transcription_time"] = time.time() - transcribe_start
        
        return True
    
    def _save_final_transcription(self):
        """保存最终的转录结果"""
        # 按时间戳排序所有片段
        all_segments = sorted(self.state["segments"], key=lambda x: x['start'])
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"视频: {self.video_path.name}\n")
            f.write(f"模型: {self.model_size}\n")
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
    
    def transcribe(self):
        """执行转录过程"""
        print(f"开始转录视频: {self.video_path.name}")
        print(f"使用模型: {self.model_size}, 语言: {self.language}")
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
            if not self._extract_audio():
                return False
            
            # 阶段2: 加载Whisper模型
            print("加载Whisper模型...")
            model_start = time.time()
            
            try:
                # 延迟导入whisper模块
                if self.whisper_module is None:
                    self.whisper_module = import_whisper()
                    if self.whisper_module is None:
                        print("错误: 无法导入whisper模块")
                        return False
                
                self.model = self.whisper_module.load_model(self.model_size)
                self.timestamps["model_loading_time"] = time.time() - model_start
                print(f"模型加载完成，耗时: {format_timedelta(self.timestamps['model_loading_time'])}")
            except Exception as e:
                print(f"模型加载失败: {e}")
                return False
            
            # 阶段3: 准备音频片段
            if not self._prepare_segments():
                return False
            
            # 阶段4: 转录所有片段
            transcription_success = self._transcribe_segments()
            
            # 阶段5: 保存结果（仅在全部成功时保存）
            if transcription_success and self.state["segments"]:
                self._save_final_transcription()
                print(f"\n转录完成！结果保存在: {self.output_file}")
            elif self.state["segments"]:
                # 部分转录完成，但不保存文件
                print(f"\n转录失败！已转录 {len(self.state['segments'])}/{self.state['total_segments']} 个片段")
                print("注意：由于转录未全部完成，未生成最终转录文件")
            
            if not transcription_success:
                return False
            
            # 阶段6: 计算总时间
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
    parser = argparse.ArgumentParser(description="Whisper语音转录程序（支持断点续传）")
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
            cleanup=args.cleanup
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
