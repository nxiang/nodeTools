"""
音频处理模块
负责音频提取、格式转换和分段处理
"""

import os
import subprocess
import tempfile

def extract_audio_segment(video_path, output_path, segment_duration=None):
    """提取音频片段（测试模式）"""
    print("🎵 提取音频...")
    
    # 编码安全处理函数
    def safe_subprocess_run(cmd):
        """安全的子进程执行函数，处理编码问题"""
        try:
            # 使用二进制模式捕获输出，避免编码问题
            result = subprocess.run(cmd, shell=True, capture_output=True, text=False)
            
            # 手动解码输出，处理编码异常
            stdout = ""
            stderr = ""
            
            if result.stdout:
                try:
                    stdout = result.stdout.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        stdout = result.stdout.decode('gbk', errors='ignore')
                    except:
                        stdout = result.stdout.decode('utf-8', errors='ignore')
            
            if result.stderr:
                try:
                    stderr = result.stderr.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        stderr = result.stderr.decode('gbk', errors='ignore')
                    except:
                        stderr = result.stderr.decode('utf-8', errors='ignore')
            
            # 创建新的结果对象
            class ProcessResult:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            return ProcessResult(result.returncode, stdout, stderr)
            
        except Exception as e:
            print(f"⚠️ 子进程执行异常: {e}")
            # 返回一个默认的结果对象
            class ProcessResult:
                def __init__(self):
                    self.returncode = 1
                    self.stdout = ""
                    self.stderr = str(e)
            return ProcessResult()
    
    # 如果是测试模式，获取视频总时长
    if segment_duration:
        # 构建FFmpeg命令（测试模式：提取前N秒）
        cmd = f'ffmpeg -i "{video_path}" -t {segment_duration} -vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{output_path}"'
        print(f"🔬 测试模式：提取前 {segment_duration} 秒音频")
    else:
        # 完整模式：提取完整音频
        cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{output_path}"'
    
    # 执行音频提取
    result = safe_subprocess_run(cmd)
    
    if result.returncode == 0:
        # 验证输出文件
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ 音频提取成功: {output_path}")
            return True
        else:
            print(f"❌ 音频文件创建失败或为空")
            return False
    else:
        print(f"❌ 音频提取失败: {result.stderr}")
        
        # 尝试备用命令格式
        print("🔄 尝试备用命令格式...")
        if segment_duration:
            cmd = f'ffmpeg -i "{video_path}" -t {segment_duration} -vn -acodec pcm_s16le -ar 16000 -ac 1 "{output_path}"'
        else:
            cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{output_path}"'
        
        result = safe_subprocess_run(cmd)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ 备用命令音频提取成功: {output_path}")
            return True
        else:
            print(f"❌ 备用命令也失败: {result.stderr}")
            return False

def cleanup_audio_files():
    """清理临时音频文件"""
    temp_files = [
        "temp/audio.wav",
        "temp/audio_segment.wav"
    ]
    
    cleaned_count = 0
    for file_path in temp_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                cleaned_count += 1
            except Exception as e:
                print(f"⚠️ 无法删除临时文件 {file_path}: {e}")
    
    if cleaned_count > 0:
        print(f"🧹 清理了 {cleaned_count} 个临时音频文件")
    else:
        print("📝 无临时音频文件需要清理")

def get_audio_duration(audio_path):
    """获取音频文件时长"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        print(f"⚠️ 无法获取音频时长: {e}")
    
    return 0

def convert_audio_format(input_path, output_path, target_format='wav', sample_rate=16000):
    """转换音频格式"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            '-y', output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 音频格式转换成功: {output_path}")
            return True
        else:
            print(f"❌ 音频格式转换失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 音频格式转换异常: {e}")
        return False
