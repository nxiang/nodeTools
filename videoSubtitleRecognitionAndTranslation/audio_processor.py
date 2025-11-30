"""
音频处理模块
负责音频提取、格式转换、分段处理和静默检测
"""

import os
import subprocess
import tempfile
import json
from pathlib import Path

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

def detect_speech_segments(video_path, silence_threshold=-30.0, min_silence_duration=0.5, max_duration=None):
    """
    使用FFmpeg进行静默检测，返回有语音的时间段列表
    
    Args:
        video_path: 视频文件路径
        silence_threshold: 静默阈值（分贝），默认-30.0
        min_silence_duration: 最小静默持续时间（秒），默认0.5
        max_duration: 最大处理时长（用于测试模式），None表示处理完整视频
        
    Returns:
        list: 语音段列表，每个元素为[开始时间, 结束时间]（秒）
    """
    print(f"🔍 开始静默检测，寻找语音片段...")
    
    # 构建FFmpeg静默检测命令
    cmd = [
        'ffmpeg', '-i', video_path,
        '-af', f'silencedetect=n={silence_threshold}dB:d={min_silence_duration}',
        '-f', 'null', '-'
    ]
    
    result = safe_subprocess_run(cmd)
    
    if result.returncode != 0:
        print(f"❌ 静默检测失败: {result.stderr}")
        return None
    
    # 解析输出，提取语音段
    speech_segments = []
    in_speech = False
    speech_start = 0.0
    
    # 处理输出
    lines = result.stderr.split('\n')
    for line in lines:
        if 'silence_start' in line:
            # 发现静默开始，意味着之前是语音
            if in_speech:
                # 静默开始前是语音结束
                silence_start = float(line.split(':')[-1].strip())
                # 只保留持续时间大于0.3秒的语音段
                if silence_start - speech_start > 0.3:
                    speech_segments.append([speech_start, silence_start])
                in_speech = False
        elif 'silence_end' in line:
            # 发现静默结束，意味着语音开始
            parts = line.split(':')
            silence_end = float(parts[-2].split()[0])
            speech_start = silence_end
            in_speech = True
    
    # 检查最后是否还有语音段
    if in_speech:
        # 尝试获取视频总时长
        try:
            duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                           'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            duration_result = safe_subprocess_run(duration_cmd)
            if duration_result.returncode == 0:
                total_duration = float(duration_result.stdout.strip())
                if total_duration - speech_start > 0.3:
                    speech_segments.append([speech_start, total_duration])
        except Exception:
            pass
    
    # 合并过于接近的语音段（间隔小于1秒的合并）
    merged_segments = []
    for segment in speech_segments:
        if not merged_segments:
            merged_segments.append(segment)
        else:
            last = merged_segments[-1]
            if segment[0] - last[1] < 1.0:
                # 合并间隔小于1秒的段
                merged_segments[-1] = [last[0], segment[1]]
            else:
                merged_segments.append(segment)
    
    # 计算总语音时长
    total_speech_duration = sum(end - start for start, end in merged_segments)
    
    print(f"✅ 静默检测完成")
    print(f"   发现语音片段数: {len(merged_segments)}")
    print(f"   总语音时长: {total_speech_duration:.2f}秒")
    print(f"   平均片段长度: {total_speech_duration/len(merged_segments):.2f}秒" if merged_segments else "   无语音片段")
    
    # 在测试模式下（max_duration不为None），过滤出指定时长内的语音段
    final_segments = merged_segments
    final_duration = total_speech_duration
    
    if max_duration:
        # 过滤出前max_duration秒内的语音段
        filtered = []
        for start, end in merged_segments:
            # 保留与测试时间段有重叠的语音段
            if not (end <= 0 or start >= max_duration):
                # 调整超出测试时间段的部分
                adjusted_start = max(start, 0)
                adjusted_end = min(end, max_duration)
                filtered.append([adjusted_start, adjusted_end])
        
        final_segments = filtered
        final_duration = sum(end - start for start, end in final_segments)
        
        # 如果是测试模式，显示过滤后的统计信息
        if final_segments != merged_segments:
            print(f"🔬 测试模式：在{max_duration}秒内发现{len(final_segments)}个语音段")
    
    # 保存语音段信息到临时文件
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    video_stem = Path(video_path).stem
    segments_file = temp_dir / f"{video_stem}_speech_segments.json"
    
    with open(segments_file, 'w', encoding='utf-8') as f:
        json.dump({
            'segments': final_segments,
            'total_speech_duration': final_duration,
            'video_path': video_path,
            'is_test_mode': max_duration is not None,
            'test_duration': max_duration
        }, f, ensure_ascii=False, indent=2)
    
    print(f"💾 语音段信息已保存到: {segments_file}")
    
    return merged_segments

def extract_audio_segment(video_path, output_path, segment_duration=None, optimize_for_low_speech=False):
    """
    提取音频片段（支持测试模式和低语音量优化模式）
    
    Args:
        video_path: 视频文件路径
        output_path: 输出音频路径
        segment_duration: 测试模式下提取的音频时长（秒）
        optimize_for_low_speech: 是否针对低语音量场景优化（进行静默检测）
        
    Returns:
        tuple: (成功标志, 语音段信息或None)
    """
    print("🎵 开始音频处理...")
    
    speech_segments = None
    
    # 针对低语音量场景的优化模式
    if optimize_for_low_speech:
        print("⚡ 启用低语音量优化模式，进行静默检测...")
        
        # 首先进行静默检测（传入max_duration支持测试模式）
        speech_segments = detect_speech_segments(video_path, max_duration=segment_duration)
    
    # 测试模式处理（优先处理，确保始终限制时长）
    if segment_duration:
        # 如果启用了低语音量优化且语音片段较少，合并提取这些片段
        if optimize_for_low_speech and speech_segments and len(speech_segments) <= 10:
            print(f"🔄 语音片段较少（{len(speech_segments)}个），合并提取...")
            
            # 构建concat文件
            concat_file = output_path + '.txt'
            temp_segments = []
            
            with open(concat_file, 'w', encoding='utf-8') as f:
                for i, (start, end) in enumerate(speech_segments):
                    temp_segment = f"{output_path}.part{i}.wav"
                    temp_segments.append(temp_segment)
                    
                    # 提取单个语音段
                    segment_cmd = f'ffmpeg -i "{video_path}" -ss {start} -to {end} -vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{temp_segment}"'
                    segment_result = safe_subprocess_run(segment_cmd)
                    
                    if segment_result.returncode == 0:
                        f.write(f"file '{os.path.abspath(temp_segment)}'\n")
                        print(f"   ✅ 提取语音段 {i+1}/{len(speech_segments)}: {start:.2f}s - {end:.2f}s")
                    else:
                        print(f"   ❌ 提取语音段 {i+1} 失败: {segment_result.stderr}")
            
            # 合并所有语音段
            merge_cmd = f'ffmpeg -f concat -safe 0 -i "{concat_file}" -c copy -y "{output_path}"'
            merge_result = safe_subprocess_run(merge_cmd)
            
            # 清理临时文件
            try:
                os.remove(concat_file)
                for temp_segment in temp_segments:
                    if os.path.exists(temp_segment):
                        os.remove(temp_segment)
            except Exception as e:
                print(f"⚠️ 清理临时文件时出错: {e}")
            
            if merge_result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                total_speech_duration = sum(end - start for start, end in speech_segments)
                print(f"✅ 合并语音段成功: {output_path}")
                print(f"   📊 优化效果: 总语音时长 {total_speech_duration:.2f}秒 (相比原视频大幅减少)")
                return True, speech_segments
            else:
                print(f"❌ 合并语音段失败: {merge_result.stderr}")
                # 合并失败时回退到标准测试模式提取
        
        # 标准测试模式：提取前N秒音频（确保始终限制时长）
        cmd = f'ffmpeg -i "{video_path}" -t {segment_duration} -vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{output_path}"'
        print(f"🔬 测试模式：提取前 {segment_duration} 秒音频")
        
        # 执行音频提取
        result = safe_subprocess_run(cmd)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ 测试模式音频提取成功: {output_path}")
            return True, None
        else:
            print(f"❌ 音频提取失败: {result.stderr}")
            return False, None
    
    # 非测试模式下的低语音量优化
    if optimize_for_low_speech and not segment_duration:
        if not speech_segments or len(speech_segments) == 0:
            print("⚠️ 未检测到语音片段，使用完整音频提取作为备选")
            # 如果没有检测到语音，回退到完整音频提取
            cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{output_path}"'
            result = safe_subprocess_run(cmd)
            
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ 回退到完整音频提取成功: {output_path}")
                return True, None
            else:
                print(f"❌ 音频提取失败: {result.stderr}")
                return False, None
        
        # 如果语音片段很少（少于10个），直接合并提取这些片段
        if len(speech_segments) <= 10:
            print(f"🔄 语音片段较少（{len(speech_segments)}个），合并提取...")
            
            # 构建concat文件
            concat_file = output_path + '.txt'
            temp_segments = []
            
            with open(concat_file, 'w', encoding='utf-8') as f:
                for i, (start, end) in enumerate(speech_segments):
                    temp_segment = f"{output_path}.part{i}.wav"
                    temp_segments.append(temp_segment)
                    
                    # 提取单个语音段
                    segment_cmd = f'ffmpeg -i "{video_path}" -ss {start} -to {end} -vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{temp_segment}"'
                    segment_result = safe_subprocess_run(segment_cmd)
                    
                    if segment_result.returncode == 0:
                        f.write(f"file '{os.path.abspath(temp_segment)}'\n")
                        print(f"   ✅ 提取语音段 {i+1}/{len(speech_segments)}: {start:.2f}s - {end:.2f}s")
                    else:
                        print(f"   ❌ 提取语音段 {i+1} 失败: {segment_result.stderr}")
            
            # 合并所有语音段
            merge_cmd = f'ffmpeg -f concat -safe 0 -i "{concat_file}" -c copy -y "{output_path}"'
            merge_result = safe_subprocess_run(merge_cmd)
            
            # 清理临时文件
            try:
                os.remove(concat_file)
                for temp_segment in temp_segments:
                    if os.path.exists(temp_segment):
                        os.remove(temp_segment)
            except Exception as e:
                print(f"⚠️ 清理临时文件时出错: {e}")
            
            if merge_result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                total_speech_duration = sum(end - start for start, end in speech_segments)
                print(f"✅ 合并语音段成功: {output_path}")
                print(f"   📊 优化效果: 总语音时长 {total_speech_duration:.2f}秒 (相比原视频大幅减少)")
                return True, speech_segments
            else:
                print(f"❌ 合并语音段失败: {merge_result.stderr}")
                return False, speech_segments
        else:
            print(f"⚠️ 语音片段较多（{len(speech_segments)}个），回退到完整音频提取")
    
    # 默认模式：提取完整音频
    cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{output_path}"'
    print("📤 提取完整音频...")
    
    # 执行音频提取
    result = safe_subprocess_run(cmd)
    
    if result.returncode == 0:
        # 验证输出文件
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ 音频提取成功: {output_path}")
            return True, None
        else:
            print(f"❌ 音频文件创建失败或为空")
            return False, None
    else:
        print(f"❌ 音频提取失败: {result.stderr}")
        
        # 尝试备用命令格式
        print("🔄 尝试备用命令格式...")
        cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{output_path}"'
        
        result = safe_subprocess_run(cmd)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ 备用命令音频提取成功: {output_path}")
            return True, None
        else:
            print(f"❌ 备用命令也失败: {result.stderr}")
            return False, None

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
