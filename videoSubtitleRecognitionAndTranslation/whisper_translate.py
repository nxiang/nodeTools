#!/usr/bin/env python3
"""
视频字幕识别与翻译工具
使用Whisper medium模型识别日语语音，并通过百度翻译API生成双语字幕
支持测试模式（仅处理前10%内容）
"""

import os
import sys
import time
import random
import hashlib
import requests
import argparse
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime
import whisper

def check_cpu_availability():
    """检查CPU信息"""
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
        logical_cpu_count = psutil.cpu_count(logical=True)  # 逻辑核心数
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1024**3
        
        return f"CPU: {cpu_count}核/{logical_cpu_count}线程, 内存: {memory_gb:.1f}GB"
    except ImportError:
        return "CPU模式（psutil未安装，无法获取详细信息）"

def setup_whisper_model(model_size='medium'):
    """设置Whisper模型（CPU模式）"""
    print("💻 使用CPU处理模式")
    
    # 设置缓存路径，避免重复下载
    cache_dir = os.path.expanduser('~/.cache/whisper')
    os.environ['WHISPER_CACHE_DIR'] = cache_dir
    
    print(f"📥 加载Whisper {model_size}模型...")
    
    # 预期的模型文件大小（字节）- 更新为实际大小
    expected_sizes = {
        'tiny': 75_572_083,
        'base': 142_000_000,
        'small': 466_000_000,
        'medium': 1_528_008_539,
        'large': 3_087_371_615,  # 修正为实际文件大小，并统一使用'large'作为参数
    }
    
    # 检查本地缓存
    model_file = os.path.join(cache_dir, f'{model_size}.pt')
    
    # 对于large模型，检查是否存在large-v3.pt文件
    if model_size == 'large' and not os.path.exists(model_file):
        large_v3_file = os.path.join(cache_dir, 'large-v3.pt')
        if os.path.exists(large_v3_file):
            print(f"🔄 发现large-v3.pt文件，创建符号链接为large.pt")
            try:
                # 创建符号链接或复制文件
                if os.name == 'nt':  # Windows系统
                    import shutil
                    shutil.copy2(large_v3_file, model_file)
                else:  # Unix系统
                    os.symlink(large_v3_file, model_file)
                print(f"✅ 已创建large.pt文件")
            except Exception as e:
                print(f"⚠️ 无法创建large.pt文件: {e}")
    
    if os.path.exists(model_file):
        # 验证文件完整性
        file_size = os.path.getsize(model_file)
        expected_size = expected_sizes.get(model_size, 0)
        
        if expected_size > 0 and file_size < expected_size * 0.9:
            print(f"⚠️ 模型文件可能损坏: {file_size:,} bytes < 预期 {expected_size:,} bytes")
            print("🗑️ 删除损坏文件并重新下载...")
            try:
                os.remove(model_file)
            except Exception as e:
                print(f"❌ 删除失败: {e}")
        else:
            print(f"✅ 使用本地缓存模型: {model_file}")
            print(f"📊 文件大小: {file_size:,} bytes")
    else:
        print(f"📡 下载模型到缓存目录: {cache_dir}")
    
    # 线程安全的模型加载
    try:
        # 设置线程异常处理
        import threading
        threading.excepthook = lambda args: print(f"⚠️ 线程异常: {args.exc_type.__name__}: {args.exc_value}")
        
        # 加载模型
        model = whisper.load_model(model_size, device="cpu")
        print("✅ 模型加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        # 尝试使用更小的模型作为备选
        if model_size != 'tiny':
            print(f"🔄 尝试使用更小的模型作为备选...")
            # 按大小顺序尝试备选模型
            model_priority = ['medium', 'small', 'base', 'tiny']
            current_index = model_priority.index(model_size) if model_size in model_priority else 0
            
            for next_model in model_priority[current_index + 1:]:
                print(f"  尝试 {next_model} 模型...")
                try:
                    return setup_whisper_model(next_model)
                except:
                    continue
        
        # 如果所有备选都失败，抛出异常
        raise e

# 百度翻译API配置
appid = '20251126002506386'
key = 'C0qK4IqU_KXjun3PhRum'

# 成人内容专业术语词典
ADULT_TERMS_DICT = {
    "おっぱい": "胸部",
    "ちんちん": "阴茎", 
    "まんこ": "阴道",
    "フェラ": "口交",
    "中出し": "内射",
    "絶頂": "高潮",
    "イク": "高潮",
    "感じる": "有感觉",
    "気持ちいい": "舒服",
    "もっと": "再",
    "ダメ": "不行",
    "やめて": "不要",
    "いく": "要去了",
    "気持ち": "感觉",
    "奥": "深处",
    "挿入": "插入",
    "発射": "射精",
    "精子": "精液"
}

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
        # 获取视频时长
        cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        result = safe_subprocess_run(cmd)
        
        if result.returncode == 0:
            try:
                total_duration = float(result.stdout.strip())
                test_duration = total_duration * 0.1  # 10% of total duration
                print(f"📏 视频总时长: {total_duration:.2f}秒，测试模式提取: {test_duration:.2f}秒")
                
                # 提取前10%的音频
                cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 -t {test_duration} -y "{output_path}"'
            except (ValueError, TypeError) as e:
                print(f"⚠️ 无法解析视频时长，使用默认10秒测试片段: {e}")
                cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 -t 10 -y "{output_path}"'
        else:
            print("⚠️ 无法获取视频时长，使用默认10秒测试片段")
            cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 -t 10 -y "{output_path}"'
    else:
        # 提取完整音频
        cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{output_path}"'
    
    # 使用安全的子进程执行函数
    result = safe_subprocess_run(cmd)
    
    if result.returncode == 0:
        print("✅ 音频提取完成")
        return True
    else:
        print("❌ 音频提取失败")
        print(f"错误信息: {result.stderr}")
        return False

def transcribe_with_whisper(audio_path, model=None, model_size='medium', language='ja', video_path=None):
    """使用Whisper进行语音识别（CPU优化模式，支持进度显示和断点续传）"""
    print(f"🎤 使用Whisper {model_size}模型进行日语识别...")
    
    try:
        # 如果未提供模型，则加载模型
        if model is None:
            model = setup_whisper_model(model_size)
        
        # 加载音频文件
        audio = whisper.load_audio(audio_path)
        
        # CPU优化转录参数
        transcription_params = {
            'audio': audio,
            'language': language,
            'task': 'transcribe',
            'word_timestamps': True,
            'temperature': 0.0,
            'best_of': 2,  # 减少候选数量以提高速度
            'beam_size': 2,  # 减少束搜索大小
            'fp16': False,  # CPU模式下不使用fp16
            'no_speech_threshold': 0.6,  # 提高无语音检测阈值
            'compression_ratio_threshold': 2.4  # 调整压缩比阈值
        }
        
        # 显示进度条
        print("🔍 语音识别进度: [" + "█" * 0 + " " * 50 + "] 0%")
        
        # 使用线程来显示进度（基于音频时长的更准确估计）
        import threading
        import time
        import os
        progress_value = 0.0
        is_running = True
        
        def progress_monitor():
            nonlocal progress_value
            
            # 获取音频文件大小，用于更准确地估算进度
            try:
                file_size = os.path.getsize(audio_path)
                estimated_chunks = max(1, file_size // (1024 * 1024))  # 每MB一个估计块
            except:
                estimated_chunks = 20  # 默认估计块数
            
            chunk_size = 1.0 / estimated_chunks
            
            while is_running and progress_value < 0.95:
                # 基于文件大小的更合理进度估算
                # 小文件块数少，进度增长快；大文件块数多，进度增长慢
                progress_value = min(progress_value + chunk_size, 0.95)
                progress_percent = int(progress_value * 100)
                filled_bars = int(progress_value * 50)
                empty_bars = 50 - filled_bars
                print(f"\r🔍 语音识别进度: [" + "█" * filled_bars + " " * empty_bars + f"] {progress_percent}%", end="", flush=True)
                
                # 如果提供了视频路径，实时更新断点续传文件
                if video_path:
                    progress_data = {
                        'video_path': video_path,
                        'model_size': model_size,
                        'transcription_progress': progress_value,
                        'last_update_time': datetime.now().isoformat(),
                        'status': 'transcribing'
                    }
                    save_progress(video_path, progress_data)
                
                # 动态调整更新间隔，大文件更新更快
                update_interval = max(0.5, 3.0 - (estimated_chunks / 10))
                time.sleep(update_interval)
        
        # 启动进度监控线程
        progress_thread = threading.Thread(target=progress_monitor)
        progress_thread.daemon = True
        progress_thread.start()
        
        # 进行语音识别
        result = model.transcribe(**transcription_params)
        
        # 停止进度监控
        is_running = False
        progress_thread.join(timeout=1)
        
        # 识别完成，更新进度显示
        print(f"\r🔍 语音识别进度: [" + "█" * 50 + "] 100%")
        print("✅ Whisper识别完成")
        
        # 更新断点续传文件
        if video_path:
            progress_data = {
                'video_path': video_path,
                'model_size': model_size,
                'transcription_result': result,
                'transcription_completed': True,
                'last_update_time': datetime.now().isoformat(),
                'status': 'completed'
            }
            save_progress(video_path, progress_data)
            print(f"💾 识别进度已保存到断点续传文件")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Whisper识别失败: {e}")
        
        # 停止进度监控（如果正在运行）
        try:
            is_running = False
        except NameError:
            pass
        
        # 保存错误信息到断点续传文件
        if video_path:
            progress_data = {
                'video_path': video_path,
                'model_size': model_size,
                'error': str(e),
                'error_time': datetime.now().isoformat(),
                'status': 'error'
            }
            save_progress(video_path, progress_data)
            print(f"💾 错误信息已保存到断点续传文件")
        
        return None

def replace_adult_terms(text):
    """替换成人内容专业术语"""
    for jp_term, cn_term in ADULT_TERMS_DICT.items():
        text = text.replace(jp_term, cn_term)
    return text

# 翻译缓存字典
_translation_cache = {}
_translation_cache_file = "translation_cache.json"

# 尝试加载缓存文件
try:
    import os
    if os.path.exists(_translation_cache_file):
        import json
        with open(_translation_cache_file, 'r', encoding='utf-8') as f:
            _translation_cache = json.load(f)
        print(f"✅ 已加载翻译缓存，缓存条目数: {len(_translation_cache)}")
except Exception as e:
    print(f"⚠️  加载翻译缓存失败: {e}")
    _translation_cache = {}

def save_translation_cache():
    """保存翻译缓存到文件"""
    try:
        import json
        with open(_translation_cache_file, 'w', encoding='utf-8') as f:
            json.dump(_translation_cache, f, ensure_ascii=False, indent=2)
        print(f"💾 翻译缓存已保存，当前缓存条目数: {len(_translation_cache)}")
    except Exception as e:
        print(f"⚠️  保存翻译缓存失败: {e}")

def baidu_translate(text, from_lang='jp', to_lang='zh', max_retries=3):
    """使用百度翻译API翻译文本（带重试机制和缓存功能）"""
    if not appid or not key:
        print("❌ 请先配置百度翻译API的appid和key")
        return text
    
    # 如果文本为空或过短，直接返回
    if not text or len(text.strip()) < 2:
        return text
    
    # 检查文本是否已经主要是中文（超过50%的字符是中文），避免重复翻译
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    if chinese_chars > len(text) * 0.5:
        print(f"⚠️  文本已包含大量中文 ({chinese_chars}/{len(text)}), 跳过翻译")
        return text
    
    # 生成缓存键
    cache_key = f"{from_lang}:{to_lang}:{text}"
    
    # 检查缓存中是否已有翻译结果
    if cache_key in _translation_cache:
        cached_result = _translation_cache[cache_key]
        print(f"✅ 使用缓存的翻译结果: {text[:20]}{'...' if len(text) > 20 else ''}")
        return cached_result
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            salt = str(random.randint(32768, 65536))
            sign = hashlib.md5((appid + text + salt + key).encode()).hexdigest()
            
            url = "https://fanyi-api.baidu.com/ait/api/aiTextTranslate"
            params = {
                'q': text,
                'from': from_lang,
                'to': to_lang,
                'appid': appid,
                'salt': salt,
                'sign': sign
            }
            
            # 增加超时时间并设置重试间隔
            timeout = 15 + (attempt * 5)  # 每次重试增加超时时间
            
            # 尝试使用POST请求（新API可能需要POST）
            try:
                response = requests.post(url, data=params, timeout=timeout)
                result = response.json()
            except:
                # 如果POST失败，尝试GET请求
                response = requests.get(url, params=params, timeout=timeout)
                result = response.json()
            
            # 调试信息：打印API响应
            print(f"🔍 API响应: {result}")
            
            # 处理多种可能的API响应格式
            translated = None
            
            # 格式1: 标准百度翻译API格式
            if 'trans_result' in result and isinstance(result['trans_result'], list):
                if result['trans_result']:
                    translated = result['trans_result'][0].get('dst', '')
            
            # 格式2: 新AI翻译API格式
            elif 'result' in result and 'trans_result' in result['result']:
                if result['result']['trans_result']:
                    translated = result['result']['trans_result'][0].get('dst', '')
            
            # 格式3: 直接返回翻译结果
            elif 'dst' in result:
                translated = result['dst']
            
            # 格式4: 其他可能的格式
            elif 'translated_text' in result:
                translated = result['translated_text']
            
            if translated:
                # 对翻译结果进行术语替换
                translated_result = replace_adult_terms(translated)
                
                # 将翻译结果添加到缓存
                cache_key = f"{from_lang}:{to_lang}:{text}"
                _translation_cache[cache_key] = translated_result
                
                # 每10个新缓存条目保存一次
                if len(_translation_cache) % 10 == 0:
                    save_translation_cache()
                
                return translated_result
            else:
                error_msg = result.get('error_msg', result.get('message', result.get('error', '未知错误')))
                print(f"❌ 翻译失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                
                # 如果是API配额或认证问题，直接返回原文
                if 'quota' in str(error_msg).lower() or 'appid' in str(error_msg).lower() or 'sign' in str(error_msg).lower():
                    print("⚠️  API配额或认证问题，直接返回原文")
                    return text
                
                # 如果是最后一次尝试，返回原文
                if attempt == max_retries - 1:
                    print(f"❌ 翻译失败，最大重试次数已达，返回原文: {text}")
                    return text
                
                # 等待后重试
                time.sleep(2 ** attempt)  # 指数退避
                
        except requests.exceptions.Timeout:
            print(f"⏰ 翻译超时 (尝试 {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return text
            time.sleep(2 ** attempt)
            
        except requests.exceptions.ConnectionError:
            print(f"🌐 网络连接错误 (尝试 {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return text
            time.sleep(2 ** attempt)
            
        except Exception as e:
            print(f"❌ 翻译异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return text
            time.sleep(2 ** attempt)
    
    return text

def check_translation_quality(translated_text, original_text=None):
    """检查翻译质量，返回True表示质量良好，False表示需要重试"""
    # 如果翻译结果为空，说明翻译失败
    if not translated_text:
        return False
    
    # 如果提供了原文，检查是否与原文相同
    if original_text and translated_text == original_text:
        return False
    
    # 如果翻译结果包含大量日文字符，说明翻译可能失败
    japanese_chars = sum(1 for char in translated_text if '぀' <= char <= 'ヿ')
    if japanese_chars > len(translated_text) * 0.3:  # 超过30%的日文字符
        return False
    
    # 如果提供了原文，检查翻译结果是否过短
    if original_text and len(translated_text) < len(original_text) * 0.2:
        return False
    
    return True

def generate_bilingual_subtitle_file(transcription_result, output_path, video_path=None, adult_content=False):
    """生成双语字幕文件（日语+中文，支持断点续传和实时进度显示）"""
    print("📝 生成双语字幕文件...")
    print("🌐 使用百度翻译API翻译日语到中文...")
    print("🎨 字幕样式: 日语(12号金色) + 中文(16号白色)")
    
    try:
        segments = transcription_result.get('segments', [])
        total_segments = len(segments)
        
        if total_segments == 0:
            print("⚠️  没有可翻译的片段")
            return False
        
        # 加载进度（如果支持断点续传）
        start_index = 0
        srt_content = ""
        
        if video_path:
            progress = load_progress(video_path)
            if progress:
                # 检查是否已完成
                if progress.get('completed', False):
                    print(f"✅ 检测到已完成的翻译进度，直接使用保存的结果")
                    if 'srt_content' in progress:
                        srt_content = progress['srt_content']
                        # 直接写入文件并返回成功
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(srt_content)
                        print(f"✅ 双语字幕文件已生成: {output_path}")
                        return True
                
                # 加载翻译进度
                last_translated = progress.get('last_translated_index', 0)
                saved_srt = progress.get('srt_content', "")
                
                # 验证进度的有效性
                if 0 <= last_translated <= total_segments:
                    start_index = last_translated
                    # 只有当索引大于0时才使用保存的SRT内容（避免使用空内容覆盖）
                    if start_index > 0 and saved_srt.strip():
                        srt_content = saved_srt
                        print(f"🔄 从断点继续: 已翻译 {start_index}/{total_segments} 个片段")
                    else:
                        print("🔄 重新开始翻译: 进度文件中的内容无效或为空")
                else:
                    print(f"⚠️ 检测到无效的进度索引: {last_translated}，重新开始翻译")
        
        # 显示进度条初始化
        print("📊 翻译进度: [" + " " * 50 + "] 0%")
        
        # 生成双语SRT格式字幕
        for i in range(start_index, total_segments):
            segment = segments[i]
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            japanese_text = segment['text'].strip()
            
            if japanese_text:  # 只处理非空文本
                # 使用百度翻译API翻译日语到中文（带重试机制）
                chinese_text = baidu_translate(japanese_text, max_retries=5)
                
                # 检查翻译质量，如果翻译失败则重试
                if not check_translation_quality(chinese_text, japanese_text):
                    print(f"⚠️  翻译质量不佳，重试片段 {i+1}...")
                    chinese_text = baidu_translate(japanese_text, max_retries=3)
                
                # 添加延迟避免请求过快
                time.sleep(0.3)
                
                srt_content += f"{i+1}\n"
                srt_content += f"{start_time} --> {end_time}\n"
                srt_content += f"<font size=\"12\" color=\"#FFD700\">{japanese_text}</font>\n"
                srt_content += f"<font size=\"16\" color=\"#FFFFFF\">{chinese_text}</font>\n\n"
                
                # 实时进度显示
                progress_percent = int((i + 1) / total_segments * 100)
                progress_bar_length = int(progress_percent / 2)
                progress_bar = "█" * progress_bar_length + " " * (50 - progress_bar_length)
                print(f"\r📊 翻译进度: [{progress_bar}] {progress_percent}% ({i+1}/{total_segments})", end="", flush=True)
                
                # 实时保存进度到磁盘（每翻译1个片段保存一次）
                if video_path:
                    # 构建完整的进度数据，包含所有必要信息
                    progress_data = {
                        'video_path': video_path,
                        'output_path': output_path,
                        'last_translated_index': i + 1,
                        'srt_content': srt_content,
                        'total_segments': total_segments,
                        'progress_percent': progress_percent,
                        'last_save_time': datetime.now().isoformat(),
                        'transcription_result': transcription_result,  # 保存完整的识别结果以便恢复
                        'status': 'translating',
                        'current_segment': {
                            'index': i + 1,
                            'japanese_text': japanese_text,
                            'chinese_text': chinese_text,
                            'start_time': start_time,
                            'end_time': end_time
                        }
                    }
                    
                    # 尝试保存进度，如果失败则继续处理（不中断流程）
                    save_success = save_progress(video_path, progress_data)
                    if not save_success:
                        print(f"⚠️ 警告：进度保存失败，继续处理片段 {i+1}")
        
        # 完成进度显示
        print(f"\r📊 翻译进度: [" + "█" * 50 + "] 100% ({total_segments}/{total_segments})")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        # 不清理进度文件，保持断点续传文件
        if video_path:
            # 构建最终的完成状态进度数据
            final_progress_data = {
                'video_path': video_path,
                'output_path': output_path,
                'last_translated_index': total_segments,
                'srt_content': srt_content,
                'total_segments': total_segments,
                'progress_percent': 100,
                'completed': True,
                'completion_time': datetime.now().isoformat(),
                'subtitle_file': output_path,
                'status': 'completed',
                'transcription_result': transcription_result,  # 保存完整的识别结果
                'execution_summary': {
                    'total_translated_segments': total_segments,
                    'file_size': len(srt_content),
                    'completion_timestamp': datetime.now().isoformat()
                }
            }
            
            # 确保最终进度保存成功
            final_save_success = save_progress(video_path, final_progress_data)
            if final_save_success:
                print(f"💾 最终进度文件已保存: {get_progress_file_path(video_path)}")
            else:
                print(f"⚠️ 警告：最终进度保存失败，但字幕文件已生成")
        
        print(f"✅ 双语字幕文件已生成: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ 生成字幕文件失败: {e}")
        # 保存错误进度以便恢复
        if video_path:
            progress_data = {
                'last_translated_index': start_index,
                'srt_content': srt_content,
                'total_segments': total_segments,
                'progress_percent': int(start_index / total_segments * 100) if total_segments > 0 else 0,
                'error': str(e),
                'error_time': datetime.now().isoformat()
            }
            save_progress(video_path, progress_data)
            print(f"💾 错误进度已保存，可断点续传")
        return False

def format_time(seconds):
    """将秒数格式化为SRT时间格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"



def auto_select_model(video_path, user_model_size='medium'):
    """根据视频时长自动选择模型大小"""
    # 如果用户指定了模型大小，优先使用用户选择
    return user_model_size

def main(video_path=None, test_mode=True, model_size='medium', enable_translation=True, output_dir=None, adult_content=False, merge_to_video=False, clean_progress=False):
    """主函数"""
    # 显示CPU状态
    cpu_info = check_cpu_availability()
    print(f"🔍 系统检测: {cpu_info}")
    
    # 如果没有指定视频文件，查找当前目录下的视频文件
    if not video_path:
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv']
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = file
                break
        
        if not video_path:
            print("❌ 未找到视频文件，请指定视频文件路径")
            return
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 自动选择模型大小
    selected_model_size = auto_select_model(video_path, model_size)
    
    # 成人内容优化提示
    if adult_content:
        print("🔞 成人内容模式已启用")
        print(f"   - 使用专业术语词典优化翻译")
        print(f"   - 建议使用 {selected_model_size} 或更高精度模型")
    
    print(f"🚀 开始处理视频: {video_path}")
    print(f"🌐 识别语言: 日语 → {'中文' if enable_translation else '仅识别'}")
    print(f"🔬 测试模式: {'开启' if test_mode else '关闭'}")
    print(f"🔧 使用Whisper {selected_model_size}模型 {'+ 百度翻译API' if enable_translation else ''}")
    
    # 创建临时目录
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 首先检查是否有断点续传文件，避免重复识别
    progress = load_progress(video_path)
    result = None
    
    if progress and 'transcription_result' in progress:
        print("✅ 使用已保存的语音识别结果，跳过识别阶段")
        result = progress['transcription_result']
    else:
        # 提取音频
        audio_path = os.path.join(temp_dir, "audio.wav")
        if not extract_audio_segment(video_path, audio_path, segment_duration=test_mode):
            return
        
        # 使用Whisper进行语音识别（CPU模式，支持进度显示和断点续传）
        model = setup_whisper_model(selected_model_size)
        result = transcribe_with_whisper(audio_path, model=model, language='ja', video_path=video_path)
    
    if not result:
        return
    
    # 生成字幕文件（与视频文件同名且在同一目录）
    subtitle_path = get_same_dir_subtitle_path(video_path, enable_translation)
    
    # 检查是否有错误状态的进度文件
    if progress and 'error' in progress:
        print(f"🔄 检测到上次中断的进度，继续处理...")
        print(f"   错误信息: {progress.get('error', '未知错误')}")
        print(f"   错误时间: {progress.get('error_time', '未知时间')}")
    
    # 保存语音识别结果到进度文件
    progress_data = {
        'transcription_result': result,
        'video_path': video_path,
        'model_size': selected_model_size,
        'enable_translation': enable_translation,
        'save_time': datetime.now().isoformat(),
        'transcription_completed': True
    }
    save_progress(video_path, progress_data)
    print(f"💾 语音识别进度已保存: {get_progress_file_path(video_path)}")
    
    if enable_translation:
        success = generate_bilingual_subtitle_file(result, subtitle_path, video_path, adult_content=adult_content)
    else:
        # 仅生成日语字幕
        success = generate_japanese_subtitle_file(result, subtitle_path)
    
    if success:
        # 显示识别结果摘要
        segments = result.get('segments', [])
        total_duration = sum(segment['end'] - segment['start'] for segment in segments)
        
        print(f"\n📊 识别结果摘要:")
        print(f"   识别片段数: {len(segments)}")
        print(f"   总识别时长: {total_duration:.2f}秒")
        print(f"   字幕文件: {subtitle_path}")
        
        # 显示前几个识别片段作为示例
        print(f"\n📋 前5个片段示例:")
        for i, segment in enumerate(segments[:5]):
            japanese_text = segment['text'].strip()
            if enable_translation:
                chinese_text = baidu_translate(japanese_text)
                print(f"   {i+1}. [{format_time(segment['start'])}]")
                print(f"       日语: {japanese_text}")
                print(f"       中文: {chinese_text}")
                time.sleep(0.2)  # 避免请求过快
            else:
                print(f"   {i+1}. [{format_time(segment['start'])}]")
                print(f"       日语: {japanese_text}")
    
    # 字幕合并到视频
    if merge_to_video and success:
        print("\n🎬 开始字幕合并到视频...")
        if check_ffmpeg_installed():
            # 确定字幕语言
            subtitle_language = 'chi' if enable_translation else 'jpn'
            
            # 合并字幕到视频
            merge_success = merge_subtitle_to_video(
                video_path=video_path,
                subtitle_path=subtitle_path,
                subtitle_language=subtitle_language
            )
            
            if merge_success:
                print("✅ 字幕已成功嵌入视频文件中")
                print("💡 播放时可在播放器字幕菜单中选择内置字幕")
            else:
                print("⚠️ 字幕合并失败，保留独立的字幕文件")
        else:
            print("❌ FFmpeg未安装，无法合并字幕到视频")
            print("💡 请安装FFmpeg或使用外部播放器加载字幕文件")
    
    # 清理临时文件
    try:
        # 检查audio_path变量是否存在且文件存在
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
            print("🧹 临时文件已清理")
        else:
            print("📝 无临时音频文件需要清理")
    except Exception as e:
        print(f"⚠️ 清理临时文件时出错: {e}")
    
    # 进度文件管理
    progress_file = get_progress_file_path(video_path)
    if clean_progress:
        # 清理进度文件
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
                print("🧹 进度文件已清理")
            except Exception as e:
                print(f"⚠️ 无法清理进度文件: {e}")
    else:
        # 默认保留进度文件以便断点续传
        if os.path.exists(progress_file):
            print(f"📁 进度文件已保留: {progress_file}")
            print("💡 如需清理进度文件，请使用 --clean-progress 参数或手动删除")

def generate_japanese_subtitle_file(transcription_result, output_path):
    """生成仅日语字幕文件"""
    print("📝 生成日语字幕文件...")
    
    try:
        # 生成SRT格式字幕
        srt_content = ""
        segments = transcription_result.get('segments', [])
        
        for i, segment in enumerate(segments, 1):
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            japanese_text = segment['text'].strip()
            
            if japanese_text:  # 只处理非空文本
                srt_content += f"{i}\n"
                srt_content += f"{start_time} --> {end_time}\n"
                srt_content += f"{japanese_text}\n\n"
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        print(f"✅ 日语字幕文件已生成: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 生成字幕文件失败: {e}")
        return False

def merge_subtitle_to_video(video_path, subtitle_path, output_path=None, subtitle_language='chi'):
    """将字幕合并到视频文件中"""
    print(f"🎬 开始合并字幕到视频...")
    
    if not output_path:
        video_name = Path(video_path).stem
        output_path = f"{video_name}_with_subtitle.mp4"
    
    try:
        # 检查FFmpeg是否可用
        if not check_ffmpeg_installed():
            print("❌ FFmpeg未安装或不可用，无法合并字幕")
            print("💡 请安装FFmpeg：https://ffmpeg.org/download.html")
            return False
        
        # 构建FFmpeg命令
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', subtitle_path,
            '-c', 'copy',  # 复制视频和音频流
            '-c:s', 'mov_text',  # 字幕编码格式
            '-metadata:s:s:0', f'language={subtitle_language}',
            '-y',  # 覆盖输出文件
            output_path
        ]
        
        print(f"   📥 输入视频: {video_path}")
        print(f"   📄 输入字幕: {subtitle_path}")
        print(f"   📤 输出文件: {output_path}")
        print(f"   🔧 字幕语言: {subtitle_language}")
        
        # 执行合并
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 字幕合并成功: {output_path}")
            print(f"💡 播放时可在播放器中选择字幕轨道")
            return True
        else:
            print(f"❌ 字幕合并失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 合并过程中出现错误: {e}")
        return False

def check_ffmpeg_installed():
    """检查FFmpeg是否已安装"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def get_progress_file_path(video_path):
    """获取进度文件路径（保存在temp目录下）"""
    video_name = Path(video_path).stem
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"{video_name}_progress.json")

def save_progress(video_path, progress_data):
    """保存进度到文件（断点续传）"""
    progress_file = get_progress_file_path(video_path)
    try:
        # 确保临时目录存在
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        
        # 添加保存时间戳
        progress_data['last_save_timestamp'] = datetime.now().isoformat()
        
        # 分块写入以避免大文件操作问题
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        
        # 验证保存是否成功
        if os.path.exists(progress_file):
            file_size = os.path.getsize(progress_file)
            if file_size > 0:
                return True
            else:
                print(f"⚠️ 进度文件创建失败：文件为空")
                return False
        return False
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON编码错误，无法保存进度: {e}")
        return False
    except IOError as e:
        print(f"⚠️ IO错误，无法保存进度: {e}")
        return False
    except Exception as e:
        print(f"⚠️ 保存进度时发生未知错误: {e}")
        return False

def load_progress(video_path):
    """从文件加载进度（断点续传）"""
    progress_file = get_progress_file_path(video_path)
    if os.path.exists(progress_file):
        try:
            # 检查文件大小
            if os.path.getsize(progress_file) == 0:
                print(f"⚠️ 进度文件为空: {progress_file}")
                return None
                
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                
            # 验证进度数据的完整性
            if not isinstance(progress, dict):
                print("⚠️ 进度数据格式错误：不是有效的字典")
                return None
                
            # 检查关键字段是否存在（根据不同阶段的需求）
            if 'status' in progress:
                status = progress['status']
                if status == 'transcribing' and 'transcription_progress' not in progress:
                    print("⚠️ 转录阶段进度数据不完整")
                elif status == 'completed' and 'transcription_result' not in progress:
                    print("⚠️ 完成阶段进度数据不完整")
                    
            return progress
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析错误，无法加载进度文件: {e}")
            # 尝试清理损坏的进度文件
            try:
                os.remove(progress_file)
                print(f"🧹 已清理损坏的进度文件: {progress_file}")
            except:
                pass
        except IOError as e:
            print(f"⚠️ IO错误，无法读取进度文件: {e}")
        except Exception as e:
            print(f"⚠️ 加载进度时发生未知错误: {e}")
    return None

def cleanup_progress(video_path):
    """清理进度文件"""
    progress_file = get_progress_file_path(video_path)
    if os.path.exists(progress_file):
        try:
            os.remove(progress_file)
            print("🧹 进度文件已清理")
        except Exception as e:
            print(f"⚠️ 无法清理进度文件: {e}")

def get_same_dir_subtitle_path(video_path, enable_translation=True):
    """获取与视频文件同目录的字幕文件路径"""
    video_dir = Path(video_path).parent
    video_name = Path(video_path).stem
    
    if enable_translation:
        subtitle_name = f"{video_name}.srt"  # 与视频文件同名
    else:
        subtitle_name = f"{video_name}_japanese.srt"
    
    return str(video_dir / subtitle_name)



if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='视频字幕识别与翻译工具')
    parser.add_argument('video_path', nargs='?', help='视频文件路径')
    parser.add_argument('--test', action='store_true', help='测试模式（仅处理前10%内容）')
    parser.add_argument('--model', default='medium', choices=['tiny', 'base', 'small', 'medium', 'large'], 
                       help='Whisper模型大小（默认：medium）')
    parser.add_argument('--no-translate', action='store_true', help='仅识别不翻译')
    parser.add_argument('--output-dir', help='输出目录')
    parser.add_argument('--adult', action='store_true', help='成人内容模式（优化专业术语翻译）')
    parser.add_argument('--merge', action='store_true', help='将字幕合并到视频文件中（需要FFmpeg）')
    parser.add_argument('--clean-progress', action='store_true', help='清理进度文件（默认保留）')
    
    args = parser.parse_args()
    
    # 模型推荐：成人内容建议使用medium或large
    if args.adult and args.model in ['tiny', 'base', 'small']:
        print("⚠️  成人内容建议使用medium或large模型以获得更好的识别精度")
    
    # 检查FFmpeg是否已安装（如果启用了合并功能）
    if args.merge and not check_ffmpeg_installed():
        print("⚠️  FFmpeg未安装，字幕合并功能将不可用")
        print("💡  请安装FFmpeg：https://ffmpeg.org/download.html")
        print("💡  或者使用外部播放器加载独立的字幕文件")
    
    try:
        main(
            video_path=args.video_path,
            test_mode=args.test,
            model_size=args.model,
            enable_translation=not args.no_translate,
            output_dir=args.output_dir,
            adult_content=args.adult,
            merge_to_video=args.merge,
            clean_progress=args.clean_progress
        )
    except KeyboardInterrupt:
        print("\n\n🛑 程序已被用户中断")
    except Exception as e:
        print(f"\n\n❌ 程序运行时发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 程序结束时保存翻译缓存
        save_translation_cache()
