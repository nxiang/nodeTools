"""
字幕生成模块
负责语音识别、字幕格式化和双语字幕生成
"""

import os
import time
import numpy as np
import wave
import contextlib
import threading
from datetime import datetime
from pathlib import Path

# 导入自定义模块
from translator import save_translation_cache, load_translation_cache, set_current_video_name, baidu_translate, batch_translate
from progress_manager import save_progress, load_progress, get_progress_file_path

# 使用translator模块中的缓存
_translation_cache = load_translation_cache()

def transcribe_with_whisper(model, audio_path, model_size='medium'):
    """使用Whisper进行语音识别"""
    # 记录开始时间
    transcribe_start_time = time.time()
    print(f"🎤 使用Whisper {model_size}模型进行日语识别...")
    
    # 已在文件顶部导入必要的库
    
    try:
        # 获取音频文件时长（用于信息显示，但不再用于估计进度百分比）
        audio_duration = 0
        try:
            with contextlib.closing(wave.open(audio_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                audio_duration = frames / float(rate)
                print(f"🎵 音频时长: {audio_duration:.2f}秒")
        except Exception as e:
            print(f"⚠️ 无法获取音频时长: {e}")
        
        # 使用线程来显示实时活动指示器，不再显示不准确的进度百分比
        stop_event = threading.Event()
        
        def activity_thread():
            start_time = time.time()
            # 动画字符，用于显示活动状态
            activity_chars = ["◐", "◑", "◒", "◓", "◔", "◕"]
            char_index = 0
            
            # 加载状态信息
            status_messages = [
                "正在加载音频数据...",
                "正在分析音频特征...",
                "正在进行语音识别...",
                "正在处理识别结果..."
            ]
            status_index = 0
            status_update_time = 0
            
            while not stop_event.is_set():
                elapsed = time.time() - start_time
                char_index = (char_index + 1) % len(activity_chars)
                
                # 每5秒更新一次状态信息
                if elapsed - status_update_time > 5:
                    status_index = (status_index + 1) % len(status_messages)
                    status_update_time = elapsed
                
                # 显示加载动画和状态信息
                bar_length = 50
                # 使用波浪形进度条来表示活动状态
                wave_position = int(elapsed * 2) % bar_length
                bar = " " * (wave_position - 2) + activity_chars[char_index] * 3 + " " * (bar_length - wave_position - 1)
                
                # 显示经过时间，让用户了解处理持续时间
                minutes, seconds = divmod(int(elapsed), 60)
                
                print(f"\r🔄 处理中 {bar} {status_messages[status_index]} ({minutes:02d}:{seconds:02d})", end="", flush=True)
                time.sleep(0.2)  # 每200毫秒更新一次，更流畅的动画效果
        
        # 启动活动线程
        thread = threading.Thread(target=activity_thread)
        thread.daemon = True
        thread.start()
        
        try:
            # 执行语音识别（不使用不支持的progress_callback参数）
            result = model.transcribe(audio_path, language='ja')
            
            # 停止活动线程
            stop_event.set()
            thread.join(timeout=0.5)
            
            # 完成时显示确认信息，不再显示百分比
            print(f"\r✅ 语音识别处理完成 [{'█' * 50}]"),
            
            # 验证识别结果
            if result and 'segments' in result and len(result['segments']) > 0:
                print(f"✅ 语音识别完成: {len(result['segments'])} 个片段")
                
                # 显示识别结果摘要
                total_duration = sum(segment['end'] - segment['start'] for segment in result['segments'])
                print(f"📊 识别结果摘要:")
                print(f"   识别片段数: {len(result['segments'])}")
                print(f"   总识别时长: {total_duration:.2f}秒")
                
                # 显示前5个片段示例
                print(f"📋 前5个片段示例:")
                for i, segment in enumerate(result['segments'][:5]):
                    text = segment['text'].strip()
                    if len(text) > 50:
                        text = text[:47] + "..."
                    print(f"   {i+1}. [{format_time(segment['start'])}] {text}")
                
                # 记录结束时间并计算总耗时
                transcribe_end_time = time.time()
                transcribe_total_time = transcribe_end_time - transcribe_start_time
                print(f"⏱️ 语音识别耗时: {transcribe_total_time:.2f}秒")
                
                return result
            else:
                print("❌ 语音识别失败：无有效片段")
                return None
        except Exception as e:
            # 发生异常时停止活动线程
            stop_event.set()
            thread.join(timeout=0.5)
            print(f"\r❌ 处理中断")
            raise e
            
    except Exception as e:
        print(f"\n❌ 语音识别异常: {e}")
        return None

# 翻译相关函数已在顶部导入

def generate_bilingual_subtitle_file(video_path, transcription_result, 
                                   enable_translation=True, adult_content=False, progress=None, 
                                   time_offset=0.0):
    """生成双语字幕文件
    
    Args:
        video_path: 视频文件路径
        transcription_result: 语音识别结果
        enable_translation: 是否启用翻译
        adult_content: 是否为成人内容
        progress: 进度信息
        time_offset: 字幕时间偏移（秒），正值表示字幕延迟，负值表示字幕提前
    """
    # 记录开始时间
    subtitle_start_time = time.time()
    
    # 更新全局时间偏移参数
    global SUBTITLE_TIME_OFFSET
    SUBTITLE_TIME_OFFSET = time_offset
    
    # 获取当前时间作为开始处理时间
    start_time = time.time()
    print(f"🔄 开始生成双语字幕，视频路径: {video_path}")
    # 确保设置了当前视频名称
    set_current_video_name(video_path)
    
    if not transcription_result or 'segments' not in transcription_result:
        print("❌ 无效的识别结果")
        return False
    
    segments = transcription_result['segments']
    total_segments = len(segments)
    
    if total_segments == 0:
        print("❌ 无识别片段")
        return False
    
    # 确定输出路径
    video_name = Path(video_path).stem
    output_path = f"temp/{video_name}.srt"
    
    # 确保输出目录存在
    os.makedirs("temp", exist_ok=True)
    
    # 初始化SRT内容
    srt_content = ""
    start_index = 0
    
    # 进度恢复逻辑
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
    
    # 批量翻译设置
    MAX_CHARS_PER_BATCH = 5000  # 进一步增加批次大小，合并更多文本以提高语义连贯性
    separator = "<>"  # 使用<>作为分隔符
    
    # batch_translate已在顶部导入
    
    # 生成双语SRT格式字幕
    i = start_index
    while i < total_segments:
        # 准备批量翻译的文本（基于字符数限制）
        batch_segments = []
        batch_japanese_texts = []
        valid_indices = []
        current_char_count = 0
        
        # 收集不超过字符限制的文本
        for j in range(i, total_segments):
            segment = segments[j]
            japanese_text = segment['text'].strip()
            
            # 计算添加这个文本后可能的总字符数（包括分隔符）
            segment_char_count = len(japanese_text)
            if batch_japanese_texts:  # 如果不是第一个元素，需要加上分隔符
                segment_char_count += len(separator)
            
            # 检查是否超过字符限制
            if current_char_count + segment_char_count > MAX_CHARS_PER_BATCH:
                break
            
            # 添加到批次
            batch_segments.append(segment)
            if japanese_text:  # 只处理非空文本
                batch_japanese_texts.append(japanese_text)
                valid_indices.append(len(batch_japanese_texts) - 1)
            else:
                valid_indices.append(-1)  # 标记为空文本
            
            # 更新字符计数
            current_char_count += segment_char_count
        
        # 计算当前批次的结束索引
        batch_end = i + len(batch_segments)
        
        # 执行批量翻译
        if batch_japanese_texts:
            print(f"\n📦 批量翻译批次 {i//len(batch_segments)+1}: 处理{len(batch_segments)}个片段")
            print(f"📊 批量翻译模式: 启用缓存，优先检查缓存")
            
            # 优先使用批量翻译以保持语义连贯性
            batch_chinese_texts = ["" for _ in range(len(batch_japanese_texts))]  # 预初始化结果列表
            cached_count = 0
            
            # 先检查缓存状态
            for idx, japanese_text in enumerate(batch_japanese_texts):
                cache_key = f"jp:zh:{japanese_text}"
                if cache_key in _translation_cache:
                    cached_data = _translation_cache[cache_key]
                    # 处理不同格式的缓存数据
                    if isinstance(cached_data, dict):
                        # 从response_result中提取翻译结果
                        if 'response_result' in cached_data and 'trans_result' in cached_data['response_result']:
                            if cached_data['response_result']['trans_result']:
                                batch_chinese_texts[idx] = cached_data['response_result']['trans_result'][0].get('dst', '')
                            else:
                                batch_chinese_texts[idx] = ''
                        else:
                            # 兼容旧格式的dict缓存
                            batch_chinese_texts[idx] = cached_data.get('result', '')
                    else:
                        # 旧格式（直接存储结果字符串）
                        batch_chinese_texts[idx] = cached_data
                    cached_count += 1
            
            # 即使有缓存文本，也尝试批量翻译整个批次以保持更好的语义连贯性
            # 但只发送未缓存的文本，避免API返回不一致的结果
            uncached_texts = []
            uncached_indices = []
            for idx, japanese_text in enumerate(batch_japanese_texts):
                cache_key = f"jp:zh:{japanese_text}"
                if cache_key not in _translation_cache:
                    uncached_texts.append(japanese_text)
                    uncached_indices.append(idx)
            
            # 尝试批量翻译未缓存的文本，保持对话的语义连贯性
            if uncached_texts:
                print(f"📊 缓存命中: {cached_count}/{len(batch_japanese_texts)}，剩余{len(uncached_texts)}个文本需要翻译")
                print(f"🔍 批量翻译触发: 优先使用批量翻译保持语义连贯性")
                print(f"📦 批量翻译文本列表: {uncached_texts}")
                
                # 优先尝试批量翻译所有未缓存的文本
                try:
                    # 正确调用批量翻译API，添加show_individual_logs=False参数以隐藏单独翻译日志
                    combined_result = batch_translate(uncached_texts, False, show_individual_logs=False)  # 默认为非成人内容，隐藏单独翻译日志
                    
                    # 检查返回结果类型，如果是字符串则进行分割
                    if isinstance(combined_result, str):
                        # 如果返回的是单个字符串，尝试用分隔符分割
                        api_translated = [text.strip() for text in combined_result.split(separator) if text.strip()]
                    else:
                        # 如果已经是列表，直接使用
                        api_translated = combined_result
                    
                    # 清理每个翻译结果中的<SEP>分隔符，确保输出干净
                    api_translated = [text.replace(separator, '') for text in api_translated]
                    
                    print(f"🔍 批量翻译返回处理后: {api_translated}")
                    
                    # 检查批量翻译返回结果是否有效
                    if api_translated and len(api_translated) == len(uncached_texts):
                        # 批量翻译成功，将结果填充到正确位置
                        for text_idx, idx in enumerate(uncached_indices):
                            japanese_text = batch_japanese_texts[idx]
                            cache_key = f"jp:zh:{japanese_text}"
                            batch_chinese_texts[idx] = api_translated[text_idx]
                            # 保存到缓存 - 只保留百度API的请求参数和响应结果格式
                            _translation_cache[cache_key] = {
                                'request_params': {
                                    'q': japanese_text,
                                    'from': 'jp',
                                    'to': 'zh'
                                },
                                'response_result': {
                                    'from': 'jp',
                                    'to': 'zh',
                                    'trans_result': [{'src': japanese_text, 'dst': api_translated[text_idx]}]
                                }
                            }
                            print(f"✅ 批量翻译填充: 日语'{japanese_text[:30]}{'...' if len(japanese_text) > 30 else ''}' -> 中文'{api_translated[text_idx][:30]}{'...' if len(api_translated[text_idx]) > 30 else ''}'")
                        
                        print(f"✅ 批量翻译成功: 处理了{len(uncached_texts)}个文本，保持了语义连贯性")
                        print(f"🔄 批量翻译策略: 保持对话语义连贯性，优化翻译质量")
                    elif api_translated:
                        # 批量翻译结果部分可用
                        print(f"⚠️ 批量翻译结果数量不匹配: {len(api_translated)} != {len(uncached_texts)}")
                        print(f"🔄 优先处理批量翻译成功的部分，剩余部分降级到单独翻译")
                        
                        # 使用可用的批量翻译结果（保持批量优先原则）
                        successful_batch_count = 0
                        for text_idx, idx in enumerate(uncached_indices):
                            if text_idx < len(api_translated):
                                japanese_text = batch_japanese_texts[idx]
                                cache_key = f"jp:zh:{japanese_text}"
                                batch_chinese_texts[idx] = api_translated[text_idx]
                                # 仍然保存到缓存 - 使用正确的格式
                                _translation_cache[cache_key] = {
                                    'request_params': {
                                        'q': japanese_text,
                                        'from': 'jp',
                                        'to': 'zh'
                                    },
                                    'response_result': {
                                        'from': 'jp',
                                        'to': 'zh',
                                        'trans_result': [{'src': japanese_text, 'dst': api_translated[text_idx]}]
                                    }
                                }
                                print(f"✅ 使用批量翻译结果: {japanese_text[:30]}{'...' if len(japanese_text) > 30 else ''} -> {api_translated[text_idx][:30]}{'...' if len(api_translated[text_idx]) > 30 else ''}")
                                successful_batch_count += 1
                        
                        # 对于超出部分，作为批量翻译失败的降级处理
                        failed_batch_count = len(uncached_texts) - successful_batch_count
                        if failed_batch_count > 0:
                            print(f"📊 批量翻译部分成功({successful_batch_count}/{len(uncached_texts)})，开始降级处理剩余{failed_batch_count}个文本")
                            for text_idx, idx in enumerate(uncached_indices[successful_batch_count:]):
                                try:
                                    japanese_text = batch_japanese_texts[idx]
                                    cache_key = f"jp:zh:{japanese_text}"
                                    print(f"🔄 降级处理: {japanese_text[:30]}{'...' if len(japanese_text) > 30 else ''}")
                                    chinese_text = baidu_translate(japanese_text, max_retries=3)
                                    # 确保单独翻译结果也干净
                                    chinese_text = chinese_text.replace(separator, '')
                                    batch_chinese_texts[idx] = chinese_text
                                    # 使用baidu_translate函数已经保存了正确格式的缓存，这里不需要重复保存
                                except Exception as inner_e:
                                    print(f"❌ 降级翻译失败: {japanese_text[:30]}... - {inner_e}")
                                    batch_chinese_texts[idx] = "[翻译失败]"
                except Exception as e:
                    # 批量翻译异常，这是预期外的错误情况，进行降级处理
                    print(f"❌ 批量翻译异常: {e}")
                    print(f"🔄 按设计降级到单独翻译作为备选方案")
                    print(f"📊 批量翻译策略: 批量优先保证语义连贯，单独翻译作为降级备份")
                    
                    # 严格作为批量翻译失败的降级处理
                    success_count = 0
                    for idx in uncached_indices:
                        japanese_text = batch_japanese_texts[idx]
                        cache_key = f"jp:zh:{japanese_text}"
                        try:
                            print(f"🔄 降级翻译: {japanese_text[:30]}{'...' if len(japanese_text) > 30 else ''}")
                            # 对于降级翻译，增加重试次数以提高成功率
                            chinese_text = baidu_translate(japanese_text, max_retries=5)
                            # 清理单独翻译结果中的<SEP>分隔符
                            chinese_text = chinese_text.replace(separator, '')
                            batch_chinese_texts[idx] = chinese_text
                            success_count += 1
                            # 使用baidu_translate函数已经保存了正确格式的缓存，这里不需要重复保存
                        except Exception as inner_e:
                            print(f"❌ 降级翻译失败: {japanese_text[:30]}... - {inner_e}")
                            batch_chinese_texts[idx] = "[翻译失败]"
                    
                    print(f"📊 降级翻译完成: 成功{success_count}/{len(uncached_indices)}个文本")
            else:
                # 所有文本都在缓存中
                if batch_count == 0:
                    print(f"✅ 全部使用缓存，开始生成字幕")
                # 确保batch_chinese_texts已正确初始化
                if not batch_chinese_texts:
                    batch_chinese_texts = [_translation_cache.get(f"jp:zh:{text}", "") for text in batch_japanese_texts]
            
            # 减少缓存保存频率
            if len(_translation_cache) % 100 == 0 and len(_translation_cache) > 0:
                save_translation_cache(_translation_cache)
            
            # 处理每个翻译结果
            for idx, segment in enumerate(batch_segments):
                global_index = i + idx
                start_time = format_time(segment['start'])
                end_time = format_time(segment['end'])
                japanese_text = segment['text'].strip()
                
                if valid_indices[idx] != -1 and valid_indices[idx] < len(batch_chinese_texts):
                    chinese_text = batch_chinese_texts[valid_indices[idx]]
                    # 移除详细的翻译结果日志
                else:
                    chinese_text = ""  # 空文本处理
                
                srt_content += f"{global_index+1}\n"
                srt_content += f"{start_time} --> {end_time}\n"
                srt_content += f"<font size=\"12\" color=\"#FFD700\">{japanese_text}</font>\n"
                srt_content += f"<font size=\"16\" color=\"#FFFFFF\">{chinese_text}</font>\n\n"
        else:
            # 处理空批次（只有空文本）
            for idx, segment in enumerate(batch_segments):
                global_index = i + idx
                start_time = format_time(segment['start'])
                end_time = format_time(segment['end'])
                japanese_text = segment['text'].strip()
                
                srt_content += f"{global_index+1}\n"
                srt_content += f"{start_time} --> {end_time}\n"
                srt_content += f"<font size=\"12\" color=\"#FFD700\">{japanese_text}</font>\n"
                srt_content += f"<font size=\"16\" color=\"#FFFFFF\"></font>\n\n"
        
        # 更新进度
        i = batch_end
        
        # 实时进度显示
        progress_percent = int(i / total_segments * 100)
        progress_bar_length = int(progress_percent / 2)
        progress_bar = "█" * progress_bar_length + " " * (50 - progress_bar_length)
        print(f"\r📊 翻译进度: [{progress_bar}] {progress_percent}% ({i}/{total_segments})", end="", flush=True)
        
        # 添加延迟避免请求过快，但对于批量翻译减少延迟以提高效率
        # 在完全复用翻译结果时（全部使用缓存）不添加延迟，提高处理速度
        # 明确检查是否有未缓存文本需要翻译
        if len(uncached_texts) > 0 and cached_count < len(batch_japanese_texts):
            # 只有在确实有文本需要通过API翻译时才添加延迟
            time.sleep(0.3)  # 批量翻译后稍微减少延迟
            print(f"⏱️ 添加翻译延迟: {0.3}秒 (存在{len(uncached_texts)}个未缓存文本)")
        else:
            # 完全复用翻译结果时，不添加任何延迟
            print("🚀 完全复用翻译缓存，无延迟处理")
        
        # 实时保存进度到磁盘（每批保存一次）
        if video_path:
            # 构建完整的进度数据，包含所有必要信息
            progress_data = {
                'video_path': video_path,
                'output_path': output_path,
                'last_translated_index': i,
                'srt_content': srt_content,
                'total_segments': total_segments,
                'progress_percent': progress_percent,
                'last_save_time': datetime.now().isoformat(),
                'transcription_result': transcription_result,  # 保存完整的识别结果以便恢复
                'status': 'translating'
            }
            
            # 尝试保存进度，如果失败则继续处理（不中断流程）
            save_success = save_progress(video_path, progress_data)
            if not save_success:
                print(f"⚠️ 警告：进度保存失败，继续处理当前批次")
    
    # 完成进度显示（修复格式字符串问题）
    print(f"\r📊 翻译进度: [" + "█" * 50 + f"] 100% ({total_segments}/{total_segments})")
    
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
    
    # 翻译完成后保存缓存（使用视频特定的缓存文件）
    if len(_translation_cache) > 0:
        # 确保设置了当前视频名称
        if video_path:
            set_current_video_name(video_path)
        save_translation_cache(_translation_cache)
    
    # 记录结束时间并计算总耗时
    subtitle_end_time = time.time()
    subtitle_total_time = subtitle_end_time - subtitle_start_time
    print(f"⏱️ 字幕生成耗时: {subtitle_total_time:.2f}秒")
    print(f"✅ 双语字幕文件已生成: {output_path}")
    return True

# 全局时间偏移参数（秒），可根据需要调整
SUBTITLE_TIME_OFFSET = 0.0  # 正值表示字幕延迟，负值表示字幕提前

def format_time(seconds):
    """将秒数格式化为SRT时间格式，支持时间偏移调整"""
    # 应用时间偏移，确保不会出现负时间
    adjusted_seconds = max(0, seconds + SUBTITLE_TIME_OFFSET)
    
    hours = int(adjusted_seconds // 3600)
    minutes = int((adjusted_seconds % 3600) // 60)
    adjusted_seconds = adjusted_seconds % 60
    milliseconds = int((adjusted_seconds - int(adjusted_seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(adjusted_seconds):02d},{milliseconds:03d}"

def generate_japanese_only_subtitle(transcription_result, output_path, time_offset=0.0):
    """仅生成日语字幕
    
    Args:
        transcription_result: 语音识别结果
        output_path: 输出文件路径
        time_offset: 字幕时间偏移（秒），正值表示字幕延迟，负值表示字幕提前
    """
    # 更新全局时间偏移参数
    global SUBTITLE_TIME_OFFSET
    original_offset = SUBTITLE_TIME_OFFSET  # 保存原始偏移值
    SUBTITLE_TIME_OFFSET = time_offset
    
    try:
        if not transcription_result or 'segments' not in transcription_result:
            print("❌ 无效的识别结果")
            return False
        
        segments = transcription_result['segments']
        
        srt_content = ""
        for i, segment in enumerate(segments):
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            japanese_text = segment['text'].strip()
            
            srt_content += f"{i+1}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{japanese_text}\n\n"    
    finally:
        # 恢复原始偏移值
        SUBTITLE_TIME_OFFSET = original_offset
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    print(f"✅ 日语字幕文件已生成: {output_path}")
    return True
