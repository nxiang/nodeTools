"""
字幕生成模块
负责语音识别、字幕格式化和双语字幕生成
"""

import os
import time
from datetime import datetime
from pathlib import Path

# 全局翻译缓存（在主模块中定义）
_translation_cache = {}

# 导入自定义模块
from translator import save_translation_cache, load_translation_cache, set_current_video_name, baidu_translate
from progress_manager import save_progress, load_progress, get_progress_file_path

# 程序启动时加载翻译缓存
_translation_cache = load_translation_cache()

def transcribe_with_whisper(model, audio_path, model_size='medium'):
    """使用Whisper进行语音识别"""
    print(f"🎤 使用Whisper {model_size}模型进行日语识别...")
    
    try:
        # 执行语音识别
        result = model.transcribe(audio_path, language='ja')
        
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
            
            return result
        else:
            print("❌ 语音识别失败：无有效片段")
            return None
            
    except Exception as e:
        print(f"❌ 语音识别异常: {e}")
        return None

# 翻译缓存函数已在顶部导入

def generate_bilingual_subtitle_file(video_path, transcription_result, 
                                   enable_translation=True, adult_content=False, progress=None):
    """生成双语字幕文件"""
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
    
    # 导入翻译函数
    from translator import batch_translate, check_translation_quality, baidu_translate
    
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
                    # 调用批量翻译API
                    combined_result = batch_translate(uncached_texts, separator)
                    
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
                            print(f"✅ 批量翻译填充: 日语'{japanese_text}' -> 中文'{api_translated[text_idx]}'")
                        
                        print(f"✅ 批量翻译成功: 处理了{len(uncached_texts)}个文本，保持了语义连贯性")
                        print(f"🔄 批量翻译策略: 保持对话语义连贯性，优化翻译质量")
                    elif api_translated:
                        # 批量翻译结果部分可用
                        print(f"⚠️ 批量翻译结果数量不匹配: {len(api_translated)} != {len(uncached_texts)}")
                        
                        # 使用可用的批量翻译结果
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
                                print(f"✅ 使用批量翻译部分结果: {japanese_text[:30]}... -> {api_translated[text_idx][:30]}...")
                            else:
                                # 对于超出部分，使用单独翻译
                                try:
                                    japanese_text = batch_japanese_texts[idx]
                                    cache_key = f"jp:zh:{japanese_text}"
                                    chinese_text = baidu_translate(japanese_text, max_retries=3)
                                    # 确保单独翻译结果也干净
                                    chinese_text = chinese_text.replace(separator, '')
                                    batch_chinese_texts[idx] = chinese_text
                                    # 使用baidu_translate函数已经保存了正确格式的缓存，这里不需要重复保存
                                    print(f"✅ 单独翻译: {japanese_text[:30]}... -> {chinese_text[:30]}...")
                                except Exception as inner_e:
                                    print(f"❌ 单独翻译失败: {japanese_text[:30]}... - {inner_e}")
                                    batch_chinese_texts[idx] = "[翻译失败]"
                except Exception as e:
                    # 批量翻译异常，尝试使用单独翻译
                    print(f"⚠️ 批量翻译异常: {e}")
                    print(f"📊 降级到单独翻译，确保功能正常")
                    
                    # 使用单独翻译
                    for idx in uncached_indices:
                        japanese_text = batch_japanese_texts[idx]
                        cache_key = f"jp:zh:{japanese_text}"
                        try:
                            # 对于单独翻译，增加重试次数以提高成功率
                            chinese_text = baidu_translate(japanese_text, max_retries=5)
                            # 清理单独翻译结果中的<SEP>分隔符
                            chinese_text = chinese_text.replace(separator, '')
                            batch_chinese_texts[idx] = chinese_text
                            # 使用baidu_translate函数已经保存了正确格式的缓存，这里不需要重复保存
                            print(f"✅ 单独翻译: {japanese_text[:30]}... -> {chinese_text[:30]}...")
                        except Exception as inner_e:
                            print(f"❌ 单独翻译失败: {japanese_text[:30]}... - {inner_e}")
                            batch_chinese_texts[idx] = "[翻译失败]"
            else:
                # 所有文本都在缓存中
                print(f"✅ 全部使用缓存: {cached_count}/{len(batch_japanese_texts)}")
                print(f"📝 注意: 缓存内容可能不如批量翻译保持语义连贯性")
                # 确保batch_chinese_texts已正确初始化
                if not batch_chinese_texts:
                    batch_chinese_texts = [_translation_cache.get(f"jp:zh:{text}", "") for text in batch_japanese_texts]
            
            # 批量翻译完成后保存缓存
            if len(_translation_cache) > 0:
                save_translation_cache(_translation_cache)
            
            # 处理每个翻译结果
            for idx, segment in enumerate(batch_segments):
                global_index = i + idx
                start_time = format_time(segment['start'])
                end_time = format_time(segment['end'])
                japanese_text = segment['text'].strip()
                
                if valid_indices[idx] != -1 and valid_indices[idx] < len(batch_chinese_texts):
                    chinese_text = batch_chinese_texts[valid_indices[idx]]
                    
                    # 直接使用批量翻译结果，不进行质量检查
                    print(f"✅ 使用批量翻译结果: {chinese_text}")
                    
                    print(f"📊 当前缓存条目数: {len(_translation_cache)}")
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
        if len(uncached_texts) > 0:
            time.sleep(0.3)  # 批量翻译后稍微减少延迟
        else:
            time.sleep(0.1)  # 全部使用缓存时减少延迟
        
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
            from progress_manager import save_progress
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
        from progress_manager import save_progress
        final_save_success = save_progress(video_path, final_progress_data)
        if final_save_success:
            from progress_manager import get_progress_file_path
            print(f"💾 最终进度文件已保存: {get_progress_file_path(video_path)}")
        else:
            print(f"⚠️ 警告：最终进度保存失败，但字幕文件已生成")
    
    # 翻译完成后保存缓存（使用视频特定的缓存文件）
    if len(_translation_cache) > 0:
        # 确保设置了当前视频名称
        if video_path:
            set_current_video_name(video_path)
        save_translation_cache(_translation_cache)
    
    print(f"✅ 双语字幕文件已生成: {output_path}")
    return True

def format_time(seconds):
    """将秒数格式化为SRT时间格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def generate_japanese_only_subtitle(transcription_result, output_path):
    """仅生成日语字幕"""
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
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    print(f"✅ 日语字幕文件已生成: {output_path}")
    return True
