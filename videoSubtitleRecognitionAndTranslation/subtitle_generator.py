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

# 导入翻译缓存函数
from translator import save_translation_cache, load_translation_cache

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

# 导入翻译缓存函数
from translator import save_translation_cache, load_translation_cache

def generate_bilingual_subtitle_file(video_path, transcription_result, 
                                   enable_translation=True, adult_content=False, progress=None):
    """生成双语字幕文件"""
    
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
    MAX_CHARS_PER_BATCH = 5000  # 百度翻译API限制6000字符，设置5000留有余地
    separator = "<>"  # 批量翻译分隔符
    
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
            
            # 检查批量文本的缓存
            batch_chinese_texts = []
            cached_count = 0
            for japanese_text in batch_japanese_texts:
                cache_key = f"jp:zh:{japanese_text}"
                if cache_key in _translation_cache:
                    batch_chinese_texts.append(_translation_cache[cache_key])
                    cached_count += 1
                else:
                    # 对于未缓存的文本，使用批量翻译
                    batch_chinese_texts.append("")
            
            # 如果有未缓存的文本，优先使用批量翻译
            if cached_count < len(batch_japanese_texts):
                print(f"📊 缓存命中: {cached_count}/{len(batch_japanese_texts)}，剩余使用批量API翻译")
                uncached_texts = [text for text in batch_japanese_texts if f"jp:zh:{text}" not in _translation_cache]
                
                # 优先尝试批量翻译
                try:
                    api_translated = batch_translate(uncached_texts, separator)
                    
                    # 检查批量翻译返回结果数量
                    if len(api_translated) == len(uncached_texts):
                        # 批量翻译成功，正常合并结果
                        api_index = 0
                        for idx, japanese_text in enumerate(batch_japanese_texts):
                            cache_key = f"jp:zh:{japanese_text}"
                            if cache_key not in _translation_cache:
                                batch_chinese_texts[idx] = api_translated[api_index]
                                # 保存到缓存
                                _translation_cache[cache_key] = api_translated[api_index]
                                api_index += 1
                        
                        # 保存批量翻译的合并文本和分隔符分隔的结果
                        batch_combined_key = f"batch_jp:zh:{separator.join(uncached_texts)}"
                        batch_combined_result = separator.join(api_translated)
                        _translation_cache[batch_combined_key] = batch_combined_result
                        
                        print(f"✅ 批量翻译成功: 处理了{len(uncached_texts)}个文本")
                        print(f"📦 批量翻译合并文本已保存到缓存")
                    else:
                        # 批量翻译结果不匹配，智能复用已有结果
                        print(f"⚠️ 批量翻译结果数量不匹配: {len(api_translated)} != {len(uncached_texts)}")
                        print(f"📊 智能复用批量翻译结果，补充缺失部分")
                        
                        # 复用已有的批量翻译结果
                        api_index = 0
                        reused_count = 0
                        missing_texts = []
                        missing_indices = []
                        
                        for idx, japanese_text in enumerate(batch_japanese_texts):
                            cache_key = f"jp:zh:{japanese_text}"
                            if cache_key not in _translation_cache:
                                if api_index < len(api_translated):
                                    # 复用已有的批量翻译结果
                                    batch_chinese_texts[idx] = api_translated[api_index]
                                    _translation_cache[cache_key] = api_translated[api_index]
                                    api_index += 1
                                    reused_count += 1
                                else:
                                    # 记录缺失的文本和索引
                                    missing_texts.append(japanese_text)
                                    missing_indices.append(idx)
                        
                        print(f"✅ 复用批量翻译结果: {reused_count}/{len(uncached_texts)} 个文本")
                        
                        # 对缺失的文本使用单独翻译
                        if missing_texts:
                            print(f"📊 补充翻译缺失部分: {len(missing_texts)} 个文本")
                            for i, japanese_text in enumerate(missing_texts):
                                idx = missing_indices[i]
                                cache_key = f"jp:zh:{japanese_text}"
                                # 使用单独翻译API
                                chinese_text = baidu_translate(japanese_text, max_retries=3)
                                batch_chinese_texts[idx] = chinese_text
                                # 保存到缓存
                                _translation_cache[cache_key] = chinese_text
                                print(f"✅ 补充翻译并缓存: {japanese_text[:30]}...")
                        
                        # 保存部分批量翻译的合并文本和分隔符分隔的结果
                        if reused_count > 0:
                            reused_texts = [batch_japanese_texts[i] for i in range(len(batch_japanese_texts)) 
                                          if f"jp:zh:{batch_japanese_texts[i]}" not in _translation_cache 
                                          and i < len(api_translated)]
                            reused_results = [api_translated[i] for i in range(min(len(api_translated), len(reused_texts)))]
                            
                            if reused_texts and reused_results:
                                batch_partial_key = f"batch_partial_jp:zh:{separator.join(reused_texts)}"
                                batch_partial_result = separator.join(reused_results)
                                _translation_cache[batch_partial_key] = batch_partial_result
                                print(f"📦 部分批量翻译结果已保存到缓存: {reused_count}个文本")
                except Exception as e:
                    # 批量翻译异常，降级到单独翻译
                    print(f"⚠️ 批量翻译异常: {e}")
                    print(f"📊 降级到单独翻译模式")
                    
                    # 使用单独翻译确保缓存完整
                    for idx, japanese_text in enumerate(batch_japanese_texts):
                        cache_key = f"jp:zh:{japanese_text}"
                        if cache_key not in _translation_cache:
                            # 使用单独翻译API
                            chinese_text = baidu_translate(japanese_text, max_retries=3)
                            batch_chinese_texts[idx] = chinese_text
                            # 保存到缓存
                            _translation_cache[cache_key] = chinese_text
                            print(f"✅ 单独翻译并缓存: {japanese_text[:30]}...")
            else:
                print(f"✅ 全部使用缓存: {cached_count}/{len(batch_japanese_texts)}")
            
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
                    
                    # 检查翻译质量
                    if not check_translation_quality(chinese_text, japanese_text):
                        print(f"⚠️  翻译质量不佳，单独重试片段 {global_index+1}...")
                        print(f"📊 单独翻译统计: 第{global_index+1}个片段质量检查失败，启动单独翻译")
                        
                        # 生成缓存键
                        cache_key = f"jp:zh:{japanese_text}"
                        
                        # 检查缓存
                        if cache_key in _translation_cache:
                            chinese_text = _translation_cache[cache_key]
                            print(f"✅ 使用缓存的翻译结果")
                            print(f"📊 单独翻译统计: 第{global_index+1}个片段使用缓存，跳过API调用")
                        else:
                            # 使用百度翻译API
                            print(f"🌐 开始API翻译: 第{global_index+1}个片段")
                            chinese_text = baidu_translate(japanese_text, max_retries=3)
                            
                            # 保存到缓存
                        _translation_cache[cache_key] = chinese_text
                        print(f"✅ 翻译完成")
                        print(f"📊 单独翻译统计: 第{global_index+1}个片段API翻译成功")
                        
                        # 每5个新缓存条目保存一次
                        if len(_translation_cache) % 5 == 0:
                            save_translation_cache(_translation_cache)
                    
                    print(f"🌐 翻译: {chinese_text}")
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
        
        # 添加延迟避免请求过快
        time.sleep(0.5)
        
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
    
    # 翻译完成后保存缓存
    if len(_translation_cache) > 0:
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
