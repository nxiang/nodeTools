"""
翻译模块
负责处理日语到中文的翻译，包括缓存管理和百度翻译API调用
"""

import os
import json
import time
import hashlib
import requests
from pathlib import Path
from config import get_baidu_config, get_system_config, get_adult_terms_dict

# 当前视频名称，用于生成视频特定的缓存文件
_current_video_name = None

# 翻译缓存
_translation_cache = {}

def set_current_video_name(video_path):
    """设置当前视频名称，用于生成视频特定的缓存文件"""
    global _current_video_name
    if video_path:
        _current_video_name = Path(video_path).stem
        print(f"📽️ 已设置当前视频名称: {_current_video_name}")
    else:
        _current_video_name = None

def get_video_specific_cache_file():
    """获取视频特定的缓存文件路径"""
    # 获取临时目录
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # 根据是否设置了视频名称决定缓存文件路径
    if _current_video_name:
        cache_file = temp_dir / f"{_current_video_name}_translation_cache.json"
        print(f"💾 使用视频特定的缓存文件: {cache_file}")
    else:
        cache_file = temp_dir / "translation_cache.json"
        print(f"⚠️ 未设置视频名称，使用默认缓存文件: {cache_file}")
    
    return cache_file

def load_translation_cache():
    """加载翻译缓存（会自动使用视频特定的缓存文件）"""
    global _translation_cache
    
    try:
        cache_file = get_video_specific_cache_file()
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                _translation_cache = json.load(f)
            print(f"✅ 已加载翻译缓存，缓存条目数: {len(_translation_cache)}")
        else:
            _translation_cache = {}
            print(f"ℹ️ 翻译缓存文件不存在，创建新缓存")
    except Exception as e:
        _translation_cache = {}
        print(f"❌ 加载翻译缓存失败: {e}")
    
    return _translation_cache

def save_translation_cache(cache=None):
    """保存翻译缓存到文件（会自动使用视频特定的缓存文件）
    
    支持无参数调用，此时会保存全局缓存
    """
    global _translation_cache
    
    try:
        # 使用传入的缓存或全局缓存
        if cache is None:
            cache = _translation_cache
        
        cache_file = get_video_specific_cache_file()
        
        # 确保目录存在
        cache_file.parent.mkdir(exist_ok=True)
        
        # 保存缓存
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        
        print(f"💾 翻译缓存已保存到 {cache_file}，当前缓存条目数: {len(cache)}")
        return True
    except Exception as e:
        print(f"❌ 保存翻译缓存失败: {e}")
        return False

def baidu_translate(text, adult_content=False, max_retries=3, show_individual_logs=True):
    """使用百度翻译API翻译文本
    
    Args:
        text: 要翻译的文本
        adult_content: 是否处理成人内容
        max_retries: 最大重试次数
        show_individual_logs: 是否显示翻译成功的单条日志
    """
    global _translation_cache
    
    # 标准缓存键格式
    cache_key = f"jp:zh:{text}"
    
    # 检查缓存
    if cache_key in _translation_cache:
        cached_result = _translation_cache[cache_key]
        # 处理不同格式的缓存数据
        if isinstance(cached_result, dict) and 'response_result' in cached_result and 'trans_result' in cached_result['response_result']:
            if cached_result['response_result']['trans_result']:
                result = cached_result['response_result']['trans_result'][0].get('dst', '')
                print(f"✅ 使用缓存的翻译结果: {text[:20]}{'...' if len(text) > 20 else ''}")
                return result
        elif isinstance(cached_result, str):
            print(f"✅ 使用缓存的翻译结果: {text[:20]}{'...' if len(text) > 20 else ''}")
            return cached_result
        elif isinstance(cached_result, dict) and 'result' in cached_result:
            print(f"✅ 使用缓存的翻译结果: {text[:20]}{'...' if len(text) > 20 else ''}")
            return cached_result['result']
    
    # 获取配置
    baidu_config = get_baidu_config()
    system_config = get_system_config()
    
    # 使用指定的重试次数或系统配置中的重试次数
    retry_count = max_retries if max_retries > 0 else system_config['max_retries']
    
    # 成人内容处理：使用专业术语词典
    if adult_content:
        translated_text = process_adult_content(text)
        if translated_text != text:  # 如果有替换
            _translation_cache[cache_key] = {
                'request_params': {'q': text, 'from': 'jp', 'to': 'zh'},
                'response_result': {'from': 'jp', 'to': 'zh', 'trans_result': [{'src': text, 'dst': translated_text}]}
            }
            return translated_text
    
    # 准备翻译API参数（适配新的API端点）
    appid = baidu_config['appid']
    secret_key = baidu_config['key']
    url = baidu_config['url']
    
    # 生成签名
    salt = str(int(time.time()))
    sign_str = appid + text + salt + secret_key
    sign = hashlib.md5(sign_str.encode()).hexdigest()
    
    # 构建请求头
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }
    
    # 根据URL判断API类型并设置相应参数
    if 'ait/api/aiTextTranslate' in url:
        # 百度AI开放平台文本翻译API参数
        data = {
            'appid': appid,
            'from': 'jp',     # 日语
            'to': 'zh',       # 中文
            'q': text,
            'salt': salt,
            'sign': sign
        }
        # 使用POST请求
        request_method = 'post'
    else:
        # 传统百度翻译API参数
        params = {
            'q': text,
            'from': 'jp',
            'to': 'zh',
            'appid': appid,
            'salt': salt,
            'sign': sign
        }
        request_method = 'get'
    
    # 发送请求（带重试机制）
    retry_delay = system_config['retry_delay']
    api_success = False
    
    for retry in range(retry_count):
        try:
            print(f"📤 发送百度翻译API请求: 文本='{text[:20]}{'...' if len(text) > 20 else ''}' 请求方式={request_method.upper()}")
            
            if request_method == 'post':
                response = requests.post(url, data=data, headers=headers, timeout=10)
            else:
                response = requests.get(url, params=params, timeout=10)
            
            print(f"📥 收到百度翻译API响应: 状态码={response.status_code}")
            
            result = response.json()
            
            result = response.json()
            
            # 检查是否有错误
            if 'error_code' in result:
                print(f"❌ 百度翻译API错误: {result.get('error_code')} - {result.get('error_msg')}")
                time.sleep(retry_delay)
                continue
            
            print(f"📊 百度翻译API响应正常: 成功获取翻译结果")
            
            # 提取翻译结果（适配不同API返回格式）
            if 'trans_result' in result and isinstance(result['trans_result'], list) and result['trans_result']:
                translated_text = result['trans_result'][0]['dst']
                api_success = True
            elif 'result' in result and 'trans_result' in result['result']:
                # 可能的新格式
                translated_text = result['result']['trans_result']
                api_success = True
            elif 'trans_result' in result and 'dst' in result['trans_result']:
                translated_text = result['trans_result']['dst']
                api_success = True
            else:
                print(f"❌ 无法解析API响应格式: {result}")
                time.sleep(retry_delay)
                continue
            
            if api_success:
                print(f"🌐 翻译成功: {text[:20]}{'...' if len(text) > 20 else ''} -> {translated_text[:20]}{'...' if len(translated_text) > 20 else ''}")
                
                # 存入缓存 - 使用一致的缓存格式
                _translation_cache[cache_key] = {
                    'request_params': {'q': text, 'from': 'jp', 'to': 'zh'},
                    'response_result': {'from': 'jp', 'to': 'zh', 'trans_result': [{'src': text, 'dst': translated_text}]}
                }
                return translated_text
            
        except Exception as e:
            print(f"❌ 翻译请求异常: {e}")
            time.sleep(retry_delay)
    
    # 如果所有重试都失败，返回带标记的原文
    print(f"⚠️ 翻译失败，使用原文: {text}")
    translated_text = f"[翻译失败] {text}"
    _translation_cache[cache_key] = translated_text
    return translated_text

def process_adult_content(text):
    """处理成人内容，使用专业术语词典进行替换"""
    adult_terms = get_adult_terms_dict()
    processed_text = text
    
    # 替换成人术语
    for term, replacement in adult_terms.items():
        if term in processed_text:
            processed_text = processed_text.replace(term, replacement)
            print(f"🔞 替换成人术语: {term} -> {replacement}")
    
    return processed_text

def batch_translate(text_list, adult_content=False, show_individual_logs=False):
    """批量翻译文本列表，优化缓存命中的处理速度
    
    Args:
        text_list: 要翻译的文本列表
        adult_content: 是否处理成人内容
        show_individual_logs: 是否显示每条翻译的单独日志，批量模式下建议设为False
    """
    # 定义百度翻译API的最大字符限制
    MAX_CHAR_LIMIT = 6000
    
    translated_results = []
    
    # 在开始处理前，预先统计真正存在的缓存命中数量
    pre_existing_cache_count = 0
    unique_texts = set()
    for text in text_list:
        if text not in unique_texts:
            cache_key = f"jp:zh:{text}"
            if cache_key in _translation_cache:
                pre_existing_cache_count += 1
            unique_texts.add(text)
    
    # 分组处理：缓存命中和需要API翻译的文本
    cache_hits = 0
    api_calls = 0
    
    # 首先处理缓存命中的文本，并收集需要API翻译的文本和它们在原列表中的位置
    texts_to_translate = []
    positions_to_fill = []
    
    for i, text in enumerate(text_list):
        cache_key = f"jp:zh:{text}"
        
        # 直接在缓存中查找
        if cache_key in _translation_cache:
            cached_result = _translation_cache[cache_key]
            # 处理不同格式的缓存数据
            if isinstance(cached_result, dict) and 'response_result' in cached_result and 'trans_result' in cached_result['response_result']:
                if cached_result['response_result']['trans_result']:
                    result = cached_result['response_result']['trans_result'][0].get('dst', '')
                    if show_individual_logs:
                        print(f"✅ 使用缓存的翻译结果: {text[:20]}{'...' if len(text) > 20 else ''}")
                    cache_hits += 1
                    translated_results.append(result)
                    continue
            elif isinstance(cached_result, str):
                if show_individual_logs:
                    print(f"✅ 使用缓存的翻译结果: {text[:20]}{'...' if len(text) > 20 else ''}")
                cache_hits += 1
                translated_results.append(cached_result)
                continue
            elif isinstance(cached_result, dict) and 'result' in cached_result:
                if show_individual_logs:
                    print(f"✅ 使用缓存的翻译结果: {text[:20]}{'...' if len(text) > 20 else ''}")
                cache_hits += 1
                translated_results.append(cached_result['result'])
                continue
        
        # 缓存未命中，添加到待翻译列表和位置记录
        texts_to_translate.append(text)
        translated_results.append(None)  # 先添加占位符
        positions_to_fill.append(i)
    
    # 如果有需要API翻译的文本，进行批量翻译
    if texts_to_translate:
        print(f"📤 开始批量API翻译: {len(texts_to_translate)} 条文本待翻译")
        
        # 使用<>拼接所有待翻译文本
        concatenated_text = "<>" .join(texts_to_translate)
        
        # 检查是否超出百度翻译API的字符限制
        if len(concatenated_text) > MAX_CHAR_LIMIT:
            print(f"⚠️  拼接后的文本超出字符限制: {len(concatenated_text)} > {MAX_CHAR_LIMIT} 字符")
            print(f"🔄 开始分批翻译处理")
            
            # 分批处理文本
            batches = []
            current_batch = []
            current_batch_size = 0
            separator_length = len("<>")
            
            for text in texts_to_translate:
                text_length = len(text)
                # 如果添加当前文本会导致批次超出限制，则将当前批次加入批次列表并开始新批次
                if current_batch_size + text_length + (separator_length if current_batch else 0) > MAX_CHAR_LIMIT:
                    if current_batch:  # 确保当前批次不为空
                        batches.append(current_batch)
                        current_batch = []
                        current_batch_size = 0
                # 添加文本到当前批次
                current_batch.append(text)
                current_batch_size += text_length + (separator_length if current_batch_size > 0 else 0)
            
            # 添加最后一个批次
            if current_batch:
                batches.append(current_batch)
            
            print(f"📊 文本已分成 {len(batches)} 个批次进行翻译")
            
            # 处理每个批次
            all_translated_parts = []
            for i, batch in enumerate(batches):
                print(f"📦 处理翻译批次 {i+1}/{len(batches)}: {len(batch)} 条文本")
                batch_text = "<>" .join(batch)
                print(f"   批次字符数: {len(batch_text)}")
                
                # 翻译当前批次
                batch_translated = baidu_translate(batch_text, adult_content, show_individual_logs=False)
                api_calls += 1
                
                # 拆分批次翻译结果
                batch_translated_parts = batch_translated.split("<>")
                all_translated_parts.extend(batch_translated_parts)
            
            # 合并所有批次的翻译结果
            translated_parts = all_translated_parts
        else:
            # 未超出字符限制，直接进行一次性翻译
            print(f"📊 当前文本字符数: {len(concatenated_text)}，未超出限制")
            translated_batch = baidu_translate(concatenated_text, adult_content, show_individual_logs=False)
            api_calls += 1
            
            # 拆分翻译结果
            translated_parts = translated_batch.split("<>")
        
        # 拆分翻译结果并填充到对应的位置
        translated_parts = translated_batch.split("<>")
        
        # 处理翻译结果拆分可能不匹配的情况
        if len(translated_parts) != len(texts_to_translate):
            print(f"⚠️  批量翻译结果拆分不匹配: 预期{len(texts_to_translate)}条，实际{len(translated_parts)}条")
            # 降级策略：对每个文本单独调用baidu_translate函数进行翻译
            print(f"🔄 降级为单独翻译模式")
            for i, pos in enumerate(positions_to_fill):
                try:
                    # 对每个文本单独调用翻译函数
                    individual_translated = baidu_translate(texts_to_translate[i], adult_content, show_individual_logs=False)
                    translated_results[pos] = individual_translated
                    # 更新缓存
                    cache_key = f"jp:zh:{texts_to_translate[i]}"
                    _translation_cache[cache_key] = individual_translated
                    api_calls += 1  # 每个单独翻译也算一次API调用
                except Exception as e:
                    print(f"❌ 单独翻译失败: {texts_to_translate[i][:20]}{'...' if len(texts_to_translate[i]) > 20 else ''}, 错误: {str(e)}")
                    # 失败时使用原文本作为后备
                    translated_results[pos] = texts_to_translate[i]
        else:
            # 正常情况：将每个翻译结果填充到对应的位置
            for i, pos in enumerate(positions_to_fill):
                translated_results[pos] = translated_parts[i]
                # 更新缓存
                cache_key = f"jp:zh:{texts_to_translate[i]}"
                _translation_cache[cache_key] = translated_parts[i]
        
        print(f"📥 批量API翻译完成: {len(texts_to_translate)} 条文本已翻译")
    
    # 打印批量翻译的总体统计信息，使用预先统计的真正缓存命中数量
    print(f"📊 批量翻译完成: 总计 {len(text_list)} 条，缓存命中 {pre_existing_cache_count} 条，API调用 {api_calls} 条")
    
    return translated_results

def check_translation_quality(original_text, translated_text):
    """检查翻译质量（简化实现）"""
    # 简单的质量检查逻辑
    if not translated_text or translated_text == original_text:
        return False, "翻译结果为空或与原文相同"
    
    # 检查长度比例（日语通常比中文短）
    if len(translated_text) < len(original_text) * 0.3 or len(translated_text) > len(original_text) * 3:
        return False, "翻译结果长度异常"
    
    return True, "翻译质量良好"
