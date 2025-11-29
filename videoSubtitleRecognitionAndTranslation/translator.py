#!/usr/bin/env python3
"""
翻译模块 - 处理百度翻译API调用和缓存管理
"""

import os
import json
import time
import random
import hashlib
import requests

# 百度翻译API配置
appid = '20251126002506386'
key = 'C0qK4IqU_KXjun3PhRum'

# 翻译缓存相关配置
_translation_cache_file = "temp/translation_cache.json"

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

def replace_adult_terms(text):
    """替换成人内容专业术语"""
    result = text
    for term, replacement in ADULT_TERMS_DICT.items():
        result = result.replace(term, replacement)
    return result

def load_translation_cache():
    """加载翻译缓存文件，只保留百度API请求参数和响应结果"""
    try:
        if os.path.exists(_translation_cache_file):
            with open(_translation_cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"✅ 已加载翻译缓存，缓存条目数: {len(cache)}")
            return cache
        return {}
    except Exception as e:
        print(f"⚠️  加载翻译缓存失败: {e}")
        return {}

def save_translation_cache(cache_data=None):
    """保存翻译缓存到文件，只保留百度API请求参数和响应结果"""
    try:
        # 确保temp目录存在
        temp_dir = os.path.dirname(_translation_cache_file)
        if temp_dir and not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            print(f"📁 创建temp目录: {temp_dir}")
            
        # 如果没有提供缓存数据，尝试从全局变量获取
        if cache_data is None:
            global _translation_cache
            cache_data = _translation_cache
            
        with open(_translation_cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"💾 翻译缓存已保存到 {_translation_cache_file}，当前缓存条目数: {len(cache_data)}")
        return True
    except Exception as e:
        print(f"⚠️  保存翻译缓存失败: {e}")
        return False

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
    
    # 加载缓存
    global _translation_cache
    _translation_cache = load_translation_cache()
    
    # 检查缓存中是否已有翻译结果
    if cache_key in _translation_cache:
        cached_data = _translation_cache[cache_key]
        # 处理不同格式的缓存数据
        if isinstance(cached_data, dict):
            # 从response_result中提取翻译结果
            if 'response_result' in cached_data:
                response_data = cached_data['response_result']
                # 格式1: 标准百度翻译API格式
                if 'trans_result' in response_data and isinstance(response_data['trans_result'], list):
                    if response_data['trans_result']:
                        cached_result = response_data['trans_result'][0].get('dst', text)
                    else:
                        cached_result = text
                # 格式2: 新AI翻译API格式
                elif 'result' in response_data and 'trans_result' in response_data['result']:
                    if response_data['result']['trans_result']:
                        cached_result = response_data['result']['trans_result'][0].get('dst', text)
                    else:
                        cached_result = text
                else:
                    # 兼容旧格式的dict缓存
                    cached_result = cached_data.get('result', text)
            else:
                # 兼容旧格式的dict缓存
                cached_result = cached_data.get('result', text)
        else:
            # 如果是旧格式（直接存储结果字符串），只返回结果不更新缓存格式
            cached_result = cached_data
        print(f"✅ 使用缓存的翻译结果: {text[:20]}{'...' if len(text) > 20 else ''}")
        return cached_result
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            # 生成百度API请求参数
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
                
                # 只保存百度API的请求参数和响应结果到缓存
                _translation_cache[cache_key] = {
                    'request_params': {
                        'q': text,
                        'from': from_lang,
                        'to': to_lang
                    },
                    'response_result': result
                }
                
                # 立即保存缓存，确保格式正确
                save_translation_cache(_translation_cache)
                
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

def batch_translate(texts, separator="<SEP>"):
    """批量翻译文本，使用<SEP>作为分隔符"""
    if not texts:
        return []
    
    # 使用分隔符连接多个文本
    batch_text = separator.join(texts)
    print(f"🔄 批量翻译请求: {len(texts)}个文本片段，总长度: {len(batch_text)}字符")
    
    # 调用百度翻译API
    batch_result = baidu_translate(batch_text, max_retries=5)
    
    # 根据分隔符分割翻译结果
    translated_texts = batch_result.split(separator)
    
    # 处理分割结果不匹配的情况
    if len(translated_texts) != len(texts):
        print(f"⚠️  批量翻译结果分割不匹配，原始: {len(texts)}，翻译: {len(translated_texts)}")
        # 如果分割失败，使用单独翻译作为回退方案
        individual_results = []
        for text in texts:
            # 直接调用百度翻译，确保缓存格式正确
            result = baidu_translate(text)
            individual_results.append(result)
        return individual_results
    
    # 为每个翻译结果单独更新缓存
    for i, text in enumerate(texts):
        if i < len(translated_texts):
            cache_key = f"jp:zh:{text}"
            if cache_key not in _translation_cache:
                # 提取响应结果中的单个翻译部分
                individual_result = {
                    'request_params': {
                        'q': text,
                        'from': 'jp',
                        'to': 'zh'
                    },
                    'response_result': {
                        'from': 'jp',
                        'to': 'zh',
                        'trans_result': [{'src': text, 'dst': translated_texts[i]}]
                    }
                }
                _translation_cache[cache_key] = individual_result
    
    # 保存更新后的缓存
    save_translation_cache(_translation_cache)
    
    return translated_texts

def check_translation_quality(original_text, translated_text):
    """检查翻译质量，确保翻译结果有效"""
    # 检查翻译结果是否为空
    if not translated_text or translated_text.strip() == "":
        return False, "翻译结果为空"
    
    # 检查翻译结果是否与原文完全相同（可能翻译失败）
    if original_text.strip() == translated_text.strip():
        return False, "翻译结果与原文相同"
    
    # 检查是否包含明显的错误标记
    error_markers = ['error', '错误', 'failed', '失败', 'exception']
    for marker in error_markers:
        if marker.lower() in translated_text.lower():
            return False, f"翻译结果包含错误标记: {marker}"
    
    # 检查长度是否合理（不应该太短或太长）
    original_len = len(original_text)
    translated_len = len(translated_text)
    
    # 允许的长度比例范围（根据语言特点调整）
    min_ratio = 0.3  # 最小允许的长度比例
    max_ratio = 3.0  # 最大允许的长度比例
    
    if translated_len < original_len * min_ratio:
        return False, f"翻译结果太短，原文长度: {original_len}，翻译长度: {translated_len}"
    
    if translated_len > original_len * max_ratio:
        return False, f"翻译结果太长，原文长度: {original_len}，翻译长度: {translated_len}"
    
    # 检查是否包含有效的中文字符（对于日译中）
    chinese_chars = sum(1 for char in translated_text if '\u4e00' <= char <= '\u9fff')
    if chinese_chars < translated_len * 0.3 and len(translated_text) > 5:
        return False, f"翻译结果中中文字符比例过低: {chinese_chars}/{translated_len}"
    
    return True, "翻译质量良好"

# 初始化全局变量
_translation_cache = load_translation_cache()
