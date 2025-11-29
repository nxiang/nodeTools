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
    """保存翻译缓存到文件（会自动使用视频特定的缓存文件）"""
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

def baidu_translate(text, adult_content=False):
    """使用百度翻译API翻译文本"""
    global _translation_cache
    
    # 检查缓存
    if text in _translation_cache:
        cached_result = _translation_cache[text]
        print(f"✅ 使用缓存的翻译结果: {text[:20]}{'...' if len(text) > 20 else ''}")
        return cached_result
    
    # 获取配置
    baidu_config = get_baidu_config()
    system_config = get_system_config()
    
    # 成人内容处理：使用专业术语词典
    if adult_content:
        translated_text = process_adult_content(text)
        if translated_text != text:  # 如果有替换
            _translation_cache[text] = translated_text
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
    max_retries = system_config['max_retries']
    retry_delay = system_config['retry_delay']
    api_success = False
    
    for retry in range(max_retries):
        try:
            if request_method == 'post':
                response = requests.post(url, data=data, headers=headers, timeout=10)
            else:
                response = requests.get(url, params=params, timeout=10)
            
            result = response.json()
            
            # 检查是否有错误
            if 'error_code' in result:
                print(f"❌ 百度翻译API错误: {result.get('error_code')} - {result.get('error_msg')}")
                time.sleep(retry_delay)
                continue
            
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
                
                # 存入缓存
                _translation_cache[text] = translated_text
                return translated_text
            
        except Exception as e:
            print(f"❌ 翻译请求异常: {e}")
            time.sleep(retry_delay)
    
    # 如果所有重试都失败，返回带标记的原文
    print(f"⚠️ 翻译失败，使用原文: {text}")
    translated_text = f"[翻译失败] {text}"
    _translation_cache[text] = translated_text
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

def batch_translate(text_list, adult_content=False):
    """批量翻译文本列表"""
    translated_results = []
    for text in text_list:
        translated = baidu_translate(text, adult_content)
        translated_results.append(translated)
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
