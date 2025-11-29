"""
翻译模块
负责百度翻译API调用、批量翻译和翻译质量检查
"""

import time
import hashlib
import requests
import random

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

def baidu_translate(text, from_lang='jp', to_lang='zh', max_retries=3):
    """使用百度翻译API进行翻译"""
    
    # 检查是否为空文本
    if not text or text.strip() == "":
        return ""
    
    # 检查是否为成人内容专业术语
    if text.strip() in ADULT_TERMS_DICT:
        return ADULT_TERMS_DICT[text.strip()]
    
    # 生成签名
    salt = str(random.randint(32768, 65536))
    sign = appid + text + salt + key
    sign = hashlib.md5(sign.encode()).hexdigest()
    
    # 构建请求URL
    url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    
    for attempt in range(max_retries):
        try:
            # 构建请求参数
            params = {
                'q': text,
                'from': from_lang,
                'to': to_lang,
                'appid': appid,
                'salt': salt,
                'sign': sign
            }
            
            # 发送请求
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                # 检查API返回状态
                if 'error_code' in result:
                    error_msg = result.get('error_msg', '未知错误')
                    print(f"❌ 翻译API错误 ({attempt+1}/{max_retries}): {error_msg}")
                    
                    # 如果是频率限制错误，等待后重试
                    if result['error_code'] == '54003':  # 访问频率受限
                        wait_time = 2 ** attempt  # 指数退避
                        print(f"⏳ 频率限制，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # 其他错误直接返回原文
                        return text
                
                # 提取翻译结果
                if 'trans_result' in result and result['trans_result']:
                    translated_text = result['trans_result'][0]['dst']
                    return translated_text
                else:
                    print(f"⚠️  API返回无翻译结果: {result}")
                    return text
            
            else:
                print(f"❌ HTTP错误 ({attempt+1}/{max_retries}): {response.status_code}")
        
        except requests.exceptions.Timeout:
            print(f"⏰ 请求超时 ({attempt+1}/{max_retries})")
        
        except requests.exceptions.ConnectionError:
            print(f"🌐 网络连接错误 ({attempt+1}/{max_retries})")
        
        except Exception as e:
            print(f"❌ 翻译异常 ({attempt+1}/{max_retries}): {e}")
        
        # 重试前等待
        if attempt < max_retries - 1:
            wait_time = 1 + attempt * 0.5  # 递增等待时间
            print(f"⏳ 等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
    
    # 所有重试都失败，返回原文
    print(f"❌ 翻译失败，返回原文")
    return text

def batch_translate(texts, separator="<>"):
    """批量翻译文本（禁用缓存，直接API翻译）"""
    
    if not texts:
        return []
    
    # 过滤有效文本（非空且非纯空白字符）
    valid_texts = [text for text in texts if text and text.strip()]
    
    if not valid_texts:
        return [""] * len(texts)
    
    print(f"📊 批量翻译模式: 禁用缓存，直接进行API翻译")
    print(f"📦 批量翻译: {len(valid_texts)} 个文本")
    
    # 合并文本进行批量翻译
    combined_text = separator.join(valid_texts)
    
    # 执行批量翻译
    combined_result = baidu_translate(combined_text)
    
    # 调试信息：打印合并文本和翻译结果
    print(f"🔍 合并文本长度: {len(combined_text)}")
    print(f"🔍 翻译结果长度: {len(combined_result) if combined_result else 0}")
    print(f"🔍 分隔符出现次数: {combined_result.count(separator) if combined_result else 0}")
    
    # 分割结果
    if combined_result:
        results = combined_result.split(separator)
        
        # 确保结果数量与输入一致
        if len(results) == len(valid_texts):
            # 构建完整的结果列表（包括空文本的位置）
            final_results = []
            valid_index = 0
            
            for text in texts:
                if text and text.strip():
                    final_results.append(results[valid_index])
                    valid_index += 1
                else:
                    final_results.append("")
            
            return final_results
        else:
            print(f"⚠️  批量翻译结果数量不匹配: 期望 {len(valid_texts)}, 实际 {len(results)}")
            print(f"🔍 实际分割结果: {results}")
            # 返回单个翻译结果
            return [baidu_translate(text) for text in texts]
    else:
        print("❌ 批量翻译失败，转为单条翻译")
        return [baidu_translate(text) for text in texts]

def check_translation_quality(translated_text, original_text):
    """检查翻译质量"""
    
    # 空文本检查
    if not translated_text or translated_text.strip() == "":
        return False
    
    # 检查是否返回原文（可能是API错误）
    if translated_text == original_text:
        return False
    
    # 检查翻译结果是否过短（可能是不完整的翻译）
    if len(translated_text) < len(original_text) * 0.3:  # 翻译结果过短
        return False
    
    # 检查是否包含明显的错误标记
    error_indicators = ['error', '错误', '失败', 'timeout', '超时']
    if any(indicator in translated_text.lower() for indicator in error_indicators):
        return False
    
    # 检查是否为乱码或异常字符
    import re
    if re.search(r'[\x00-\x1f\x7f-\xff]', translated_text):
        return False
    
    return True

def apply_adult_content_filter(text, adult_content=False):
    """应用成人内容过滤"""
    if adult_content:
        # 在成人内容模式下，使用专业术语词典
        for jp_term, zh_term in ADULT_TERMS_DICT.items():
            if jp_term in text:
                text = text.replace(jp_term, zh_term)
    return text

def save_translation_cache(cache_data):
    """保存翻译缓存到文件"""
    import json
    import os
    
    try:
        # 确保temp目录存在
        os.makedirs("temp", exist_ok=True)
        
        cache_file = "temp/translation_cache.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"❌ 翻译缓存保存失败: {e}")
        return False

def load_translation_cache():
    """从文件加载翻译缓存"""
    import json
    import os
    
    cache_file = "temp/translation_cache.json"
    
    if not os.path.exists(cache_file):
        return {}
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        return cache_data if isinstance(cache_data, dict) else {}
    except Exception as e:
        print(f"❌ 翻译缓存加载失败: {e}")
        return {}
