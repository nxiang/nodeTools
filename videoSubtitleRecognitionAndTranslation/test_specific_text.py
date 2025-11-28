#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试特定文本的翻译"""

import requests
import hashlib
import random
import time

def baidu_translate_test(text, from_lang='jp', to_lang='zh'):
    """测试百度翻译API"""
    # 百度翻译API配置
    appid = '20251126002506386'
    key = 'C0qK4IqU_KXjun3PhRum'
    url = 'https://fanyi-api.baidu.com/ait/api/aiTextTranslate'
    
    # 生成签名
    salt = str(random.randint(32768, 65536))
    sign_str = appid + text + salt + key
    sign = hashlib.md5(sign_str.encode()).hexdigest()
    
    # 请求参数
    params = {
        'q': text,
        'from': from_lang,
        'to': to_lang,
        'appid': appid,
        'salt': salt,
        'sign': sign
    }
    
    print(f"🔍 测试文本: {text}")
    print(f"🔍 请求参数: {params}")
    
    try:
        # 先尝试POST
        response = requests.post(url, data=params, timeout=15)
        result = response.json()
        print(f"🔍 POST响应: {result}")
        
        # 如果POST失败，尝试GET
        if 'error_code' in result:
            response = requests.get(url, params=params, timeout=15)
            result = response.json()
            print(f"🔍 GET响应: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return {'error': str(e)}

# 测试特定文本
test_texts = [
    "ヒーブを上げそうなんだからは…",
    "おいちゃん、ちいちゃん、しいちゃんかもー!",
    "ちいちゃん、すいなのいでー?",
    "うーん…",
    "どうしたの?"
]

for i, test_text in enumerate(test_texts, 1):
    print(f"\n=== 测试 {i}/{len(test_texts)} ===")
    result = baidu_translate_test(test_text)

    print("\n=== 测试结果分析 ===")
    if 'trans_result' in result:
        print("✅ 翻译成功")
        print(f"原文: {test_text}")
        print(f"译文: {result['trans_result'][0]['dst']}")
    elif 'error_code' in result:
        print(f"❌ 翻译失败 - 错误代码: {result['error_code']}")
        print(f"错误信息: {result.get('error_msg', '未知错误')}")
    else:
        print(f"❓ 未知响应格式: {result}")
