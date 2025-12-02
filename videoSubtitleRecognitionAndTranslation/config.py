"""
配置管理模块
负责API配置、模型配置和系统参数管理
"""

import os

# 百度翻译API配置
BAIDU_TRANSLATE_CONFIG = {
    'appid': '20251126002506386',
    'key': 'C0qK4IqU_KXjun3PhRum',
    'url': 'https://fanyi-api.baidu.com/ait/api/aiTextTranslate'
}

# 成人内容专业术语词典
ADULT_TERMS_DICT = {
    "おっぱい": "胸部",
    "おちんちん": "男性生殖器",
    "まんこ": "女性生殖器",
    "ちんこ": "阴茎",
    "ちんぽ": "阴茎",
    "まんまん": "女性生殖器",
    "おまんこ": "女性生殖器",
    "おちんぽ": "阴茎",
    "ちんちん": "阴茎",
    "おっぱ": "胸部",
    "おっぱい": "胸部",
    "ぱいおつ": "胸部",
    "ぱいぱい": "胸部",
    "おっぱい": "胸部",
    "おっぱ": "胸部",
    "おちん": "阴茎",
    "おちんちん": "阴茎",
    "おちんぽ": "阴茎",
    "ちんこ": "阴茎",
    "ちんぽ": "阴茎",
    "ちんちん": "阴茎",
    "まんこ": "女性生殖器",
    "まんまん": "女性生殖器",
    "おまんこ": "女性生殖器",
    "おまんまん": "女性生殖器",
    "ぱい": "胸部",
    "ぱいおつ": "胸部",
    "ぱいぱい": "胸部",
    "おっぱい": "胸部",
    "おっぱ": "胸部",
    "おちん": "阴茎",
    "おちんちん": "阴茎",
    "おちんぽ": "阴茎",
    "ちんこ": "阴茎",
    "ちんぽ": "阴茎",
    "ちんちん": "阴茎",
    "まんこ": "女性生殖器",
    "まんまん": "女性生殖器",
    "おまんこ": "女性生殖器",
    "おまんまん": "女性生殖器"
}

# Whisper模型配置
WHISPER_MODEL_CONFIG = {
    'tiny': {
        'size': 'tiny',
        'description': '最小模型，速度最快，精度最低',
        'recommended_duration': 300  # 5分钟以下
    },
    'base': {
        'size': 'base', 
        'description': '基础模型，平衡速度和精度',
        'recommended_duration': 600  # 10分钟以下
    },
    'small': {
        'size': 'small',
        'description': '小型模型，精度较好',
        'recommended_duration': 1200  # 20分钟以下
    },
    'medium': {
        'size': 'medium',
        'description': '中等模型，精度高',
        'recommended_duration': 1800  # 30分钟以下
    },
    'large': {
        'size': 'large',
        'description': '大型模型，精度最高，速度最慢',
        'recommended_duration': 3600  # 60分钟以下
    }
}

# 系统配置
SYSTEM_CONFIG = {
    'max_retries': 3,  # API重试次数
    'retry_delay': 2,  # 重试延迟（秒）
    'max_chars_per_batch': 5000,  # 批量翻译最大字符数
    'batch_separator': '<S>',  # 批量翻译分隔符
    'cache_enabled': True,  # 是否启用翻译缓存
    'temp_dir': 'temp',  # 临时文件目录
    'model_cache_dir': 'models',  # 模型缓存目录
    'progress_file_suffix': '_progress.json',  # 进度文件后缀
    'cache_file': 'translation_cache.json'  # 缓存文件名
}

def get_model_config(model_size):
    """获取指定模型的配置"""
    return WHISPER_MODEL_CONFIG.get(model_size, WHISPER_MODEL_CONFIG['medium'])

def get_baidu_config():
    """获取百度翻译配置"""
    return BAIDU_TRANSLATE_CONFIG

def get_system_config():
    """获取系统配置"""
    return SYSTEM_CONFIG

def get_adult_terms_dict():
    """获取成人内容术语词典"""
    return ADULT_TERMS_DICT

def update_baidu_config(appid=None, key=None, url=None):
    """更新百度翻译配置"""
    if appid:
        BAIDU_TRANSLATE_CONFIG['appid'] = appid
    if key:
        BAIDU_TRANSLATE_CONFIG['key'] = key
    if url:
        BAIDU_TRANSLATE_CONFIG['url'] = url

def update_system_config(**kwargs):
    """更新系统配置"""
    for key, value in kwargs.items():
        if key in SYSTEM_CONFIG:
            SYSTEM_CONFIG[key] = value

def validate_config():
    """验证配置完整性"""
    errors = []
    
    # 验证百度翻译配置
    baidu_config = get_baidu_config()
    if not baidu_config['appid'] or baidu_config['appid'] == '20240426002033339':
        errors.append("百度翻译APPID需要配置")
    if not baidu_config['key'] or baidu_config['key'] == '3Y7J9v9J9v9J9v9J':
        errors.append("百度翻译密钥需要配置")
    
    # 验证临时目录
    temp_dir = SYSTEM_CONFIG['temp_dir']
    if not os.path.exists(temp_dir):
        try:
            os.makedirs(temp_dir)
        except Exception as e:
            errors.append(f"无法创建临时目录 {temp_dir}: {e}")
    
    return errors

def get_config_summary():
    """获取配置摘要"""
    baidu_config = get_baidu_config()
    system_config = get_system_config()
    
    return {
        'baidu_translate': {
            'appid': baidu_config['appid'][:8] + '...' if len(baidu_config['appid']) > 8 else baidu_config['appid'],
            'key': baidu_config['key'][:8] + '...' if len(baidu_config['key']) > 8 else baidu_config['key'],
            'url': baidu_config['url']
        },
        'system': {
            'max_retries': system_config['max_retries'],
            'cache_enabled': system_config['cache_enabled'],
            'temp_dir': system_config['temp_dir'],
            'model_cache_dir': system_config['model_cache_dir']
        },
        'models': list(WHISPER_MODEL_CONFIG.keys())
    }
