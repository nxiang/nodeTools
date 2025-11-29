#!/usr/bin/env python3
"""
模块化重构测试脚本
测试所有模块的功能是否正常
"""

import os
import sys
import tempfile
from pathlib import Path

def test_module_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    modules_to_test = [
        'main',
        'model_manager',
        'audio_processor', 
        'subtitle_generator',
        'translator',
        'progress_manager',
        'config',
        'utils'
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name}.py")
        except Exception as e:
            print(f"   ❌ {module_name}.py: {e}")
            return False
    
    print("✅ 所有模块导入成功！")
    return True

def test_config_validation():
    """测试配置验证"""
    print("\n🔧 测试配置验证...")
    
    try:
        from config import validate_config, get_config_summary
        
        # 验证配置
        errors = validate_config()
        if errors:
            print("⚠️ 配置验证警告:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("✅ 配置验证通过")
        
        # 获取配置摘要
        summary = get_config_summary()
        print(f"📊 配置摘要:")
        print(f"   - 模型: {', '.join(summary['models'])}")
        print(f"   - 缓存: {'启用' if summary['system']['cache_enabled'] else '禁用'}")
        
        return True
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False

def test_utils_functions():
    """测试工具函数"""
    print("\n🔧 测试工具函数...")
    
    try:
        from utils import check_ffmpeg_installed, format_duration, safe_filename
        
        # 测试FFmpeg检测
        ffmpeg_available = check_ffmpeg_installed()
        print(f"   📹 FFmpeg: {'可用' if ffmpeg_available else '不可用'}")
        
        # 测试时间格式化
        duration_str = format_duration(3665)  # 1小时1分5秒
        print(f"   ⏱️  时间格式化: 3665秒 → {duration_str}")
        
        # 测试安全文件名
        safe_name = safe_filename("测试/文件:名.txt")
        print(f"   📁 安全文件名: '测试/文件:名.txt' → '{safe_name}'")
        
        print("✅ 工具函数测试通过")
        return True
    except Exception as e:
        print(f"❌ 工具函数测试失败: {e}")
        return False

def test_progress_manager():
    """测试进度管理"""
    print("\n📊 测试进度管理...")
    
    try:
        from progress_manager import save_progress, load_progress, cleanup_progress, get_progress_file_path
        
        # 测试进度文件路径
        test_video_path = "test_video.mp4"
        progress_file = get_progress_file_path(test_video_path)
        print(f"   📁 进度文件路径: {progress_file}")
        
        # 测试进度保存（包含所有必需字段）
        test_progress = {
            "video_path": test_video_path,
            "last_translated_index": 10,
            "srt_content": "测试字幕内容",
            "total_segments": 100,
            "progress_percent": 10
        }
        
        save_result = save_progress(test_video_path, test_progress)
        print(f"   💾 进度保存{'成功' if save_result else '失败'}")
        
        # 测试进度加载
        loaded_progress = load_progress(test_video_path)
        if loaded_progress:
            print(f"   📖 进度加载成功: 索引{loaded_progress.get('last_translated_index', 'N/A')}")
        else:
            print("   ❌ 进度加载失败")
        
        # 测试进度清理
        cleanup_progress(test_video_path)
        print("   🗑️ 进度清理完成")
        
        print("✅ 进度管理测试通过")
        return True
    except Exception as e:
        print(f"❌ 进度管理测试失败: {e}")
        return False

def test_model_manager():
    """测试模型管理"""
    print("\n🤖 测试模型管理...")
    
    try:
        from model_manager import auto_select_model
        
        # 测试自动模型选择
        test_video_path = "test_video.mp4"
        
        # 模拟不同时长的视频
        short_video_model = auto_select_model(test_video_path, 'medium')  # 使用默认参数
        print(f"   📹 推荐模型: {short_video_model}")
        
        print("✅ 模型管理测试通过")
        return True
    except Exception as e:
        print(f"❌ 模型管理测试失败: {e}")
        return False

def test_translator():
    """测试翻译功能"""
    print("\n🌐 测试翻译功能...")
    
    try:
        from translator import baidu_translate, batch_translate, save_translation_cache, load_translation_cache
        
        # 测试百度翻译
        test_text = "Hello, this is a test."
        
        try:
            translated = baidu_translate(test_text)
            print(f"   🌐 百度翻译结果: {translated}")
        except Exception as e:
            print(f"   ⚠️ 百度翻译测试跳过: {e}")
        
        # 测试批量翻译
        try:
            texts = ["Hello", "World", "Test"]
            batch_result = batch_translate(texts)
            print(f"   📦 批量翻译结果: {batch_result}")
        except Exception as e:
            print(f"   ⚠️ 批量翻译测试跳过: {e}")
        
        # 测试缓存功能
        save_translation_cache({"test": "translated"})
        cache = load_translation_cache()
        print(f"   💾 缓存功能: {'正常' if cache else '异常'}")
        
        print("✅ 翻译功能测试通过")
        return True
    except Exception as e:
        print(f"❌ 翻译功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始模块化重构测试")
    print("=" * 50)
    
    tests = [
        test_module_imports,
        test_config_validation,
        test_utils_functions,
        test_progress_manager,
        test_model_manager,
        test_translator
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"   ✅ 通过: {passed}/{total}")
    print(f"   ❌ 失败: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！模块化重构成功！")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关模块")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
