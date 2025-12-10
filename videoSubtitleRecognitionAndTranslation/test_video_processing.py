#!/usr/bin/env python3
"""
测试视频处理修复效果
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from whisper_transcription_vad import VADProcessor, VADConfig

def test_video_loading():
    """测试视频加载功能"""
    print("=== 测试视频加载功能 ===")
    
    # 创建配置
    config = VADConfig()
    processor = VADProcessor(config)
    
    # 测试视频文件路径
    test_video_path = "Z:\\视频\\成人内容\\bt\\ap-547.mp4"
    
    if not os.path.exists(test_video_path):
        print(f"测试视频文件不存在: {test_video_path}")
        # 创建一个虚拟的测试文件路径
        test_video_path = "test_video.mp4"
        print(f"使用虚拟路径: {test_video_path}")
    
    try:
        print(f"尝试加载视频: {test_video_path}")
        audio, sr = processor.load_audio(test_video_path)
        print(f"✅ 视频加载成功! 音频长度: {len(audio)/sr:.1f}秒")
        return True
    except Exception as e:
        print(f"❌ 视频加载失败: {e}")
        return False

def test_moviepy_import():
    """测试moviepy导入和版本"""
    print("\n=== 测试moviepy导入 ===")
    
    try:
        from moviepy import VideoFileClip
        print("✅ moviepy导入成功")
        
        # 检查moviepy版本
        import moviepy
        print(f"moviepy版本: {moviepy.__version__}")
        
        # 检查是否有change_settings函数
        try:
            from moviepy.config import change_settings
            print("✅ 找到change_settings函数")
        except ImportError:
            print("❌ 未找到change_settings函数（新版本moviepy）")
            
        return True
    except ImportError as e:
        print(f"❌ moviepy导入失败: {e}")
        return False

def test_ffmpeg_availability():
    """测试ffmpeg可用性"""
    print("\n=== 测试ffmpeg可用性 ===")
    
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ffmpeg可用")
            # 提取版本信息
            version_line = result.stdout.split('\n')[0]
            print(f"ffmpeg版本: {version_line}")
            return True
        else:
            print("❌ ffmpeg不可用")
            return False
    except Exception as e:
        print(f"❌ ffmpeg检查失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试视频处理修复...")
    
    # 测试moviepy导入
    moviepy_ok = test_moviepy_import()
    
    # 测试ffmpeg可用性
    ffmpeg_ok = test_ffmpeg_availability()
    
    # 测试视频加载（如果moviepy可用）
    if moviepy_ok:
        video_ok = test_video_loading()
    else:
        print("\n⚠️ moviepy不可用，跳过视频加载测试")
        video_ok = False
    
    print("\n=== 测试总结 ===")
    print(f"moviepy导入: {'✅ 成功' if moviepy_ok else '❌ 失败'}")
    print(f"ffmpeg可用: {'✅ 成功' if ffmpeg_ok else '❌ 失败'}")
    print(f"视频加载: {'✅ 成功' if video_ok else '❌ 失败'}")
    
    if not moviepy_ok:
        print("\n💡 建议: 安装或更新moviepy")
        print("pip install moviepy")
        
    if not ffmpeg_ok:
        print("\n💡 建议: 确保ffmpeg已安装并添加到PATH")
        
    if moviepy_ok and ffmpeg_ok and not video_ok:
        print("\n💡 建议: 检查视频文件格式和路径")
