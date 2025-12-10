#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试moviepy视频加载修复
"""

import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_moviepy_loading():
    """测试moviepy视频加载功能"""
    try:
        from moviepy.editor import VideoFileClip
        
        # 测试视频文件路径
        test_video = "Z:\\视频\\成人内容\\bt\\ap-547.mp4"
        
        if not os.path.exists(test_video):
            logger.warning(f"测试视频文件不存在: {test_video}")
            # 尝试使用当前目录下的其他视频文件
            for file in os.listdir('.'):
                if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    test_video = file
                    logger.info(f"使用测试视频: {test_video}")
                    break
            else:
                logger.error("没有找到可用的测试视频文件")
                return False
        
        logger.info(f"开始测试moviepy视频加载: {test_video}")
        
        # 方法1: 使用修复后的参数
        try:
            logger.info("方法1: 使用修复后的参数 (verbose=False, audio_fps=44100)")
            os.environ["FFMPEG_BINARY"] = "ffmpeg"
            video = VideoFileClip(test_video, verbose=False, audio_fps=44100)
            logger.info(f"方法1成功! 视频时长: {video.duration:.1f}秒")
            video.close()
            return True
        except Exception as e:
            logger.error(f"方法1失败: {e}")
        
        # 方法2: 使用简单模式
        try:
            logger.info("方法2: 使用简单模式 (无额外参数)")
            video = VideoFileClip(test_video)
            logger.info(f"方法2成功! 视频时长: {video.duration:.1f}秒")
            video.close()
            return True
        except Exception as e:
            logger.error(f"方法2失败: {e}")
        
        logger.error("所有方法都失败了")
        return False
        
    except ImportError as e:
        logger.error(f"无法导入moviepy: {e}")
        return False
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始测试moviepy修复")
    
    # 检查ffmpeg可用性
    try:
        result = os.system("ffmpeg -version")
        if result == 0:
            logger.info("ffmpeg可用")
        else:
            logger.warning("ffmpeg可能不可用")
    except Exception as e:
        logger.warning(f"检查ffmpeg时出错: {e}")
    
    # 测试moviepy加载
    success = test_moviepy_loading()
    
    if success:
        logger.info("✅ moviepy修复测试成功!")
    else:
        logger.error("❌ moviepy修复测试失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
