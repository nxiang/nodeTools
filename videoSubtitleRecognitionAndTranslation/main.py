#!/usr/bin/env python3
"""
视频字幕识别与翻译工具 - 主入口模块
使用Whisper模型识别日语语音，并通过百度翻译API生成双语字幕
支持测试模式（仅处理前10%内容）
"""

import os
import sys
import time
import argparse
from pathlib import Path

# 导入自定义模块
from model_manager import setup_whisper_model, auto_select_model
from audio_processor import extract_audio_segment
from subtitle_generator import transcribe_with_whisper, generate_bilingual_subtitle_file, generate_japanese_only_subtitle
from progress_manager import load_progress, save_progress, get_progress_file_path, get_same_dir_subtitle_path, cleanup_progress
from utils import check_ffmpeg_installed, print_section_header, print_success, print_warning, print_error, print_info
from config import validate_config, get_config_summary

def check_cpu_availability():
    """检查CPU信息"""
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
        logical_cpu_count = psutil.cpu_count(logical=True)  # 逻辑核心数
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1024**3
        
        return f"CPU: {cpu_count}核/{logical_cpu_count}线程, 内存: {memory_gb:.1f}GB"
    except ImportError:
        return "CPU模式（psutil未安装，无法获取详细信息）"

def merge_subtitle_to_video(video_path, subtitle_path, output_path=None, subtitle_language='chi'):
    """将字幕合并到视频文件中"""
    import subprocess
    
    print(f"🎬 开始合并字幕到视频...")
    
    if not output_path:
        video_name = Path(video_path).stem
        output_path = f"{video_name}_with_subtitle.mp4"
    
    try:
        # 检查FFmpeg是否可用
        if not check_ffmpeg_installed():
            print("❌ FFmpeg未安装或不可用，无法合并字幕")
            print("💡 请安装FFmpeg：https://ffmpeg.org/download.html")
            return False
        
        # 构建FFmpeg命令
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', subtitle_path,
            '-c', 'copy',  # 复制视频和音频流
            '-c:s', 'mov_text',  # 字幕编码格式
            '-metadata:s:s:0', f'language={subtitle_language}',
            '-y',  # 覆盖输出文件
            output_path
        ]
        
        print(f"   📥 输入视频: {video_path}")
        print(f"   📄 输入字幕: {subtitle_path}")
        print(f"   📤 输出文件: {output_path}")
        print(f"   🔧 字幕语言: {subtitle_language}")
        
        # 执行合并
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 字幕合并成功: {output_path}")
            print(f"💡 播放时可在播放器中选择字幕轨道")
            return True
        else:
            print(f"❌ 字幕合并失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 合并过程中出现错误: {e}")
        return False

def main(video_path=None, test_mode=True, model_size='medium', enable_translation=True, 
         output_dir=None, adult_content=False, merge_to_video=False, clean_progress=False):
    """主函数"""
    
    # 显示程序标题
    print_section_header("视频字幕识别与翻译工具")
    
    # 显示CPU状态
    cpu_info = check_cpu_availability()
    print(f"🔍 系统检测: {cpu_info}")
    
    # 验证配置
    config_errors = validate_config()
    if config_errors:
        print("⚠️ 配置验证警告:")
        for error in config_errors:
            print(f"   - {error}")
    
    # 显示配置摘要
    config_summary = get_config_summary()
    print(f"🔧 配置摘要:")
    print(f"   - 模型: {', '.join(config_summary['models'])}")
    print(f"   - 缓存: {'启用' if config_summary['system']['cache_enabled'] else '禁用'}")
    print(f"   - 临时目录: {config_summary['system']['temp_dir']}")
    
    # 如果没有指定视频文件，查找当前目录下的视频文件
    if not video_path:
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv']
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = file
                break
        
        if not video_path:
            print("❌ 未找到视频文件，请指定视频文件路径")
            return
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 自动选择模型大小
    selected_model_size = auto_select_model(video_path, model_size)
    
    # 成人内容优化提示
    if adult_content:
        print("🔞 成人内容模式已启用")
        print(f"   - 使用专业术语词典优化翻译")
        print(f"   - 建议使用 {selected_model_size} 或更高精度模型")
    
    print(f"🚀 开始处理视频: {video_path}")
    print(f"🌐 识别语言: 日语 → {'中文' if enable_translation else '仅识别'}")
    print(f"🔬 测试模式: {'开启' if test_mode else '关闭'}")
    print(f"🔧 使用Whisper {selected_model_size}模型 {'+ 百度翻译API' if enable_translation else ''}")
    
    # 创建临时目录
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 加载翻译缓存
    from translator import load_translation_cache
    global _translation_cache
    _translation_cache = load_translation_cache()
    print(f"💾 翻译缓存已加载，当前缓存条目数: {len(_translation_cache)}")
    
    # 首先检查是否有断点续传文件，避免重复识别
    progress = load_progress(video_path)
    result = None
    
    if progress and 'transcription_result' in progress:
        print("✅ 使用已保存的语音识别结果，跳过识别阶段")
        result = progress['transcription_result']
    else:
        # 提取音频
        audio_path = os.path.join(temp_dir, "audio.wav")
        # 测试模式下只提取前60秒音频
        segment_duration = 60 if test_mode else None
        if not extract_audio_segment(video_path, audio_path, segment_duration=segment_duration):
            return
        
        # 使用Whisper进行语音识别（CPU模式，支持进度显示和断点续传）
        model = setup_whisper_model(selected_model_size)
        result = transcribe_with_whisper(model, audio_path, selected_model_size)
    
    if not result:
        return
    
    # 生成字幕文件（与视频文件同名且在同一目录）
    subtitle_path = get_same_dir_subtitle_path(video_path)
    
    # 检查是否有错误状态的进度文件
    if progress and 'error' in progress:
        print(f"🔄 检测到上次中断的进度，继续处理...")
        print(f"   错误信息: {progress.get('error', '未知错误')}")
        print(f"   错误时间: {progress.get('error_time', '未知时间')}")
    
    # 保存语音识别结果到进度文件
    progress_data = {
        'transcription_result': result,
        'video_path': video_path,
        'model_size': selected_model_size,
        'enable_translation': enable_translation,
        'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'transcription_completed': True
    }
    save_progress(video_path, progress_data)
    print(f"💾 语音识别进度已保存: {get_progress_file_path(video_path)}")
    
    if enable_translation:
        success = generate_bilingual_subtitle_file(video_path, result, enable_translation=True, 
                                                 adult_content=adult_content, progress=progress)
    else:
        # 仅生成日语字幕
        success = generate_japanese_only_subtitle(result, subtitle_path)
    
    if success:
        # 显示识别结果摘要
        segments = result.get('segments', [])
        total_duration = sum(segment['end'] - segment['start'] for segment in segments)
        
        print_section_header("处理完成摘要")
        print(f"📊 识别结果摘要:")
        print(f"   识别片段数: {len(segments)}")
        print(f"   总识别时长: {total_duration:.2f}秒")
        print(f"   字幕文件: {subtitle_path}")
        
        # 显示前几个识别片段作为示例
        print(f"\n📋 前5个片段示例:")
        for i, segment in enumerate(segments[:5]):
            japanese_text = segment['text'].strip()
            if enable_translation:
                from translator import baidu_translate
                chinese_text = baidu_translate(japanese_text)
                print(f"   {i+1}. 日语: {japanese_text}")
                print(f"      中文: {chinese_text}")
                time.sleep(0.2)  # 避免请求过快
            else:
                print(f"   {i+1}. 日语: {japanese_text}")
    
    # 字幕合并到视频
    if merge_to_video and success:
        print("\n🎬 开始字幕合并到视频...")
        if check_ffmpeg_installed():
            # 确定字幕语言
            subtitle_language = 'chi' if enable_translation else 'jpn'
            
            # 合并字幕到视频
            merge_success = merge_subtitle_to_video(
                video_path=video_path,
                subtitle_path=subtitle_path,
                subtitle_language=subtitle_language
            )
            
            if merge_success:
                print("✅ 字幕已成功嵌入视频文件中")
                print("💡 播放时可在播放器字幕菜单中选择内置字幕")
            else:
                print("⚠️ 字幕合并失败，保留独立的字幕文件")
        else:
            print("❌ FFmpeg未安装，无法合并字幕到视频")
            print("💡 请安装FFmpeg或使用外部播放器加载字幕文件")
    
    # 清理临时文件
    try:
        # 检查audio_path变量是否存在且文件存在
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
            print("🧹 临时文件已清理")
        else:
            print("📝 无临时音频文件需要清理")
    except Exception as e:
        print(f"⚠️ 清理临时文件时出错: {e}")
    
    # 进度文件管理
    progress_file = get_progress_file_path(video_path)
    if clean_progress:
        # 清理进度文件
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
                print("🧹 进度文件已清理")
            except Exception as e:
                print(f"⚠️ 无法清理进度文件: {e}")
    else:
        # 默认保留进度文件以便断点续传
        if os.path.exists(progress_file):
            print(f"📁 进度文件已保留: {progress_file}")
            print("💡 如需清理进度文件，请使用 --clean-progress 参数或手动删除")
    
    print_section_header("处理完成")
    print_success("视频字幕处理已完成！")

if __name__ == "__main__":
    import traceback
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='视频字幕识别与翻译工具')
    parser.add_argument('video_path', nargs='?', help='视频文件路径')
    parser.add_argument('--test', action='store_true', help='测试模式（仅处理前10%%内容）')
    parser.add_argument('--model', default='medium', choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper模型大小（默认：medium）')
    parser.add_argument('--no-translate', action='store_true', help='仅识别不翻译')
    parser.add_argument('--output-dir', help='输出目录')
    parser.add_argument('--adult', action='store_true', help='成人内容模式（优化专业术语翻译）')
    parser.add_argument('--merge', action='store_true', help='将字幕合并到视频文件中（需要FFmpeg）')
    parser.add_argument('--clean-progress', action='store_true', help='清理进度文件（默认保留）')
    
    args = parser.parse_args()
    
    # 模型推荐：成人内容建议使用medium或large
    if args.adult and args.model in ['tiny', 'base', 'small']:
        print("⚠️  成人内容建议使用medium或large模型以获得更好的识别精度")
    
    # 检查FFmpeg是否已安装（如果启用了合并功能）
    if args.merge and not check_ffmpeg_installed():
        print("⚠️  FFmpeg未安装，字幕合并功能将不可用")
        print("💡  请安装FFmpeg：https://ffmpeg.org/download.html")
        print("💡  或者使用外部播放器加载独立的字幕文件")
    
    try:
        main(
            video_path=args.video_path,
            test_mode=args.test,
            model_size=args.model,
            enable_translation=not args.no_translate,
            output_dir=args.output_dir,
            adult_content=args.adult,
            merge_to_video=args.merge,
            clean_progress=args.clean_progress
        )
    except KeyboardInterrupt:
        print("\n\n🛑 程序已被用户中断")
    except Exception as e:
        print(f"\n\n❌ 程序运行时发生未捕获的异常: {e}")
        traceback.print_exc()
    finally:
        # 程序结束时保存翻译缓存
        from translator import save_translation_cache, load_translation_cache
        cache_data = load_translation_cache()
        save_translation_cache(cache_data)
