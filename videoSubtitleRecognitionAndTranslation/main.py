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

def main(video_path=None, test_mode=None, model_size='medium', enable_translation=True, 
         output_dir=None, adult_content=False, merge_to_video=False, clean=False, optimize_low_speech=False):
    """主函数"""
    
    # 记录总处理时间开始
    total_start_time = time.time()
    
    # 初始化各阶段耗时统计字典
    time_stats = {
        'total': 0,
        'audio_extraction': 0,
        'speech_recognition': 0,
        'subtitle_generation': 0,
        'subtitle_merging': 0
    }
    
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
    
    # 清理模式：在程序开始时删除temp目录下除视频文件外的所有文件
    if clean:
        try:
            video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv']
            temp_dir = "temp"
            files_cleaned = 0
            
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    # 只删除文件，不删除子目录
                    if os.path.isfile(file_path):
                        # 检查是否是视频文件
                        is_video = any(file.lower().endswith(ext) for ext in video_extensions)
                        if not is_video:
                            os.remove(file_path)
                            files_cleaned += 1
            
            print(f"🧹 已清理temp目录中{files_cleaned}个非视频文件")
        except Exception as e:
            print(f"⚠️ 清理temp目录失败: {e}")
    
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
    print(f"🔬 测试模式: {'开启' if test_mode else '关闭'} {'(' + str(test_mode) + '% 视频内容)' if test_mode else ''}")
    print(f"🔧 使用Whisper {selected_model_size}模型 {'+ 百度翻译API' if enable_translation else ''}")
    print(f"⚡ 低语音量优化: {'启用' if optimize_low_speech else '禁用'} {'(仅处理有语音的部分)' if optimize_low_speech else ''}")
    if enable_translation and args.time_offset != 0:
        print(f"⏱️  字幕时间偏移: {args.time_offset}秒 {'(延迟)' if args.time_offset > 0 else '(提前)'}")
    
    # 创建临时目录
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 设置当前视频名称并加载翻译缓存
    from translator import load_translation_cache, set_current_video_name
    set_current_video_name(video_path)
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
        # 提取音频（使用视频名称作为音频文件名，便于缓存和断点续传）
        video_name = Path(video_path).stem
        audio_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
        
        # 检查是否已有音频文件，避免重复提取
        if os.path.exists(audio_path):
            print(f"✅ 发现已存在的音频文件: {audio_path}，跳过提取步骤")
            speech_segments = None
        else:
            # 测试模式下根据视频总时长的百分比计算提取时长
            segment_duration = None
            if test_mode:
                try:
                    # 使用ffprobe获取视频总时长
                    import subprocess
                    duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                                   'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
                    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                    if duration_result.returncode == 0:
                        total_duration = float(duration_result.stdout.strip())
                        # 计算测试时长（总时长的N%）
                        segment_duration = total_duration * (test_mode / 100)
                        print(f"🔬 测试模式：提取前 {test_mode}% 的视频内容（约 {segment_duration:.2f} 秒）")
                except Exception as e:
                    print(f"⚠️ 获取视频时长失败: {e}，默认使用前60秒进行测试")
                    segment_duration = 60
            # 记录音频提取开始时间
            audio_start_time = time.time()
            # 提取音频，启用低语音量优化
            extract_result = extract_audio_segment(video_path, audio_path, segment_duration=segment_duration, optimize_for_low_speech=optimize_low_speech)
            # 兼容原函数返回值
            if isinstance(extract_result, tuple):
                audio_success, speech_segments = extract_result
            else:
                audio_success, speech_segments = extract_result, None
                
            if not audio_success:
                return
            # 记录音频提取完成时间
            time_stats['audio_extraction'] = time.time() - audio_start_time
            print(f"💾 音频文件已保存: {audio_path}，用于后续断点续传")
            print(f"⏱️  音频提取耗时: {time_stats['audio_extraction']:.2f}秒")
            
            # 如果有语音段信息，保存到进度中
            if speech_segments:
                progress['speech_segments'] = speech_segments
                save_progress(video_path, progress)
        
        # 使用Whisper进行语音识别（CPU模式，支持进度显示和断点续传）
        model = setup_whisper_model(selected_model_size)
        # 记录语音识别开始时间
        recognition_start_time = time.time()
        result = transcribe_with_whisper(model, audio_path, selected_model_size)
        # 记录语音识别完成时间
        time_stats['speech_recognition'] = time.time() - recognition_start_time
        print(f"⏱️  语音识别耗时: {time_stats['speech_recognition']:.2f}秒")
    
    if not result:
        return
    
    # 生成字幕文件（与视频文件同名且在同一目录）
    subtitle_path = get_same_dir_subtitle_path(video_path)
    
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
    
    # 记录字幕生成开始时间
    subtitle_start_time = time.time()
    
    if enable_translation:
        success = generate_bilingual_subtitle_file(video_path, result, enable_translation=True, 
                                                 adult_content=adult_content, progress=progress, 
                                                 time_offset=args.time_offset)
        if args.time_offset != 0:
            print(f"⏱️  字幕时间偏移已设置: {args.time_offset}秒")
    else:
        # 仅生成日语字幕
        success = generate_japanese_only_subtitle(result, subtitle_path, time_offset=args.time_offset)
        if args.time_offset != 0:
            print(f"⏱️  字幕时间偏移已设置: {args.time_offset}秒")
    
    # 记录字幕生成完成时间
    time_stats['subtitle_generation'] = time.time() - subtitle_start_time
    print(f"⏱️  字幕生成耗时: {time_stats['subtitle_generation']:.2f}秒")
    
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
                print(f"   {i+1}. 日语: {japanese_text[:40]}{'...' if len(japanese_text) > 40 else ''}")
                print(f"      中文: {chinese_text[:40]}{'...' if len(chinese_text) > 40 else ''}")
                time.sleep(0.2)  # 避免请求过快
            else:
                print(f"   {i+1}. 日语: {japanese_text[:40]}{'...' if len(japanese_text) > 40 else ''}")
    
    # 字幕合并到视频
    if merge_to_video and success:
        print("\n🎬 开始字幕合并到视频...")
        if check_ffmpeg_installed():
            # 确定字幕语言
            subtitle_language = 'chi' if enable_translation else 'jpn'
            
            # 记录字幕合并开始时间
            merge_start_time = time.time()
            
            # 合并字幕到视频
            merge_success = merge_subtitle_to_video(
                video_path=video_path,
                subtitle_path=subtitle_path,
                subtitle_language=subtitle_language
            )
            
            # 记录字幕合并完成时间
            time_stats['subtitle_merging'] = time.time() - merge_start_time
            
            if merge_success:
                print("✅ 字幕已成功嵌入视频文件中")
                print("💡 播放时可在播放器字幕菜单中选择内置字幕")
                print(f"⏱️  字幕合并耗时: {time_stats['subtitle_merging']:.2f}秒")
            else:
                print("⚠️ 字幕合并失败，保留独立的字幕文件")
        else:
            print("❌ FFmpeg未安装，无法合并字幕到视频")
    
    # 不再清理音频文件，保留用于断点续传
    if 'audio_path' in locals() and os.path.exists(audio_path):
        print("💾 音频文件已保留，用于后续断点续传")
    
    # 保留进度文件以便断点续传（如果没有在开始时清理）
    progress_file = get_progress_file_path(video_path)
    if not clean and os.path.exists(progress_file):
        print("📁 进度文件已保留，用于后续断点续传")
    
    # 计算总处理时间
    time_stats['total'] = time.time() - total_start_time
    
    # 显示各阶段耗时统计
    print_section_header("处理完成")
    print("⏱️  处理阶段耗时统计（秒）:")
    print(f"   - 音频提取: {time_stats['audio_extraction']:.2f}秒")
    print(f"   - 语音识别: {time_stats['speech_recognition']:.2f}秒")
    print(f"   - 字幕生成: {time_stats['subtitle_generation']:.2f}秒")
    if merge_to_video:
        print(f"   - 字幕合并: {time_stats['subtitle_merging']:.2f}秒")
    print(f"   - 总处理时间: {time_stats['total']:.2f}秒")
    
    print_success("视频字幕处理已完成！")

if __name__ == "__main__":
    import traceback
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='视频字幕识别与翻译工具')
    parser.add_argument('video_path', nargs='?', help='视频文件路径')
    parser.add_argument('--test', type=int, default=None, nargs='?', const=10, choices=range(1, 101), help='测试模式：指定语音识别前N%%视频时间长度（1-100，默认10）')
    parser.add_argument('--model', default='medium', choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper模型大小（默认：medium）')
    parser.add_argument('--no-translate', action='store_true', help='仅识别不翻译')
    parser.add_argument('--output-dir', help='输出目录')
    parser.add_argument('--adult', action='store_true', help='成人内容模式（优化专业术语翻译）')
    parser.add_argument('--merge', action='store_true', help='将字幕合并到视频文件中（需要FFmpeg）')
    parser.add_argument('--clean', action='store_true', help='清理temp目录下除视频文件外的所有文件')
    parser.add_argument('--optimize-low-speech', action='store_true', help='针对低语音量场景优化处理速度（例如2小时视频但说话很少）')
    parser.add_argument('--time-offset', type=float, default=0.0, help='字幕时间偏移（秒），正值表示字幕延迟，负值表示字幕提前')
    
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
            test_mode=args.test,  # 传入整数值而不是布尔值
            model_size=args.model,
            enable_translation=not args.no_translate,
            output_dir=args.output_dir,
            adult_content=args.adult,
            merge_to_video=args.merge,
            clean=args.clean,
            optimize_low_speech=getattr(args, 'optimize_low_speech', False)
        )
    except KeyboardInterrupt:
        print("\n\n🛑 程序已被用户中断")
    except Exception as e:
        print(f"\n\n❌ 程序运行时发生未捕获的异常: {e}")
        traceback.print_exc()
    finally:
        # 程序结束时保存翻译缓存
        try:
            # 确保当前视频名称已设置
            if 'video_path' in locals():
                set_current_video_name(video_path)
            # 保存缓存（使用全局缓存，避免重复加载）
            from translator import save_translation_cache
            save_translation_cache()
        except Exception as e:
            print(f"⚠️ 保存翻译缓存失败: {e}")
