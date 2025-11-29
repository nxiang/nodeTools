"""
进度管理模块
负责翻译进度的保存、加载和清理
"""

import os
import json
import time
from pathlib import Path

def get_progress_file_path(video_path):
    """获取进度文件路径"""
    video_name = Path(video_path).stem
    return f"temp/{video_name}_progress.json"

def save_progress(video_path, progress_data):
    """保存翻译进度到文件"""
    try:
        progress_file = get_progress_file_path(video_path)
        
        # 确保目录存在
        os.makedirs("temp", exist_ok=True)
        
        # 添加时间戳
        progress_data['last_save_time'] = time.time()
        
        # 写入文件
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"❌ 进度保存失败: {e}")
        return False

def load_progress(video_path):
    """从文件加载翻译进度"""
    progress_file = get_progress_file_path(video_path)
    
    # 检查文件是否存在
    if not os.path.exists(progress_file):
        print(f"📝 进度文件不存在: {progress_file}")
        return None
    
    # 检查文件大小
    file_size = os.path.getsize(progress_file)
    if file_size == 0:
        print(f"⚠️ 进度文件为空: {progress_file}")
        cleanup_progress(video_path)
        return None
    
    try:
        # 读取并解析JSON
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        
        # 验证数据完整性
        if not isinstance(progress_data, dict):
            print(f"❌ 进度文件格式错误: 不是字典类型")
            cleanup_progress(video_path)
            return None
        
        # 检查关键字段
        required_fields = ['video_path', 'last_translated_index', 'srt_content']
        for field in required_fields:
            if field not in progress_data:
                print(f"❌ 进度文件缺少关键字段: {field}")
                cleanup_progress(video_path)
                return None
        
        # 验证视频路径匹配
        if progress_data.get('video_path') != video_path:
            print(f"⚠️ 进度文件与当前视频不匹配，可能已更换视频文件")
            cleanup_progress(video_path)
            return None
        
        print(f"✅ 进度文件加载成功: {progress_file}")
        return progress_data
        
    except json.JSONDecodeError as e:
        print(f"❌ 进度文件JSON解析错误: {e}")
        cleanup_progress(video_path)
        return None
    except IOError as e:
        print(f"❌ 进度文件读取错误: {e}")
        return None
    except Exception as e:
        print(f"❌ 进度文件加载异常: {e}")
        cleanup_progress(video_path)
        return None

def cleanup_progress(video_path):
    """清理损坏的进度文件"""
    progress_file = get_progress_file_path(video_path)
    
    if os.path.exists(progress_file):
        try:
            os.remove(progress_file)
            print(f"🗑️ 已清理损坏的进度文件: {progress_file}")
        except Exception as e:
            print(f"⚠️ 进度文件清理失败: {e}")

def get_same_dir_subtitle_path(video_path):
    """获取与视频同目录的字幕文件路径"""
    video_dir = Path(video_path).parent
    video_name = Path(video_path).stem
    return str(video_dir / f"{video_name}.srt")

def check_progress_completion(video_path):
    """检查翻译是否已完成"""
    progress_data = load_progress(video_path)
    
    if progress_data and progress_data.get('completed', False):
        output_path = progress_data.get('output_path', '')
        if output_path and os.path.exists(output_path):
            return {
                'completed': True,
                'subtitle_file': output_path,
                'progress_data': progress_data
            }
    
    return {'completed': False}

def get_progress_summary(video_path):
    """获取进度摘要信息"""
    progress_data = load_progress(video_path)
    
    if not progress_data:
        return {
            'status': 'not_started',
            'message': '翻译尚未开始'
        }
    
    if progress_data.get('completed', False):
        return {
            'status': 'completed',
            'message': '翻译已完成',
            'subtitle_file': progress_data.get('output_path', ''),
            'completion_time': progress_data.get('completion_time', '')
        }
    
    # 正在翻译中
    total_segments = progress_data.get('total_segments', 0)
    last_translated = progress_data.get('last_translated_index', 0)
    progress_percent = progress_data.get('progress_percent', 0)
    
    return {
        'status': 'in_progress',
        'message': f'翻译进行中: {last_translated}/{total_segments} ({progress_percent}%)',
        'progress': progress_percent,
        'translated_segments': last_translated,
        'total_segments': total_segments,
        'last_save_time': progress_data.get('last_save_time', '')
    }
