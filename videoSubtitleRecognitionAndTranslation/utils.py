"""
工具函数模块
包含各种辅助函数和工具
"""

import os
import sys
import time
import hashlib
import subprocess
from pathlib import Path

def check_ffmpeg_installed():
    """检查FFmpeg是否已安装"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                             capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False

def get_file_hash(file_path, algorithm='md5', chunk_size=8192):
    """计算文件的哈希值"""
    hash_func = getattr(hashlib, algorithm)()
    
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        print(f"❌ 文件哈希计算失败: {e}")
        return None

def format_file_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def format_duration(seconds):
    """格式化时间长度"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}时{minutes}分{secs:.1f}秒"

def safe_filename(filename, max_length=255):
    """生成安全的文件名"""
    # 移除非法字符
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # 限制长度
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    return filename

def progress_bar(percentage, width=50, fill='█', empty=' '):
    """生成进度条字符串"""
    filled_length = int(width * percentage / 100)
    bar = fill * filled_length + empty * (width - filled_length)
    return f"[{bar}] {percentage}%"

def count_files_in_directory(directory, pattern="*"):
    """统计目录中匹配模式的文件数量"""
    try:
        path = Path(directory)
        if not path.exists():
            return 0
        return len(list(path.glob(pattern)))
    except Exception as e:
        print(f"❌ 文件统计失败: {e}")
        return 0

def cleanup_old_files(directory, max_age_days=7, pattern="*"):
    """清理指定目录中的旧文件"""
    try:
        path = Path(directory)
        if not path.exists():
            return 0
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        deleted_count = 0
        
        for file_path in path.glob(pattern):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    deleted_count += 1
        
        if deleted_count > 0:
            print(f"🗑️ 已清理 {deleted_count} 个超过 {max_age_days} 天的旧文件")
        
        return deleted_count
    except Exception as e:
        print(f"❌ 文件清理失败: {e}")
        return 0

def ensure_directory(directory):
    """确保目录存在"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"❌ 目录创建失败: {e}")
        return False

def is_video_file(file_path):
    """检查是否为视频文件"""
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
    return Path(file_path).suffix.lower() in video_extensions

def is_audio_file(file_path):
    """检查是否为音频文件"""
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    return Path(file_path).suffix.lower() in audio_extensions

def get_file_info(file_path):
    """获取文件信息"""
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        
        stat = path.stat()
        return {
            'name': path.name,
            'size': stat.st_size,
            'size_formatted': format_file_size(stat.st_size),
            'modified_time': time.ctime(stat.st_mtime),
            'created_time': time.ctime(stat.st_ctime),
            'is_video': is_video_file(file_path),
            'is_audio': is_audio_file(file_path)
        }
    except Exception as e:
        print(f"❌ 文件信息获取失败: {e}")
        return None

def retry_with_backoff(func, max_retries=3, base_delay=1, max_delay=10):
    """带指数退避的重试装饰器"""
    def wrapper(*args, **kwargs):
        retries = 0
        while retries <= max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                
                delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                print(f"⚠️ 操作失败，{delay}秒后重试 ({retries}/{max_retries})...")
                time.sleep(delay)
    
    return wrapper

def measure_execution_time(func):
    """测量函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"⏱️  {func.__name__} 执行时间: {execution_time:.2f}秒")
        
        return result
    
    return wrapper

def validate_file_path(file_path, check_exists=True):
    """验证文件路径"""
    try:
        path = Path(file_path)
        
        if check_exists and not path.exists():
            return False, f"文件不存在: {file_path}"
        
        if not path.is_file():
            return False, f"不是有效的文件: {file_path}"
        
        return True, ""
    except Exception as e:
        return False, f"文件路径验证失败: {e}"

def get_available_disk_space(directory='.'):
    """获取可用磁盘空间"""
    try:
        stat = os.statvfs(directory)
        free_space = stat.f_bavail * stat.f_frsize
        return free_space
    except AttributeError:
        # Windows系统使用不同的方法
        import ctypes
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(directory), None, None, ctypes.pointer(free_bytes)
        )
        return free_bytes.value
    except Exception as e:
        print(f"❌ 磁盘空间获取失败: {e}")
        return 0

def print_section_header(title, width=60):
    """打印章节标题"""
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width)

def print_success(message):
    """打印成功消息"""
    print(f"✅ {message}")

def print_warning(message):
    """打印警告消息"""
    print(f"⚠️  {message}")

def print_error(message):
    """打印错误消息"""
    print(f"❌ {message}")

def print_info(message):
    """打印信息消息"""
    print(f"ℹ️  {message}")

def create_backup(file_path, backup_suffix='.bak'):
    """创建文件备份"""
    try:
        backup_path = file_path + backup_suffix
        import shutil
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        print(f"❌ 备份创建失败: {e}")
        return None
