"""
模型管理模块
负责Whisper模型的加载、缓存和配置
"""

import os
import whisper

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

def setup_whisper_model(model_size='medium'):
    """设置Whisper模型（CPU模式）"""
    print("💻 使用CPU处理模式")
    
    # 设置缓存路径，避免重复下载
    cache_dir = os.path.expanduser('~/.cache/whisper')
    os.environ['WHISPER_CACHE_DIR'] = cache_dir
    
    print(f"📥 加载Whisper {model_size}模型...")
    
    # 预期的模型文件大小（字节）- 更新为实际大小
    expected_sizes = {
        'tiny': 75_572_083,
        'base': 142_000_000,
        'small': 466_000_000,
        'medium': 1_528_008_539,
        'large': 3_087_371_615,  # 修正为实际文件大小，并统一使用'large'作为参数
    }
    
    # 检查本地缓存
    model_file = os.path.join(cache_dir, f'{model_size}.pt')
    
    # 对于large模型，检查是否存在large-v3.pt文件
    if model_size == 'large' and not os.path.exists(model_file):
        large_v3_file = os.path.join(cache_dir, 'large-v3.pt')
        if os.path.exists(large_v3_file):
            print(f"🔄 发现large-v3.pt文件，创建符号链接为large.pt")
            try:
                # 创建符号链接或复制文件
                if os.name == 'nt':  # Windows系统
                    import shutil
                    shutil.copy2(large_v3_file, model_file)
                else:  # Unix系统
                    os.symlink(large_v3_file, model_file)
                print(f"✅ 已创建large.pt文件")
            except Exception as e:
                print(f"⚠️ 无法创建large.pt文件: {e}")
    
    if os.path.exists(model_file):
        # 验证文件完整性
        file_size = os.path.getsize(model_file)
        expected_size = expected_sizes.get(model_size, 0)
        
        if expected_size > 0 and file_size < expected_size * 0.9:
            print(f"⚠️ 模型文件可能损坏: {file_size:,} bytes < 预期 {expected_size:,} bytes")
            print("🗑️ 删除损坏文件并重新下载...")
            try:
                os.remove(model_file)
            except Exception as e:
                print(f"❌ 删除失败: {e}")
        else:
            print(f"✅ 使用本地缓存模型: {model_file}")
            print(f"📊 文件大小: {file_size:,} bytes")
    else:
        print(f"📡 下载模型到缓存目录: {cache_dir}")
    
    # 线程安全的模型加载
    try:
        # 设置线程异常处理
        import threading
        threading.excepthook = lambda args: print(f"⚠️ 线程异常: {args.exc_type.__name__}: {args.exc_value}")
        
        # 抑制FP16警告
        import warnings
        warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
        
        # 加载模型
        model = whisper.load_model(model_size, device="cpu")
        print("✅ 模型加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        # 尝试使用更小的模型作为备选
        if model_size != 'tiny':
            print(f"🔄 尝试使用更小的模型作为备选...")
            # 按大小顺序尝试备选模型
            model_priority = ['medium', 'small', 'base', 'tiny']
            current_index = model_priority.index(model_size) if model_size in model_priority else 0
            
            for next_model in model_priority[current_index + 1:]:
                print(f"  尝试 {next_model} 模型...")
                try:
                    return setup_whisper_model(next_model)
                except:
                    continue
        
        # 如果所有备选都失败，抛出异常
        raise e

def auto_select_model(video_path, user_model_size='medium'):
    """根据视频时长自动选择模型大小"""
    try:
        import subprocess
        
        # 获取视频时长
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            
            # 根据时长推荐模型
            if duration <= 300:  # 5分钟以内
                recommended = 'small'
            elif duration <= 1800:  # 30分钟以内
                recommended = 'medium'
            else:  # 超过30分钟
                recommended = 'large'
            
            # 如果用户指定的模型比推荐的小，使用用户指定的
            model_sizes = ['tiny', 'base', 'small', 'medium', 'large']
            user_index = model_sizes.index(user_model_size) if user_model_size in model_sizes else 2
            recommended_index = model_sizes.index(recommended) if recommended in model_sizes else 2
            
            if user_index < recommended_index:
                print(f"⚠️  视频时长 {duration:.1f}秒 建议使用 {recommended} 模型，但将使用用户指定的 {user_model_size} 模型")
                return user_model_size
            else:
                print(f"📊 视频时长 {duration:.1f}秒，自动选择 {recommended} 模型")
                return recommended
        
    except Exception as e:
        print(f"⚠️  无法获取视频时长，使用默认模型: {e}")
    
    return user_model_size
