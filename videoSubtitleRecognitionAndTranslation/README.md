# 视频字幕识别与翻译工具

一个基于Whisper模型和百度翻译API的视频字幕识别与翻译工具，支持日语语音识别和中文翻译，生成双语字幕。

## 项目结构

模块化重构后的项目结构清晰，功能分离明确：

```
videoSubtitleRecognitionAndTranslation/
├── main.py                 # 主入口模块，整合所有功能
├── model_manager.py        # 模型管理模块
├── audio_processor.py      # 音频处理模块
├── subtitle_generator.py   # 字幕生成模块
├── translator.py           # 翻译模块
├── progress_manager.py     # 进度管理模块
├── config.py               # 配置管理模块
├── utils.py                # 工具函数模块
├── temp/                   # 临时文件目录
└── README.md              # 项目说明
```

## 模块功能说明

### main.py - 主入口模块
- 程序入口点和命令行参数解析
- 系统检测和配置验证
- 整体流程控制
- 异常处理和用户交互

### model_manager.py - 模型管理模块
- Whisper模型加载和缓存管理
- 模型完整性验证和错误处理
- 基于视频时长的自动模型选择
- CPU/GPU模式适配

### audio_processor.py - 音频处理模块
- 视频音频提取和格式转换
- FFmpeg命令执行和错误处理
- 音频时长获取和测试模式支持
- 临时文件清理

### subtitle_generator.py - 字幕生成模块
- Whisper语音识别执行
- 双语字幕生成和格式处理
- 进度显示和断点续传
- 时间格式化和字幕文件输出

### translator.py - 翻译模块
- 百度翻译API调用和签名生成
- 批量翻译和智能分割
- 翻译质量检查和错误重试
- 成人内容专业术语过滤

### progress_manager.py - 进度管理模块
- 断点续传进度文件管理
- 进度保存、加载和验证
- 进度清理和状态恢复
- 字幕文件路径管理

### config.py - 配置管理模块
- 百度翻译API配置管理
- 成人内容专业术语词典
- Whisper模型配置参数
- 系统配置和验证

### utils.py - 工具函数模块
- 文件操作和格式转换
- 进度显示和装饰器
- 安全处理和错误恢复
- 目录统计和文件清理

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用
```bash
# 处理当前目录下的视频文件
python main.py

# 指定视频文件路径
python main.py video.mp4

# 测试模式（仅处理前10%内容）
python main.py --test

# 指定模型大小
python main.py --model large

# 仅识别不翻译
python main.py --no-translate
```

### 高级功能
```bash
# 成人内容模式（优化专业术语翻译）
python main.py --adult

# 将字幕合并到视频文件中（需要FFmpeg）
python main.py --merge

# 清理进度文件
python main.py --clean-progress
```

## 配置说明

### 百度翻译API配置
在代码中配置百度翻译API的APP ID和密钥：
```python
# 在config.py中配置
BAIDU_TRANSLATE_CONFIG = {
    'appid': 'your_app_id',
    'key': 'your_secret_key',
    'url': 'http://api.fanyi.baidu.com/api/trans/vip/translate'
}
```

### 成人内容专业术语
支持成人内容专业术语的优化翻译，相关词汇在`config.py`中配置。

## 功能特点

1. **智能模型选择** - 根据视频时长自动选择最优的Whisper模型
2. **断点续传** - 支持进度保存和恢复，避免重复处理
3. **批量翻译优化** - 智能合并文本，减少API调用次数
4. **翻译质量检查** - 自动检测翻译结果质量
5. **成人内容优化** - 专业术语词典支持
6. **多格式支持** - 支持MP4、MKV、AVI等常见视频格式
7. **字幕合并** - 可选将字幕嵌入视频文件

## 注意事项

1. 需要安装FFmpeg用于音频提取和字幕合并
2. 需要百度翻译API的APP ID和密钥
3. 首次运行需要下载Whisper模型文件
4. 建议使用测试模式验证功能后再进行完整处理

## 故障排除

### 常见问题
1. **FFmpeg未安装** - 安装FFmpeg并确保在PATH中
2. **百度翻译API错误** - 检查APP ID和密钥配置
3. **模型下载失败** - 检查网络连接或手动下载模型
4. **内存不足** - 使用较小的模型或增加系统内存

### 日志和调试
- 程序会显示详细的处理进度和状态信息
- 临时目录包含进度文件和翻译缓存
- 错误信息会详细显示以便排查问题

## 许可证

本项目仅供学习和研究使用，请遵守相关法律法规。
