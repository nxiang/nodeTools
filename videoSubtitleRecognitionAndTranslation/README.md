# 视频字幕识别与翻译工具

一个基于Node.js和AI技术的视频字幕生成与翻译工具，支持离线运行和ASS格式字幕文件生成。

## 功能特性

- 🎯 **支持多种视频格式**: MP4, AVI, MKV, MOV, WMV, FLV, WebM
- 🎤 **离线语音识别**: 使用Vosk进行本地语音识别，无需网络连接
- 🌐 **字幕翻译**: 支持中英文互译（基于规则翻译）
- 📝 **ASS格式输出**: 生成标准的ASS字幕文件
- 🔧 **批量处理**: 支持批量处理目录中的所有视频文件
- 🚀 **高性能**: 利用FFmpeg进行高效音频提取

## 系统要求

- Node.js >= 14.0.0
- FFmpeg (必须安装)
- 至少2GB可用磁盘空间用于模型文件

## 安装步骤

### 1. 安装FFmpeg

**Windows:**
- 下载FFmpeg: https://ffmpeg.org/download.html
- 解压并添加bin目录到系统PATH环境变量

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update && sudo apt install ffmpeg
```

### 2. 安装项目依赖

#### 使用 npm (推荐新手)
```bash
# 克隆或下载项目文件
cd nodeTools

# 安装Node.js依赖
npm install
```

#### 使用 pnpm (推荐，更快更节省空间)
```bash
# 克隆或下载项目文件
cd nodeTools

# 安装pnpm (如果未安装)
npm install -g pnpm

# 使用pnpm安装依赖
pnpm install
```

#### 使用安装脚本 (推荐)
```bash
# 运行安装脚本，自动检测包管理器
install.bat
```

### 3. 下载语音识别模型

```bash
# 创建模型目录
mkdir vosk-models

# 下载中文语音识别模型
cd vosk-models
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip vosk-model-small-cn-0.22.zip
```

或者手动下载并解压到 `vosk-models/vosk-model-small-cn-0.22/` 目录

## 使用方法

### 基本使用

```bash
# 处理单个视频文件
node VideoSubtitleRecognitionAndTranslation.js video.mp4

# 启用翻译功能
node VideoSubtitleRecognitionAndTranslation.js video.mp4 --translate

# 指定输出目录
node VideoSubtitleRecognitionAndTranslation.js video.mp4 --output-dir ./subtitles

# 批量处理目录中的所有视频
node VideoSubtitleRecognitionAndTranslation.js --batch ./videos --translate
```

### 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--output-dir <目录>` | 输出目录 | 视频所在目录 |
| `--translate` | 启用翻译功能 | false |
| `--lang <语言代码>` | 目标语言 | en |
| `--batch <目录>` | 批量处理目录 | - |

### 编程方式使用

```javascript
const VideoProcessor = require('./VideoSubtitleRecognitionAndTranslation.js');

const processor = new VideoProcessor();

// 处理单个视频
processor.processVideo('video.mp4', {
    outputDir: './subtitles',
    enableTranslation: true,
    targetLanguage: 'en'
}).then(result => {
    console.log('处理完成:', result);
});

// 批量处理
processor.batchProcess('./videos', {
    enableTranslation: true
}).then(results => {
    console.log('批量处理完成:', results);
});
```

## 输出文件格式

生成的ASS字幕文件包含：

- **原始字幕**: 白色字体，20px大小，位于屏幕底部
- **翻译字幕**: 黄色字体，16px大小，位于原始字幕上方
- **时间轴**: 精确到百分之一秒
- **样式定义**: 符合ASS标准格式

## 技术架构

```
视频文件 → FFmpeg音频提取 → Vosk语音识别 → 文本处理 → ASS字幕生成
                    ↓
                翻译引擎（可选）
```

### 核心组件

1. **FFmpeg**: 视频音频提取和格式转换
2. **Vosk**: 离线语音识别引擎
3. **自定义翻译器**: 基于规则的翻译系统
4. **ASS生成器**: 标准字幕格式生成

## 性能优化建议

1. **模型选择**: 使用小型模型提高处理速度
2. **批量处理**: 合理安排处理顺序，避免内存溢出
3. **硬件加速**: 如有GPU，可考虑使用GPU加速版本
4. **分段处理**: 长视频可分段处理后再合并

## 故障排除

### 常见问题

**Q: 提示FFmpeg未安装**
A: 请确保FFmpeg已正确安装并添加到系统PATH

**Q: 语音识别准确率低**
A: 尝试使用更大的Vosk模型或调整音频参数

**Q: 处理速度慢**
A: 检查系统资源使用情况，考虑使用更小的模型

**Q: 翻译效果不佳**
A: 当前使用基于规则的翻译，可考虑集成更先进的翻译API

### 日志调试

启用详细日志输出：
```bash
DEBUG=* node VideoSubtitleRecognitionAndTranslation.js video.mp4
```

## 开发计划

- [ ] 集成更先进的翻译引擎
- [ ] 支持更多语言识别
- [ ] 添加GUI界面
- [ ] 支持实时字幕生成
- [ ] 添加字幕编辑功能

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 技术支持

如有问题，请提交Issue或联系开发团队。
