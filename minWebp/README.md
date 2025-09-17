# 智能去水印工具

这是一个基于Node.js的智能去水印工具，集成了多种云端AI服务，提供高质量的水印去除功能。

## 功能特性

- 🎯 **多算法融合**: 结合深度学习、PatchMatch和云端AI服务
- ☁️ **云端API集成**: 支持Remove.bg、Cleanup.pictures和LaMa等云端服务
- 🤖 **智能修复**: 基于AI的图像修复和纹理合成
- 🎨 **自然融合**: 高级色彩校正和边缘感知处理
- 📁 **批量处理**: 支持目录批量处理

## 安装依赖

```bash
npm install
```

## 配置API密钥

1. 复制环境变量配置文件：
```bash
copy .env.example .env
```

2. 编辑 `.env` 文件，填入您的API密钥：

```env
# Remove.bg API密钥
# 注册地址: https://www.remove.bg/api
REMOVE_BG_API_KEY=your_remove_bg_api_key_here

# Cleanup.pictures API密钥
# 注册地址: https://cleanup.pictures/api
CLEANUP_PICTURES_API_KEY=your_cleanup_pictures_api_key_here

# LaMa API密钥 (使用Replicate)
# 注册地址: https://replicate.com/
LAMA_API_KEY=your_replicate_api_key_here

# 可选：设置API请求超时时间（毫秒）
API_TIMEOUT=300000

# 可选：设置重试次数
MAX_RETRIES=3
```

## API服务注册

### Remove.bg
1. 访问 [Remove.bg官网](https://www.remove.bg/api)
2. 注册账户并获取API密钥
3. 免费额度：每月50次调用

### Cleanup.pictures
1. 访问 [Cleanup.pictures官网](https://cleanup.pictures/api)
2. 注册账户并获取API密钥
3. 提供免费试用额度

### LaMa (通过Replicate)
1. 访问 [Replicate官网](https://replicate.com/)
2. 注册账户并获取API令牌
3. 搜索并使用LaMa模型进行图像修复

## 使用方法

### 处理单个目录
```bash
node removeWatermark.js [输入目录] [输出目录] [处理方法]
```

参数说明：
- `输入目录`: 包含待处理图片的目录路径
- `输出目录`: 处理后图片的保存目录
- `处理方法`: 可选，支持以下方法：
  - `hybrid`: 混合多算法策略（默认）
  - `deep_learning`: 深度学习修复
  - `patchmatch`: PatchMatch算法
  - `professional`: 专业修复算法
  - `text_only`: 仅去除文字水印

### 示例
```bash
# 使用混合算法处理当前目录的图片
node removeWatermark.js ./input ./output hybrid

# 使用深度学习算法
node removeWatermark.js ./input ./output deep_learning

# 仅去除文字水印
node removeWatermark.js ./input ./output text_only
```

## 支持的图片格式

- WebP (.webp)
- JPEG (.jpg, .jpeg)
- PNG (.png)

## 处理流程

1. **水印检测**: 自动检测图像中的水印区域
2. **多算法修复**: 
   - 深度学习修复（40%权重）
   - PatchMatch算法（30%权重）
   - 云端AI服务（30%权重）
3. **结果融合**: 智能融合多种算法的结果
4. **后处理**: 色彩校正和边缘平滑

## 云端API集成

### Remove.bg API
- 专用于背景移除和图像清理
- 高质量的人像和物体分割
- 自动处理复杂背景

### Cleanup.pictures API
- 专业的图像清理和修复
- 去除不需要的物体和瑕疵
- 保持图像质量

### LaMa API
- 基于深度学习的图像修复
- 大型遮罩修复
- 保持纹理和结构

## 错误处理

- 如果某个云端API调用失败，系统会自动尝试其他API
- 如果所有云端API都不可用，会回退到本地算法
- 详细的错误日志会显示在控制台

## 性能优化

- 支持异步并行处理
- 智能缓存机制
- 自动重试失败请求
- 超时控制

## 注意事项

1. **API配额**: 各云端服务都有调用限制，请合理使用
2. **网络连接**: 需要稳定的网络连接来调用云端API
3. **图片大小**: 建议单张图片不超过10MB
4. **处理时间**: 云端API处理时间可能较长，请耐心等待

## 故障排除

### 常见问题

**Q: API调用失败**
A: 检查API密钥是否正确，网络连接是否正常

**Q: 处理结果不理想**
A: 尝试使用不同的处理方法，或调整水印检测参数

**Q: 程序运行缓慢**
A: 云端API处理需要时间，可以尝试减少同时处理的图片数量

### 调试模式

设置环境变量启用调试模式：
```bash
set DEBUG=watermark
node removeWatermark.js
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### v1.0.0
- 初始版本发布
- 集成云端API服务
- 支持多种去水印算法
- 批量处理功能
