# 百度翻译API集成配置指南

## 概述

本工具已成功集成百度翻译API，支持高质量的在线翻译功能。当配置了百度翻译API密钥后，系统将优先使用百度翻译API进行字幕翻译，提供更准确、更流畅的翻译体验。

## 配置步骤

### 1. 获取百度翻译API密钥

1. 访问 [百度翻译开放平台](https://api.fanyi.baidu.com/)
2. 注册/登录百度账号
3. 进入"管理控制台"
4. 创建新的翻译服务应用
5. 获取以下信息：
   - **APP ID** (应用ID)
   - **密钥** (Secret Key)

### 2. 配置API密钥

有三种方式配置API密钥：

#### 方式一：环境变量（推荐）

在系统环境变量中设置：

```bash
# Windows PowerShell
$env:BAIDU_TRANSLATE_APPID = "你的APP_ID"
$env:BAIDU_TRANSLATE_KEY = "你的SECRET_KEY"

# Windows CMD
set BAIDU_TRANSLATE_APPID=你的APP_ID
set BAIDU_TRANSLATE_KEY=你的SECRET_KEY

# Linux/Mac
 export BAIDU_TRANSLATE_APPID="你的APP_ID"
 export BAIDU_TRANSLATE_KEY="你的SECRET_KEY"
```

#### 方式二：修改配置文件

编辑 `config.js` 文件，直接设置密钥：

```javascript
const config = {
    baidu: {
        appid: '你的APP_ID',  // 直接填写
        key: '你的SECRET_KEY', // 直接填写
        apiUrl: 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    },
    // ... 其他配置
};
```

#### 方式三：临时环境变量

在运行命令前设置：

```bash
# Windows
set BAIDU_TRANSLATE_APPID=你的APP_ID && set BAIDU_TRANSLATE_KEY=你的SECRET_KEY && node videoSubtitleRecognitionAndTranslation.js

# Linux/Mac
BAIDU_TRANSLATE_APPID="你的APP_ID" BAIDU_TRANSLATE_KEY="你的SECRET_KEY" node videoSubtitleRecognitionAndTranslation.js
```

## 功能特性

### 智能翻译策略

1. **优先级管理**：百度翻译API > Google翻译API > 增强规则翻译
2. **自动降级**：当在线API不可用时，自动切换到本地增强翻译
3. **错误重试**：网络错误时自动重试（最多3次）
4. **超时控制**：10秒超时保护，避免长时间等待

### 支持的语言对

- 中文 ↔ 英文
- 日语 ↔ 中文
- 日语 ↔ 英文
- 中文 ↔ 日语
- 英文 ↔ 中文

### 翻译质量对比

| 翻译方式 | 准确性 | 流畅度 | 速度 | 成本 |
|---------|--------|--------|------|------|
| 百度翻译API | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 付费 |
| Google翻译API | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 免费/付费 |
| 增强规则翻译 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | 免费 |

## 使用示例

### 基本使用

```javascript
const processor = new VideoSubtitleRecognitionAndTranslation();

// 启用翻译功能
await processor.processVideo('input.mp4', {
    enableTranslation: true,
    sourceLanguage: 'ja',  // 日语
    targetLanguage: 'zh',  // 中文
    outputFormat: 'ass'
});
```

### 高级配置

```javascript
const processor = new VideoSubtitleRecognitionAndTranslation();

await processor.processVideo('input.mp4', {
    enableTranslation: true,
    sourceLanguage: 'ja',
    targetLanguage: 'zh',
    outputFormat: 'ass',
    translationMode: 'online',  // 强制使用在线翻译
    // 或者使用混合模式
    // translationMode: 'hybrid'  // 智能选择最佳翻译方式
});
```

## 故障排除

### 常见问题

1. **"百度翻译API密钥未配置"错误**
   - 检查环境变量是否正确设置
   - 重启终端或IDE使环境变量生效
   - 验证密钥是否有效

2. **"请求超时"错误**
   - 检查网络连接
   - 尝试增加超时时间（修改config.js中的timeout配置）
   - 使用代理服务器（如果需要）

3. **API调用频率限制**
   - 百度翻译API有调用频率限制
   - 考虑升级API套餐或使用备用翻译服务

### 调试模式

启用详细日志输出：

```javascript
const processor = new VideoSubtitleRecognitionAndTranslation();
processor.setDebugMode(true);  // 启用调试模式
```

## 成本控制

### 百度翻译API定价

- 标准版：每月免费额度 + 按量计费
- 高级版：更高配额和优先级
- 具体价格参考[百度翻译定价页面](https://api.fanyi.baidu.com/doc/21)

### 优化建议

1. **批量处理**：尽量批量处理视频，减少API调用次数
2. **缓存结果**：相同文本的翻译结果可以缓存
3. **本地预处理**：使用增强规则翻译处理简单词汇
4. **监控使用量**：定期检查API使用情况

## 技术支持

- 百度翻译API文档：https://api.fanyi.baidu.com/doc/21
- 项目问题反馈：GitHub Issues
- 配置问题咨询：查看项目README.md

## 更新日志

### v1.1.0 (当前版本)
- ✅ 集成百度翻译API
- ✅ 智能错误处理和重试机制
- ✅ 多API备用策略
- ✅ 完整的配置管理系统

---

**注意**：请妥善保管您的API密钥，不要将密钥提交到公开代码仓库中。
