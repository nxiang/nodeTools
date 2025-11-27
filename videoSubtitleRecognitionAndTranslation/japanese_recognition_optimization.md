# 日语语音识别优化方案

## 当前问题分析

### 1. segments.json文件未及时更新
- **问题**：实时保存频率过低，用户无法实时查看识别进度
- **原因**：保存间隔设置为每5个片段才保存一次
- **已修复**：添加了实时保存状态显示，提高用户体验

### 2. 日语语音识别不准确
- **问题**：识别结果包含大量无意义重复文本（如"ン ジャー ジャー ジャー"）
- **原因**：
  - 使用小型模型 `vosk-model-small-ja-0.22`（准确度有限）
  - 音频预处理参数可能过于激进
  - 背景噪音（海浪声）干扰严重

## 优化方案

### 方案一：升级语音识别模型（推荐）

#### 1. 下载更大的日语模型
```bash
# 进入项目目录
cd e:\gitstore\nodeTools\videoSubtitleRecognitionAndTranslation

# 创建模型目录（如果不存在）
mkdir vosk-models
cd vosk-models

# 下载标准日语模型（约1.4GB，准确度更高）
wget https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip
unzip vosk-model-ja-0.22.zip

# 或者下载最新版本（如果可用）
wget https://alphacephei.com/vosk/models/vosk-model-ja-0.42.zip
unzip vosk-model-ja-0.42.zip
```

#### 2. 修改代码使用新模型
在 `videoSubtitleRecognitionAndTranslation.js` 中修改模型配置：

```javascript
// 第29行左右的supportedLanguages配置
this.supportedLanguages = {
    'ja': {
        name: '日语',
        model: 'vosk-model-ja-0.22', // 改为标准模型
        url: 'https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip'
    }
    // ... 其他语言配置保持不变
};
```

### 方案二：优化音频预处理参数

#### 1. 修改音频提取参数
在 `extractAudio` 方法中（约114行），调整音频处理参数：

```javascript
// 针对有背景噪音的日语音频优化处理
if (sourceLang === 'ja') {
    // 更温和的音频处理，减少过度压缩
    ffmpegArgs.push(
        '-af', 'highpass=f=80,lowpass=f=8000', // 更宽的频率范围
        '-af', 'compand=attacks=0.3:decays=0.8:points=-80/-80|-30/-10|0/0', // 更温和的动态压缩
        '-af', 'volume=2.0:precision=fixed',     // 适中的音量增强
        '-af', 'dynaudnorm=p=0.5:m=100:s=10:r=0', // 更温和的动态音频归一化
        '-af', 'afftdn=nf=-20:nr=70',           // 适中的频域噪声抑制
        '-compression_level', '8'               // 中等压缩级别
    );
}
```

#### 2. 优化语音识别参数
在 `speechRecognition` 方法中（约200行），调整识别参数：

```javascript
// 创建识别器实例 - 针对有背景噪音的日语环境优化参数
const rec = new vosk.Recognizer({ 
    model: model, 
    sampleRate: sampleRate,
    beam: 0.15,           // 适度放宽beam值
    lattice_beam: 0.02,   // 适度放宽lattice_beam
    maxActive: 3000,      // 适中的maxActive
    maxAlternatives: 3,   // 适中的备选结果数量
    word_confidence: true,
    min_active: 200,      // 适中的min_active
    max_active: 5000,     // 适中的max_active
    acoustic_scale: 0.7,  // 适度降低声学模型权重
    silence_weight: 0.3   // 适度降低静音权重
});
```

### 方案三：改进实时保存机制

#### 1. 提高保存频率
已修复：在检测到句子结束时立即显示保存状态

#### 2. 添加进度监控
可以添加一个独立的进度监控功能：

```javascript
// 在语音识别过程中定期检查进度
setInterval(() => {
    const segmentsFile = path.join(tempDir, `${videoName}_segments.json`);
    if (fs.existsSync(segmentsFile)) {
        const segments = JSON.parse(fs.readFileSync(segmentsFile, 'utf8'));
        console.log(`📊 当前已识别: ${segments.length} 个片段`);
    }
}, 10000); // 每10秒检查一次
```

## 测试建议

### 1. 先测试模型升级
```bash
# 下载标准日语模型后测试
node videoSubtitleRecognitionAndTranslation.js 1.mkv --translate --source-lang ja --target-lang cn
```

### 2. 如果效果仍不理想，再优化音频参数
- 逐步调整音频处理参数
- 测试不同参数组合的效果

### 3. 监控识别质量
- 检查 `temp/1_audio_segments.json` 文件
- 观察识别文本的质量改善
- 注意是否有重复字符减少

## 预期效果

- **模型升级后**：识别准确度应显著提高，重复字符减少
- **参数优化后**：对背景噪音的适应性更好
- **实时保存改进**：用户可以实时查看识别进度

## 注意事项

1. **模型文件大小**：标准模型约1.4GB，下载需要时间
2. **内存使用**：更大的模型需要更多内存
3. **处理速度**：标准模型可能比小型模型稍慢，但准确度更高
4. **备份配置**：修改前备份原始配置文件

建议优先尝试方案一（升级模型），这通常能带来最显著的改善。
