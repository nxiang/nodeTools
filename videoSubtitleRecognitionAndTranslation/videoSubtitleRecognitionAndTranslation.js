#!/usr/bin/env node

/**
 * 视频字幕识别与翻译工具
 * 支持常见视频格式的字幕生成与翻译
 * 可断网运行，生成ASS格式字幕文件
 */

import fs from 'fs';
import path from 'path';
import { spawn, execSync } from 'child_process';
import { fileURLToPath } from 'url';
import config from './config.js';
import https from 'https';
import crypto from 'crypto';

// 动态导入fetch（Node.js 18+ 内置，旧版本需要polyfill）
let fetch;
if (typeof globalThis.fetch === 'undefined') {
    fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
} else {
    fetch = globalThis.fetch;
}

// ESM兼容性处理
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class VideoSubtitleRecognitionAndTranslation {
    constructor() {
        this.supportedFormats = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'];
        this.voskModelPath = null;
        this.translationModel = null;
        this.currentLanguage = 'cn'; // 默认中文
        
        // 支持的语言配置
        this.supportedLanguages = {
            'cn': {
                name: '中文',
                model: 'vosk-model-small-cn-0.22',
                url: 'https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip'
            },
            'ja': {
                name: '日语',
                model: 'vosk-model-small-ja-0.22',
                url: 'https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip'
            },
            'en': {
                name: '英语',
                model: 'vosk-model-small-en-us-0.15',
                url: 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip'
            }
        };
    }

    /**
     * 检查依赖
     */
    async checkDependencies(sourceLang = 'cn') {
        console.log('🔍 检查依赖...');
        
        // 检查ffmpeg
        try {
            execSync('ffmpeg -version', { stdio: 'ignore' });
            console.log('✅ FFmpeg 已安装');
        } catch (error) {
            console.error('❌ FFmpeg 未安装，请先安装 FFmpeg');
            return false;
        }
        
        // 检查模型文件
        const modelName = this.supportedLanguages[sourceLang]?.model;
        const modelPath = path.join(__dirname, 'vosk-models', modelName);
        
        if (!fs.existsSync(modelPath)) {
            console.error(`❌ 语音识别模型未找到: ${modelPath}`);
            console.error(`   请下载 ${modelName} 模型并解压到 vosk-models 目录`);
            return false;
        }
        
        this.voskModelPath = modelPath;
        console.log(`✅ 语音识别模型已找到: ${modelName}`);
        return true;
    }

    /**
     * 提取音频 - 针对日语优化
     */
    async extractAudio(videoPath, outputPath, options = {}) {
        const { testMode = false, sourceLang = 'ja' } = options;
        console.log(`🎵 提取音频${testMode ? '（测试模式 - 仅提取前10%）' : ''}...`);
        
        return new Promise((resolve, reject) => {
            let ffmpegArgs = [
                '-i', videoPath,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1'
            ];
            
            // 针对日语音频优化处理
            if (sourceLang === 'ja') {
                // 日语音频处理优化 - 合并到一个-af参数中
                ffmpegArgs.push(
                    '-af', 'highpass=f=80,lowpass=f=8000,volume=1.5', // 合并音频滤镜
                    '-compression_level', '10'            // 提高压缩级别
                );
            }
            
            ffmpegArgs.push('-y');
            
            // 如果是测试模式，只提取视频前10%的音频
            if (testMode) {
                // 先获取视频总时长
                try {
                    const durationOutput = execSync(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${videoPath}"`, {
                        encoding: 'utf8',
                        stdio: ['ignore', 'pipe', 'ignore']
                    });
                    const totalDuration = parseFloat(durationOutput.trim());
                    const testDuration = totalDuration * 0.1; // 10% of total duration
                    
                    console.log(`📏 视频总时长: ${totalDuration.toFixed(2)}秒，测试模式提取: ${testDuration.toFixed(2)}秒`);
                    ffmpegArgs = [
                        '-i', videoPath,
                        '-vn',
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',
                        '-ac', '1'
                    ];
                    
                    // 日语优化参数
                    if (sourceLang === 'ja') {
                        ffmpegArgs.push(
                            '-af', 'highpass=f=80,lowpass=f=8000',
                            '-compression_level', '10',
                            '-af', 'volume=1.5'
                        );
                    }
                    
                    ffmpegArgs.push('-t', testDuration.toString(), '-y');
                } catch (durationError) {
                    console.warn('⚠️  无法获取视频时长，使用默认10秒测试片段');
                    ffmpegArgs.push('-t', '10');
                }
            }
            
            ffmpegArgs.push(outputPath);
            
            const ffmpeg = spawn('ffmpeg', ffmpegArgs);
            
            let progress = 0;
            ffmpeg.stderr.on('data', (data) => {
                const str = data.toString();
                const durationMatch = str.match(/Duration: ([0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+)/);
                const timeMatch = str.match(/time=([0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+)/);
                
                if (durationMatch && timeMatch) {
                    const durationStr = durationMatch[1];
                    const timeStr = timeMatch[1];
                    
                    const duration = this.timeToSeconds(durationStr);
                    const currentTime = this.timeToSeconds(timeStr);
                    
                    if (duration > 0) {
                        const newProgress = Math.min(100, Math.floor((currentTime / duration) * 100));
                        if (newProgress > progress && newProgress % 10 === 0) {
                            progress = newProgress;
                            console.log(`  - 音频提取进度: ${progress}%`);
                        }
                    }
                }
            });
            
            ffmpeg.on('close', (code) => {
                if (code === 0) {
                    console.log('✅ 音频提取完成');
                    resolve(outputPath);
                } else {
                    console.error('❌ 音频提取失败');
                    reject(new Error('FFmpeg 音频提取失败'));
                }
            });
            
            ffmpeg.on('error', (err) => {
                console.error('❌ FFmpeg 执行错误:', err.message);
                reject(err);
            });
        });
    }

    /**
     * 时间字符串转秒数
     */
    timeToSeconds(timeStr) {
        const [hours, minutes, seconds] = timeStr.split(':').map(parseFloat);
        return hours * 3600 + minutes * 60 + seconds;
    }

    /**
     * 获取音频文件时长（秒）
     */
    async getAudioDuration(audioPath) {
        return new Promise((resolve, reject) => {
            try {
                const durationOutput = execSync(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${audioPath}"`, {
                    encoding: 'utf8',
                    stdio: ['ignore', 'pipe', 'ignore']
                });
                const duration = parseFloat(durationOutput.trim());
                resolve(duration);
            } catch (error) {
                console.warn('⚠️  无法获取音频时长，使用默认估算:', error.message);
                // 如果无法获取时长，使用文件大小估算（假设16kHz 16bit单声道）
                const stats = fs.statSync(audioPath);
                const estimatedDuration = stats.size / (16000 * 2); // 16kHz * 2 bytes per sample
                resolve(estimatedDuration);
            }
        });
    }

    /**
     * 语音识别 - 流式处理版本（改进的实时识别）
     */
    async speechRecognition(audioPath, options = {}) {
        console.log('🎤 开始语音识别（流式处理）...');
        
        // 动态导入vosk以避免预加载
        const voskModule = await import('vosk');
        const vosk = voskModule.default;
        
        // 初始化Vosk
        vosk.setLogLevel(-1); // 禁用日志
        
        const model = new vosk.Model(this.voskModelPath);
        const sampleRate = 16000;
        
        // 获取文件大小用于进度计算
        const fileStats = fs.statSync(audioPath);
        const fileSize = fileStats.size;
        
        console.log(`📊 音频文件大小: ${(fileSize / (1024 * 1024)).toFixed(2)}MB`);
        
        // 获取音频实际时长
        const audioDuration = await this.getAudioDuration(audioPath);
        console.log(`⏱️  音频实际时长: ${audioDuration.toFixed(2)}秒`);
        
        const allSegments = [];
        let processedBytes = 0;
        let lastPartialText = '';
        let lastPartialTime = 0;
        let segmentBuffer = [];
        let segmentCount = 0; // 添加segmentCount变量定义
        const segmentThreshold = 1.2; // 降低静音检测阈值到1.2秒，提高时间精度
        
        // 创建识别器实例
        const rec = new vosk.Recognizer({ 
            model: model, 
            sampleRate: sampleRate,
            beam: 0.1,
            lattice_beam: 0.01,
            maxActive: 5000,
            maxAlternatives: 5,
            word_confidence: true
        });
        
        // 创建流式读取器
        const audioStream = fs.createReadStream(audioPath, {
            highWaterMark: 64 * 1024 // 64KB 块大小
        });
        
        // 实时保存间隔计数器
        let saveCounter = 0;
        const saveInterval = 5; // 每5个片段保存一次
        
        return new Promise((resolve, reject) => {
            audioStream.on('data', (chunk) => {
                try {
                    rec.acceptWaveform(chunk);
                    processedBytes += chunk.length;
                    const progress = ((processedBytes / fileSize) * 100).toFixed(2);
                    
                    // 实时显示进度 - 限制频率避免性能问题
                    if (processedBytes % (fileSize / 100) === 0) { // 每1%更新一次
                        process.stdout.write(`\r🔄 语音识别进度: ${progress}%`);
                    }
                    
                    // 获取部分结果用于实时显示
                    const partialResult = rec.partialResult();
                    if (partialResult && partialResult.partial) {
                        const currentText = partialResult.partial.trim();
                        
                        if (currentText && currentText.length > 0) {
                            // 计算当前音频时间 - 基于实际时长和字节比例
                            const currentTime = (processedBytes / fileSize) * audioDuration;
                            
                            // 检查是否有新内容
                            if (currentText !== lastPartialText) {
                                // 文本发生变化，可能是新内容
                                const timeDiff = currentTime - lastPartialTime;
                                
                                // 降低静音检测阈值，提高时间精度
                                if (timeDiff > segmentThreshold && lastPartialText.length > 3) {
                                    // 静音时间较长，认为是句子结束
                                    console.log(`\n🎯 检测到句子结束 (${currentTime.toFixed(2)}s): ${lastPartialText.substring(0, 50)}...`);
                                    
                                    // 保存当前句子到缓冲区
                                    if (lastPartialText.length > 5) {
                                        const segment = {
                                            text: lastPartialText,
                                            start: lastPartialTime,
                                            end: currentTime,
                                            confidence: this.calculateJapaneseConfidence(lastPartialText)
                                        };
                                        
                                        // 过滤低质量片段
                                        if (segment.confidence > 0.4) {
                                            segmentBuffer.push(segment);
                                            segmentCount++;
                                            
                                            // 实时保存到磁盘 - 提高保存频率
                                            saveCounter++;
                                            if (saveCounter >= saveInterval) {
                                                this.saveSegmentsToDisk(segmentBuffer, audioPath, {
                                                    processedBytes: processedBytes,
                                                    totalBytes: fileSize,
                                                    isPartial: true
                                                });
                                                saveCounter = 0;
                                                
                                                // 显示保存状态
                                                process.stdout.write(` 💾 已保存 ${segmentBuffer.length} 个片段`);
                                            }
                                        } else {
                                            console.log(`⚠️  跳过低质量片段 (置信度: ${segment.confidence.toFixed(2)}): ${lastPartialText.substring(0, 30)}...`);
                                        }
                                    }
                                }
                                
                                lastPartialText = currentText;
                                lastPartialTime = currentTime;
                            }
                            
                            // 显示当前识别的文本
                            if (currentText.length > 30) {
                                process.stdout.write(` (${currentText.substring(0, 30)}...)`);
                            } else {
                                process.stdout.write(` (${currentText})`);
                            }
                        }
                    }
                    
                } catch (err) {
                    console.error('❌ 处理音频数据时出错:', err.message);
                    // 继续处理，不中断流
                }
            });
            
            audioStream.on('end', async () => {
                try {
                    // 获取最终结果
                    const finalResult = rec.finalResult();
                    
                    if (finalResult && typeof finalResult === 'object' && finalResult.text) {
                        const finalText = finalResult.text.trim();
                        
                        if (finalText) {
                            // 处理最终结果，使用实际音频时长
                            const segments = this.processRecognitionResult(finalText, 0, audioDuration);
                            const uniqueSegments = this.removeDuplicateSegments(segments);
                            allSegments.push(...uniqueSegments);
                        }
                    }
                    
                    // 处理缓冲区中的实时片段
                    if (segmentBuffer.length > 0) {
                        console.log(`\n📝 处理实时识别片段: ${segmentBuffer.length} 个`);
                        segmentBuffer.forEach((segment, index) => {
                            // 使用实际检测到的时间戳，而不是重新计算
                            const processedSegments = [{
                                text: segment.text,
                                start: segment.start,
                                end: segment.end
                            }];
                            allSegments.push(...processedSegments);
                        });
                    }
                    
                    // 最终保存到磁盘
                    this.saveSegmentsToDisk(allSegments, audioPath, {
                        processedBytes: processedBytes,
                        totalBytes: fileSize,
                        isPartial: false
                    });
                    
                    // 释放资源
                    rec.free();
                    model.free();
                    
                    // 完成进度显示
                    process.stdout.write('\n');
                    console.log(`✅ 语音识别完成，共识别 ${allSegments.length} 个片段`);
                    
                    resolve(allSegments);
                    
                } catch (err) {
                    console.error('❌ 处理最终结果时出错:', err.message);
                    reject(err);
                }
            });
            
            audioStream.on('error', (err) => {
                console.error('❌ 读取音频文件时出错:', err.message);
                reject(err);
            });
        });
    }

    /**
     * 处理识别结果 - 针对日语优化
     */
    processRecognitionResult(text, startTime, audioDuration = null) {
        if (!text || text.trim().length === 0) {
            return [];
        }
        
        // 验证audioDuration参数
        if (audioDuration !== null && (isNaN(audioDuration) || audioDuration <= 0)) {
            console.warn('⚠️  音频时长参数无效，使用基于语速的估算');
            audioDuration = null;
        }
        
        // 预处理文本，提高识别质量
        const processedText = this.preprocessJapaneseText(text);
        
        const segments = [];
        let currentTime = startTime;
        
        // 改进的日语分割规则：更智能的句子分割
        const sentenceDelimiters = /[。！？.!?]+/;
        const sentences = processedText.split(sentenceDelimiters).filter(s => s.trim().length > 0);
        
        // 如果提供了音频时长，基于实际时长分配时间
        if (audioDuration && audioDuration > 0) {
            const totalCharacters = sentences.reduce((sum, sentence) => sum + sentence.trim().length, 0);
            
            sentences.forEach(sentence => {
                const sentenceText = sentence.trim();
                if (sentenceText.length === 0) return;
                
                // 基于字符比例分配时间，但限制最小和最大时长
                const proportion = sentenceText.length / totalCharacters;
                const duration = Math.max(1.0, Math.min(10.0, audioDuration * proportion)); // 限制在1-10秒之间
                
                const segment = {
                    start: currentTime,
                    end: currentTime + duration,
                    text: sentenceText,
                    confidence: this.calculateJapaneseConfidence(sentenceText)
                };
                
                segments.push(segment);
                currentTime += duration;
            });
        } else {
            // 如果没有音频时长，使用改进的基于语速的估算方法
            const avgWordsPerSecond = 4; // 提高日语平均语速到4字/秒
            
            sentences.forEach(sentence => {
                const sentenceText = sentence.trim();
                if (sentenceText.length === 0) return;
                
                // 计算假名比例，调整语速
                const hiraganaCount = (sentenceText.match(/[\u3040-\u309F]/g) || []).length;
                const katakanaCount = (sentenceText.match(/[\u30A0-\u30FF]/g) || []).length;
                const kanjiCount = (sentenceText.match(/[\u4E00-\u9FAF]/g) || []).length;
                const totalJapaneseChars = hiraganaCount + katakanaCount + kanjiCount;
                
                let speedFactor = 1.0;
                if (totalJapaneseChars > 0) {
                    const kanaRatio = (hiraganaCount + katakanaCount) / totalJapaneseChars;
                    // 假名比例高时语速较快，汉字比例高时语速较慢
                    speedFactor = 0.8 + (kanaRatio * 0.4);
                }
                
                // 基于字符数和语速计算时长，限制合理范围
                const textLength = sentenceText.length;
                const duration = Math.max(1.0, Math.min(8.0, textLength / (avgWordsPerSecond * speedFactor)));
                
                const segment = {
                    start: currentTime,
                    end: currentTime + duration,
                    text: sentenceText,
                    confidence: this.calculateJapaneseConfidence(sentenceText)
                };
                
                segments.push(segment);
                currentTime += duration;
            });
        }
        
        return segments;
    }
    
    /**
     * 计算日语文本的置信度
     */
    calculateJapaneseConfidence(text) {
        if (!text || text.trim().length === 0) return 0;
        
        const trimmedText = text.trim();
        let confidence = 0.7; // 基础置信度
        
        // 日语特征检查
        const japaneseChars = trimmedText.match(/[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]/g) || [];
        const japaneseRatio = japaneseChars.length / trimmedText.length;
        
        // 日语文本比例高则置信度高
        if (japaneseRatio > 0.8) {
            confidence += 0.2;
        } else if (japaneseRatio > 0.5) {
            confidence += 0.1;
        }
        
        // 检查是否有明显的日语语法特征
        const japaneseParticles = ['は', 'が', 'を', 'に', 'で', 'と', 'も', 'か', 'ね', 'よ'];
        const hasParticles = japaneseParticles.some(particle => trimmedText.includes(particle));
        if (hasParticles) {
            confidence += 0.1;
        }
        
        // 检查句子长度合理性
        if (trimmedText.length >= 3 && trimmedText.length <= 50) {
            confidence += 0.1;
        }
        
        return Math.min(1.0, confidence);
    }
    
    /**
     * 移除重复或相似的片段 - 针对日语优化
     */
    removeDuplicateSegments(segments) {
        if (!segments || segments.length === 0) {
            return [];
        }
        
        const uniqueSegments = [];
        const seenTexts = new Set();
        
        segments.forEach(segment => {
            if (!segment.text || segment.text.trim().length === 0) {
                return; // 跳过空文本
            }
            
            // 日语文本预处理
            const processedText = this.preprocessJapaneseText(segment.text.trim());
            const normalizedText = processedText.toLowerCase();
            
            // 检查是否为重复或相似的文本
            let isDuplicate = false;
            for (const seenText of seenTexts) {
                // 使用文本相似度检查（针对日语优化）
                if (this.isJapaneseDuplicate(normalizedText, seenText)) {
                    isDuplicate = true;
                    break;
                }
            }
            
            if (!isDuplicate) {
                seenTexts.add(normalizedText);
                uniqueSegments.push(segment);
            } else {
                console.log(`🔍 跳过重复片段: "${segment.text}" (置信度: ${segment.confidence?.toFixed(2) || 'N/A'})`);
            }
        });
        
        return uniqueSegments;
    }
    
    /**
     * 日语文本预处理
     */
    preprocessJapaneseText(text) {
        if (!text) return '';
        
        let processed = text;
        
        // 移除常见的识别错误和噪声
        processed = processed.replace(/[\s\n\r]+/g, ' '); // 标准化所有空白字符
        processed = processed.replace(/[、，]/g, '、'); // 统一日语逗号
        
        // 移除常见的无意义字符和噪声
        processed = processed.replace(/[\u0000-\u001F\u007F-\u009F]/g, ''); // 移除控制字符
        processed = processed.replace(/[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w\s、。！？.!?]/g, ''); // 移除非日语字符
        
        // 修正常见的日语识别错误
        const correctionMap = {
            'ん': 'ん',
            'ン': 'ン',
            'は': 'は',
            'が': 'が', 
            'を': 'を',
            'に': 'に',
            'で': 'で',
            'と': 'と',
            'も': 'も',
            'か': 'か',
            'ね': 'ね',
            'よ': 'よ',
            'え': 'え',
            'お': 'お',
            'あ': 'あ',
            'い': 'い',
            'う': 'う'
        };
        
        // 简单的字符修正
        Object.keys(correctionMap).forEach(wrong => {
            const correct = correctionMap[wrong];
            processed = processed.replace(new RegExp(wrong, 'g'), correct);
        });
        
        // 移除重复的假名（如"んんん"）
        processed = processed.replace(/([\u3040-\u309F\u30A0-\u30FF])\1{2,}/g, '$1$1');
        
        return processed.trim();
    }
    
    /**
     * 检查日语文本是否重复 - 改进版本
     */
    isJapaneseDuplicate(text1, text2) {
        if (!text1 || !text2) return false;
        
        // 如果文本完全相同，直接返回true
        if (text1 === text2) return true;
        
        // 如果文本长度差异太大，不认为是重复
        const lengthRatio = Math.min(text1.length, text2.length) / Math.max(text1.length, text2.length);
        if (lengthRatio < 0.6) return false;
        
        // 计算Jaccard相似度（针对日语优化）
        const set1 = new Set(text1.split(''));
        const set2 = new Set(text2.split(''));
        const intersection = new Set([...set1].filter(x => set2.has(x)));
        const union = new Set([...set1, ...set2]);
        const similarity = intersection.size / union.size;
        
        // 提高相似度阈值到0.85
        if (similarity > 0.85) {
            // 检查是否有相同的日语助词
            const japaneseParticles = ['は', 'が', 'を', 'に', 'で', 'と', 'も', 'か', 'ね', 'よ'];
            const hasCommonParticles = japaneseParticles.some(particle => 
                text1.includes(particle) && text2.includes(particle)
            );
            
            return hasCommonParticles;
        }
        
        return false;
    }
    
    /**
     * 计算文本相似度（简单的Jaccard相似度）
     */
    calculateTextSimilarity(text1, text2) {
        if (!text1 || !text2) return 0;
        
        const words1 = new Set(text1.split(/\s+/));
        const words2 = new Set(text2.split(/\s+/));
        
        const intersection = new Set([...words1].filter(word => words2.has(word)));
        const union = new Set([...words1, ...words2]);
        
        return union.size === 0 ? 0 : intersection.size / union.size;
    }
    
    /**
     * 实时保存识别结果到磁盘 - 优化合并机制
     */
    saveSegmentsToDisk(segments, audioPath, options) {
        try {
            const videoName = path.basename(audioPath, path.extname(audioPath));
            const tempDir = path.dirname(audioPath);
            
            // 保存识别结果
            const segmentsFile = path.join(tempDir, `${videoName}_segments.json`);
            
            // 保存进度信息（用于断点续传）
            const progressFile = path.join(tempDir, `${videoName}_progress.json`);
            
            // 如果是部分保存，同时更新进度信息和片段文件
            if (options.isPartial) {
                let allSegments = [];
                
                // 如果已有片段文件，读取并合并历史数据
                if (fs.existsSync(segmentsFile)) {
                    try {
                        const existingData = fs.readFileSync(segmentsFile, 'utf8');
                        const existingSegments = JSON.parse(existingData);
                        allSegments = existingSegments;
                    } catch (error) {
                        console.warn('⚠️  读取历史片段文件失败，将创建新文件:', error.message);
                    }
                }
                
                // 合并新片段（去重处理）
                let newSegmentsCount = 0;
                segments.forEach(newSegment => {
                    const isDuplicate = allSegments.some(existingSegment => 
                        this.isJapaneseDuplicate(newSegment.text, existingSegment.text) ||
                        (Math.abs(existingSegment.start - newSegment.start) < 0.5 && 
                         Math.abs(existingSegment.end - newSegment.end) < 0.5)
                    );
                    if (!isDuplicate) {
                        allSegments.push(newSegment);
                        newSegmentsCount++;
                    }
                });
                
                // 按时间戳排序
                allSegments.sort((a, b) => a.start - b.start);
                
                const progressInfo = {
                    totalSegments: allSegments.length,
                    lastUpdate: new Date().toISOString(),
                    segmentsFile: segmentsFile,
                    processedBytes: options.processedBytes || 0,
                    totalBytes: options.totalBytes || 0,
                    isPartial: true
                };
                
                // 实时保存片段文件和进度信息
                fs.writeFileSync(segmentsFile, JSON.stringify(allSegments, null, 2));
                fs.writeFileSync(progressFile, JSON.stringify(progressInfo, null, 2));
                process.stdout.write(` 💾 实时保存 ${allSegments.length} 个片段 (新增 ${newSegmentsCount} 个)`);
            } else {
                // 最终保存：合并所有历史数据并保存
                let allSegments = [];
                
                // 如果已有片段文件，读取并合并历史数据
                if (fs.existsSync(segmentsFile)) {
                    try {
                        const existingData = fs.readFileSync(segmentsFile, 'utf8');
                        const existingSegments = JSON.parse(existingData);
                        allSegments = existingSegments;
                        console.log(`📂 加载历史片段: ${existingSegments.length} 个`);
                    } catch (error) {
                        console.warn('⚠️  读取历史片段文件失败，将创建新文件:', error.message);
                    }
                }
                
                // 合并新片段（去重处理）
                let newSegmentsCount = 0;
                segments.forEach(newSegment => {
                    const isDuplicate = allSegments.some(existingSegment => 
                        this.isJapaneseDuplicate(newSegment.text, existingSegment.text) ||
                        (Math.abs(existingSegment.start - newSegment.start) < 0.5 && 
                         Math.abs(existingSegment.end - newSegment.end) < 0.5)
                    );
                    if (!isDuplicate) {
                        allSegments.push(newSegment);
                        newSegmentsCount++;
                    }
                });
                
                // 按时间戳排序
                allSegments.sort((a, b) => a.start - b.start);
                
                // 保存合并后的完整片段和进度信息
                const progressInfo = {
                    totalSegments: allSegments.length,
                    lastUpdate: new Date().toISOString(),
                    segmentsFile: segmentsFile,
                    processedBytes: options.processedBytes || 0,
                    totalBytes: options.totalBytes || 0,
                    isPartial: false
                };
                
                fs.writeFileSync(segmentsFile, JSON.stringify(allSegments, null, 2));
                fs.writeFileSync(progressFile, JSON.stringify(progressInfo, null, 2));
                console.log(`\n💾 最终保存识别结果: ${allSegments.length} 个片段（新增 ${newSegmentsCount} 个）`);
            }
            
        } catch (error) {
            console.warn('⚠️  保存识别结果到磁盘时出错:', error.message);
        }
    }

    /**
     * 翻译文本 - 改进版本
     */
    async translateText(text, sourceLang, targetLang) {
        try {
            // 如果没有文本，直接返回空字符串
            if (!text || text.trim().length === 0) {
                return '';
            }
            
            // 预处理文本，提高翻译质量
            const processedText = this.preprocessJapaneseText(text);
            
            // 检查文本质量，过滤低质量文本
            const confidence = this.calculateJapaneseConfidence(processedText);
            if (confidence < 0.5) {
                console.log(`⚠️  跳过低质量文本翻译 (置信度: ${confidence.toFixed(2)}): ${processedText.substring(0, 30)}...`);
                return '';
            }
            
            // 检查文本长度，过滤过短或过长的文本
            if (processedText.length < 3) {
                console.log(`⚠️  跳过过短文本翻译: ${processedText}`);
                return '';
            }
            
            if (processedText.length > 200) {
                console.log(`⚠️  文本过长，截断处理: ${processedText.substring(0, 50)}...`);
                // 截断过长的文本
                const truncatedText = processedText.substring(0, 200);
                return await this.baiduTranslate(truncatedText, sourceLang, targetLang);
            }
            
            // 使用百度翻译API
            console.log(`🌐 开始翻译片段 (置信度: ${confidence.toFixed(2)}): ${processedText.substring(0, 30)}${processedText.length > 30 ? '...' : ''}`);
            const translated = await this.baiduTranslate(processedText, sourceLang, targetLang);
            console.log(`✅ 翻译成功: ${processedText.substring(0, 20)}${processedText.length > 20 ? '...' : ''} -> ${translated.substring(0, 20)}${translated.length > 20 ? '...' : ''}`);
            return translated;
        } catch (error) {
            console.error(`❌ 翻译失败: ${error.message}`);
            // 错误码58001表示语言参数无效，提供更明确的错误信息
            if (error.message.includes('58001')) {
                console.error(`   提示: 请检查语言代码是否正确，百度API使用'jp'而非'ja'表示日语`);
            }
            return ''; // 失败时返回空字符串，避免显示错误翻译
        }
    }
    
    /**
     * 使用百度翻译API进行翻译
     */
    async baiduTranslate(text, sourceLang, targetLang) {
        // 语言代码映射 - 修正日语代码为'jp'
        const langMap = {
            'cn': 'zh',
            'zh': 'zh',
            'ja': 'jp',
            'en': 'en'
        };
        
        const from = langMap[sourceLang] || 'auto';
        const to = langMap[targetLang] || 'zh';
        
        console.log(`🌐 调用百度翻译: ${from} -> ${to}`);
        
        // 获取配置
        const { appid, key, apiUrl } = config.baidu;
        
        // 生成签名
        const salt = Math.floor(Math.random() * 10000000000);
        const sign = crypto.createHash('md5').update(appid + text + salt + key).digest('hex');
        
        // 构建请求参数
        const params = new URLSearchParams({
            q: text,
            from: from,
            to: to,
            appid: appid,
            salt: salt,
            sign: sign
        });
        
        // 构建URL
        const url = `${apiUrl}?${params.toString()}`;
        
        // 发送请求
        const response = await new Promise((resolve, reject) => {
            const req = https.get(url, {
                timeout: config.timeout || 10000,
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            }, (res) => {
                let data = '';
                
                res.on('data', (chunk) => {
                    data += chunk;
                });
                
                res.on('end', () => {
                    try {
                        const result = JSON.parse(data);
                        if (result.error_code) {
                            reject(new Error(`百度翻译API错误: ${result.error_code} - ${result.error_msg}`));
                        } else {
                            resolve(result);
                        }
                    } catch (err) {
                        reject(new Error('解析翻译结果失败'));
                    }
                });
            });
            
            req.on('error', (err) => {
                reject(err);
            });
            
            req.on('timeout', () => {
                req.destroy();
                reject(new Error('请求超时'));
            });
        });
        
        // 提取翻译结果
        if (response.trans_result && response.trans_result.length > 0) {
            return response.trans_result[0].dst;
        }
        
        throw new Error('未获取到翻译结果');
    }

    /**
     * 应用日语语法规则
     */
    applyJapaneseGrammarRules(text) {
        let result = text;
        
        // 处理常见的日语语法结构 - 使用更合适的正则表达式
        result = result.replace(/([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)は/g, '$1');
        result = result.replace(/が/g, '');
        result = result.replace(/を/g, '');
        result = result.replace(/に/g, '在');
        result = result.replace(/ます/g, '');
        result = result.replace(/ました/g, '了');
        result = result.replace(/ません/g, '不');
        result = result.replace(/です/g, '是');
        result = result.replace(/でした/g, '曾是');
        result = result.replace(/か/g, '吗');
        
        // 清理多余的空格和标点
        result = result.replace(/\s+/g, ' ').trim();
        
        return result;
    }

    /**
     * 生成ASS字幕
     */
    generateASSSubtitle(segments, outputPath, translatedSegments = null) {
        console.log('📝 生成ASS字幕...');
        
        // 使用正确的编码格式和换行符，确保中文显示正常
        let assContent = `[Script Info]
Title: Auto-generated subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei UI,32,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,0
Style: Translation,Microsoft YaHei UI,36,&H00FF00FF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,2,8,10,10,40,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;
        
        segments.forEach((segment, index) => {
            const startTime = this.formatTime(segment.start);
            const endTime = this.formatTime(segment.end);
            
            // 添加原始字幕
            assContent += `Dialogue: 0,${startTime},${endTime},Default,,0,0,0,,${segment.text}\N\N\N\N
`;
            
            // 如果有翻译，添加翻译字幕
            if (translatedSegments && translatedSegments[index]) {
                assContent += `Dialogue: 0,${startTime},${endTime},Translation,,0,0,0,,${translatedSegments[index].text}\N
`;
            }
        });
        
        // 使用UTF-8 BOM编码保存，确保中文在各种播放器中正常显示
        const bom = Buffer.from([0xEF, 0xBB, 0xBF]);
        const contentBuffer = Buffer.from(assContent, 'utf8');
        const finalBuffer = Buffer.concat([bom, contentBuffer]);
        fs.writeFileSync(outputPath, finalBuffer);
        console.log('✅ ASS字幕文件生成完成:', outputPath);
        console.log('📝 字幕使用UTF-8 BOM编码，支持中文显示');
    }

    /**
     * 格式化时间（ASS格式）
     */
    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        const centisecs = Math.floor((seconds % 1) * 100);
        
        return `${hours.toString().padStart(1, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${centisecs.toString().padStart(2, '0')}`;
    }

    /**
     * 处理视频文件
     */
    async processVideo(videoPath, options = {}) {
        const {
            outputDir = path.dirname(videoPath),
            enableTranslation = false,
            targetLanguage = 'en',
            sourceLanguage = 'cn',
            skipCleanup = false,
            testMode = false
        } = options;

        console.log('🚀 开始处理视频:', videoPath);
        console.log(`🌐 识别语言: ${this.supportedLanguages[sourceLanguage]?.name || sourceLanguage}`);
        if (enableTranslation) {
            console.log(`🌐 翻译目标: ${targetLanguage}`);
        }
        if (testMode) {
            console.log('🔬 测试模式: 仅处理视频前10%内容');
        }

        // 检查文件格式
        const ext = path.extname(videoPath).toLowerCase();
        if (!this.supportedFormats.includes(ext)) {
            throw new Error(`不支持的文件格式: ${ext}`);
        }

        // 检查依赖
        const depsReady = await this.checkDependencies(sourceLanguage);
        if (!depsReady) {
            throw new Error('依赖检查失败');
        }

        // 创建临时目录
        const tempDir = path.join(outputDir, 'temp');
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir, { recursive: true });
        }

        const baseName = path.basename(videoPath, ext);
        const audioPath = path.join(tempDir, `${baseName}_audio.wav`);
        const outputPath = path.join(outputDir, `${baseName}.ass`);
        // 将JSON文件保存到temp目录下
        const progressFile = path.join(tempDir, `${baseName}_progress.json`);
        const segmentsFile = path.join(tempDir, `${baseName}_segments.json`);
        const translatedSegmentsFile = path.join(tempDir, `${baseName}_translated_segments.json`);
        
        let segments = [];
        let translatedSegments = [];
        let shouldSkipAudioExtraction = false;
        
        // 检查是否已有字幕文件
        if (fs.existsSync(outputPath)) {
            console.log('⚠️  字幕文件已存在:', outputPath);
            console.log('   如需重新生成，请先删除该文件');
            return {
                success: false,
                message: '字幕文件已存在'
            };
        }
        
        // 检查temp目录中的JSON文件
        if (fs.existsSync(segmentsFile)) {
            try {
                segments = JSON.parse(fs.readFileSync(segmentsFile, 'utf8'));
                console.log(`📊 从temp目录发现 ${segments.length} 个已识别的片段，继续处理...`);
            } catch (error) {
                console.warn('⚠️  读取已识别片段失败，重新识别:', error.message);
                segments = [];
            }
        }
        
        // 检查是否有已翻译的片段
        if (enableTranslation) {
            if (fs.existsSync(translatedSegmentsFile)) {
                try {
                    translatedSegments = JSON.parse(fs.readFileSync(translatedSegmentsFile, 'utf8'));
                    console.log(`📊 从temp目录发现 ${translatedSegments.length} 个已翻译的片段`);
                } catch (error) {
                    console.warn('⚠️  读取已翻译片段失败，重新翻译:', error.message);
                    translatedSegments = [];
                }
            }
        }
        
        // 检查是否需要重新提取音频
        if (fs.existsSync(audioPath)) {
            if (testMode) {
                console.log('⚠️  测试模式下，重新提取音频以确保只处理前10%');
                shouldSkipAudioExtraction = false;
            } else {
                console.log('📁 发现已提取的音频文件，复用...');
                shouldSkipAudioExtraction = true;
            }
        }
        
        const startTime = Date.now();
        
        try {
            // 1. 提取音频
            if (!shouldSkipAudioExtraction) {
                await this.extractAudio(videoPath, audioPath, { testMode });
            }
            
            // 2. 语音识别
            const recognizeStartTime = Date.now();
            if (segments.length === 0) {
                const speechRecognitionOptions = {
                    language: sourceLanguage
                };
                segments = await this.speechRecognition(audioPath, speechRecognitionOptions);
                
                // 保存识别结果到temp目录
                fs.writeFileSync(segmentsFile, JSON.stringify(segments), 'utf8');
                console.log(`💾 已保存识别结果到temp目录: ${path.basename(segmentsFile)}`);
            } else {
                console.log(`📊 已有 ${segments.length} 个字幕片段，跳过语音识别`);
            }
            
            console.log(`✅ 语音识别完成 (耗时: ${((Date.now() - recognizeStartTime) / 1000).toFixed(2)}秒)`);
            
            if (segments.length === 0) {
                console.warn('⚠️  未识别到有效内容');
            }

            // 3. 翻译（如果启用）
            if (enableTranslation && segments.length > 0) {
                console.log('🌐 开始翻译...');
                
                // 找出尚未翻译的片段
                const translatedSegmentIds = new Set(translatedSegments.map(seg => `${seg.start}-${seg.end}`));
                const segmentsToTranslate = segments.filter(segment => 
                    segment.text && segment.text.trim().length > 0 && 
                    !translatedSegmentIds.has(`${segment.start}-${segment.end}`)
                );
                
                console.log(`🔍 已有翻译片段: ${translatedSegments.length}`);
                console.log(`🔍 需要翻译的片段: ${segmentsToTranslate.length}`);
                
                if (segmentsToTranslate.length > 0) {
                    // 批量翻译
                    for (const segment of segmentsToTranslate) {
                        try {
                            const translatedText = await this.translateText(segment.text, sourceLanguage, targetLanguage);
                            const newTranslation = {
                                ...segment,
                                text: translatedText
                            };
                            translatedSegments.push(newTranslation);
                            
                            // 实时保存翻译结果到temp目录
                            fs.writeFileSync(translatedSegmentsFile, JSON.stringify(translatedSegments), 'utf8');
                            console.log(`💾 实时保存翻译结果到temp目录`);
                        } catch (err) {
                            console.error(`❌ 翻译失败: ${err.message}`);
                            translatedSegments.push(segment);
                        }
                    }
                }
                
                console.log('✅ 翻译完成');
            }

            // 4. 生成ASS字幕
            this.generateASSSubtitle(segments, outputPath, translatedSegments);

            // 5. 清理临时文件，但保留音频和JSON文件用于调试
            if (!skipCleanup) {
                this.cleanupTempFiles(tempDir);
            } else {
                console.log('📁 保留临时文件用于调试');
            }
            
            const processingTime = (Date.now() - startTime) / 1000;
            console.log(`🎉 处理完成！`);
            console.log(`⏱️  总耗时: ${processingTime.toFixed(2)}秒`);
            console.log(`📁 输出文件: ${outputPath}`);
            
            return {
                success: true,
                subtitleFile: outputPath,
                segments: segments,
                translatedSegments: translatedSegments,
                processingTime: processingTime
            };

        } catch (error) {
            console.error('❌ 处理失败:', error.message);
            console.error('🔍 错误详情:', error.stack);
            
            throw error;
        }
    }

    /**
     * 清理临时文件，保留音频和JSON文件
     */
    cleanupTempFiles(tempDir) {
        if (fs.existsSync(tempDir)) {
            const files = fs.readdirSync(tempDir);
            const filesToKeep = [];
            
            files.forEach(file => {
                const filePath = path.join(tempDir, file);
                
                // 保留音频文件和JSON文件
                if (file.endsWith('.wav') || file.endsWith('.json')) {
                    filesToKeep.push(file);
                } else {
                    try {
                        fs.unlinkSync(filePath);
                    } catch (error) {
                        console.warn('⚠️  无法删除临时文件:', error.message);
                    }
                }
            });
            
            if (filesToKeep.length > 0) {
                console.log(`📁 保留的文件: ${filesToKeep.join(', ')}`);
            }
        }
    }
    
    /**
     * 批量处理视频
     */
    async batchProcess(videoDir, options = {}) {
        const files = fs.readdirSync(videoDir);
        const videoFiles = files.filter(file => 
            this.supportedFormats.includes(path.extname(file).toLowerCase())
        );

        console.log(`📁 发现 ${videoFiles.length} 个视频文件`);

        const results = [];
        for (const file of videoFiles) {
            const videoPath = path.join(videoDir, file);
            try {
                console.log(`\n🔧 处理: ${file}`);
                const result = await this.processVideo(videoPath, options);
                results.push(result);
            } catch (error) {
                console.error(`❌ 处理失败 ${file}:`, error.message);
                results.push({ success: false, file: file, error: error.message });
            }
        }

        return results;
    }
}

// 导出类
const normalizePath = (path) => {
    return path.replace(/\\/g, '/');
};

const importPath = normalizePath(import.meta.url.replace('file:///', ''));
const argvPath = normalizePath(process.argv[1]);

if (importPath === argvPath) {
    const args = process.argv.slice(2);
    
    if (args.length === 0) {
        const processor = new VideoSubtitleRecognitionAndTranslation();
        console.log(`
🎯 视频字幕识别与翻译工具 - 优化版

使用方法:
  node videoSubtitleRecognitionAndTranslation.js <视频文件路径> [选项]

选项:
  --output-dir <目录>     输出目录 (默认: 视频所在目录)
  --translate             启用翻译功能
  --source-lang <代码>    源语言 (默认: cn, 支持: cn, ja, en)
  --target-lang <代码>    目标语言 (默认: en, 支持: zh, en)
  --batch <目录>          批量处理目录中的所有视频
  --skip-cleanup          保留临时文件（用于调试）
  --test-mode             测试模式（仅处理视频前10%内容）

语言说明:
  cn - 中文识别, ja - 日语识别, en - 英语识别
  zh - 中文翻译, en - 英文翻译

示例:
  node videoSubtitleRecognitionAndTranslation.js video.mp4
  node videoSubtitleRecognitionAndTranslation.js japanese_video.mp4 --source-lang ja --translate --target-lang zh
  node videoSubtitleRecognitionAndTranslation.js video.mp4 --translate
  node videoSubtitleRecognitionAndTranslation.js --batch ./videos --translate
  node videoSubtitleRecognitionAndTranslation.js video.mp4 --skip-cleanup  # 保留临时文件用于调试
  node videoSubtitleRecognitionAndTranslation.js video.mp4 --test-mode  # 仅处理视频前10%

支持格式: ${processor.supportedFormats.join(', ')}
`);
        process.exit(0);
    }
    
    // 解析命令行参数
    const options = {};
    let videoPath = null;
    let batchDir = null;
    
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        
        if (arg === '--output-dir' && i + 1 < args.length) {
            options.outputDir = args[++i];
        } else if (arg === '--translate') {
            options.enableTranslation = true;
        } else if (arg === '--source-lang' && i + 1 < args.length) {
            options.sourceLanguage = args[++i];
        } else if (arg === '--target-lang' && i + 1 < args.length) {
            options.targetLanguage = args[++i];
        } else if (arg === '--batch' && i + 1 < args.length) {
            batchDir = args[++i];
        } else if (arg === '--skip-cleanup') {
            options.skipCleanup = true;
        } else if (arg === '--test-mode') {
            options.testMode = true;
        } else if (!videoPath && !arg.startsWith('--')) {
            videoPath = arg;
        }
    }
    
    // 验证必需的参数
    if (!batchDir && !videoPath) {
        console.error('❌ 错误: 必须指定视频文件或批量处理目录');
        process.exit(1);
    }
    
    // 执行处理
    const processor = new VideoSubtitleRecognitionAndTranslation();
    
    if (batchDir) {
        processor.batchProcess(batchDir, options)
            .then(results => {
                console.log('\n📊 批量处理完成！');
                console.log(`成功: ${results.filter(r => r.success).length}`);
                console.log(`失败: ${results.filter(r => !r.success).length}`);
            })
            .catch(error => {
                console.error('❌ 批量处理失败:', error.message);
                process.exit(1);
            });
    } else {
        processor.processVideo(videoPath, options)
            .catch(error => {
                console.error('❌ 处理失败:', error.message);
                process.exit(1);
            });
    }
}

export default VideoSubtitleRecognitionAndTranslation;
