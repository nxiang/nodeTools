#!/usr/bin/env node

/**
 * 海浪背景音优化测试脚本
 * 专门针对有海浪背景音的日语语音识别进行测试
 * 使用优化后的主程序直接处理音频文件
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 测试音频文件路径
const audioFile = path.join(__dirname, 'temp', 'noise_reduced_audio.wav');

async function testSeawaveOptimization() {
    console.log('🌊 海浪背景音优化测试');
    console.log('='.repeat(50));
    
    // 检查音频文件是否存在
    if (!fs.existsSync(audioFile)) {
        console.log('❌ 音频文件不存在，请先运行降噪测试脚本');
        console.log('💡 运行命令: node noise_reduction_test.js');
        return;
    }
    
    console.log('📁 音频文件:', audioFile);
    
    // 获取音频文件信息
    const stats = fs.statSync(audioFile);
    const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);
    console.log(`📊 文件大小: ${fileSizeMB}MB`);
    
    // 检查音频时长
    try {
        const durationOutput = execSync(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${audioFile}"`, {
            encoding: 'utf8'
        });
        const duration = parseFloat(durationOutput.trim());
        console.log(`⏱️  音频时长: ${duration.toFixed(2)}秒`);
    } catch (error) {
        console.log('⚠️  无法获取音频时长');
    }
    
    console.log('\n🎯 优化参数说明:');
    console.log('   - 语音识别: 放宽beam值，提高对噪音的宽容度');
    console.log('   - 音频处理: 窄带滤波，专门过滤海浪噪音');
    console.log('   - 噪声抑制: 频域噪声抑制 + 语音归一化');
    console.log('   - 动态压缩: 快速响应语音信号');
    
    console.log('\n🚀 开始优化测试...');
    
    try {
        // 使用优化后的主程序直接处理音频文件
        const command = `node videoSubtitleRecognitionAndTranslation.js --audio "${audioFile}" --source-lang ja --translate --target-lang zh`;
        console.log(`💻 执行命令: ${command}`);
        
        execSync(command, { 
            stdio: 'inherit',
            cwd: __dirname 
        });
        
        console.log('✅ 优化测试完成');
        
        // 检查生成的字幕文件（字幕文件应该生成在temp目录中）
        const subtitleFile = path.join(__dirname, 'temp', 'noise_reduced_audio.ass');
        if (fs.existsSync(subtitleFile)) {
            console.log(`📝 生成的字幕文件: ${subtitleFile}`);
            
            // 读取字幕文件内容
            const subtitleContent = fs.readFileSync(subtitleFile, 'utf8');
            const segmentCount = (subtitleContent.match(/Dialogue:/g) || []).length;
            console.log(`📊 识别到的字幕片段: ${segmentCount}个`);
            
            // 显示部分字幕内容
            const lines = subtitleContent.split('\n').filter(line => line.startsWith('Dialogue:'));
            if (lines.length > 0) {
                console.log('\n📄 部分字幕内容:');
                lines.slice(0, 5).forEach((line, index) => {
                    const textMatch = line.match(/,,([^\\]+)/);
                    if (textMatch) {
                        console.log(`   ${index + 1}. ${textMatch[1].trim()}`);
                    }
                });
                if (lines.length > 5) {
                    console.log(`   ... 还有 ${lines.length - 5} 个片段`);
                }
            }
        }
        
    } catch (error) {
        console.error('❌ 优化测试失败:', error.message);
        
        // 检查是否有其他错误信息
        if (error.stderr) {
            console.error('📋 详细错误信息:');
            console.error(error.stderr.toString());
        }
    }
}

// 运行测试
testSeawaveOptimization().catch(console.error);
