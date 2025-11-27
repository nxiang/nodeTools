import { execSync } from 'child_process';
import fs from 'fs';

async function noiseReductionTest() {
    const videoPath = 'OAE-233.mp4';
    const audioPath = 'temp/noise_reduced_audio.wav';
    
    console.log('🎯 针对背景噪音环境的音频处理优化...');
    console.log('🌊 检测到海浪背景音，使用更强的噪声抑制');
    
    // 先检查视频文件是否存在
    if (!fs.existsSync(videoPath)) {
        console.error('❌ 视频文件不存在:', videoPath);
        return;
    }
    
    // 针对海浪背景音的优化参数
    const ffmpegArgs = [
        '-i', videoPath,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-t', '60', // 处理前60秒
        // 针对海浪噪声的优化处理
        '-af', 'highpass=f=200,lowpass=f=4000', // 更窄的频率范围，聚焦人声
        '-af', 'volume=6.0',                    // 更强的音量增强
        '-af', 'compand=attacks=0.05:decays=0.1:points=-90/-90|-60/-30|-20/-10|0/0', // 快速压缩
        '-af', 'afftdn=nf=-25:nr=90',          // 频域噪声抑制
        '-af', 'speechnorm=e=6:r=0.0001:l=1', // 语音归一化
        '-y',
        audioPath
    ];
    
    try {
        console.log('🎵 提取并处理音频（强噪声抑制）...');
        execSync(`ffmpeg ${ffmpegArgs.map(arg => `"${arg}"`).join(' ')}`, { stdio: 'inherit' });
        
        // 检查音频文件
        const stats = fs.statSync(audioPath);
        console.log(`✅ 音频提取完成，文件大小: ${(stats.size / (1024 * 1024)).toFixed(2)}MB`);
        
        // 检查音量
        const volumeOutput = execSync(`ffmpeg -i ${audioPath} -af "volumedetect" -f null - 2>&1`).toString();
        const meanVolumeMatch = volumeOutput.match(/mean_volume: ([\-0-9.]+) dB/);
        if (meanVolumeMatch) {
            console.log(`📊 音频平均音量: ${meanVolumeMatch[1]} dB`);
        }
        
        console.log('\n🎤 现在使用优化的语音识别参数进行测试...');
        
        // 创建优化的语音识别测试
        await testSpeechRecognitionWithOptimizedParams(audioPath);
        
    } catch (error) {
        console.error('❌ 音频处理失败:', error.message);
    }
}

async function testSpeechRecognitionWithOptimizedParams(audioPath) {
    console.log('🔧 使用优化的语音识别参数...');
    
    // 这里需要修改主程序的语音识别参数
    // 创建一个临时的优化版本
    const optimizedProcessorCode = `
        // 针对背景噪音优化的语音识别参数
        const rec = new vosk.Recognizer({ 
            model: model, 
            sampleRate: sampleRate,
            beam: 0.1,           // 放宽beam值，提高识别灵敏度
            lattice_beam: 0.01,  // 放宽lattice_beam
            maxActive: 5000,     // 增加maxActive
            maxAlternatives: 5,  // 增加备选结果数量
            word_confidence: true,
            min_active: 50,      // 降低min_active，提高灵敏度
            max_active: 10000    // 增加max_active
        });
    `;
    
    console.log('📝 优化参数已准备，建议修改主程序的语音识别部分');
    console.log('💡 主要优化方向：');
    console.log('   - 降低识别阈值（beam, lattice_beam）');
    console.log('   - 增加备选结果数量（maxAlternatives）');
    console.log('   - 提高识别灵敏度（min_active）');
    
    // 直接运行主程序进行测试
    console.log('\n🚀 运行主程序进行优化测试...');
    try {
        execSync(`node videoSubtitleRecognitionAndTranslation.js --audio ${audioPath} --source-lang ja --translate --target-lang zh --skip-cleanup`, {
            stdio: 'inherit'
        });
    } catch (error) {
        console.log('⚠️  主程序不支持直接处理音频文件，需要修改主程序');
        console.log('💡 建议修改主程序以支持：');
        console.log('   1. 直接处理音频文件');
        console.log('   2. 使用优化的语音识别参数');
        console.log('   3. 增强背景噪音抑制能力');
    }
}

noiseReductionTest();
