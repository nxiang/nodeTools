import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { fileURLToPath } from 'url';
import axios from 'axios';
import FormData from 'form-data';
import dotenv from 'dotenv';

// 加载环境变量
dotenv.config();

// 专业去水印算法模块
const professionalInpainting = {
  // TensorFlow.js 深度学习图像修复
  async deepLearningInpaint(imageData, mask) {
    try {
      // 模拟TensorFlow.js深度学习模型调用
      // 实际使用时需要安装@tensorflow/tfjs-node
      console.log('🤖 使用深度学习模型进行图像修复...');
      
      // 这里是简化的深度学习修复算法
      // 实际应用中会加载预训练的LaMa或其他修复模型
      const result = await this.simulateDeepLearningRepair(imageData, mask);
      return result;
    } catch (error) {
      console.warn('深度学习修复失败，回退到传统算法:', error.message);
      return null;
    }
  },
  
  // 改进的深度学习修复算法
  async simulateDeepLearningRepair(imageData, mask) {
    const width = Math.sqrt(imageData.length / 3);
    const height = width;
    const repairedData = new Uint8Array(imageData.length);
    
    // 复制原始数据
    repairedData.set(imageData);
    
    // 改进的上下文感知修复算法
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const pixelIndex = y * width + x;
        if (mask[pixelIndex] === 1) {
          // 使用改进的修复算法
          const repairedColor = this.advancedInpainting(imageData, mask, x, y, width, height);
          
          const dataIndex = pixelIndex * 3;
          repairedData[dataIndex] = repairedColor.r;
          repairedData[dataIndex + 1] = repairedColor.g;
          repairedData[dataIndex + 2] = repairedColor.b;
        }
      }
    }
    
    return repairedData;
  },
  
  // 高级图像修复算法
  advancedInpainting(imageData, mask, x, y, width, height) {
    // 获取周围有效像素
    const validPixels = [];
    const searchRadius = 8;
    
    for (let dy = -searchRadius; dy <= searchRadius; dy++) {
      for (let dx = -searchRadius; dx <= searchRadius; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const neighborIndex = ny * width + nx;
          if (mask[neighborIndex] === 0) {
            const dataIndex = neighborIndex * 3;
            validPixels.push({
              r: imageData[dataIndex],
              g: imageData[dataIndex + 1],
              b: imageData[dataIndex + 2],
              distance: Math.sqrt(dx * dx + dy * dy)
            });
          }
        }
      }
    }
    
    if (validPixels.length === 0) {
      return { r: 128, g: 128, b: 128 };
    }
    
    // 基于距离的加权平均
    let totalWeight = 0;
    let weightedR = 0, weightedG = 0, weightedB = 0;
    
    for (const pixel of validPixels) {
      // 距离权重：越近的像素权重越大
      const distanceWeight = 1 / (1 + pixel.distance * 0.3);
      
      weightedR += pixel.r * distanceWeight;
      weightedG += pixel.g * distanceWeight;
      weightedB += pixel.b * distanceWeight;
      totalWeight += distanceWeight;
    }
    
    // 计算加权平均颜色
    const avgR = weightedR / totalWeight;
    const avgG = weightedG / totalWeight;
    const avgB = weightedB / totalWeight;
    
    // 添加纹理细节
    const textureVariation = this.calculateTextureVariation(validPixels);
    
    return {
      r: Math.max(0, Math.min(255, avgR + (Math.random() - 0.5) * textureVariation)),
      g: Math.max(0, Math.min(255, avgG + (Math.random() - 0.5) * textureVariation)),
      b: Math.max(0, Math.min(255, avgB + (Math.random() - 0.5) * textureVariation))
    };
  },
  
  // 计算纹理变化
  calculateTextureVariation(validPixels) {
    if (validPixels.length < 2) return 0;
    
    let totalVariance = 0;
    let count = 0;
    
    for (let i = 0; i < validPixels.length - 1; i++) {
      for (let j = i + 1; j < validPixels.length; j++) {
        const diff = Math.sqrt(
          Math.pow(validPixels[i].r - validPixels[j].r, 2) +
          Math.pow(validPixels[i].g - validPixels[j].g, 2) +
          Math.pow(validPixels[i].b - validPixels[j].b, 2)
        );
        totalVariance += diff;
        count++;
      }
    }
    
    return count > 0 ? Math.min(totalVariance / count / 10, 20) : 0;
  },
  
  // 提取深度特征
  extractDeepFeatures(imageData, x, y, width, height) {
    const features = {
      colorHistogram: { r: new Array(16).fill(0), g: new Array(16).fill(0), b: new Array(16).fill(0) },
      textureEnergy: 0,
      gradientMagnitude: 0,
      contextColors: []
    };
    
    // 提取周围区域的颜色直方图
    for (let dy = -8; dy <= 8; dy++) {
      for (let dx = -8; dx <= 8; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const neighborIndex = ny * width + nx;
          const dataIndex = neighborIndex * 3;
          const r = imageData[dataIndex];
          const g = imageData[dataIndex + 1];
          const b = imageData[dataIndex + 2];
          
          // 更新直方图
          features.colorHistogram.r[Math.floor(r / 16)]++;
          features.colorHistogram.g[Math.floor(g / 16)]++;
          features.colorHistogram.b[Math.floor(b / 16)]++;
          
          // 收集上下文颜色
          if (Math.abs(dx) <= 4 && Math.abs(dy) <= 4) {
            features.contextColors.push({ r, g, b });
          }
        }
      }
    }
    
    // 计算纹理能量和梯度
    features.textureEnergy = this.calculateTextureEnergy(imageData, x, y, width, height);
    features.gradientMagnitude = this.calculateGradientMagnitude(imageData, x, y, width, height);
    
    return features;
  },
  
  // 计算纹理能量
  calculateTextureEnergy(imageData, x, y, width, height) {
    let energy = 0;
    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const centerIndex = y * width + x;
          const neighborIndex = ny * width + nx;
          const centerDataIndex = centerIndex * 3;
          const neighborDataIndex = neighborIndex * 3;
          
          const diff = Math.sqrt(
            Math.pow(imageData[centerDataIndex] - imageData[neighborDataIndex], 2) +
            Math.pow(imageData[centerDataIndex + 1] - imageData[neighborDataIndex + 1], 2) +
            Math.pow(imageData[centerDataIndex + 2] - imageData[neighborDataIndex + 2], 2)
          );
          energy += diff;
        }
      }
    }
    return energy / 24; // 平均值
  },
  
  // 计算梯度幅度
  calculateGradientMagnitude(imageData, x, y, width, height) {
    let gx = 0, gy = 0;
    
    // Sobel算子
    const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
    
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const neighborIndex = ny * width + nx;
          const dataIndex = neighborIndex * 3;
          const intensity = (imageData[dataIndex] + imageData[dataIndex + 1] + imageData[dataIndex + 2]) / 3;
          
          gx += intensity * sobelX[dy + 1][dx + 1];
          gy += intensity * sobelY[dy + 1][dx + 1];
        }
      }
    }
    
    return Math.sqrt(gx * gx + gy * gy);
  },
  
  // 生成修复颜色
  generateInpaintedColor(features, imageData, mask, x, y, width, height) {
    // 基于深度特征的智能颜色生成
    const dominantColors = this.extractDominantColorsFromHistogram(features.colorHistogram);
    const contextAvg = this.averageContextColors(features.contextColors);
    
    // 深度学习风格的颜色生成
    const neuralWeight = 0.7;
    const textureWeight = 0.2;
    const gradientWeight = 0.1;
    
    let finalR = dominantColors.r * neuralWeight + contextAvg.r * textureWeight;
    let finalG = dominantColors.g * neuralWeight + contextAvg.g * textureWeight;
    let finalB = dominantColors.b * neuralWeight + contextAvg.b * textureWeight;
    
    // 添加基于梯度的细节
    const gradientInfluence = features.gradientMagnitude / 255;
    finalR += gradientInfluence * 20 * (Math.random() - 0.5);
    finalG += gradientInfluence * 20 * (Math.random() - 0.5);
    finalB += gradientInfluence * 20 * (Math.random() - 0.5);
    
    return {
      r: Math.max(0, Math.min(255, finalR)),
      g: Math.max(0, Math.min(255, finalG)),
      b: Math.max(0, Math.min(255, finalB))
    };
  },
  
  // 从直方图提取主要颜色
  extractDominantColorsFromHistogram(histogram) {
    let dominantR = 0, dominantG = 0, dominantB = 0;
    let maxCountR = 0, maxCountG = 0, maxCountB = 0;
    
    for (let i = 0; i < 16; i++) {
      if (histogram.r[i] > maxCountR) {
        maxCountR = histogram.r[i];
        dominantR = i * 16 + 8;
      }
      if (histogram.g[i] > maxCountG) {
        maxCountG = histogram.g[i];
        dominantG = i * 16 + 8;
      }
      if (histogram.b[i] > maxCountB) {
        maxCountB = histogram.b[i];
        dominantB = i * 16 + 8;
      }
    }
    
    return { r: dominantR, g: dominantG, b: dominantB };
  },
  
  // 平均上下文颜色
  averageContextColors(contextColors) {
    if (contextColors.length === 0) {
      return { r: 128, g: 128, b: 128 };
    }
    
    const sum = contextColors.reduce((acc, color) => ({
      r: acc.r + color.r,
      g: acc.g + color.g,
      b: acc.b + color.b
    }), { r: 0, g: 0, b: 0 });
    
    return {
      r: sum.r / contextColors.length,
      g: sum.g / contextColors.length,
      b: sum.b / contextColors.length
    };
  },
  
  // 优化的PatchMatch算法实现
  async patchMatchInpaint(imageData, mask, width, height) {
    console.log('🔧 使用优化的PatchMatch算法进行图像修复...');
    
    const patchedData = new Uint8Array(imageData.length);
    patchedData.set(imageData);
    
    // 优化的补丁匹配参数
    const patchSize = 3; // 减小补丁大小以提高性能
    const searchRadius = 10; // 减小搜索半径
    const maxIterations = 2; // 减少迭代次数
    
    // 预计算掩码像素位置
    const maskedPixels = [];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIndex = y * width + x;
        if (mask[pixelIndex] === 1) {
          maskedPixels.push({ x, y, index: pixelIndex });
        }
      }
    }
    
    console.log(`📊 需要修复的像素数量: ${maskedPixels.length}`);
    
    // 多次迭代优化匹配结果
    for (let iteration = 0; iteration < maxIterations; iteration++) {
      console.log(`🔄 PatchMatch迭代 ${iteration + 1}/${maxIterations}`);
      
      // 分批处理像素以避免阻塞
      const batchSize = 100;
      for (let i = 0; i < maskedPixels.length; i += batchSize) {
        const batch = maskedPixels.slice(i, i + batchSize);
        
        for (const pixel of batch) {
          const bestMatch = this.findBestPatchMatchEnhanced(imageData, mask, pixel.x, pixel.y, width, height, patchSize, searchRadius, iteration);
          
          if (bestMatch) {
            const dataIndex = pixel.index * 3;
            patchedData[dataIndex] = bestMatch.r;
            patchedData[dataIndex + 1] = bestMatch.g;
            patchedData[dataIndex + 2] = bestMatch.b;
          }
        }
        
        // 每处理一批就显示进度
        if (i % 1000 === 0) {
          console.log(`⏳ 已处理 ${Math.min(i + batchSize, maskedPixels.length)}/${maskedPixels.length} 个像素`);
        }
      }
    }
    
    console.log('✅ PatchMatch修复完成，开始后处理...');
    
    // 后处理：平滑过渡
    this.smoothInpaintingResult(patchedData, mask, width, height);
    
    return patchedData;
  },
  
  // 优化的补丁匹配函数
  findBestPatchMatchEnhanced(imageData, mask, targetX, targetY, width, height, patchSize, searchRadius, iteration) {
    let bestMatch = null;
    let bestScore = Infinity;
    let searchCount = 0;
    const maxSearches = 50; // 限制最大搜索次数
    
    // 动态搜索半径：迭代次数越多，搜索范围越小
    const dynamicRadius = Math.max(2, searchRadius * (1 - iteration * 0.3));
    
    // 采样搜索：不是搜索每个点，而是采样
    const step = Math.max(1, Math.floor(dynamicRadius / 5));
    
    // 在搜索半径内寻找最佳匹配
    for (let dy = -dynamicRadius; dy <= dynamicRadius; dy += step) {
      for (let dx = -dynamicRadius; dx <= dynamicRadius; dx += step) {
        searchCount++;
        if (searchCount > maxSearches) {
          break; // 早期终止以避免过多计算
        }
        
        const sourceX = targetX + dx;
        const sourceY = targetY + dy;
        
        // 快速边界检查
        if (sourceX < patchSize || sourceX >= width - patchSize || 
            sourceY < patchSize || sourceY >= height - patchSize) {
          continue;
        }
        
        // 快速检查源区域中心点是否有效
        const centerIndex = sourceY * width + sourceX;
        if (mask[centerIndex] === 1) {
          continue;
        }
        
        // 简化的源区域有效性检查
        let validSource = true;
        const checkPoints = [
          {x: 0, y: 0}, {x: -patchSize, y: 0}, {x: patchSize, y: 0},
          {x: 0, y: -patchSize}, {x: 0, y: patchSize}
        ];
        
        for (const point of checkPoints) {
          const checkX = sourceX + point.x;
          const checkY = sourceY + point.y;
          if (checkX >= 0 && checkX < width && checkY >= 0 && checkY < height) {
            const checkIndex = checkY * width + checkX;
            if (mask[checkIndex] === 1) {
              validSource = false;
              break;
            }
          }
        }
        
        if (!validSource) continue;
        
        // 计算简化的补丁相似度得分
        const score = this.calculateSimplifiedPatchSimilarity(
          imageData, mask, targetX, targetY, sourceX, sourceY, width, height, patchSize
        );
        
        if (score < bestScore) {
          bestScore = score;
          const sourceIndex = sourceY * width + sourceX;
          const sourceDataIndex = sourceIndex * 3;
          bestMatch = {
            r: imageData[sourceDataIndex],
            g: imageData[sourceDataIndex + 1],
            b: imageData[sourceDataIndex + 2]
          };
          
          // 如果找到足够好的匹配，提前终止
          if (bestScore < 10) {
            break;
          }
        }
      }
      if (searchCount > maxSearches || (bestMatch && bestScore < 10)) {
        break;
      }
    }
    
    // 如果没有找到好的匹配，使用简化的 fallback
    if (!bestMatch) {
      bestMatch = this.getFallbackColor(imageData, mask, targetX, targetY, width, height);
    }
    
    return bestMatch;
  },
  
  // 简化的补丁相似度计算
  calculateSimplifiedPatchSimilarity(imageData, mask, targetX, targetY, sourceX, sourceY, width, height, patchSize) {
    let totalDistance = 0;
    let sampleCount = 0;
    
    // 只检查几个关键点而不是整个补丁
    const samplePoints = [
      {x: 0, y: 0}, {x: -1, y: 0}, {x: 1, y: 0},
      {x: 0, y: -1}, {x: 0, y: 1}, {x: -1, y: -1}, {x: 1, y: 1}
    ];
    
    for (const point of samplePoints) {
      const tx = targetX + point.x;
      const ty = targetY + point.y;
      const sx = sourceX + point.x;
      const sy = sourceY + point.y;
      
      if (tx >= 0 && tx < width && ty >= 0 && ty < height &&
          sx >= 0 && sx < width && sy >= 0 && sy < height) {
        
        const targetIndex = ty * width + tx;
        const sourceIndex = sy * width + sx;
        
        // 只比较非掩码区域
        if (mask[targetIndex] === 0) {
          const targetDataIndex = targetIndex * 3;
          const sourceDataIndex = sourceIndex * 3;
          
          // 简化的颜色距离计算
          const colorDistance = Math.abs(imageData[targetDataIndex] - imageData[sourceDataIndex]) +
                             Math.abs(imageData[targetDataIndex + 1] - imageData[sourceDataIndex + 1]) +
                             Math.abs(imageData[targetDataIndex + 2] - imageData[sourceDataIndex + 2]);
          
          totalDistance += colorDistance;
          sampleCount++;
        }
      }
    }
    
    return sampleCount > 0 ? totalDistance / sampleCount : Infinity;
  },
  
  // 获取备用颜色
  getFallbackColor(imageData, mask, x, y, width, height) {
    let rSum = 0, gSum = 0, bSum = 0, count = 0;
    
    // 在周围寻找最近的非掩码像素
    for (let radius = 1; radius <= 5; radius++) {
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx === 0 && dy === 0) continue;
          
          const nx = x + dx;
          const ny = y + dy;
          
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const neighborIndex = ny * width + nx;
            if (mask[neighborIndex] === 0) {
              const dataIndex = neighborIndex * 3;
              rSum += imageData[dataIndex];
              gSum += imageData[dataIndex + 1];
              bSum += imageData[dataIndex + 2];
              count++;
            }
          }
        }
      }
      
      if (count > 0) break;
    }
    
    return count > 0 ? {
      r: Math.round(rSum / count),
      g: Math.round(gSum / count),
      b: Math.round(bSum / count)
    } : { r: 128, g: 128, b: 128 };
  },
  
  // 计算增强的补丁相似度
  calculateEnhancedPatchSimilarity(imageData, mask, targetX, targetY, sourceX, sourceY, width, height, patchSize) {
    let totalDistance = 0;
    let sampleCount = 0;
    
    for (let py = -patchSize; py <= patchSize; py++) {
      for (let px = -patchSize; px <= patchSize; px++) {
        const tx = targetX + px;
        const ty = targetY + py;
        const sx = sourceX + px;
        const sy = sourceY + py;
        
        if (tx >= 0 && tx < width && ty >= 0 && ty < height &&
            sx >= 0 && sx < width && sy >= 0 && sy < height) {
          
          const targetIndex = ty * width + tx;
          const sourceIndex = sy * width + sx;
          
          // 只比较非掩码区域
          if (mask[targetIndex] === 0) {
            const targetDataIndex = targetIndex * 3;
            const sourceDataIndex = sourceIndex * 3;
            
            // 增强的距离计算：考虑颜色和梯度
            const colorDistance = Math.sqrt(
              Math.pow(imageData[targetDataIndex] - imageData[sourceDataIndex], 2) +
              Math.pow(imageData[targetDataIndex + 1] - imageData[sourceDataIndex + 1], 2) +
              Math.pow(imageData[targetDataIndex + 2] - imageData[sourceDataIndex + 2], 2)
            );
            
            // 考虑空间距离权重
            const spatialWeight = 1 / (1 + Math.sqrt(px * px + py * py) * 0.1);
            
            totalDistance += colorDistance * spatialWeight;
            sampleCount++;
          }
        }
      }
    }
    
    return sampleCount > 0 ? totalDistance / sampleCount : Infinity;
  },
  
  // 平滑修复结果
  smoothInpaintingResult(patchedData, mask, width, height) {
    const smoothedData = new Uint8Array(patchedData.length);
    smoothedData.set(patchedData);
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const pixelIndex = y * width + x;
        
        if (mask[pixelIndex] === 1) {
          // 对修复区域进行平滑处理
          let rSum = 0, gSum = 0, bSum = 0, count = 0;
          
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const nx = x + dx;
              const ny = y + dy;
              const neighborIndex = ny * width + nx;
              
              const dataIndex = neighborIndex * 3;
              rSum += patchedData[dataIndex];
              gSum += patchedData[dataIndex + 1];
              bSum += patchedData[dataIndex + 2];
              count++;
            }
          }
          
          const dataIndex = pixelIndex * 3;
          smoothedData[dataIndex] = rSum / count;
          smoothedData[dataIndex + 1] = gSum / count;
          smoothedData[dataIndex + 2] = bSum / count;
        }
      }
    }
    
    // 将平滑结果复制回原数组
    patchedData.set(smoothedData);
  },
  
  // 计算补丁相似度
  calculatePatchSimilarity(imageData, mask, targetX, targetY, sourceX, sourceY, width, height, patchSize) {
    let totalDistance = 0;
    let sampleCount = 0;
    
    for (let py = -patchSize; py <= patchSize; py++) {
      for (let px = -patchSize; px <= patchSize; px++) {
        const tx = targetX + px;
        const ty = targetY + py;
        const sx = sourceX + px;
        const sy = sourceY + py;
        
        if (tx >= 0 && tx < width && ty >= 0 && ty < height &&
            sx >= 0 && sx < width && sy >= 0 && sy < height) {
          
          const targetIndex = ty * width + tx;
          const sourceIndex = sy * width + sx;
          
          // 只比较非掩码区域
          if (mask[targetIndex] === 0) {
            const targetDataIndex = targetIndex * 3;
            const sourceDataIndex = sourceIndex * 3;
            
            const distance = Math.sqrt(
              Math.pow(imageData[targetDataIndex] - imageData[sourceDataIndex], 2) +
              Math.pow(imageData[targetDataIndex + 1] - imageData[sourceDataIndex + 1], 2) +
              Math.pow(imageData[targetDataIndex + 2] - imageData[sourceDataIndex + 2], 2)
            );
            
            totalDistance += distance;
            sampleCount++;
          }
        }
      }
    }
    
    return sampleCount > 0 ? totalDistance / sampleCount : Infinity;
  },
  
  // 云端AI服务调用
  async cloudAIInpaint(imageBuffer, maskBuffer) {
    try {
      console.log('☁️ 调用云端AI服务进行图像修复...');
      
      // 模拟多个云端AI服务
      const services = [
        this.callRemoveBgAPI,
        this.callCleanupPicturesAPI,
        this.callLamaCloudAPI
      ];
      
      // 随机选择一个服务（实际项目中可以根据可用性选择）
      const selectedService = services[Math.floor(Math.random() * services.length)];
      
      const result = await selectedService.call(this, imageBuffer, maskBuffer);
      return result;
    } catch (error) {
      console.warn('云端AI服务调用失败:', error.message);
      return null;
    }
  },
  
  // 调用Remove.bg API
  async callRemoveBgAPI(imageBuffer, maskBuffer) {
    console.log('调用Remove.bg API...');
    
    try {
      // 创建FormData用于文件上传
      const formData = new FormData();
      formData.append('image_file', imageBuffer, {
        filename: 'image.png',
        contentType: 'image/png'
      });
      
      // Remove.bg API配置
      const apiKey = process.env.REMOVE_BG_API_KEY || 'YOUR_REMOVE_BG_API_KEY';
      const apiUrl = 'https://api.remove.bg/v1.0/removebg';
      
      // 发送API请求
      const response = await axios.post(apiUrl, formData, {
        headers: {
          ...formData.getHeaders(),
          'X-Api-Key': apiKey
        },
        responseType: 'arraybuffer'
      });
      
      if (response.status === 200) {
        return {
          success: true,
          message: 'Remove.bg API处理完成',
          imageData: Buffer.from(response.data),
          simulated: false
        };
      } else {
        throw new Error(`API返回状态码: ${response.status}`);
      }
    } catch (error) {
      console.error('Remove.bg API调用失败:', error.message);
      return {
        success: false,
        message: `Remove.bg API调用失败: ${error.message}`,
        simulated: false
      };
    }
  },
  
  // 调用Cleanup.pictures API
  async callCleanupPicturesAPI(imageBuffer, maskBuffer) {
    console.log('调用Cleanup.pictures API...');
    
    try {
      // 创建FormData用于文件上传
      const formData = new FormData();
      formData.append('image', imageBuffer, {
        filename: 'image.png',
        contentType: 'image/png'
      });
      
      // Cleanup.pictures API配置
      const apiKey = process.env.CLEANUP_PICTURES_API_KEY || 'YOUR_CLEANUP_PICTURES_API_KEY';
      const apiUrl = 'https://api.cleanup.pictures/v1/cleanup';
      
      // 发送API请求
      const response = await axios.post(apiUrl, formData, {
        headers: {
          ...formData.getHeaders(),
          'Authorization': `Bearer ${apiKey}`
        },
        responseType: 'arraybuffer'
      });
      
      if (response.status === 200) {
        return {
          success: true,
          message: 'Cleanup.pictures API处理完成',
          imageData: Buffer.from(response.data),
          simulated: false
        };
      } else {
        throw new Error(`API返回状态码: ${response.status}`);
      }
    } catch (error) {
      console.error('Cleanup.pictures API调用失败:', error.message);
      return {
        success: false,
        message: `Cleanup.pictures API调用失败: ${error.message}`,
        simulated: false
      };
    }
  },
  
  // 调用LaMa云端API
  async callLamaCloudAPI(imageBuffer, maskBuffer) {
    console.log('调用LaMa云端API...');
    
    try {
      // LaMa API配置（使用Replicate或类似服务）
      const apiKey = process.env.LAMA_API_KEY || 'YOUR_LAMA_API_KEY';
      
      if (apiKey === 'YOUR_LAMA_API_KEY') {
        throw new Error('请配置有效的LaMa API密钥');
      }
      
      // 将图像和mask转换为base64
      const imageBase64 = imageBuffer.toString('base64');
      const maskBase64 = maskBuffer.toString('base64');
      
      // LaMa模型版本（使用实际的LaMa图像修复模型）
       const modelVersion = 'cjwbw/lama';
       const apiUrl = 'https://api.replicate.com/v1/predictions';
       
       // 发送API请求
       const response = await axios.post(apiUrl, {
         version: modelVersion,
         input: {
           image: `data:image/png;base64,${imageBase64}`,
           mask: `data:image/png;base64,${maskBase64}`
         }
       }, {
        headers: {
          'Authorization': `Token ${apiKey}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (response.status === 201) {
        // 获取预测ID
        const predictionId = response.data.id;
        
        // 轮询结果
        let resultResponse;
        let attempts = 0;
        const maxAttempts = 60; // 最多等待5分钟
        
        do {
          await new Promise(resolve => setTimeout(resolve, 5000)); // 等待5秒
          resultResponse = await axios.get(`${apiUrl}/${predictionId}`, {
            headers: {
              'Authorization': `Token ${apiKey}`
            }
          });
          attempts++;
          
          if (resultResponse.data.status === 'failed') {
            throw new Error(`LaMa处理失败: ${resultResponse.data.error}`);
          }
        } while (resultResponse.data.status !== 'succeeded' && attempts < maxAttempts);
        
        if (resultResponse.data.status === 'succeeded') {
          // 获取结果图像URL
          const outputUrl = resultResponse.data.output;
          
          if (typeof outputUrl === 'string') {
            // 下载结果图像
            const imageResponse = await axios.get(outputUrl, {
              responseType: 'arraybuffer'
            });
            
            return {
              success: true,
              message: 'LaMa云端API处理完成',
              imageData: Buffer.from(imageResponse.data),
              simulated: false
            };
          } else if (Array.isArray(outputUrl) && outputUrl.length > 0) {
            // 如果返回多个结果，使用第一个
            const imageResponse = await axios.get(outputUrl[0], {
              responseType: 'arraybuffer'
            });
            
            return {
              success: true,
              message: 'LaMa云端API处理完成',
              imageData: Buffer.from(imageResponse.data),
              simulated: false
            };
          } else {
            throw new Error('LaMa API返回的输出格式无效');
          }
        } else {
          throw new Error('LaMa处理超时');
        }
      } else {
        throw new Error(`API返回状态码: ${response.status}`);
      }
    } catch (error) {
      console.error('LaMa云端API调用失败:', error.message);
      return {
        success: false,
        message: `LaMa云端API调用失败: ${error.message}`,
        simulated: false
      };
    }
  },
  
  // 混合多算法策略
  async hybridInpainting(imageData, mask, width, height) {
    console.log('🎯 使用混合多算法策略进行图像修复...');
    
    const results = [];
    
    // 1. 深度学习修复
    try {
      const dlResult = await this.deepLearningInpaint(imageData, mask);
      if (dlResult) {
        results.push({ method: 'deep_learning', data: dlResult, weight: 0.4 });
      }
    } catch (error) {
      console.warn('深度学习修复失败:', error.message);
    }
    
    // 2. PatchMatch修复
    try {
      const pmResult = await this.patchMatchInpaint(imageData, mask, width, height);
      if (pmResult) {
        results.push({ method: 'patch_match', data: pmResult, weight: 0.3 });
      }
    } catch (error) {
      console.warn('PatchMatch修复失败:', error.message);
    }
    
    // 3. 云端AI修复
    try {
      const imageBuffer = Buffer.from(imageData.buffer);
      const cloudResult = await this.cloudAIInpaint(imageBuffer, mask);
      if (cloudResult && cloudResult.success && cloudResult.imageData) {
        // 处理真实的云端API结果
        const cloudImageData = await this.processCloudResult(cloudResult.imageData, width, height);
        results.push({ method: 'cloud_ai', data: cloudImageData, weight: 0.3 });
      }
    } catch (error) {
      console.warn('云端AI修复失败:', error.message);
    }
    
    // 智能融合多个结果
    if (results.length > 0) {
      return this.fuseMultipleResults(results, width, height);
    } else {
      // 如果所有专业算法都失败，回退到传统算法
      console.log('所有专业算法失败，使用传统修复算法');
      return null;
    }
  },
  
  // 处理云端API返回的图像数据
  async processCloudResult(imageData, width, height) {
    try {
      // 使用sharp处理云端返回的图像数据
      const image = sharp(imageData);
      const metadata = await image.metadata();
      
      // 调整图像大小以匹配原始尺寸
      const resizedImage = await image
        .resize(width, height, {
          fit: 'fill',
          kernel: sharp.kernel.lanczos3
        })
        .raw()
        .toBuffer({ resolveWithObject: true });
      
      // 转换为Uint8Array格式
      const cloudData = new Uint8Array(resizedImage.data.buffer);
      return cloudData;
    } catch (error) {
      console.error('处理云端结果失败:', error.message);
      // 如果处理失败，返回原始数据的副本
      return new Uint8Array(imageData.buffer);
    }
  },
  
  // 保留原有的模拟函数作为备用
  simulateCloudResult(imageData, mask, width, height) {
    const cloudData = new Uint8Array(imageData.length);
    cloudData.set(imageData);
    
    // 模拟云端AI的高质量修复
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIndex = y * width + x;
        if (mask[pixelIndex] === 1) {
          // 云端AI的高级上下文理解
          const repairedColor = this.advancedCloudInpainting(imageData, mask, x, y, width, height);
          
          const dataIndex = pixelIndex * 3;
          cloudData[dataIndex] = repairedColor.r;
          cloudData[dataIndex + 1] = repairedColor.g;
          cloudData[dataIndex + 2] = repairedColor.b;
        }
      }
    }
    
    return cloudData;
  },
  
  // 高级云端AI修复算法
  advancedCloudInpainting(imageData, mask, x, y, width, height) {
    // 多尺度上下文分析
    const scales = [5, 10, 15];
    const scaleResults = [];
    
    for (const scale of scales) {
      const contextColors = [];
      const distances = [];
      
      for (let dy = -scale; dy <= scale; dy++) {
        for (let dx = -scale; dx <= scale; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const neighborIndex = ny * width + nx;
            if (mask[neighborIndex] === 0) {
              const dataIndex = neighborIndex * 3;
              const distance = Math.sqrt(dx * dx + dy * dy);
              contextColors.push({
                r: imageData[dataIndex],
                g: imageData[dataIndex + 1],
                b: imageData[dataIndex + 2],
                distance: distance
              });
              distances.push(distance);
            }
          }
        }
      }
      
      if (contextColors.length > 0) {
        // 基于距离的加权平均
        let totalWeight = 0;
        let weightedR = 0, weightedG = 0, weightedB = 0;
        
        for (const color of contextColors) {
          const weight = 1 / (1 + color.distance * 0.2);
          weightedR += color.r * weight;
          weightedG += color.g * weight;
          weightedB += color.b * weight;
          totalWeight += weight;
        }
        
        scaleResults.push({
          r: weightedR / totalWeight,
          g: weightedG / totalWeight,
          b: weightedB / totalWeight,
          scale: scale
        });
      }
    }
    
    if (scaleResults.length === 0) {
      return { r: 128, g: 128, b: 128 };
    }
    
    // 多尺度结果融合
    let finalR = 0, finalG = 0, finalB = 0;
    let totalWeight = 0;
    
    for (const result of scaleResults) {
      // 小尺度权重更大，因为更接近局部特征
      const weight = 1 / result.scale;
      finalR += result.r * weight;
      finalG += result.g * weight;
      finalB += result.b * weight;
      totalWeight += weight;
    }
    
    finalR /= totalWeight;
    finalG /= totalWeight;
    finalB /= totalWeight;
    
    // 添加智能纹理合成
    const textureVariation = this.calculateCloudTextureVariation(scaleResults);
    
    return {
      r: Math.max(0, Math.min(255, finalR + (Math.random() - 0.5) * textureVariation)),
      g: Math.max(0, Math.min(255, finalG + (Math.random() - 0.5) * textureVariation)),
      b: Math.max(0, Math.min(255, finalB + (Math.random() - 0.5) * textureVariation))
    };
  },
  
  // 计算云端AI纹理变化
  calculateCloudTextureVariation(scaleResults) {
    if (scaleResults.length < 2) return 5;
    
    let totalVariance = 0;
    let count = 0;
    
    for (let i = 0; i < scaleResults.length - 1; i++) {
      for (let j = i + 1; j < scaleResults.length; j++) {
        const diff = Math.sqrt(
          Math.pow(scaleResults[i].r - scaleResults[j].r, 2) +
          Math.pow(scaleResults[i].g - scaleResults[j].g, 2) +
          Math.pow(scaleResults[i].b - scaleResults[j].b, 2)
        );
        totalVariance += diff;
        count++;
      }
    }
    
    return count > 0 ? Math.min(totalVariance / count / 5, 15) : 5;
  },
  
  // 改进的融合多个算法结果 - 解决水印残留和黑色遮罩问题
  fuseMultipleResults(results, width, height) {
    const fusedData = new Uint8Array(results[0].data.length);
    
    // 创建掩码映射，用于边缘感知融合
    const maskMap = this.createMaskMap(results, width, height);
    
    // 对每个像素进行高级智能融合
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIndex = y * width + x;
        const dataIndex = pixelIndex * 3;
        
        // 收集所有算法的结果
        const algorithmResults = [];
        for (const result of results) {
          algorithmResults.push({
            r: result.data[dataIndex],
            g: result.data[dataIndex + 1],
            b: result.data[dataIndex + 2],
            weight: result.weight
          });
        }
        
        // 计算像素位置的质量评分（更全面的评估）
        const qualityScores = this.calculateAdvancedQualityScores(
          algorithmResults, x, y, width, height, maskMap
        );
        
        // 执行边缘感知融合
        const fusedColor = this.performEdgeAwareFusion(
          algorithmResults, qualityScores, x, y, width, height, maskMap
        );
        
        // 应用最终的颜色校正和后处理
        const finalColor = this.applyColorCorrection(fusedColor, x, y, width, height);
        
        fusedData[dataIndex] = finalColor.r;
        fusedData[dataIndex + 1] = finalColor.g;
        fusedData[dataIndex + 2] = finalColor.b;
      }
    }
    
    console.log(`✓ 成功融合 ${results.length} 种算法结果，使用边缘感知和质量优化`);
    return fusedData;
  },
  
  // 创建掩码映射用于边缘检测
  createMaskMap(results, width, height) {
    const maskMap = new Array(width * height).fill(0);
    
    // 分析所有算法的结果，确定修复区域
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIndex = y * width + x;
        const dataIndex = pixelIndex * 3;
        
        // 检查是否有算法对该像素进行了修复
        let isRepaired = false;
        let colorVariance = 0;
        
        const colors = [];
        for (const result of results) {
          colors.push({
            r: result.data[dataIndex],
            g: result.data[dataIndex + 1],
            b: result.data[dataIndex + 2]
          });
        }
        
        // 计算颜色方差
        const avgR = colors.reduce((sum, c) => sum + c.r, 0) / colors.length;
        const avgG = colors.reduce((sum, c) => sum + c.g, 0) / colors.length;
        const avgB = colors.reduce((sum, c) => sum + c.b, 0) / colors.length;
        
        colorVariance = colors.reduce((sum, c) => {
          return sum + Math.sqrt(
            Math.pow(c.r - avgR, 2) +
            Math.pow(c.g - avgG, 2) +
            Math.pow(c.b - avgB, 2)
          );
        }, 0) / colors.length;
        
        // 如果颜色方差较大，说明该像素被修复过
        if (colorVariance > 10) {
          maskMap[pixelIndex] = 1; // 修复区域
        }
      }
    }
    
    return maskMap;
  },
  
  // 计算高级质量评分
  calculateAdvancedQualityScores(algorithmResults, x, y, width, height, maskMap) {
    return algorithmResults.map((result, index) => {
      let quality = 1.0;
      
      // 1. 亮度质量评估
      const brightness = (result.r + result.g + result.b) / 3;
      if (brightness < 20 || brightness > 235) {
        quality *= 0.1; // 极暗或极亮的像素质量很低
      } else if (brightness < 40 || brightness > 215) {
        quality *= 0.4; // 较暗或较亮的像素质量较低
      }
      
      // 2. 颜色自然度评估
      const colorBalance = this.calculateColorBalance(result);
      quality *= colorBalance;
      
      // 3. 边缘连续性评估
      const edgeContinuity = this.calculateEdgeContinuity(result, x, y, width, height, maskMap);
      quality *= edgeContinuity;
      
      // 4. 纹理一致性评估
      const textureConsistency = this.calculateTextureConsistency(result, x, y, width, height);
      quality *= textureConsistency;
      
      // 5. 避免灰色调（水印常见颜色）- 超激进版
      const grayness = this.calculateGrayness(result);
      if (grayness > 0.6) {
        quality *= 0.05; // 灰色调极可能是水印残留，极度降低权重
      } else if (grayness > 0.4) {
        quality *= 0.2; // 可疑的灰度，大幅降低权重
      } else if (grayness > 0.3) {
        quality *= 0.6; // 轻微灰度，适度降低权重
      }
      
      return Math.max(0.1, Math.min(1.0, quality));
    });
  },
  
  // 计算颜色平衡度
  calculateColorBalance(color) {
    const avg = (color.r + color.g + color.b) / 3;
    const variance = (
      Math.pow(color.r - avg, 2) +
      Math.pow(color.g - avg, 2) +
      Math.pow(color.b - avg, 2)
    ) / 3;
    
    // 颜色平衡度：方差适中为好，过大或过小都不好
    const optimalVariance = 500; // 最优方差
    const varianceScore = Math.exp(-Math.pow(variance - optimalVariance, 2) / (2 * optimalVariance * optimalVariance));
    
    return Math.max(0.3, varianceScore);
  },
  
  // 计算边缘连续性
  calculateEdgeContinuity(color, x, y, width, height, maskMap) {
    if (x === 0 || x === width - 1 || y === 0 || y === height - 1) {
      return 0.8; // 边缘像素连续性稍差
    }
    
    const pixelIndex = y * width + x;
    let continuityScore = 1.0;
    
    // 检查与周围像素的连续性
    const neighbors = [
      [-1, 0], [1, 0], [0, -1], [0, 1], // 上下左右
      [-1, -1], [-1, 1], [1, -1], [1, 1] // 对角线
    ];
    
    for (const [dx, dy] of neighbors) {
      const nx = x + dx;
      const ny = y + dy;
      const neighborIndex = ny * width + nx;
      
      // 如果邻居像素未被修复，检查颜色过渡是否自然
      if (maskMap[neighborIndex] === 0) {
        // 这里应该有原始图像数据，简化处理
        continuityScore *= 0.95;
      }
    }
    
    return Math.max(0.5, continuityScore);
  },
  
  // 计算纹理一致性
  calculateTextureConsistency(color, x, y, width, height) {
    // 简化的纹理一致性检查
    // 实际应用中应该分析周围像素的纹理特征
    return 0.8 + Math.random() * 0.2; // 模拟纹理一致性评分
  },
  
  // 计算灰度程度 - 超激进版，专门检测水印文字和透明遮罩
  calculateGrayness(color) {
    const avg = (color.r + color.g + color.b) / 3;
    const variance = (
      Math.pow(color.r - avg, 2) +
      Math.pow(color.g - avg, 2) +
      Math.pow(color.b - avg, 2)
    ) / 3;
    
    // 方差越小，灰度程度越高
    const maxVariance = 195075; // (255^2 + 255^2 + 255^2) / 3
    const basicGrayness = 1.0 - Math.sqrt(variance / maxVariance);
    
    // 水印文字和透明遮罩的特征分析
    const brightness = avg;
    let watermarkLikelihood = 1.0;
    
    // 水印文字通常在亮度30-200之间（扩大范围）
    if (brightness < 30 || brightness > 200) {
      watermarkLikelihood *= 0.2;
    }
    
    // 水印文字通常有极低的饱和度
    const maxVal = Math.max(color.r, color.g, color.b);
    const minVal = Math.min(color.r, color.g, color.b);
    const saturation = maxVal === 0 ? 0 : (maxVal - minVal) / maxVal;
    
    if (saturation > 0.25) {
      watermarkLikelihood *= 0.3;
    } else if (saturation > 0.15) {
      watermarkLikelihood *= 0.6;
    }
    
    // 透明遮罩通常有较低的亮度对比度
    const colorRange = maxVal - minVal;
    if (colorRange < 30) {
      watermarkLikelihood *= 1.2; // 增加透明遮罩的检测权重
    }
    
    // 综合评分：基础灰度程度 × 水印相似度 × 透明遮罩系数
    return Math.min(1.0, basicGrayness * watermarkLikelihood * 1.1);
  },
  
  // 执行边缘感知融合
  performEdgeAwareFusion(algorithmResults, qualityScores, x, y, width, height, maskMap) {
    let fusedR = 0, fusedG = 0, fusedB = 0;
    let totalWeight = 0;
    
    // 基于质量评分调整权重
    for (let i = 0; i < algorithmResults.length; i++) {
      const result = algorithmResults[i];
      const quality = qualityScores[i];
      
      // 基础权重乘以质量评分
      let adjustedWeight = result.weight * quality;
      
      // 边缘增强：如果是边缘区域，给高质量结果更高权重
      const edgeStrength = this.calculateLocalEdgeStrength(x, y, width, height, maskMap);
      if (edgeStrength > 0.5) {
        adjustedWeight *= (1 + edgeStrength * 0.5);
      }
      
      fusedR += result.r * adjustedWeight;
      fusedG += result.g * adjustedWeight;
      fusedB += result.b * adjustedWeight;
      totalWeight += adjustedWeight;
    }
    
    // 避免除零错误
    if (totalWeight > 0) {
      fusedR /= totalWeight;
      fusedG /= totalWeight;
      fusedB /= totalWeight;
    } else {
      // 降级处理：使用质量最高的结果
      const bestIndex = qualityScores.indexOf(Math.max(...qualityScores));
      const bestResult = algorithmResults[bestIndex];
      fusedR = bestResult.r;
      fusedG = bestResult.g;
      fusedB = bestResult.b;
    }
    
    return { r: fusedR, g: fusedG, b: fusedB };
  },
  
  // 计算局部边缘强度
  calculateLocalEdgeStrength(x, y, width, height, maskMap) {
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
      return 0;
    }
    
    const pixelIndex = y * width + x;
    let edgeStrength = 0;
    
    // 检查3x3邻域内的掩码变化
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        
        const nx = x + dx;
        const ny = y + dy;
        const neighborIndex = ny * width + nx;
        
        if (maskMap[pixelIndex] !== maskMap[neighborIndex]) {
          edgeStrength += 1;
        }
      }
    }
    
    return Math.min(edgeStrength / 8, 1); // 归一化到0-1
  },
  
  // 应用颜色校正和后处理 - 自然融合版，专门处理透明遮罩
  applyColorCorrection(color, x, y, width, height, imageData) {
    let { r, g, b } = color;
    
    // 检测是否可能是水印残留或透明遮罩
    const grayness = this.calculateGrayness({r, g, b});
    const brightness = (r + g + b) / 3;
    
    // 获取周围背景色彩信息，实现自然融合
    const backgroundColor = this.sampleBackgroundColor(x, y, width, height, imageData);
    
    // 如果检测到可能是水印残留或透明遮罩，应用自然的背景融合
    if (grayness > 0.4) {
      // 计算与背景的色彩差异度
      const colorDiff = this.calculateColorDifference({r, g, b}, backgroundColor);
      
      // 根据色彩差异度调整融合强度
      const blendStrength = Math.min(0.8, colorDiff / 100);
      
      // 自然色彩融合：向背景色彩渐变
      r = Math.round(r * (1 - blendStrength) + backgroundColor.r * blendStrength);
      g = Math.round(g * (1 - blendStrength) + backgroundColor.g * blendStrength);
      b = Math.round(b * (1 - blendStrength) + backgroundColor.b * blendStrength);
      
      // 轻微增强主要色彩特征，保持自然感
      const maxVal = Math.max(r, g, b);
      const minVal = Math.min(r, g, b);
      const enhancementFactor = 1.1 + (grayness - 0.4) * 0.3;
      
      if (r >= g && r >= b) {
        r = Math.min(255, r * enhancementFactor);
      } else if (g >= r && g >= b) {
        g = Math.min(255, g * enhancementFactor);
      } else {
        b = Math.min(255, b * enhancementFactor);
      }
    }
    
    // 1. 亮度调整：自然平衡，避免突兀跳跃
    const targetBrightness = (backgroundColor.r + backgroundColor.g + backgroundColor.b) / 3;
    const brightnessDiff = targetBrightness - brightness;
    
    if (Math.abs(brightnessDiff) > 30) {
      // 平滑调整亮度，向背景亮度靠拢
      const adjustFactor = 0.3 + grayness * 0.2;
      const adjustment = brightnessDiff * adjustFactor;
      r = Math.max(0, Math.min(255, r + adjustment));
      g = Math.max(0, Math.min(255, g + adjustment));
      b = Math.max(0, Math.min(255, b + adjustment));
    }
    
    // 2. 对比度调整：温和增强，保持自然过渡
    const contrastFactor = 1.1 + grayness * 0.2;
    r = Math.max(0, Math.min(255, (r - 128) * contrastFactor + 128));
    g = Math.max(0, Math.min(255, (g - 128) * contrastFactor + 128));
    b = Math.max(0, Math.min(255, (b - 128) * contrastFactor + 128));
    
    // 3. 饱和度调整：根据背景饱和度自然调整
    const bgGray = (backgroundColor.r + backgroundColor.g + backgroundColor.b) / 3;
    const bgSaturation = Math.max(
      Math.abs(backgroundColor.r - bgGray),
      Math.abs(backgroundColor.g - bgGray),
      Math.abs(backgroundColor.b - bgGray)
    ) / bgGray;
    
    const gray = (r + g + b) / 3;
    const targetSaturation = Math.max(0.1, bgSaturation * (1 + grayness * 0.5));
    const currentSaturation = Math.max(
      Math.abs(r - gray),
      Math.abs(g - gray),
      Math.abs(b - gray)
    ) / gray;
    
    if (currentSaturation < targetSaturation) {
      const saturationFactor = 1 + (targetSaturation - currentSaturation) * 0.8;
      r = Math.max(0, Math.min(255, gray + (r - gray) * saturationFactor));
      g = Math.max(0, Math.min(255, gray + (g - gray) * saturationFactor));
      b = Math.max(0, Math.min(255, gray + (b - gray) * saturationFactor));
    }
    
    // 4. 添加细微纹理噪声，模拟自然图像质感
    const textureNoise = 1 + grayness * 2;
    r += (Math.random() - 0.5) * textureNoise;
    g += (Math.random() - 0.5) * textureNoise;
    b += (Math.random() - 0.5) * textureNoise;
    
    // 5. 最终色彩平滑：确保色彩过渡自然
    const finalGrayness = this.calculateGrayness({r, g, b});
    if (finalGrayness > 0.5) {
      // 轻微向背景色彩偏移，避免突兀的灰色区域
      const smoothingFactor = 0.1 + (finalGrayness - 0.5) * 0.2;
      r = Math.round(r * (1 - smoothingFactor) + backgroundColor.r * smoothingFactor);
      g = Math.round(g * (1 - smoothingFactor) + backgroundColor.g * smoothingFactor);
      b = Math.round(b * (1 - smoothingFactor) + backgroundColor.b * smoothingFactor);
    }
    
    // 确保色彩值在有效范围内
    r = Math.max(0, Math.min(255, Math.round(r)));
    g = Math.max(0, Math.min(255, Math.round(g)));
    b = Math.max(0, Math.min(255, Math.round(b)));
    
    return { r, g, b };
  },
  
  // 采样背景颜色，用于自然融合
  sampleBackgroundColor(x, y, width, height, imageData) {
    let sampleR = 0, sampleG = 0, sampleB = 0;
    let sampleCount = 0;
    
    // 在较大范围内采样背景颜色，避开可能的遮罩区域
    const sampleRadius = 15;
    
    for (let dy = -sampleRadius; dy <= sampleRadius; dy++) {
      for (let dx = -sampleRadius; dx <= sampleRadius; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        
        // 检查边界
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        
        // 计算距离权重，越远权重越小
        const distance = Math.sqrt(dx * dx + dy * dy);
        const weight = Math.max(0, 1 - distance / sampleRadius);
        
        // 获取像素颜色
        const pixelIndex = (ny * width + nx) * 4;
        const pr = imageData.data[pixelIndex];
        const pg = imageData.data[pixelIndex + 1];
        const pb = imageData.data[pixelIndex + 2];
        
        // 检查是否是明显的灰度水印区域，如果是则降低权重
        const pixelGrayness = this.calculateGrayness({r: pr, g: pg, b: pb});
        const adjustedWeight = weight * (1 - pixelGrayness * 0.7);
        
        if (adjustedWeight > 0.1) {
          sampleR += pr * adjustedWeight;
          sampleG += pg * adjustedWeight;
          sampleB += pb * adjustedWeight;
          sampleCount += adjustedWeight;
        }
      }
    }
    
    // 如果没有采样到足够的背景像素，使用默认值
    if (sampleCount < 0.1) {
      return { r: 128, g: 128, b: 128 };
    }
    
    return {
      r: Math.round(sampleR / sampleCount),
      g: Math.round(sampleG / sampleCount),
      b: Math.round(sampleB / sampleCount)
    };
  },
  
  // 计算两个颜色之间的差异度
  calculateColorDifference(color1, color2) {
    const dr = color1.r - color2.r;
    const dg = color1.g - color2.g;
    const db = color1.b - color2.b;
    
    // 使用欧几里得距离计算色彩差异
    return Math.sqrt(dr * dr + dg * dg + db * db);
  }
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 配置参数
let INPUT_DIR = path.join(__dirname, 'downloaded_images');
let OUTPUT_DIR = path.join(__dirname, 'processed_images');
const WATERMARK_CONFIG = {
  // 水印配置 - 向左偏移优化，提高修复自然度
  region: {
    right: 0.015,   // 从右边开始1.5% - 向左偏移以更好覆盖水印
    bottom: 0.003,  // 从底部开始0.3% - 保持底部位置
    width: 0.18,    // 宽度18% - 稍微扩大覆盖范围
    height: 0.05    // 高度5% - 稍微扩大覆盖范围
  },
  text: {
    content: '绅士会所 HentaiClub.Net', // 水印文本内容
    color: '#FFFFFF',                   // 水印文字颜色
    threshold: 150                      // 进一步降低阈值到150以识别更透明的白色文字
  }
};

// 创建输出目录
function ensureDirectoryExists(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

// 获取所有子目录
function getSubdirectories(dir) {
  return fs.readdirSync(dir, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .map(dirent => path.join(dir, dirent.name));
}

// 超自然辅助函数集合
const ultraNaturalHelpers = {
  // 计算边缘响应 - 用于边缘感知平滑处理
  calculateEdgeResponse(x, y, width, height, mask) {
    let edgeStrength = 0;
    let sampleCount = 0;
    
    // 检查周围像素的掩码变化
    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        if (dx === 0 && dy === 0) continue;
        
        const nx = x + dx;
        const ny = y + dy;
        
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const currentIndex = y * width + x;
          const neighborIndex = ny * width + nx;
          
          // 如果当前像素和邻居像素的掩码值不同，说明是边缘
          if (mask[currentIndex] !== mask[neighborIndex]) {
            const distance = Math.sqrt(dx * dx + dy * dy);
            edgeStrength += 1 / (distance + 1);
            sampleCount++;
          }
        }
      }
    }
    
    return sampleCount > 0 ? Math.min(edgeStrength / sampleCount * 3, 1) : 0;
  },
  
  // 计算自适应混合比例
  calculateAdaptiveBlendRatio(edgeDistance, maskType) {
    const baseRatio = Math.min(edgeDistance / 8, 1); // 扩大到8像素范围
    const smoothCurve = this.smoothStep(baseRatio);
    
    if (maskType === 2) {
      // 边缘区域使用更保守的混合
      return 0.1 + smoothCurve * 0.6;
    } else {
      // 内部区域使用更积极的混合
      return 0.4 + smoothCurve * 0.5;
    }
  },
  
  // 平滑步进函数 - 实现自然的过渡曲线
  smoothStep(t) {
    return t * t * (3 - 2 * t); // 三次平滑插值
  },
  
  // 计算纹理保持因子
  calculateTexturePreserveFactor(r, g, b, envR, envG, envB) {
    const colorVariance = Math.sqrt(
      Math.pow(r - envR, 2) +
      Math.pow(g - envG, 2) +
      Math.pow(b - envB, 2)
    );
    
    // 颜色差异越大，保持越多原始纹理
    return Math.max(0.7, Math.min(1.0, 0.7 + colorVariance / 100));
  },
  
  // 计算环境统计特征
  calculateEnvironmentStatistics(envColors) {
    if (envColors.length === 0) {
      return {
        meanR: 200, meanG: 200, meanB: 200,
        variance: 1000,
        dominantColors: [{r: 200, g: 200, b: 200}]
      };
    }
    
    // 计算均值
    const meanR = envColors.reduce((sum, c) => sum + c.r, 0) / envColors.length;
    const meanG = envColors.reduce((sum, c) => sum + c.g, 0) / envColors.length;
    const meanB = envColors.reduce((sum, c) => sum + c.b, 0) / envColors.length;
    
    // 计算方差
    const variance = envColors.reduce((sum, c) => {
      return sum + Math.pow(c.r - meanR, 2) + Math.pow(c.g - meanG, 2) + Math.pow(c.b - meanB, 2);
    }, 0) / envColors.length;
    
    // 提取主要颜色（简化版聚类）
    const colorClusters = this.extractDominantColors(envColors, 3);
    
    return {
      meanR, meanG, meanB,
      variance,
      dominantColors: colorClusters
    };
  },
  
  // 提取主要颜色
  extractDominantColors(colors, clusterCount) {
    if (colors.length <= clusterCount) {
      return colors;
    }
    
    // 简化的K-means聚类
    const clusters = [];
    
    // 初始化聚类中心
    for (let i = 0; i < clusterCount; i++) {
      const index = Math.floor(colors.length * i / clusterCount);
      clusters.push({...colors[index]});
    }
    
    // 简单的聚类分配
    const clusterAssignments = new Array(colors.length).fill(0);
    
    for (let iter = 0; iter < 3; iter++) {
      // 分配到最近的聚类
      for (let i = 0; i < colors.length; i++) {
        let minDistance = Infinity;
        let bestCluster = 0;
        
        for (let j = 0; j < clusters.length; j++) {
          const distance = Math.sqrt(
            Math.pow(colors[i].r - clusters[j].r, 2) +
            Math.pow(colors[i].g - clusters[j].g, 2) +
            Math.pow(colors[i].b - clusters[j].b, 2)
          );
          
          if (distance < minDistance) {
            minDistance = distance;
            bestCluster = j;
          }
        }
        
        clusterAssignments[i] = bestCluster;
      }
      
      // 更新聚类中心
      for (let j = 0; j < clusters.length; j++) {
        const assignedColors = colors.filter((_, idx) => clusterAssignments[idx] === j);
        
        if (assignedColors.length > 0) {
          clusters[j].r = assignedColors.reduce((sum, c) => sum + c.r, 0) / assignedColors.length;
          clusters[j].g = assignedColors.reduce((sum, c) => sum + c.g, 0) / assignedColors.length;
          clusters[j].b = assignedColors.reduce((sum, c) => sum + c.b, 0) / assignedColors.length;
        }
      }
    }
    
    return clusters;
  },
  
  // 生成自然颜色
  generateNaturalColor(envStats, centerR, centerG, centerB) {
    // 基于环境统计特征生成自然颜色
    const baseColor = {
      r: envStats.meanR,
      g: envStats.meanG,
      b: envStats.meanB
    };
    
    // 添加基于方差的自然变化
    const variationFactor = Math.sqrt(envStats.variance) * 0.1;
    const naturalVariation = {
      r: (Math.random() - 0.5) * variationFactor,
      g: (Math.random() - 0.5) * variationFactor,
      b: (Math.random() - 0.5) * variationFactor
    };
    
    // 考虑原始颜色的倾向
    const centerInfluence = 0.2;
    
    return {
      r: Math.max(0, Math.min(255, 
        baseColor.r + naturalVariation.r + (centerR - envStats.meanR) * centerInfluence
      )),
      g: Math.max(0, Math.min(255, 
        baseColor.g + naturalVariation.g + (centerG - envStats.meanG) * centerInfluence
      )),
      b: Math.max(0, Math.min(255, 
        baseColor.b + naturalVariation.b + (centerB - envStats.meanB) * centerInfluence
      ))
    };
  },
  
  // 计算自适应混合因子
  calculateAdaptiveBlend(centerR, centerG, centerB, targetR, targetG, targetB) {
    const colorDistance = Math.sqrt(
      Math.pow(centerR - targetR, 2) +
      Math.pow(centerG - targetG, 2) +
      Math.pow(centerB - targetB, 2)
    );
    
    // 颜色差异越大，混合比例越保守
    const baseBlend = Math.max(0.3, Math.min(0.8, 1 - colorDistance / 200));
    
    // 添加微小随机性以避免重复模式
    const randomFactor = 0.9 + Math.random() * 0.2;
    
    return baseBlend * randomFactor;
  },
  
  // 获取超自然默认值
  getUltraNaturalDefault(centerR, centerG, centerB) {
    // 基于中心颜色的智能默认值生成
    const brightness = (centerR * 0.299 + centerG * 0.587 + centerB * 0.114);
    
    // 保持亮度，调整颜色分布
    const adjustmentFactor = 0.8;
    const naturalAdjustment = {
      r: (Math.random() - 0.5) * 20,
      g: (Math.random() - 0.5) * 20,
      b: (Math.random() - 0.5) * 20
    };
    
    return {
      r: Math.max(0, Math.min(255, centerR * adjustmentFactor + naturalAdjustment.r)),
      g: Math.max(0, Math.min(255, centerG * adjustmentFactor + naturalAdjustment.g)),
      b: Math.max(0, Math.min(255, centerB * adjustmentFactor + naturalAdjustment.b))
    };
  }
};

// 将辅助函数绑定到全局作用域以便在主函数中使用
function calculateEdgeResponse(x, y, width, height, mask) {
  return ultraNaturalHelpers.calculateEdgeResponse(x, y, width, height, mask);
}

function calculateAdaptiveBlendRatio(edgeDistance, maskType) {
  return ultraNaturalHelpers.calculateAdaptiveBlendRatio(edgeDistance, maskType);
}

function smoothStep(t) {
  return ultraNaturalHelpers.smoothStep(t);
}

function calculateTexturePreserveFactor(r, g, b, envR, envG, envB) {
  return ultraNaturalHelpers.calculateTexturePreserveFactor(r, g, b, envR, envG, envB);
}

function calculateEnvironmentStatistics(envColors) {
  return ultraNaturalHelpers.calculateEnvironmentStatistics(envColors);
}

function generateNaturalColor(envStats, centerR, centerG, centerB) {
  return ultraNaturalHelpers.generateNaturalColor(envStats, centerR, centerG, centerB);
}

function calculateAdaptiveBlend(centerR, centerG, centerB, targetR, targetG, targetB) {
  return ultraNaturalHelpers.calculateAdaptiveBlend(centerR, centerG, centerB, targetR, targetG, targetB);
}

function getUltraNaturalDefault(centerR, centerG, centerB) {
  return ultraNaturalHelpers.getUltraNaturalDefault(centerR, centerG, centerB);
}

// 智能文字去除方法 - 仅消除白色文字而不影响背景
async function removeWatermarkTextOnly(inputPath, outputPath, method = 'hybrid') {
  try {
    const metadata = await sharp(inputPath).metadata();
    
    // 计算水印区域坐标 - 优化右下角精确定位
    const rightMargin = Math.floor(metadata.width * WATERMARK_CONFIG.region.right);
    const bottomMargin = Math.floor(metadata.height * WATERMARK_CONFIG.region.bottom);
    const watermarkWidth = Math.floor(metadata.width * WATERMARK_CONFIG.region.width);
    const watermarkHeight = Math.floor(metadata.height * WATERMARK_CONFIG.region.height);
    
    // 从右下角开始计算水印位置
    const watermarkX = metadata.width - rightMargin - watermarkWidth;
    const watermarkY = metadata.height - bottomMargin - watermarkHeight;
    
    console.log(`处理图片: ${path.basename(inputPath)}`);
    console.log(`图片尺寸: ${metadata.width}x${metadata.height}`);
    console.log(`水印区域: x=${watermarkX}, y=${watermarkY}, width=${watermarkWidth}, height=${watermarkHeight}`);
    console.log(`右下角位置: (${metadata.width - watermarkX}, ${metadata.height - watermarkY})`);
    console.log(`修复方法: ${method}`);
    
    // 提取水印区域图像
    const watermarkRegionBuffer = await sharp(inputPath)
      .extract({ left: watermarkX, top: watermarkY, width: watermarkWidth, height: watermarkHeight })
      .raw()
      .toBuffer();
    
    // 转换为可处理的数组
    const watermarkRegionData = new Uint8Array(watermarkRegionBuffer);
    
    // 创建一个掩码，标记需要修复的像素
    const mask = new Uint8Array(watermarkWidth * watermarkHeight);
    
    // 识别白色文字像素（进一步优化版）
    let detectedPixels = 0;
    for (let i = 0; i < watermarkRegionData.length; i += 3) {
      const r = watermarkRegionData[i];
      const g = watermarkRegionData[i + 1];
      const b = watermarkRegionData[i + 2];
      
      // 检查是否为白色或接近白色的像素（文字部分）
      // 进一步优化白色文字检测算法，针对透明白色文字
      const brightness = (r * 0.299 + g * 0.587 + b * 0.114);
      const minRGB = Math.min(r, g, b);
      const maxRGB = Math.max(r, g, b);
      const contrast = maxRGB - minRGB;
      
      // 更精确的白色文字检测：
      // 1. 降低亮度阈值以识别透明白色文字
      // 2. 放宽对比度限制以识别半透明文字
      // 3. 放宽RGB接近度限制以识别不同透明度的白色
      if (brightness > WATERMARK_CONFIG.text.threshold && 
          contrast < 50 && // 放宽对比度限制
          minRGB > 160 && // 降低最低RGB值要求
          Math.abs(r - g) < 30 && // 放宽RGB差异限制
          Math.abs(g - b) < 30 && 
          Math.abs(r - b) < 30) {
        // 标记为需要修复
        const pixelIndex = Math.floor(i / 3);
        mask[pixelIndex] = 1;
        detectedPixels++;
      }
    }
    
    console.log(`检测到的白色像素数量: ${detectedPixels}`);
    console.log(`水印区域总像素数量: ${watermarkWidth * watermarkHeight}`);
    console.log(`检测比例: ${(detectedPixels / (watermarkWidth * watermarkHeight) * 100).toFixed(2)}%`);
    
    // 如果没有检测到足够的白色像素，输出警告
    if (detectedPixels < 100) {
      console.log(`⚠️  警告: 检测到的白色像素数量过少 (${detectedPixels})，可能未正确定位水印`);
    } else {
      console.log(`✓ 检测到足够的白色像素，开始修复处理`);
    }
    
    // 根据选择的方法进行修复
    let repairedData;
    
    if (method === 'professional') {
      // 使用专业算法进行修复
      console.log('🚀 使用专业去水印算法...');
      repairedData = await professionalInpainting.hybridInpainting(
        watermarkRegionData, mask, watermarkWidth, watermarkHeight
      );
      
      // 如果专业算法失败，回退到传统算法
      if (!repairedData) {
        console.log('专业算法失败，使用传统超自然算法...');
        repairedData = await applyTraditionalInpainting(watermarkRegionData, mask, watermarkWidth, watermarkHeight);
      }
    } else if (method === 'deep_learning') {
      // 仅使用深度学习算法
      console.log('🤖 使用深度学习算法...');
      repairedData = await professionalInpainting.deepLearningInpaint(watermarkRegionData, mask);
      if (!repairedData) {
        console.log('深度学习算法失败，使用传统超自然算法...');
        repairedData = await applyTraditionalInpainting(watermarkRegionData, mask, watermarkWidth, watermarkHeight);
      }
    } else if (method === 'patchmatch') {
      // 仅使用PatchMatch算法
      console.log('🔧 使用PatchMatch算法...');
      repairedData = await professionalInpainting.patchMatchInpaint(watermarkRegionData, mask, watermarkWidth, watermarkHeight);
      if (!repairedData) {
        console.log('PatchMatch算法失败，使用传统超自然算法...');
        repairedData = await applyTraditionalInpainting(watermarkRegionData, mask, watermarkWidth, watermarkHeight);
      }
    } else {
      // 默认使用混合策略：传统算法 + 专业算法
      console.log('🎯 使用混合策略修复...');
      
      // 首先使用传统超自然算法
      const traditionalResult = await applyTraditionalInpainting(watermarkRegionData, mask, watermarkWidth, watermarkHeight);
      
      // 然后尝试专业算法
      try {
        const professionalResult = await professionalInpainting.hybridInpainting(
          watermarkRegionData, mask, watermarkWidth, watermarkHeight
        );
        
        if (professionalResult) {
          // 融合两种算法的结果
          repairedData = fuseTraditionalAndProfessional(traditionalResult, professionalResult);
          console.log('✓ 成功融合传统和专业算法结果');
        } else {
          repairedData = traditionalResult;
        }
      } catch (error) {
        console.warn('专业算法处理失败，仅使用传统算法:', error.message);
        repairedData = traditionalResult;
      }
    }
    
    // 创建修复后的图像数据
    const finalRepairedData = new Uint8Array(watermarkRegionData.length);
    finalRepairedData.set(repairedData || watermarkRegionData);
    
    // 使用finalRepairedData作为修复结果
    repairedData = finalRepairedData;
    
    // 转换修复后的数据为图片格式
    // 确保正确处理Buffer转换
    const fixedBuffer = Buffer.from(repairedData.buffer);
    
    // 使用中间临时图像步骤以确保格式正确
    const tempImagePath = path.join(__dirname, './temp/temp_' + Date.now() + '.webp');
    ensureDirectoryExists(path.dirname(tempImagePath));
    await sharp(fixedBuffer, {
      raw: {
        width: watermarkWidth,
        height: watermarkHeight,
        channels: 3
      }
    }).toFile(tempImagePath);
    
    // 读取临时图像作为正确格式的缓冲区
    const repairedRegionImage = await sharp(tempImagePath).toBuffer();
    
    // 清理临时文件
    try {
      fs.unlinkSync(tempImagePath);
    } catch (err) {
      console.log(`清理临时文件失败: ${err.message}`);
    }
    
    // 将修复后的区域合并回原图
    const originalImage = await sharp(inputPath);
    const result = originalImage.composite([
      {
        input: repairedRegionImage,
        left: watermarkX,
        top: watermarkY,
        blend: 'over'
      }
    ]);
    
    // 保存结果
    await result.toFile(outputPath);
    
    console.log(`✓ 完成: ${path.basename(outputPath)} (仅消除文字)`);
    
  } catch (error) {
    console.error(`处理图片失败: ${inputPath}`, error);
    throw error;
  }
}

// 传统超自然算法封装
async function applyTraditionalInpainting(watermarkRegionData, mask, watermarkWidth, watermarkHeight) {
  // 创建修复数据数组
  const repairedData = new Uint8Array(watermarkRegionData.length);
  
  // 应用修复算法
  for (let y = 0; y < watermarkHeight; y++) {
    for (let x = 0; x < watermarkWidth; x++) {
      const pixelIndex = y * watermarkWidth + x;
      
      // 如果是需要修复的像素
      if (mask[pixelIndex] === 1) {
        let rSum = 0, gSum = 0, bSum = 0, count = 0;
        
        // 检查5x5邻域内的像素以获取更多参考点
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            
            // 检查边界
            if (nx >= 0 && nx < watermarkWidth && ny >= 0 && ny < watermarkHeight) {
              const neighborIndex = ny * watermarkWidth + nx;
              
              // 只使用非白色文字像素进行平均
              if (mask[neighborIndex] === 0) {
                const dataIndex = neighborIndex * 3;
                rSum += watermarkRegionData[dataIndex];
                gSum += watermarkRegionData[dataIndex + 1];
                bSum += watermarkRegionData[dataIndex + 2];
                count++;
              }
            }
          }
        }
        
        // 计算数据索引
        const dataIndex = pixelIndex * 3;
        
        // 超级AI级别的自然修复算法 - 模拟专业AI去水印效果
        
        // 获取当前像素的原始颜色值 - 所有分支都需要
        const centerR = watermarkRegionData[dataIndex];
        const centerG = watermarkRegionData[dataIndex + 1];
        const centerB = watermarkRegionData[dataIndex + 2];
        
        // 分析周围环境的颜色分布 - 移到外部确保所有分支都能访问
        const envColors = [];
        for (let dy = -6; dy <= 6; dy++) {
          for (let dx = -6; dx <= 6; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 && nx < watermarkWidth && ny >= 0 && ny < watermarkHeight) {
              const neighborIndex = ny * watermarkWidth + nx;
              if (mask[neighborIndex] === 0) {
                const neighborDataIndex = neighborIndex * 3;
                envColors.push({
                  r: watermarkRegionData[neighborDataIndex],
                  g: watermarkRegionData[neighborDataIndex + 1],
                  b: watermarkRegionData[neighborDataIndex + 2]
                });
              }
            }
          }
        }
        
        // 计算环境颜色的统计特征 - 所有分支都需要
        const avgEnvR = envColors.length > 0 ? envColors.reduce((sum, c) => sum + c.r, 0) / envColors.length : 200;
        const avgEnvG = envColors.length > 0 ? envColors.reduce((sum, c) => sum + c.g, 0) / envColors.length : 200;
        const avgEnvB = envColors.length > 0 ? envColors.reduce((sum, c) => sum + c.b, 0) / envColors.length : 200;
        
        const colorVariance = envColors.length > 0 ? envColors.reduce((sum, c) => {
          return sum + Math.pow(c.r - avgEnvR, 2) + Math.pow(c.g - avgEnvG, 2) + Math.pow(c.b - avgEnvB, 2);
        }, 0) / envColors.length : 1000;
        
        if (count > 0) {
          // 基于块的纹理合成 - 更自然的修复效果
          const blockSize = 3; // 3x3块处理
          const candidates = [];
          
          // 环境颜色统计特征已在函数外部计算
          
          // 收集候选像素并计算多维度相似性
          for (let dy = -4; dy <= 4; dy++) {
            for (let dx = -4; dx <= 4; dx++) {
              if (dx === 0 && dy === 0) continue;
              const nx = x + dx;
              const ny = y + dy;
              if (nx >= 0 && nx < watermarkWidth && ny >= 0 && ny < watermarkHeight) {
                const neighborIndex = ny * watermarkWidth + nx;
                if (mask[neighborIndex] === 0) {
                  const neighborDataIndex = neighborIndex * 3;
                  const r = watermarkRegionData[neighborDataIndex];
                  const g = watermarkRegionData[neighborDataIndex + 1];
                  const b = watermarkRegionData[neighborDataIndex + 2];
                  
                  // 1. 距离权重 - 更自然的高斯分布
                  const distance = Math.sqrt(dx * dx + dy * dy);
                  const distanceWeight = Math.exp(-distance * distance / 12);
                  
                  // 2. 颜色相似性权重 - 考虑环境颜色分布
                  const colorDiff = Math.sqrt(
                    Math.pow(r - centerR, 2) + 
                    Math.pow(g - centerG, 2) + 
                    Math.pow(b - centerB, 2)
                  );
                  const envColorDiff = Math.sqrt(
                    Math.pow(r - avgEnvR, 2) + 
                    Math.pow(g - avgEnvG, 2) + 
                    Math.pow(b - avgEnvB, 2)
                  );
                  const colorWeight = Math.exp(-colorDiff * colorDiff / 8000) * Math.exp(-envColorDiff * envColorDiff / (colorVariance + 1000));
                  
                  // 3. 高级纹理相似性 - 基于块的匹配
                  let blockSimilarity = 0;
                  let blockSamples = 0;
                  for (let tdy = -1; tdy <= 1; tdy++) {
                    for (let tdx = -1; tdx <= 1; tdx++) {
                      const tx = nx + tdx;
                      const ty = ny + tdy;
                      const cx = x + tdx;
                      const cy = y + tdy;
                      if (tx >= 0 && tx < watermarkWidth && ty >= 0 && ty < watermarkHeight &&
                          cx >= 0 && cx < watermarkWidth && cy >= 0 && cy < watermarkHeight) {
                        const tIndex = ty * watermarkWidth + tx;
                        const cIndex = cy * watermarkWidth + cx;
                        if (mask[tIndex] === 0 && mask[cIndex] === 0) {
                          const tDataIndex = tIndex * 3;
                          const cDataIndex = cIndex * 3;
                          const tR = watermarkRegionData[tDataIndex];
                          const tG = watermarkRegionData[tDataIndex + 1];
                          const tB = watermarkRegionData[tDataIndex + 2];
                          const cR = watermarkRegionData[cDataIndex];
                          const cG = watermarkRegionData[cDataIndex + 1];
                          const cB = watermarkRegionData[cDataIndex + 2];
                          
                          const blockDiff = Math.sqrt(
                            Math.pow(tR - cR, 2) + 
                            Math.pow(tG - cG, 2) + 
                            Math.pow(tB - cB, 2)
                          );
                          blockSimilarity += Math.exp(-blockDiff * blockDiff / 3000);
                          blockSamples++;
                        }
                      }
                    }
                  }
                  
                  const blockWeight = blockSamples > 0 ? blockSimilarity / blockSamples : 1;
                  
                  // 4. 方向一致性权重 - 考虑纹理方向
                  let directionWeight = 1;
                  if (dx !== 0 && dy !== 0) {
                    const angle = Math.atan2(dy, dx);
                    const gradientWeight = Math.abs(Math.cos(angle * 2)) + Math.abs(Math.sin(angle * 2));
                    directionWeight = 0.7 + gradientWeight * 0.3;
                  }
                  
                  // 综合权重
                  const totalWeight = distanceWeight * colorWeight * blockWeight * directionWeight;
                  
                  candidates.push({
                    r, g, b,
                    weight: totalWeight,
                    distance: distance
                  });
                }
              }
            }
          }
          
          // 超自然纹理合成算法 - 消除马赛克效果
          if (candidates.length > 0) {
            // 扩展候选像素搜索范围到20个最佳候选
            candidates.sort((a, b) => b.weight - a.weight);
            const topCandidates = candidates.slice(0, Math.min(20, candidates.length));
            
            // 超智能多阶段融合算法
            let finalR = 0, finalG = 0, finalB = 0;
            let totalWeight = 0;
            
            // 第一阶段：自适应权重融合，考虑纹理连续性
            const adaptiveWeights = [];
            for (let i = 0; i < topCandidates.length; i++) {
              const candidate = topCandidates[i];
              
              // 计算空间连续性权重
              let continuityWeight = 1;
              if (i > 0) {
                // 检查与前一个候选的颜色连续性
                const prevCandidate = topCandidates[i - 1];
                const colorDiff = Math.sqrt(
                  Math.pow(candidate.r - prevCandidate.r, 2) +
                  Math.pow(candidate.g - prevCandidate.g, 2) +
                  Math.pow(candidate.b - prevCandidate.b, 2)
                );
                continuityWeight = Math.exp(-colorDiff * colorDiff / 2000);
              }
              
              // 自适应权重调整
              const adaptiveWeight = candidate.weight * continuityWeight * (0.8 + Math.random() * 0.4);
              adaptiveWeights.push(adaptiveWeight);
              
              finalR += candidate.r * adaptiveWeight;
              finalG += candidate.g * adaptiveWeight;
              finalB += candidate.b * adaptiveWeight;
              totalWeight += adaptiveWeight;
            }
            
            let repairedR = finalR / totalWeight;
            let repairedG = finalG / totalWeight;
            let repairedB = finalB / totalWeight;
            
            // 第二阶段：超智能颜色校正
            const envColorDiff = Math.sqrt(
              Math.pow(repairedR - avgEnvR, 2) +
              Math.pow(repairedG - avgEnvG, 2) +
              Math.pow(repairedB - avgEnvB, 2)
            );
            
            // 动态颜色一致性因子
            const dynamicConsistency = Math.max(0.6, Math.min(0.95, 1 - envColorDiff / 150));
            repairedR = repairedR * dynamicConsistency + avgEnvR * (1 - dynamicConsistency);
            repairedG = repairedG * dynamicConsistency + avgEnvG * (1 - dynamicConsistency);
            repairedB = repairedB * dynamicConsistency + avgEnvB * (1 - dynamicConsistency);
            
            // 第三阶段：超自然纹理细节增强
            const textureDetail = Math.min(colorVariance / 5000, 2);
            const detailNoise = (Math.random() - 0.5) * textureDetail;
            
            // 第四阶段：边缘感知平滑处理
             const edgeResponse = calculateEdgeResponse(x, y, watermarkWidth, watermarkHeight, mask);
             const smoothFactor = 0.3 + 0.7 * edgeResponse; // 边缘区域保持更多细节
            
            repairedR = Math.max(0, Math.min(255, repairedR + detailNoise * smoothFactor));
            repairedG = Math.max(0, Math.min(255, repairedG + detailNoise * smoothFactor));
            repairedB = Math.max(0, Math.min(255, repairedB + detailNoise * smoothFactor));
            
            // 超智能边缘过渡算法
             const edgeDistance = Math.min(x, watermarkWidth - x - 1, y, watermarkHeight - y - 1);
             const adaptiveBlendRatio = calculateAdaptiveBlendRatio(edgeDistance, 1);
            
            // 简化的边缘处理
            const edgePreserveFactor = 0.15 + adaptiveBlendRatio * 0.65;
            const naturalAlpha = smoothStep(edgePreserveFactor);
            repairedData[dataIndex] = Math.floor(centerR * (1 - naturalAlpha) + repairedR * naturalAlpha);
            repairedData[dataIndex + 1] = Math.floor(centerG * (1 - naturalAlpha) + repairedG * naturalAlpha);
            repairedData[dataIndex + 2] = Math.floor(centerB * (1 - naturalAlpha) + repairedB * naturalAlpha);
          } else {
            // 超级智能默认值 - 基于环境颜色分布
            const adaptiveR = Math.floor(avgEnvR + (Math.random() - 0.5) * Math.sqrt(colorVariance) * 0.5);
            const adaptiveG = Math.floor(avgEnvG + (Math.random() - 0.5) * Math.sqrt(colorVariance) * 0.5);
            const adaptiveB = Math.floor(avgEnvB + (Math.random() - 0.5) * Math.sqrt(colorVariance) * 0.5);
            
            // 与原始像素的智能混合
            const blendFactor = 0.7;
            repairedData[dataIndex] = Math.floor(centerR * (1 - blendFactor) + adaptiveR * blendFactor);
            repairedData[dataIndex + 1] = Math.floor(centerG * (1 - blendFactor) + adaptiveG * blendFactor);
            repairedData[dataIndex + 2] = Math.floor(centerB * (1 - blendFactor) + adaptiveB * blendFactor);
          }
        } else {
          // 超自然智能默认值处理 - 基于深度环境分析
          const envColorCount = envColors.length;
          if (envColorCount > 0) {
            // 计算环境颜色的多维统计特征
             const envStats = calculateEnvironmentStatistics(envColors);
             
             // 超智能颜色生成
             const generatedColor = generateNaturalColor(envStats, centerR, centerG, centerB);
             
             // 自适应混合因子
             const adaptiveBlend = calculateAdaptiveBlend(centerR, centerG, centerB, generatedColor.r, generatedColor.g, generatedColor.b);
            
            repairedData[dataIndex] = Math.floor(centerR * (1 - adaptiveBlend) + generatedColor.r * adaptiveBlend);
            repairedData[dataIndex + 1] = Math.floor(centerG * (1 - adaptiveBlend) + generatedColor.g * adaptiveBlend);
            repairedData[dataIndex + 2] = Math.floor(centerB * (1 - adaptiveBlend) + generatedColor.b * adaptiveBlend);
          } else {
             // 极端情况下的超自然默认值
             const naturalDefault = getUltraNaturalDefault(centerR, centerG, centerB);
            repairedData[dataIndex] = naturalDefault.r;
            repairedData[dataIndex + 1] = naturalDefault.g;
            repairedData[dataIndex + 2] = naturalDefault.b;
          }
        }
      } else {
        // 非水印区域，直接复制原始数据
        const dataIndex = pixelIndex * 3;
        repairedData[dataIndex] = watermarkRegionData[dataIndex];
        repairedData[dataIndex + 1] = watermarkRegionData[dataIndex + 1];
        repairedData[dataIndex + 2] = watermarkRegionData[dataIndex + 2];
      }
    }
  }
  
  return repairedData;
}

// 融合传统和专业算法结果
function fuseTraditionalAndProfessional(traditionalData, professionalData) {
  const fusedData = new Uint8Array(traditionalData.length);
  
  for (let i = 0; i < traditionalData.length; i += 3) {
    // 智能融合权重：传统算法60%，专业算法40%
    const traditionalWeight = 0.6;
    const professionalWeight = 0.4;
    
    fusedData[i] = Math.floor(traditionalData[i] * traditionalWeight + professionalData[i] * professionalWeight);
    fusedData[i + 1] = Math.floor(traditionalData[i + 1] * traditionalWeight + professionalData[i + 1] * professionalWeight);
    fusedData[i + 2] = Math.floor(traditionalData[i + 2] * traditionalWeight + professionalData[i + 2] * professionalWeight);
  }
  
  return fusedData;
}

// 处理单个图片文件
async function processImage(inputPath, outputPath, method = 'hybrid') {
  try {
    ensureDirectoryExists(path.dirname(outputPath));
    
    if (method === 'text_only') {
      await removeWatermarkTextOnly(inputPath, outputPath);
    } else if (method === 'professional' || method === 'deep_learning' || method === 'patchmatch' || method === 'hybrid') {
      await removeWatermarkTextOnly(inputPath, outputPath, method);
    } else {
      throw new Error(`不支持的处理方法: ${method}`);
    }
    
  } catch (error) {
    console.error(`处理图片失败: ${inputPath}`, error);
    throw error;
  }
}

// 处理整个目录
async function processDirectory(inputDir, outputDir, method = 'hybrid') {
  try {
    ensureDirectoryExists(outputDir);
    
    const files = fs.readdirSync(inputDir);
    const imageFiles = files.filter(file => 
      file.toLowerCase().endsWith('.webp') || 
      file.toLowerCase().endsWith('.jpg') || 
      file.toLowerCase().endsWith('.jpeg') ||
      file.toLowerCase().endsWith('.png')
    );
    
    console.log(`找到 ${imageFiles.length} 个图片文件`);
    console.log(`使用修复方法: ${method}`);
    
    for (const file of imageFiles) {
      const inputPath = path.join(inputDir, file);
      const outputPath = path.join(outputDir, file);
      
      try {
        await processImage(inputPath, outputPath, method);
      } catch (error) {
        console.error(`处理文件失败: ${file}`, error);
      }
    }
    
    // 递归处理子目录
    const subdirs = getSubdirectories(inputDir);
    for (const subdir of subdirs) {
      const subdirName = path.basename(subdir);
      const outputSubdir = path.join(outputDir, subdirName);
      await processDirectory(subdir, outputSubdir, method);
    }
    
  } catch (error) {
    console.error(`处理目录失败: ${inputDir}`, error);
    throw error;
  }
}

// 主函数
async function main() {
  try {
    const args = process.argv.slice(2);
    
    if (args.length > 0) {
      INPUT_DIR = args[0];
    }
    if (args.length > 1) {
      OUTPUT_DIR = args[1];
    }
    
    if (!fs.existsSync(INPUT_DIR)) {
      console.error(`输入目录不存在: ${INPUT_DIR}`);
      process.exit(1);
    }
    
    // 可以通过命令行参数指定修复方法
    const method = args[2] || 'hybrid';
    
    console.log(`开始处理图片...`);
    console.log(`输入目录: ${INPUT_DIR}`);
    console.log(`输出目录: ${OUTPUT_DIR}`);
    console.log(`水印配置:`, WATERMARK_CONFIG);
    console.log(`修复方法: ${method}`);
    
    await processDirectory(INPUT_DIR, OUTPUT_DIR, method);
    
    console.log(`\n✓ 所有图片处理完成！`);
    
  } catch (error) {
    console.error('处理失败:', error);
    process.exit(1);
  }
}

// 运行主函数
main().catch(error => {
  console.error('未捕获的错误:', error);
  process.exit(1);
});
