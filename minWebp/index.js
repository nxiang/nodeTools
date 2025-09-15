import os from 'os';
import fs from 'fs';
import path from 'path';
import pMap from 'p-map';
import sharp from 'sharp';
import cliProgress from 'cli-progress';

function log(...args) {
  console.log(...args, '\n');
}

function logError(...args) {
  console.error(...args, '\n');
}

const concurrency = Math.floor(os.cpus().length * 1.2); // 比CPU核心数稍高
console.log('concurrency', concurrency)

/** 检查文件名是否有_skip标记 */
function hasSkipMark(filePath) {
  // 匹配 _skip 或 _skip_任意内容
  return /_skip(_\d+)?\.(png|jpe?g|gif)$/i.test(path.basename(filePath));
}

/** 给文件加_skip标记 */
async function markSkipFile(filePath) {
  const dir = path.dirname(filePath);
  const ext = path.extname(filePath);
  const base = path.basename(filePath, ext);
  let newPath = path.join(dir, `${base}_skip${ext}`);
  // 如果已存在，则加时间戳或随机数
  if (fs.existsSync(newPath)) {
    const timestamp = Date.now();
    newPath = path.join(dir, `${base}_skip_${timestamp}${ext}`);
  }
  await fs.promises.rename(filePath, newPath);
  log('已重命名为不压缩标记:', newPath);
}

/**
 * 获取指定目录下所有图片的绝对路径
 * @param {string} dir 绝对路径
 * @returns {string[]} 文件绝对路径数组
 */
// async function getAllPicFilePaths(dir) {
//   let results = [];
//   const list = await fs.promises.readdir(dir);
//   await pMap(list, async (file) => {
//     const filePath = path.join(dir, file);
//     if (hasSkipMark(filePath)) return;
//     let stat;
//     try {
//       stat = await fs.promises.stat(filePath);
//     } catch (err) {
//       logError('读取文件状态失败，跳过:', filePath, err);
//       return;
//     }
//     if (stat.isFile() && /\.(png|jpe?g|gif)$/i.test(file)) {
//       results.push(filePath);
//       console.count('找到图片文件');
//     } else if (stat.isDirectory()) {
//       try {
//         const subResults = await getAllPicFilePaths(filePath);

//         results = results.concat(subResults);
//       } catch (err) {
//         logError('递归子目录失败，跳过:', filePath, err);
//       }
//     }
//   }, {
//     concurrency
//   }); // 可根据实际情况调整并发数
//   return results;
// }

/**
 * 高效获取指定目录下所有图片的绝对路径
 * @param {string} dir 绝对路径
 * @returns {string[]} 文件绝对路径数组
 */
async function getAllPicFilePaths(dir) {
  const results = [];
  const queue = [dir]; // 使用队列代替递归，避免调用栈溢出

  // 创建一个Set用于去重（处理可能的链接等情况）
  const visitedDirs = new Set([dir]);

  // 使用更高的并发度用于文件扫描
  const scanConcurrency = Math.max(concurrency * 2, 8);

  while (queue.length > 0) {
    // 每次处理一批目录，而不是单个目录
    const currentDirs = queue.splice(0, Math.min(queue.length, scanConcurrency));

    // 并发处理多个目录
    const dirResults = await Promise.all(
      currentDirs.map(async (currentDir) => {
        try {
          const entries = await fs.promises.readdir(currentDir, {
            withFileTypes: true
          });
          const subDirs = [];
          const files = [];

          // 一次遍历完成文件分类
          for (const entry of entries) {
            const fullPath = path.join(currentDir, entry.name);

            if (hasSkipMark(fullPath)) continue;

            if (entry.isDirectory()) {
              if (!visitedDirs.has(fullPath)) {
                visitedDirs.add(fullPath);
                subDirs.push(fullPath);
              }
            } else if (entry.isFile() && /\.(png|jpe?g|gif)$/i.test(entry.name)) {
              files.push(fullPath);
            }
          }

          return {
            files,
            subDirs
          };
        } catch (err) {
          logError('读取目录失败，跳过:', currentDir, err);
          return {
            files: [],
            subDirs: []
          };
        }
      })
    );

    // 合并结果
    for (const {
        files,
        subDirs
      } of dirResults) {
      results.push(...files);
      queue.push(...subDirs);
    }

    // 每处理100个文件才输出一次进度，减少日志开销
    if (results.length % 100 === 0) {
      console.log(`已找到 ${results.length} 个图片文件`);
    }
  }

  console.log(`总共找到 ${results.length} 个图片文件`);
  return results;
}

/**
 * 根据文件的绝对路径删除文件
 * @param {string} filePath 文件绝对路径
 * @param {number} maxRetries 最大重试次数
 * @param {number} retryDelay 重试延迟(毫秒)
 * @returns {Promise<boolean>} 是否成功删除
 */
async function deleteFileByPath(filePath, maxRetries = 3, retryDelay = 1000) {
  let retries = 0;

  while (retries <= maxRetries) {
    try {
      await fs.promises.unlink(filePath);
      log('文件已删除:', filePath);
      return true; // 删除成功
    } catch (err) {
      if (retries === maxRetries) {
        logError(`删除文件失败(已重试${retries}次):`, filePath, err);
        log('警告: 原文件删除失败，可能需要手动清理:', filePath);
        return false; // 达到最大重试次数，返回失败
      }

      logError(`删除文件时发生错误，将在${retryDelay}ms后重试(${retries + 1}/${maxRetries}):`, err);
      retries++;

      // 等待一段时间后重试
      await new Promise(resolve => setTimeout(resolve, retryDelay));
    }
  }

  return false; // 不应该到达这里，但为了安全返回false
}

async function main() {
  const targetDir = '\\\\DXP4800PLUS-BE5\\personal_folder\\视频\\成人内容\\写真';
  // const targetDir = '\\\\DXP4800PLUS-BE5\\personal_folder\\视频\\成人内容\\写真\\[蠢沫沫]\\蠢沫沫 黑天使 [2V] - 副本';
  // const targetDir = '\\\\DXP4800PLUS-BE5\\personal_folder\\视频\\成人内容\\写真\\[Hane Ame雨波]\\[HaneAme Collection] - 副本'; // 请替换为你的目标目录
  // const targetDir = '\\\\DXP4800PLUS-BE5\\personal_folder\\视频\\成人内容\\写真\\[Hane Ame雨波]\\Haneame 23年4月 新 - 副本'; // 请替换为你的目标目录
  // const targetDir = '\\\\DXP4800PLUS-BE5\\personal_folder\\视频\\成人内容\\写真\\[Hane Ame雨波]\\Hane Ame雨波 原创 柴犬學妹波波 - 副本'; // 请替换为你的目标目录
  if (!fs.existsSync(targetDir)) {
    logError('目标目录不存在:', targetDir);
    return;
  }
  const startTime = Date.now(); // 记录开始时间
  const imgPaths = await getAllPicFilePaths(targetDir);
  log('待处理图片数量:', imgPaths.length);
  // 初始化进度条
  const bar = new cliProgress.SingleBar({
    format: '压缩进度 |{bar}| {percentage}% | {value}/{total} | 当前: {filename}\n',
    barCompleteChar: '\u2588',
    barIncompleteChar: '\u2591',
    hideCursor: true
  });
  bar.start(imgPaths.length, 0, {
    filename: ''
  });
  let processed = 0;
  await pMap(imgPaths, async (imgPath) => {
    const outputDir = path.dirname(imgPath);
    const ext = path.extname(imgPath).toLowerCase();
    let outputFile = path.join(outputDir, path.basename(imgPath, ext) + '.webp');
    try {
      if (fs.existsSync(outputFile)) {
        await deleteFileByPath(imgPath);
        log('文件已存在，结束循环:', outputFile);
        return;
      }
      // 使用sharp库进行图片压缩（不需要外部可执行文件）
      try {
        // 直接读取文件到Buffer
        const buffer = await fs.promises.readFile(imgPath);

        // 根据文件类型设置不同的压缩参数
        let webpBuffer;
        if (ext === '.png') {
          // PNG采用无损压缩
          webpBuffer = await sharp(buffer)
            .webp({
              lossless: true,
              quality: 100,
              effort: 3, // 降低压缩等级以提高速度
              reductionEffort: 4 // 添加此参数平衡质量和速度
            })
            .toBuffer();
        } else if (ext === '.gif') {
          // GIF转换为WebP（精简版，保留有效的动画处理）
          try {
            // 获取GIF的元数据
            const imageInfo = await sharp(buffer).metadata();
            const frameCount = imageInfo.pages || 1;
            log(`处理GIF: ${path.basename(imgPath)}, 帧数: ${frameCount}`);

            // 创建Sharp实例，启用动画模式
            const sharpInstance = sharp(buffer, {
              animated: true // 初始化时就启用动画模式
            });

            // 使用有效的方法A处理所有GIF
            webpBuffer = await sharpInstance
              .toFormat('webp', {
                quality: 80,
                effort: 3,
                animated: true,
                loop: 0, // 无限循环
                // 确保正确处理帧的关键参数
                pageHeight: imageInfo.height,
                pageWidth: imageInfo.width,
                // Sharp 0.34.3版本特定参数
                qualityProfile: 'default',
                minQuality: 60
              })
              .toBuffer();

            log(`GIF转换完成，WebP大小: ${(webpBuffer.length / 1024).toFixed(2)}KB`);
          } catch (gifError) {
            logError('GIF转换出错:', gifError.message);
            // 降级处理：保留第一帧
            webpBuffer = await sharp(buffer)
              .webp({
                quality: 80,
                effort: 3
              })
              .toBuffer();
          }
        } else {
          // JPG采用有损压缩
          webpBuffer = await sharp(buffer)
            .webp({
              quality: 80,
              effort: 3, // 降低压缩等级
            })
            .toBuffer();
        }

        // 比较压缩前后的文件大小
        if (webpBuffer.length < buffer.length) {
          if (fs.existsSync(outputFile)) {
            log(`检测到文件冲突: ${outputFile}`);
            const timestamp = Date.now();
            outputFile = path.join(outputDir, `${path.basename(imgPath, ext)}_${timestamp}.webp`);
            log('已重命名输出文件为:', outputFile);
          }
          // 写入压缩后的文件
          await fs.promises.writeFile(outputFile, webpBuffer);


          log('图片已压缩并保存到:', outputFile);
          await deleteFileByPath(imgPath);
        } else {
          log('原图比 webp 更小，跳过:', imgPath);
          await markSkipFile(imgPath);
        }
      } catch (error) {
        // 错误日志中添加文件名和详细错误信息
        logError('当前:', path.basename(imgPath), '图片压缩失败:', imgPath, error.message || error);
        logError('错误分析: 这可能是由于文件访问权限、文件损坏或路径问题导致的。');

        // 标记文件为跳过状态
        try {
          await markSkipFile(imgPath);
          log('标记文件为跳过状态，后续不会再次尝试压缩。');
        } catch (markErr) {
          logError('标记为跳过失败:', markErr.message || markErr);
        }
      }
    } catch (error) {
      logError('处理图片时发生未知错误:', imgPath, error);
    } finally {
      // 处理完成后计数+1
      processed++;
      bar.update(processed, {
        filename: path.basename(imgPath)
      });
    }
  }, {
    concurrency
  });
  bar.stop();
  const endTime = Date.now(); // 记录结束时间
  const duration = ((endTime - startTime) / 1000).toFixed(2);
  log(`全部处理完成，总耗时：${duration} 秒`);
}

main().catch(err => {
  logError('发生错误:', err);
});
