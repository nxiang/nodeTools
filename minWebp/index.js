import fs from 'fs';
import os from 'os';
import path from 'path';
import pMap from 'p-map';
import imagemin from 'imagemin';
import cliProgress from 'cli-progress';
import imageminWebp from 'imagemin-webp';

function log(...args) {
  console.log(...args, '\n');
}

function logError(...args) {
  console.error(...args, '\n');
}

const concurrency = Math.max(os.cpus().length - 1, 1); // 留一个核心给系统

/** 检查文件名是否有_skip标记 */
function hasSkipMark(filePath) {
  // 匹配 _skip 或 _skip_任意内容
  return /_skip(_\d+)?\.(png|jpe?g)$/i.test(path.basename(filePath));
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
async function getAllPicFilePaths(dir) {
  let results = [];
  const list = await fs.promises.readdir(dir);
  await pMap(list, async (file) => {
    const filePath = path.join(dir, file);
    if (hasSkipMark(filePath)) return;
    let stat;
    try {
      stat = await fs.promises.stat(filePath);
    } catch (err) {
      logError('读取文件状态失败，跳过:', filePath, err);
      return;
    }
    if (stat.isFile() && /\.(png|jpe?g)$/i.test(file)) {
      results.push(filePath);
    } else if (stat.isDirectory()) {
      try {
        const subResults = await getAllPicFilePaths(filePath);

        results = results.concat(subResults);
      } catch (err) {
        logError('递归子目录失败，跳过:', filePath, err);
      }
    }
  }, {
    concurrency
  }); // 可根据实际情况调整并发数
  return results;
}

/**
 * 根据文件的绝对路径删除文件
 * @param {string} filePath 文件绝对路径
 * @returns {Promise<void>}
 */
async function deleteFileByPath(filePath) {
  try {
    await fs.promises.unlink(filePath);
    log('文件已删除:', filePath);
  } catch (err) {
    logError('删除文件时发生错误:', err);
  }
}

async function main() {
  const targetDir = '\\\\DXP4800PLUS-BE5\\personal_folder\\视频\\成人内容\\写真\\[Hane Ame雨波]\\[HaneAme Collection] - 副本'; // 请替换为你的目标目录
  // const targetDir = '\\\\DXP4800PLUS-BE5\\personal_folder\\视频\\成人内容\\写真\\[Hane Ame雨波]\\Haneame 23年4月 新 - 副本'; // 请替换为你的目标目录
  if (!fs.existsSync(targetDir)) {
    logError('目标目录不存在:', targetDir);
    return;
  }
  const startTime = Date.now(); // 记录开始时间
  const imgPaths = await getAllPicFilePaths(targetDir);
  log('待处理图片数量:', imgPaths.length);
  // 初始化进度条
  const bar = new cliProgress.SingleBar({
    format: '压缩进度 |{bar}| {percentage}% | {value}/{total} | 当前: {filename}',
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
    const outputFile = path.join(outputDir, path.basename(imgPath, ext) + '.webp');
    try {
      if (fs.existsSync(outputFile)) {
        await deleteFileByPath(imgPath);
        log('文件已存在，结束循环:', outputFile);
        return;
      }
      let buffer;
      try {
        buffer = await fs.promises.readFile(imgPath);
      } catch (readErr) {
        logError('读取图片失败，可能图片损坏:', imgPath, readErr);
        return;
      }
      let webpBuffer;
      try {
        webpBuffer = await imagemin.buffer(buffer, {
          plugins: [
            imageminWebp(
              ext === '.png' ? {
                lossless: 9
              } : {
                quality: 85
              }
            )
          ]
        });
      } catch (compressErr) {
        logError('图片压缩失败:', imgPath, compressErr);
        return;
      }
      // 比较文件大小
      if (webpBuffer.length < buffer.length) {
        try {
          await fs.promises.writeFile(outputFile, webpBuffer);
          log('图片已压缩并保存到:', outputFile);
          await deleteFileByPath(imgPath);
        } catch (writeErr) {
          logError('写入压缩图片失败:', outputFile, writeErr);
        }
      } else {
        try {
          await markSkipFile(imgPath); // 重命名为_skip
          log('原图比 webp 更小，跳过:', imgPath);
        } catch (error) {
          logError('重命名为_skip失败:', imgPath, error);
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
