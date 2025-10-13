import fs from 'fs';
import path from 'path';
import {
  pageBasedDownload
} from './utils.js';

/**
 * 获取目录下符合 [xxx] 格式的第一级文件夹名称
 * @param {string} dirPath - 要扫描的目录路径
 * @returns {string[]} - 符合格式的文件夹名称数组
 */
function getBracketFolderNames(dirPath) {
  try {
    const items = fs.readdirSync(dirPath, {
      withFileTypes: true
    });
    const bracketNames = [];

    for (const item of items) {
      if (item.isDirectory()) {
        const match = item.name.match(/\[([^\]]+)\]/);
        if (match && match[1]) {
          bracketNames.push(match[1]);
        }
      }
    }

    return bracketNames;
  } catch (error) {
    console.error(`读取目录失败: ${dirPath}`, error.message);
    return [];
  }
}

/**
 * 并发处理控制器
 * @param {string[]} items - 要处理的项数组
 * @param {number} concurrency - 并发数
 * @param {Function} processor - 处理函数
 * @returns {Promise<void>}
 */
async function processWithConcurrency(items, concurrency, processor) {
  const queue = [...items];
  let activeCount = 0;
  let completedCount = 0;
  const totalCount = items.length;

  return new Promise((resolve, reject) => {
    function processNext() {
      if (queue.length === 0 && activeCount === 0) {
        console.log(`\n所有任务处理完成！总计: ${totalCount}`);
        resolve();
        return;
      }

      if (queue.length === 0) {
        return;
      }

      if (activeCount >= concurrency) {
        return;
      }

      const item = queue.shift();
      activeCount++;
      completedCount++;

      processor(item, completedCount, totalCount)
        .then(() => {
          activeCount--;
          processNext();
        })
        .catch(error => {
          activeCount--;
          console.error(`处理项失败: ${item}, 错误:`, error.message);
          processNext();
        });

      // 立即尝试处理下一个，保持并发数
      setImmediate(processNext);
    }

    // 启动初始的并发任务
    for (let i = 0; i < Math.min(concurrency, queue.length); i++) {
      processNext();
    }
  });
}

/**
 * 主函数 - 处理命令行参数
 */
async function main() {
  // 获取自定义目录名参数
  let customDirName = '\\\\DXP4800PLUS-BE5\\personal_folder\\视频\\成人内容\\写真';

  // 获取符合 [xxx] 格式的文件夹名称
  const bracketNames = getBracketFolderNames(customDirName);

  console.log(`找到 ${bracketNames.length} 个符合 [xxx] 格式的文件夹:`);
  bracketNames.forEach(name => console.log(`  - ${name}`));

  if (bracketNames.length === 0) {
    console.log('没有找到符合格式的文件夹，程序结束。');
    return;
  }

  // 使用并发控制处理所有文件夹
  await processWithConcurrency(
    bracketNames,
    5, // 并发数为5
    async (name, current, total) => {
      console.log(`\n[${current}/${total}] 开始处理文件夹: [${name}]`);
      try {
        const listPageUrl = `https://www.hentaiclub.net/tag/${encodeURIComponent(name)}/`;
        const firstCustomDirName = `${customDirName}\\[${name}]`;
        await pageBasedDownload(listPageUrl, firstCustomDirName);
        console.log(`[${current}/${total}] 完成处理: ${listPageUrl}`);
      } catch (error) {
        console.error(`[${current}/${total}] 处理 [${name}] 时出错:`, error.message);
        throw error;
      }
    }
  );

  console.log('\n所有文件夹处理完成！');
}

// 运行程序
main().catch(console.error);
