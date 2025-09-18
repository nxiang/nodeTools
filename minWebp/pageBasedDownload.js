import * as cheerio from 'cheerio';
import {
  getPageWithRetry,
  findNextPageUrl,
  ensureFullUrl,
  crawlWebsite
} from './utils.js';

/**
 * 分页下载主函数 
 */
async function pageBasedDownload(listPageUrl, firstCustomDirName = null) {
  try {
    console.log(`开始分页爬取: ${listPageUrl}`);

    let currentPageUrl = listPageUrl;
    let pageCount = 1;
    let totalDownloadedPages = 0;

    // 获取第一页内容以设置初始pageCount
    const firstResponse = await getPageWithRetry(currentPageUrl);
    const $first = cheerio.load(firstResponse.data);

    // 获取类名current的文本作为pageCount初始值
    const currentText = $first('.current').text().trim();
    if (currentText && !isNaN(parseInt(currentText))) {
      pageCount = parseInt(currentText);
    }

    while (currentPageUrl) {
      console.log(`\n=== 处理第 ${pageCount} 页 ===`);
      console.log(`当前页面URL: ${currentPageUrl}`);

      // 获取当前页面内容
      const response = await getPageWithRetry(currentPageUrl);
      const $ = cheerio.load(response.data);

      // 提取所有item-link类的a标签href属性
      const itemLinks = $('.item-link');
      console.log(`找到 ${itemLinks.length} 个项目链接`);

      if (itemLinks.length === 0) {
        console.log('未找到类名为item-link的a标签');
      }

      // 准备项目处理任务
      const projectTasks = [];

      for (let i = 0; i < itemLinks.length; i++) {
        const link = $(itemLinks[i]);
        const href = link.attr('href');

        if (!href) {
          console.warn(`第 ${i + 1} 个链接缺少href属性`);
          continue;
        }

        // 确保URL是完整的
        let fullUrl = ensureFullUrl(href, currentPageUrl);


        // 获取项目名（item-link下的item-link-text类元素的文本内容）
        const projectName = $(link).find('.item-link-text').text().trim() || `项目_${i + 1}`;

        // 添加到项目任务列表
        projectTasks.push({
          url: fullUrl,
          name: projectName,
          index: i + 1
        });
      }

      // 并发处理项目（最多5个并发）
      const concurrencyLimit = 2;
      let pageDownloadCount = 0;

      // 创建并发控制器
      async function processProjectsWithConcurrency(tasks, limit) {
        const executing = new Set();
        const results = [];

        for (const task of tasks) {
          // 如果当前并发数达到限制，等待其中一个完成
          if (executing.size >= limit) {
            await Promise.race(executing);
          }

          // 创建项目处理任务
          const promise = (async () => {
            console.log(`\n处理第 ${task.index}/${itemLinks.length} 个项目: ${task.name}`);

            try {
              // 使用imageCrawler.js的crawlWebsite方法处理每个项目页面
              await crawlWebsite(task.url, null, firstCustomDirName);
              executing.delete(promise);
              return {
                success: true,
                index: task.index
              };
            } catch (error) {
              console.error(`处理项目失败 ${task.url}:`, error.message);
              executing.delete(promise);
              return {
                success: false,
                index: task.index
              };
            }
          })();

          executing.add(promise);
          results.push(promise);
        }

        return Promise.all(results);
      }
      // 执行并发处理
      const processResults = await processProjectsWithConcurrency(projectTasks, concurrencyLimit);

      // 统计成功数量
      pageDownloadCount = processResults.filter(result => result.success).length;
      totalDownloadedPages += pageDownloadCount;

      console.log(`第 ${pageCount} 页处理完成，成功下载 ${pageDownloadCount} 个项目`);

      // 查找下一页URL
      const nextPageUrl = findNextPageUrl($, currentPageUrl);

      if (nextPageUrl) {
        console.log(`找到下一页: ${nextPageUrl}`);
        currentPageUrl = nextPageUrl;
        pageCount++;

        // 添加延迟，避免请求过于频繁
        await new Promise(resolve => setTimeout(resolve, 2000));
      } else {
        console.log('没有找到下一页，爬取结束');
        currentPageUrl = null;
      }
    }

    console.log(`\n=== 分页爬取完成 ===`);
    console.log(`总共处理 ${pageCount} 页`);
    console.log(`成功下载 ${totalDownloadedPages} 个项目`);

  } catch (error) {
    console.error('分页爬取失败:', error.message);
  }
}

/**
 * 主函数 - 处理命令行参数
 */
async function main() {
  const listPageUrl = process.argv[2];

  // 获取自定义目录名参数 - 从listPageUrl以反斜杠分隔的最后一个全为数字的字符串
  let firstCustomDirName = 'downloaded_images';
  if (listPageUrl) {
    const urlParts = listPageUrl.split('/');
    // 从后往前查找第一个不全为数字的部分
    for (let i = urlParts.length - 1; i >= 0; i--) {
      const part = decodeURIComponent(urlParts[i].trim());
      if (part && !/^\d+$/.test(part)) {
        firstCustomDirName += `_${part}`;
        break;
      }
    }
  }

  if (!listPageUrl) {
    console.log('请提供列表页面URL作为参数:');
    console.log('node pageBasedDownload.js https://your-list-page-url.com');
    console.log('\n功能说明:');
    console.log('1. 提取类名为item-link的所有a标签href属性');
    console.log('2. 使用imageCrawler.js的crawlWebsite方法处理每个链接');
    console.log('3. 自动处理分页器（查找current类li元素的下一个兄弟li元素中的a标签）');
    console.log('4. 支持完整的分页爬取');
    return;
  }

  await pageBasedDownload(listPageUrl, firstCustomDirName);
}

// 运行程序
main().catch(console.error);
