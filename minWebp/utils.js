import crypto from 'crypto';
import axios from 'axios';
import * as cheerio from 'cheerio';
import fs from 'fs';
import path from 'path';
import { promisify } from 'util';
import stream from 'stream';
import https from 'https';
import http from 'http';

/**
 * 获取页面内容（带重试机制和HTML缓存）
 */
export async function getPageWithRetry(url, retries = 5, firstCustomDirName = null, useCache = true) {
  // 只有在useCache为true时才使用缓存功能
  if (useCache) {
    // 创建HTML缓存目录
    let htmlCacheDir;
    if (firstCustomDirName && path.isAbsolute(firstCustomDirName)) {
      // 如果firstCustomDirName是绝对路径，直接使用它作为基础目录
      htmlCacheDir = path.join(firstCustomDirName, '.html');
    } else {
      // 否则使用原来的逻辑
      htmlCacheDir = path.join(process.cwd(), firstCustomDirName || '.html');
    }
    
    if (!fs.existsSync(htmlCacheDir)) {
      fs.mkdirSync(htmlCacheDir, { recursive: true });
    }

    // 生成URL的hash值作为缓存文件名
    const urlHash = crypto.createHash('md5').update(url).digest('hex');
    const cacheFilePath = path.join(htmlCacheDir, `${urlHash}.html`);

    // 检查缓存文件是否存在
    if (fs.existsSync(cacheFilePath)) {
      try {
        console.log(`✓ 从缓存读取HTML: ${url}`);
        const cachedHtml = fs.readFileSync(cacheFilePath, 'utf8');
        // 返回模拟的response对象
        return {
          data: cachedHtml,
          status: 200,
          statusText: 'OK',
          headers: {
            'content-type': 'text/html'
          },
          config: {},
          request: {}
        };
      } catch (error) {
        console.warn(`读取缓存文件失败: ${error.message}`);
      }
    }
  }

  // 缓存不存在或读取失败，进行网络请求
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await axios.get(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
          'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
          'Accept-Encoding': 'gzip, deflate',
          'Connection': 'keep-alive',
        },
        timeout: 60000,
        httpsAgent: new https.Agent({
          keepAlive: true,
          maxSockets: 10,
          maxFreeSockets: 5,
          timeout: 60000,
          freeSocketTimeout: 30000
        }),
        httpAgent: new http.Agent({
          keepAlive: true,
          maxSockets: 10,
          maxFreeSockets: 5,
          timeout: 60000,
          freeSocketTimeout: 30000
        })
      });
      
      // 只有在useCache为true时才保存缓存文件
      if (useCache) {
        // 创建HTML缓存目录
        let htmlCacheDir;
        if (firstCustomDirName && path.isAbsolute(firstCustomDirName)) {
          // 如果firstCustomDirName是绝对路径，直接使用它作为基础目录
          htmlCacheDir = path.join(firstCustomDirName, '.html');
        } else {
          // 否则使用原来的逻辑
          htmlCacheDir = path.join(process.cwd(), firstCustomDirName || '.html');
        }
        
        if (!fs.existsSync(htmlCacheDir)) {
          fs.mkdirSync(htmlCacheDir, { recursive: true });
        }

        // 生成URL的hash值作为缓存文件名
        const urlHash = crypto.createHash('md5').update(url).digest('hex');
        const cacheFilePath = path.join(htmlCacheDir, `${urlHash}.html`);
        
        // 将HTML内容保存到缓存文件
        try {
          fs.writeFileSync(cacheFilePath, response.data, 'utf8');
          console.log(`✓ HTML已缓存到: ${cacheFilePath}`);
        } catch (cacheError) {
          console.warn(`保存缓存文件失败: ${cacheError.message}`);
        }
      }
      
      return response;
    } catch (error) {
      // 如果是404错误，立即停止重试
      if (error.response && error.response.status === 404) {
        console.error(`404错误 - 页面不存在: ${url}`);
        throw error;
      }
      
      if (attempt === retries) {
        throw error;
      }
      console.warn(`第 ${attempt} 次请求失败，${retries - attempt} 次重试剩余:`, error.message);
      if (error.response) {
        console.warn(`HTTP状态码: ${error.response.status}`);
        console.warn(`响应头:`, error.response.headers);
      } else if (error.request) {
        // console.warn('请求已发送但无响应:', error.request);
        console.warn('请求已发送但无响应:');
      } else {
        console.warn('错误详情:', error);
      }
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
}

/**
 * 清理文件名，移除非法字符
 */
export function cleanFilename(filename) {
  return filename.replace(/[<>:"\/\\|?*]/g, '_')
    .replace(/\s+/g, '_')
    .substring(0, 100);
}

/**
 * 清理页面标题，移除特殊字符和后缀
 */
export function cleanPageTitle(title) {
  // 移除最后的' - 绅士会所'后缀
  const cleanedTitle = title.replace(/\s*-\s*绅士会所\s*$/, '');
  return cleanedTitle.replace(/[<>:"\/\\|?*]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * 确保URL是完整的（处理相对URL）
 */
export function ensureFullUrl(url, baseUrl) {
  if (!url.startsWith('http')) {
    const base = new URL(baseUrl);
    return new URL(url, base.origin).href;
  }
  return url;
}

/**
 * 查找下一页URL
 */
export function findNextPageUrl($, currentPageUrl) {
  // 查找当前页的li元素（类名为current）
  const currentLi = $('.current');

  if (!currentLi.length) {
    console.log('未找到当前页标记（类名为current的li元素）');
    return null;
  }

  // 获取下一个兄弟li元素
  const nextLi = currentLi.next('li');

  if (!nextLi.length) {
    console.log('没有找到下一页，可能是最后一页');
    return null;
  }

  // 获取下一个兄弟li元素中的a标签href属性
  const nextLink = nextLi.find('a');

  if (!nextLink.length) {
    console.log('下一页li元素中没有找到a标签');
    return null;
  }

  const nextUrl = nextLink.attr('href');

  if (!nextUrl) {
    console.log('下一页a标签没有href属性');
    return null;
  }

  // 确保URL是完整的
  return ensureFullUrl(nextUrl, currentPageUrl);
}

const pipeline = promisify(stream.pipeline);

/**
 * 下载图片
 */
export async function downloadImage(url, filename, directoryName = null, firstCustomDirName = null) {
  try {
    // 清理文件名
    const cleanedFilename = cleanFilename(filename);

    // 确定目标目录 - 优先使用传入的directoryName参数
    let targetDir;
    if (firstCustomDirName && path.isAbsolute(firstCustomDirName)) {
      // 如果firstCustomDirName是绝对路径，直接使用它作为基础目录
      targetDir = directoryName ? path.join(firstCustomDirName, directoryName) : firstCustomDirName;
    } else {
      // 否则使用原来的逻辑
      targetDir = directoryName ? path.join(process.cwd(), firstCustomDirName || 'downloaded_images', directoryName) : path.join(process.cwd(), firstCustomDirName || 'downloaded_images');
    }

    // 先检查文件是否已存在（在发起网络请求前）
    // 由于扩展名未知，检查所有可能的扩展名
    const possibleExtensions = ['jpg', 'jpeg', 'png', 'gif', 'webp'];
    for (const ext of possibleExtensions) {
      const filePath = path.join(targetDir, `${cleanedFilename}.${ext}`);
      if (fs.existsSync(filePath)) {
        // console.log(`✓ 文件已存在，跳过下载: ${cleanedFilename}.${ext}`);
        // console.log(`✓ 文件已存在，跳过下载: ${filePath}`);
        return true;
      }
    }

    let response;
    let retries = 3;

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        response = await axios({
          method: 'GET',
          url: url,
          responseType: 'stream',
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
          },
          timeout: 60000,
          httpsAgent: new https.Agent({
            keepAlive: true,
            maxSockets: 10,
            maxFreeSockets: 5,
            timeout: 60000,
            freeSocketTimeout: 30000
          }),
          httpAgent: new http.Agent({
            keepAlive: true,
            maxSockets: 10,
            maxFreeSockets: 5,
            timeout: 60000,
            freeSocketTimeout: 30000
          })
        });
        break; // 成功则跳出循环
      } catch (error) {
        if (attempt === retries) {
          throw error; // 最后一次尝试仍然失败
        }
        console.warn(`第 ${attempt} 次下载尝试失败，${retries - attempt} 次重试剩余:`, error.message);
        await new Promise(resolve => setTimeout(resolve, 1000)); // 1秒后重试
      }
    }

    // 获取文件扩展名
    const contentType = response.headers['content-type'];
    let extension = 'jpg';

    if (contentType) {
      if (contentType.includes('jpeg')) extension = 'jpg';
      else if (contentType.includes('png')) extension = 'png';
      else if (contentType.includes('gif')) extension = 'gif';
      else if (contentType.includes('webp')) extension = 'webp';
    }

    const filePath = path.join(targetDir, `${cleanedFilename}.${extension}`);

    // 写入文件
    const writer = fs.createWriteStream(filePath);
    await pipeline(response.data, writer);

    console.log(`✓ 下载成功: ${filePath}`);
    return true;

  } catch (error) {
    console.error(`✗ 下载失败 ${filename}:`, error.message);
    return false;
  }
}

/**
 * 爬取网站并下载图片
 */
export async function crawlWebsite(targetUrl, customDirName = null, firstCustomDirName = null) {
  try {
    console.log(`开始爬取: ${targetUrl}`);

    // 获取网页内容（使用utils中的重试机制）
    const response = await getPageWithRetry(targetUrl, 5, firstCustomDirName);

    const $ = cheerio.load(response.data);

    // 获取页面标题作为目录名
    const pageTitle = $('title').text().trim() || 'untitled_page';
    const cleanedPageTitle = cleanPageTitle(pageTitle);

    // 优先使用传入的自定义目录名，如果没有则使用页面标题
    const dirName = customDirName || cleanedPageTitle;

    // 创建下载目录
    let downloadDir;
    if (firstCustomDirName && path.isAbsolute(firstCustomDirName)) {
      // 如果firstCustomDirName是绝对路径，直接使用它作为基础目录
      downloadDir = path.join(firstCustomDirName, dirName);
    } else {
      // 否则使用原来的逻辑
      downloadDir = path.join(process.cwd(), firstCustomDirName || 'downloaded_images', dirName);
    }
    if (!fs.existsSync(downloadDir)) {
      fs.mkdirSync(downloadDir, {
        recursive: true
      });
      console.log(`创建下载目录: ${cleanedPageTitle}`);
    }

    // 查找目标图片
    const images = $('.post-item-img.lazy');
    console.log(`找到 ${images.length} 张图片`);

    if (images.length === 0) {
      console.log('未找到class为"post-item-img lazy"的图片');
      return;
    }

    let successCount = 0;

    // 准备图片下载任务
    const downloadTasks = [];

    for (let i = 0; i < images.length; i++) {
      const img = $(images[i]);
      const imageUrl = img.attr('data-original');
      const title = img.attr('title') || `image_${i + 1}`;

      if (!imageUrl) {
        console.warn(`第 ${i + 1} 张图片缺少 data-original 属性`);
        continue;
      }

      // console.log(`准备下载第 ${i + 1}/${images.length} 张图片: ${title}`);

      // 确保URL是完整的
      let fullImageUrl = ensureFullUrl(imageUrl, targetUrl);

      // 添加到下载任务列表
      downloadTasks.push({
        url: fullImageUrl,
        filename: title,
        index: i + 1
      });
    }

    // 并发下载控制（限制并发数为5，不阻塞式）
    const concurrencyLimit = 5;
    const results = [];

    // 创建一个并发控制器
    async function processWithConcurrency(tasks, limit) {
      const results = [];
      const executing = new Set();

      for (const task of tasks) {
        // 如果当前并发数达到限制，等待其中一个完成
        if (executing.size >= limit) {
          await Promise.race(executing);
        }

        // 创建下载任务
        const promise = (async () => {
          // console.log(`开始下载第 ${task.index}/${images.length} 张图片: ${task.filename}`);
          const success = await downloadImage(task.url, task.filename, dirName, firstCustomDirName);
          executing.delete(promise);
          return {
            success,
            index: task.index
          };
        })();

        executing.add(promise);
        results.push(promise);
      }

      return Promise.all(results);
    }

    // 执行并发下载
    const downloadResults = await processWithConcurrency(downloadTasks, concurrencyLimit);
    results.push(...downloadResults);

    // 统计成功数量
    successCount = results.filter(result => result.success).length;

    console.log(`\n下载完成! 成功: ${successCount}/${images.length}`);

  } catch (error) {
    console.error('爬取失败:', error.message);
  }
}

/**
 * 分页下载主函数 
 */
export async function pageBasedDownload(listPageUrl, firstCustomDirName = null) {
  try {
    console.log(`开始分页爬取: ${listPageUrl}`);

    let currentPageUrl = listPageUrl;
    let pageCount = 1;
    let totalDownloadedPages = 0;

    // 获取第一页内容以设置初始pageCount
    const firstResponse = await getPageWithRetry(currentPageUrl, 5, firstCustomDirName, false);
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
      const response = await getPageWithRetry(currentPageUrl, 5, firstCustomDirName, false);
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
        await new Promise(resolve => setTimeout(resolve, 500));
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
