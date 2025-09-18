import axios from 'axios';
import * as cheerio from 'cheerio';
import fs from 'fs';
import path from 'path';
import {
  promisify
} from 'util';
import stream from 'stream';
import https from 'https';
import http from 'http';

const pipeline = promisify(stream.pipeline);

// 创建下载目录（在crawlWebsite函数中动态创建）
let downloadDir;

async function downloadImage(url, filename, directoryName = null) {
  try {
    // 清理文件名
    const cleanFilename = filename.replace(/[<>:"\/\\|?*]/g, '_')
      .replace(/\s+/g, '_')
      .substring(0, 100);

    // 确定目标目录 - 优先使用传入的directoryName参数
    const targetDir = directoryName ? path.join(process.cwd(), 'downloaded_images', directoryName) : downloadDir;
    
    // 先检查文件是否已存在（在发起网络请求前）
    // 由于扩展名未知，检查所有可能的扩展名
    const possibleExtensions = ['jpg', 'jpeg', 'png', 'gif', 'webp'];
    for (const ext of possibleExtensions) {
      const filePath = path.join(targetDir, `${cleanFilename}.${ext}`);
      if (fs.existsSync(filePath)) {
        console.log(`✓ 文件已存在，跳过下载: ${cleanFilename}.${ext}`);
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

    const filePath = path.join(targetDir, `${cleanFilename}.${extension}`);

    // 写入文件
    const writer = fs.createWriteStream(filePath);
    await pipeline(response.data, writer);

    console.log(`✓ 下载成功: ${cleanFilename}.${extension}`);
    return true;

  } catch (error) {
    console.error(`✗ 下载失败 ${filename}:`, error.message);
    return false;
  }
}

async function crawlWebsite(targetUrl, customDirName = null) {
  try {
    console.log(`开始爬取: ${targetUrl}`);

    // 获取网页内容（添加重试机制）
    let response;
    let retries = 5; // 增加重试次数
    
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        response = await axios.get(targetUrl, {
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
        break; // 成功则跳出循环
      } catch (error) {
        if (attempt === retries) {
          throw error; // 最后一次尝试仍然失败
        }
        console.warn(`第 ${attempt} 次请求失败，${retries - attempt} 次重试剩余:`, error.message);
        if (error.response) {
          console.warn(`HTTP状态码: ${error.response.status}`);
          console.warn(`响应头:`, error.response.headers);
        } else if (error.request) {
          console.warn('请求已发送但无响应:', error.request);
        } else {
          console.warn('错误详情:', error);
        }
        await new Promise(resolve => setTimeout(resolve, 5000)); // 增加到5秒后重试
      }
    }

    const $ = cheerio.load(response.data);

    // 获取页面标题作为目录名
    const pageTitle = $('title').text().trim() || 'untitled_page';
    // 移除最后的' - 绅士会所'后缀
    const cleanedTitle = pageTitle.replace(/\s*-\s*绅士会所\s*$/, '');
    const cleanPageTitle = cleanedTitle.replace(/[<>:"\/\\|?*]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    
    // 优先使用传入的自定义目录名，如果没有则使用页面标题
    const dirName = customDirName || cleanPageTitle;
    
    // 创建下载目录
    downloadDir = path.join(process.cwd(), 'downloaded_images', dirName);
    if (!fs.existsSync(downloadDir)) {
      fs.mkdirSync(downloadDir, {
        recursive: true
      });
      console.log(`创建下载目录: ${cleanPageTitle}`);
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

      console.log(`准备下载第 ${i + 1}/${images.length} 张图片: ${title}`);

      // 确保URL是完整的
      let fullImageUrl = imageUrl;
      if (!imageUrl.startsWith('http')) {
        const baseUrl = new URL(targetUrl).origin;
        fullImageUrl = new URL(imageUrl, baseUrl).href;
      }

      // 添加到下载任务列表
      downloadTasks.push({ url: fullImageUrl, filename: title, index: i + 1 });
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
          console.log(`开始下载第 ${task.index}/${images.length} 张图片: ${task.filename}`);
          const success = await downloadImage(task.url, task.filename, dirName);
          executing.delete(promise);
          return { success, index: task.index };
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

// 使用示例
async function main() {
  // 请替换为您的目标网址
  const targetUrl = process.argv[2] || 'https://example.com';
  // 获取自定义目录名参数
  const customDirName = process.argv[3] || null;

  if (!targetUrl || targetUrl === 'https://example.com') {
    console.log('请提供目标网址作为参数:');
    console.log('node imageCrawler.js https://your-target-website.com [custom-directory-name]');
    return;
  }

  await crawlWebsite(targetUrl, customDirName);
}

// 运行程序
main().catch(console.error);

// 导出函数供其他模块使用
export { downloadImage, crawlWebsite };
