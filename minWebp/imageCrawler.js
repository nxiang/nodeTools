import {
  crawlWebsite
} from './utils.js';



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
