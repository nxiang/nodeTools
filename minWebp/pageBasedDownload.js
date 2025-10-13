import {
  pageBasedDownload
} from './utils.js';

/**
 * 主函数 - 处理命令行参数
 */
async function main() {
  const listPageUrl = process.argv[2];

  // 获取自定义目录名参数 - 从listPageUrl以反斜杠分隔的最后一个全为数字的字符串
  // let firstCustomDirName = 'downloaded_images';
  let firstCustomDirName = '\\\\DXP4800PLUS-BE5\\personal_folder\\视频\\成人内容\\写真';
  if (listPageUrl) {
    const urlParts = listPageUrl.split('/');
    // 从后往前查找第一个不全为数字的部分
    for (let i = urlParts.length - 1; i >= 0; i--) {
      const part = decodeURIComponent(urlParts[i].trim());
      if (part && !/^\d+$/.test(part)) {
        firstCustomDirName += `\\[${part}]`;
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
