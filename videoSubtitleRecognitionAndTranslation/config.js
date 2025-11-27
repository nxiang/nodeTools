// 翻译API配置
// 请将您的API密钥设置到环境变量中，或直接在此文件中配置

const config = {
    // 百度翻译API配置
    baidu: {
        appid: '20251126002506386',
        key: 'C0qK4IqU_KXjun3PhRum',
        apiUrl: 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    },
    // 翻译服务优先级
    translationPriority: ['baidu', 'google'],

    // 请求超时时间（毫秒）
    timeout: 10000,

    // 重试次数
    maxRetries: 3,

    // 重试延迟（毫秒）
    retryDelay: 1000
};

// 检查配置是否有效
export function validateConfig() {
    const errors = [];

    if (!config.baidu.appid || !config.baidu.key) {
        errors.push('百度翻译API配置不完整，请设置BAIDU_TRANSLATE_APPID和BAIDU_TRANSLATE_KEY环境变量');
    }

    if (!config.google.apiKey) {
        errors.push('Google翻译API密钥未配置，请设置GOOGLE_TRANSLATE_API_KEY环境变量');
    }

    return {
        isValid: errors.length === 0,
        errors,
        hasBaidu: !!(config.baidu.appid && config.baidu.key),
        hasGoogle: !!config.google.apiKey
    };
}

export default config;
