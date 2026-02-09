import requests
from bs4 import BeautifulSoup
import datetime
import re
import os
import urllib3

# 禁用 SSL 警告（某些代理环境需要）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_proxies_from_env():
    """
    从环境变量自动检测代理设置
    支持 HTTP_PROXY/http_proxy 和 HTTPS_PROXY/https_proxy
    """
    proxies = {}
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    
    if http_proxy:
        proxies['http'] = http_proxy
    if https_proxy:
        proxies['https'] = https_proxy
    
    return proxies

def get_daily_papers(date_str=None):
    """
    获取 HuggingFace Daily Papers 的 Arxiv ID 列表
    date_str: '2024-05-20'，默认当天
    """
    if not date_str:
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    
    url = f"https://huggingface.co/papers?date={date_str}"
    print(f"正在抓取: {url}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
    # 从环境变量获取代理（如果有设置）
    proxies = get_proxies_from_env()
    if proxies:
        print(f"检测到代理: {proxies}")
    
    # 请求配置
    req_kwargs = {
        'headers': headers,
        'timeout': 30,
    }
    
    # 只有在检测到代理时才使用代理
    if proxies:
        req_kwargs['proxies'] = proxies
        # 某些代理环境需要跳过 SSL 验证
        req_kwargs['verify'] = False
    
    try:
        resp = requests.get(url, **req_kwargs)
    except Exception as e:
        print(f"请求失败: {e}")
        return []
    
    if resp.status_code != 200:
        print(f"无法访问 HuggingFace Papers (状态码: {resp.status_code})")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    
    papers = set()
    
    # 查找所有的文章链接
    for link in soup.find_all('a', href=True):
        href = link['href']
        
        # 链接必须以 /papers/ 开头
        if href.startswith('/papers/') and not 'submit' in href:
            # 提取最后一部分，例如 2405.12345#community
            raw_id = href.split('/')[-1]
            
            # 1. 清理：去掉 ? 后面的参数和 # 后面的锚点
            clean_id = raw_id.split('?')[0].split('#')[0]
            
            # 2. 校验：Arxiv ID 格式通常是 4位数字.4或5位数字 (例如 2405.12345)
            # 这一步会过滤掉日期链接 (2026-02-05)
            if re.match(r'^\d{4}\.\d{4,5}$', clean_id):
                papers.add(clean_id)
    
    # 转换为列表
    valid_papers = list(papers)
    print(f"解析到 {len(valid_papers)} 个有效 Arxiv ID")
    return valid_papers
