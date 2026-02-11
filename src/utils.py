import os
import re
import requests
import arxiv
import time

def sanitize_filename(filename):
    """清理文件名中的非法字符"""
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip().replace(' ', '_')

def download_pdf(arxiv_id, save_path):
    """通过 arxiv ID 下载 PDF"""
    # 如果文件已存在且大小正常(>10KB)，跳过下载
    if os.path.exists(save_path) and os.path.getsize(save_path) > 10240:
        return True

    try:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        # 增加 headers 伪装
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, headers=headers, stream=True, timeout=60)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            print(f"下载失败 {arxiv_id} Status: {response.status_code}")
    except Exception as e:
        print(f"下载异常 {arxiv_id}: {e}")
    return False

def get_arxiv_metadata(arxiv_ids, chunk_size=10, delay=3):
    """批量获取 arxiv 元数据，分批处理防止 URL 过长和限流"""
    results = {}
    
    # 将列表切片，每次查 chunk_size 个（减小批次大小）
    for i in range(0, len(arxiv_ids), chunk_size):
        chunk = arxiv_ids[i:i + chunk_size]
        print(f"正在获取 Arxiv 元数据 ({i+1}-{min(i+chunk_size, len(arxiv_ids))}/{len(arxiv_ids)})...")
        
        max_retries = 5  # 增加重试次数
        for attempt in range(max_retries):
            try:
                # 第一批之前先等待，避免初始请求过快
                if i == 0 and attempt == 0:
                    time.sleep(5)
                elif i > 0 or attempt > 0:
                    # 递增延迟：基础延迟 + 重试次数 * 5秒
                    wait_time = delay + (attempt * 5)
                    print(f"  等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                
                search = arxiv.Search(id_list=chunk)
                for r in search.results():
                    # 获取 ID 的纯数字部分 (去除 v1, v2)
                    clean_id = r.get_short_id().split('v')[0]
                    results[clean_id] = {
                        'title': r.title,
                        'authors': [a.name for a in r.authors],
                        'summary': r.summary.replace('\n', ' '), # 摘要去换行
                        'published': r.published,
                        'pdf_url': r.pdf_url,
                        'arxiv_id': clean_id
                    }
                print(f"  ✓ 批次成功，获取 {len(chunk)} 篇")
                break  # 成功则跳出重试循环
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg and attempt < max_retries - 1:
                    wait_time = (attempt + 2) * 10  # 更长的递增等待：10, 20, 30, 40秒
                    print(f"  ⚠️ 遇到限流(429)，等待 {wait_time} 秒后重试 ({attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ 获取元数据批次失败: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        break
    
    print(f"成功获取 {len(results)}/{len(arxiv_ids)} 篇论文元数据")
    return results
