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

def get_arxiv_metadata(arxiv_ids, chunk_size=20):
    """批量获取 arxiv 元数据，分批处理防止 URL 过长"""
    results = {}
    
    # 将列表切片，每次查 chunk_size 个
    for i in range(0, len(arxiv_ids), chunk_size):
        chunk = arxiv_ids[i:i + chunk_size]
        print(f"正在获取 Arxiv 元数据 ({i+1}/{len(arxiv_ids)})...")
        
        try:
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
            # 礼貌性延时，防止请求过快
            time.sleep(1) 
            
        except Exception as e:
            print(f"获取元数据批次失败: {e}")
            
    return results
