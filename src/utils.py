import os
import re
import requests
import arxiv
import time

def sanitize_filename(filename):
    """清理文件名中的非法字符"""
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip().replace(' ', '_')

def download_pdf(arxiv_id, save_path, max_retries=3, retry_delay=2):
    """通过 arxiv ID 下载 PDF，支持重试机制"""
    # 如果文件已存在且大小正常(>10KB)，跳过下载
    if os.path.exists(save_path) and os.path.getsize(save_path) > 10240:
        return True

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for attempt in range(max_retries):
        try:
            response = requests.get(pdf_url, headers=headers, stream=True, timeout=60)

            if response.status_code == 200:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(8192):  # 增大chunk size
                        if chunk:
                            f.write(chunk)
                # 验证文件是否正确下载
                if os.path.exists(save_path) and os.path.getsize(save_path) > 10240:
                    if attempt > 0:
                        print(f"   ✅ PDF下载成功 (第{attempt + 1}次尝试): {arxiv_id}")
                    return True
                else:
                    print(f"   ⚠️  PDF文件大小异常: {arxiv_id}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    return False
            else:
                print(f"   ⚠️  PDF下载失败 (HTTP {response.status_code}): {arxiv_id} (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return False
        except requests.exceptions.Timeout:
            print(f"   ⚠️  PDF下载超时 ({arxiv_id}): 尝试 {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            return False
        except Exception as e:
            print(f"   ⚠️  PDF下载异常 ({arxiv_id}): {str(e)[:50]} (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            return False

    return False

def get_arxiv_metadata_stream(arxiv_ids, chunk_size=10, delay=3):
    """
    流式获取 arXiv 论文元数据，支持生成器模式
    """
    for i in range(0, len(arxiv_ids), chunk_size):
        chunk = arxiv_ids[i:i + chunk_size]
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # 第一批之前先等待，避免初始请求过快
                if i == 0 and attempt == 0:
                    time.sleep(2)  # 减少初始等待
                elif i > 0 or attempt > 0:
                    wait_time = delay + (attempt * 5)
                    time.sleep(wait_time)
                
                search = arxiv.Search(id_list=chunk)
                batch_results = []
                for r in search.results():
                    clean_id = r.get_short_id().split('v')[0]
                    paper_data = {
                        'title': r.title,
                        'authors': [a.name for a in r.authors],
                        'summary': r.summary.replace('\n', ' '),
                        'published': r.published,
                        'pdf_url': r.pdf_url,
                        'arxiv_id': clean_id
                    }
                    batch_results.append((clean_id, paper_data))
                
                for res in batch_results:
                    yield res
                break
                
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    time.sleep((attempt + 2) * 10)
                else:
                    if attempt >= max_retries - 1:
                        break
                    time.sleep(5)

def get_arxiv_metadata(arxiv_ids, chunk_size=10, delay=3):
    """保持向后兼容的同步版本"""
    results = {}
    for aid, data in get_arxiv_metadata_stream(arxiv_ids, chunk_size, delay):
        results[aid] = data
    return results
