import os
import yaml
import datetime
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src import hf_scraper, utils, llm_agent, zotero_ops

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

ZOTERO_STRUCTURE = {}

def process_single_paper(args):
    arxiv_id, meta, local_dir, target_date = args
    
    # 1. RAG 分析
    try:
        analysis = llm_agent.analyze_paper_with_structure(
            meta['title'], meta['summary'], ZOTERO_STRUCTURE
        )
    except Exception as e:
        return None

    # 【修改点】如果不感兴趣，返回一个特殊的标记对象，而不是 None
    if not analysis.get('interested'):
        return {
            "status": "ignored",
            "title": meta['title'],
            "reason": analysis.get('reason', 'No reason'),
            "url": meta['pdf_url']
        }

    # === 下面是感兴趣的处理逻辑 (保持不变) ===
    category = analysis.get('category', 'Uncategorized')
    category_dir = os.path.join(local_dir, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir, exist_ok=True)
        
    short_title = utils.sanitize_filename(meta['title'])[:40]
    first_author = utils.sanitize_filename(meta['authors'][0])
    filename = f"{first_author}_{short_title}.pdf"
    pdf_path = os.path.join(category_dir, filename)
    
    utils.download_pdf(arxiv_id, pdf_path)
    
    note_md = llm_agent.generate_reading_note(
        meta['title'], ", ".join(meta['authors']), meta['summary'], analysis
    )
    
    note_path = pdf_path.replace('.pdf', '.md')
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write(note_md)
        
    tags = analysis.get('tags', [])
    tags.append(f"Date:{target_date}")
    
    zotero_ops.upload_paper_linked(meta, pdf_path, note_md, tags, category)

    return {
        "status": "interested",
        "title": meta['title'],
        "url": meta['pdf_url'],
        "category": category,
        "summary": analysis.get('summary_cn', '无总结'),
        "tricks": analysis.get('tricks_cn', '无'),
        "reason": analysis.get('reason', ''), # 推荐理由
        "local_path": pdf_path,
        "note_path": note_path
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='YYYY-MM-DD', default=None)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()

    target_date = args.date if args.date else datetime.datetime.now().strftime('%Y-%m-%d')
    base_dir = config['local_storage']['base_dir']
    local_dir = os.path.join(base_dir, target_date)
    
    print(f"📅 日期: {target_date} | 本地目录: {local_dir}")
    
    global ZOTERO_STRUCTURE
    ZOTERO_STRUCTURE = zotero_ops.get_existing_structure()
    
    arxiv_ids = hf_scraper.get_daily_papers(target_date)
    if not arxiv_ids:
        print("今天没有新论文。")
        return

    print(f"🔍 抓取到 {len(arxiv_ids)} 篇，获取元数据...")
    papers_meta = utils.get_arxiv_metadata(arxiv_ids)
    
    print("🚀 开始 AI 处理...")
    
    tasks = []
    for arxiv_id, meta in papers_meta.items():
        tasks.append((arxiv_id, meta, local_dir, target_date))

    # 收集结果
    interested_results = []
    ignored_results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_paper = {executor.submit(process_single_paper, task): task for task in tasks}
        for future in tqdm(as_completed(future_to_paper), total=len(tasks)):
            try:
                res = future.result()
                if res:
                    if res['status'] == 'interested':
                        interested_results.append(res)
                    else:
                        ignored_results.append(res)
            except Exception as exc:
                print(f"异常: {exc}")

    # 生成新版日报
    generate_daily_report(interested_results, ignored_results, target_date, local_dir)

def generate_daily_report(interested, ignored, date, local_dir):
    print("\n📝 正在生成深度日报...")
    
    # 1. 生成宏观综述 (调用 LLM)
    overview = llm_agent.generate_daily_overview(interested, ignored)

    md = [f"# 📅 AI 科研情报 - {date}", ""]
    
    # 2. 写入综述
    md.append("## 1. 今日概览 (Executive Summary)")
    md.append(overview)
    md.append("")

    # 3. 写入感兴趣的论文 (详细)
    md.append(f"## 2. 核心精读 ({len(interested)} 篇)")
    
    # 按分类聚合
    grouped = {}
    for r in interested:
        cat = r['category']
        if cat not in grouped: grouped[cat] = []
        grouped[cat].append(r)

    for cat, items in grouped.items():
        md.append(f"### 📂 {cat}")
        for item in items:
            rel_pdf = os.path.relpath(item['local_path'], local_dir)
            rel_note = os.path.relpath(item['note_path'], local_dir)
            
            md.append(f"#### 📄 [{item['title']}]({item['url']})")
            md.append(f"> **推荐理由**: {item['reason']}")
            md.append(f"- **核心贡献**: {item['summary']}") # 这里复用summary，因为我们在analysis里已经要求summary是一句话总结了
            md.append(f"- **关键结论/Tricks**: {item['tricks']}")
            md.append(f"- 🔗 [本地PDF]({rel_pdf}) | 📝 [深度笔记]({rel_note})")
            md.append("")
        md.append("---")

    # 4. 写入不感兴趣的论文 (列表)
    if ignored:
        md.append(f"## 3. 其他论文一览 ({len(ignored)} 篇)")
        md.append("| 标题 | 过滤原因 |")
        md.append("|---|---|")
        for item in ignored:
            # 表格里标题太长可以截断
            short_title = item['title'][:80] + "..." if len(item['title']) > 80 else item['title']
            md.append(f"| [{short_title}]({item['url']}) | {item['reason']} |")

    report_path = os.path.join(local_dir, "00_Daily_Report_CN.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
    
    print(f"✅ 日报生成完毕: {report_path}")

if __name__ == "__main__":
    main()
