import os
import yaml
import datetime
import argparse
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from src import hf_scraper, utils, llm_agent, zotero_ops, paper_analyzer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import box

console = Console()

# 全局变量
ZOTERO_STRUCTURE = {}

async def process_paper_async(arxiv_id, meta, local_dir, target_date, skip_deep, progress, task_id, semaphores):
    """异步处理单篇论文的流水线"""
    filter_sem, download_sem, ocr_sem, llm_sem = semaphores
    
    try:
        # 1. RAG 分析 (筛选)
        progress.update(task_id, description=f"[cyan]🔍 筛选中: {meta['title'][:30]}...")
        async with filter_sem:
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(
                None, 
                llm_agent.analyze_paper_with_structure,
                meta['title'], meta['summary'], ZOTERO_STRUCTURE
            )
        
        if not analysis.get('interested'):
            progress.update(task_id, description=f"[grey50]⏭️  已跳过: {meta['title'][:30]}", completed=100)
            return {
                "status": "ignored",
                "title": meta['title'],
                "reason": analysis.get('reason', '不感兴趣'),
                "url": meta['pdf_url']
            }

        # 2. 准备目录
        category = analysis.get('category', 'Uncategorized')
        category_dir = os.path.join(local_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        short_title = utils.sanitize_filename(meta['title'])[:40]
        first_author = utils.sanitize_filename(meta['authors'][0])
        paper_subdir_name = f"{first_author}_{short_title}"
        paper_dir = os.path.join(category_dir, paper_subdir_name)
        os.makedirs(paper_dir, exist_ok=True)
        
        filename = f"{first_author}_{short_title}.pdf"
        pdf_path = os.path.join(paper_dir, filename)

        # 3. 下载 PDF
        progress.update(task_id, description=f"[blue]📥 下载中: {meta['title'][:30]}...", advance=20)
        async with download_sem:
            download_success = await loop.run_in_executor(None, utils.download_pdf, arxiv_id, pdf_path)

        if not download_success:
            print(f"   ❌ PDF下载失败: {meta['title'][:40]}...")
            progress.update(task_id, description=f"[red]❌ 下载失败: {meta['title'][:30]}", completed=100)
            return {
                "status": "failed",
                "title": meta['title'],
                "reason": "PDF下载失败",
                "url": meta['pdf_url']
            }

        # 4. 深度分析 (OCR + LLM) - 使用独立的信号量，实现真正流水线
        deep_analysis_result = None
        if not skip_deep:
            paper_info = {
                'title': meta['title'],
                'authors': meta['authors'],
                'date': target_date,
                'arxiv_id': arxiv_id
            }

            # 4.1 OCR阶段 - 独立信号量，带重试机制
            progress.update(task_id, description=f"[magenta]📄 OCR识别: {meta['title'][:30]}...", advance=15)
            ocr_result = None
            max_ocr_retries = 2

            for ocr_attempt in range(max_ocr_retries):
                async with ocr_sem:
                    ocr_result = await loop.run_in_executor(
                        None,
                        paper_analyzer.extract_ocr_only,
                        pdf_path, paper_info, category_dir
                    )

                if ocr_result is not None:
                    if ocr_attempt > 0:
                        print(f"   ✅ OCR成功 (第{ocr_attempt + 1}次尝试): {meta['title'][:40]}...")
                    break
                else:
                    print(f"   ⚠️  OCR失败 (尝试 {ocr_attempt + 1}/{max_ocr_retries}): {meta['title'][:40]}...")
                    if ocr_attempt < max_ocr_retries - 1:
                        await asyncio.sleep(2 * (ocr_attempt + 1))  # 递增延迟

            if ocr_result is None:
                print(f"   ❌ OCR阶段最终失败，跳过LLM分析: {meta['title'][:40]}...")
                deep_analysis_result = None
            else:
                # 4.2 LLM分析阶段 - 独立信号量，带重试机制
                progress.update(task_id, description=f"[magenta]🧠 LLM分析: {meta['title'][:30]}...", advance=15)
                deep_analysis_result = None
                max_llm_retries = 2

                for llm_attempt in range(max_llm_retries):
                    async with llm_sem:
                        deep_analysis_result = await loop.run_in_executor(
                            None,
                            paper_analyzer.analyze_with_llm,
                            ocr_result, paper_info, category_dir
                        )

                    if deep_analysis_result is not None:
                        if llm_attempt > 0:
                            print(f"   ✅ LLM分析成功 (第{llm_attempt + 1}次尝试): {meta['title'][:40]}...")
                        break
                    else:
                        print(f"   ⚠️  LLM分析失败 (尝试 {llm_attempt + 1}/{max_llm_retries}): {meta['title'][:40]}...")
                        if llm_attempt < max_llm_retries - 1:
                            await asyncio.sleep(2 * (llm_attempt + 1))
        
        # 5. 上传 Zotero & 笔记
        progress.update(task_id, description=f"[green]📤 上传中: {meta['title'][:30]}...", advance=30)
        tags = analysis.get('tags', [])
        tags.append(f"Date:{target_date}")
        
        note_content = ""
        if deep_analysis_result and deep_analysis_result.get('note_path'):
            try:
                with open(deep_analysis_result['note_path'], 'r', encoding='utf-8') as f:
                    note_content = f.read()
            except: pass
        
        if not note_content:
            note_content = await loop.run_in_executor(
                None,
                llm_agent.generate_reading_note,
                meta['title'], ", ".join(meta['authors']), meta['summary'], analysis
            )
        
        await loop.run_in_executor(
            None,
            zotero_ops.upload_paper_linked,
            meta, pdf_path, note_content, tags, category
        )
        
        progress.update(task_id, description=f"[bold green]✅ 已完成: {meta['title'][:30]}", completed=100)
        return {
            "status": "interested",
            "title": meta['title'],
            "url": meta['pdf_url'],
            "category": category,
            "summary": analysis.get('summary_cn', '无总结'),
            "tricks": analysis.get('tricks_cn', '无'),
            "reason": analysis.get('reason', ''),
            "local_path": pdf_path,
            "deep_analysis": deep_analysis_result
        }

    except Exception as e:
        error_msg = str(e)
        progress.update(task_id, description=f"[red]❌ 失败: {meta['title'][:30]}")
        print(f"\n❌ 论文处理失败: {meta['title']}")
        print(f"   错误信息: {error_msg}")
        import traceback
        print(f"   堆栈跟踪:\n{traceback.format_exc()}")
        return None

def generate_daily_report(interested, ignored, date, local_dir):
    """生成汇总式日报 - 基于LLM的分批次汇总"""
    print("\n📝 正在生成汇总日报...")
    os.makedirs(local_dir, exist_ok=True)

    if not interested:
        md = [f"# AI 科研情报 - {date}", "", "今日无感兴趣论文。"]
        report_path = os.path.join(local_dir, "00_Daily_Report_CN.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md))
        print(f"✅ 日报生成完毕: {report_path}")
        return

    # 1. 收集所有论文的note内容
    print(f"   📚 正在读取 {len(interested)} 篇论文的详细笔记...")
    papers_notes = []
    for item in interested:
        if not item:
            continue
        note_path = item.get('deep_analysis', {}).get('note_path', '') if item.get('deep_analysis') else ''
        if note_path and os.path.exists(note_path):
            try:
                with open(note_path, 'r', encoding='utf-8') as f:
                    note_content = f.read()
                # 解析note内容
                parsed = llm_agent.parse_note_content(note_content)
                parsed['category'] = item.get('category', 'Uncategorized')
                parsed['url'] = item.get('url', '')
                parsed['note_path'] = note_path
                papers_notes.append(parsed)
            except Exception as e:
                print(f"   ⚠️  读取笔记失败: {note_path}, 错误: {e}")
                continue
        else:
            # 如果没有详细笔记，使用基础信息
            deep = item.get('deep_analysis', {})
            analysis = deep.get('analysis', {}) if deep else {}
            papers_notes.append({
                'title': item.get('title', ''),
                'title_cn': analysis.get('title_cn', ''),
                'authors': analysis.get('authors', []),
                'category': item.get('category', 'Uncategorized'),
                'core_problem': analysis.get('core_problem', ''),
                'core_contribution': analysis.get('core_contribution', []),
                'method_summary': analysis.get('method_summary', ''),
                'key_results': analysis.get('key_results', ''),
                'pros': analysis.get('pros', []),
                'url': item.get('url', ''),
                'note_path': note_path
            })

    if not papers_notes:
        print("   ⚠️  没有可用的论文笔记")
        return

    print(f"   ✅ 成功读取 {len(papers_notes)} 篇论文笔记")

    # 2. 分批次进行小汇总（每10篇一批）
    batch_size = 10
    batches = [papers_notes[i:i+batch_size] for i in range(0, len(papers_notes), batch_size)]
    print(f"   🔄 分为 {len(batches)} 个批次进行汇总...")

    batch_summaries = []
    for batch_idx, batch in enumerate(batches):
        print(f"      📦 批次 {batch_idx + 1}/{len(batches)}: {len(batch)} 篇论文")
        summary = llm_agent.summarize_papers_batch(batch, batch_idx)
        batch_summaries.append(summary)

    # 3. 基于小汇总生成最终日报
    print(f"   🧠 正在生成最终日报...")
    final_report = llm_agent.generate_final_daily_report(batch_summaries, len(papers_notes), date)

    # 4. 构建Markdown日报
    md = [f"# AI 科研情报 - {date}", ""]

    # 4.1 今日概览
    md.append("## 1. 今日概览")
    md.append(final_report.get('daily_overview', '今日概览生成失败'))
    md.append("")

    # 4.2 核心洞察
    key_insights = final_report.get('key_insights', [])
    if key_insights:
        md.append("## 2. 核心洞察")
        md.append("")
        for idx, insight in enumerate(key_insights, 1):
            md.append(f"{idx}. {insight}")
        md.append("")

    # 4.3 方向小结
    direction_summary = final_report.get('direction_summary', {})
    if direction_summary:
        md.append("## 3. 方向小结")
        md.append("")
        for direction, summary in direction_summary.items():
            if summary:
                md.append(f"### {direction}")
                md.append(summary)
                md.append("")

    # 4.4 各批次详细汇总
    md.append("## 4. 论文详细汇总")
    md.append("")

    for batch_idx, (batch, summary) in enumerate(zip(batches, batch_summaries), 1):
        md.append(f"### 4.{batch_idx} 批次 {batch_idx} ({len(batch)} 篇)")
        md.append("")

        # 批次整体趋势
        batch_summary_text = summary.get('batch_summary', '')
        if batch_summary_text:
            md.append(f"**整体趋势**: {batch_summary_text}")
            md.append("")

        # 技术趋势
        tech_trends = summary.get('technical_trends', [])
        if tech_trends:
            md.append(f"**技术趋势**: {', '.join(tech_trends)}")
            md.append("")

        # 每篇论文要点
        papers_highlights = summary.get('papers_highlights', [])
        md.append("**论文要点**:")
        md.append("")

        for paper_idx, (paper, highlight) in enumerate(zip(batch, papers_highlights), 1):
            md.append(f"{paper_idx}. **{paper.get('title', '')}**")
            if paper.get('title_cn'):
                md.append(f"   - 中文标题: {paper['title_cn']}")

            # 使用LLM提取的要点
            if highlight:
                md.append(f"   - 方法: {highlight.get('key_method', '')}")
                md.append(f"   - 发现: {highlight.get('key_finding', '')}")
                if highlight.get('result_highlight'):
                    md.append(f"   - 结果: {highlight.get('result_highlight')}")
            else:
                # 备用：使用解析的内容
                if paper.get('method_summary'):
                    method = paper['method_summary'][:100] + "..." if len(paper['method_summary']) > 100 else paper['method_summary']
                    md.append(f"   - 方法: {method}")

            # 链接
            if paper.get('note_path') and os.path.exists(paper['note_path']):
                rel_note = os.path.relpath(paper['note_path'], local_dir)
                md.append(f"   - 📄 [详细笔记]({rel_note}) | 🔗 [arXiv原文]({paper['url']})")

            md.append("")

        md.append("---")
        md.append("")

    # 4.5 值得关注论文
    notable_papers = final_report.get('notable_papers', [])
    if notable_papers:
        md.append("## 5. 值得关注论文")
        md.append("")
        for idx, paper in enumerate(notable_papers, 1):
            md.append(f"{idx}. **{paper.get('title', '')}**")
            md.append(f"   - {paper.get('why_notable', '')}")
            md.append("")

    # 4.6 未来趋势
    future_trends = final_report.get('future_trends', '')
    if future_trends:
        md.append("## 6. 未来趋势展望")
        md.append(future_trends)
        md.append("")

    # 4.7 忽略的论文
    if ignored:
        md.append(f"## 7. 其他论文 ({len(ignored)} 篇)")
        md.append("")
        md.append("以下论文因不符合当前研究方向被过滤：")
        md.append("")
        md.append("| 序号 | 标题 | 过滤原因 |")
        md.append("|:---:|:---|:---|")
        for idx, item in enumerate(ignored[:15], 1):
            short_title = item['title'][:55] + "..." if len(item['title']) > 55 else item['title']
            reason = item.get('reason', '未知原因')
            if len(reason) > 50:
                reason = reason[:50] + "..."
            md.append(f"| {idx} | [{short_title}]({item['url']}) | {reason} |")
        md.append("")

    # 5. 保存日报
    report_path = os.path.join(local_dir, "00_Daily_Report_CN.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
    print(f"✅ 日报生成完毕: {report_path}")

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='YYYY-MM-DD', default=None)
    parser.add_argument('--skip-deep-analysis', action='store_true', help='跳过深度分析')
    args = parser.parse_args()

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    target_date = args.date if args.date else datetime.datetime.now().strftime('%Y-%m-%d')
    base_dir = config['local_storage']['base_dir']
    local_dir = os.path.join(base_dir, target_date)
    
    concurrency_config = config.get('concurrency', {})
    # 增加并发度 - 各阶段独立控制
    filter_limit = concurrency_config.get('paper_workers', 4)
    download_limit = 10  # 下载可以更多
    ocr_limit = concurrency_config.get('ocr_workers', 4)  # OCR并发
    llm_limit = concurrency_config.get('llm_workers', 4)  # LLM分析并发

    print(f"📅 日期: {target_date} | 本地目录: {local_dir}")
    print(f"⚙️  并发配置: 筛选={filter_limit}, 下载={download_limit}, OCR={ocr_limit}, LLM={llm_limit}")

    global ZOTERO_STRUCTURE
    ZOTERO_STRUCTURE = zotero_ops.get_existing_structure()

    arxiv_ids = hf_scraper.get_daily_papers(target_date)
    if not arxiv_ids:
        print("今天没有新论文。")
        return

    print(f"🔍 抓取到 {len(arxiv_ids)} 篇，开始异步流水线处理...")

    # 信号量控制 - 4个独立信号量
    semaphores = (
        asyncio.Semaphore(filter_limit),   # 0: 筛选
        asyncio.Semaphore(download_limit), # 1: 下载
        asyncio.Semaphore(ocr_limit),      # 2: OCR
        asyncio.Semaphore(llm_limit)       # 3: LLM分析
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True
    ) as progress:
        
        # 预先创建所有任务占位
        tasks_map = {}
        for aid in arxiv_ids:
            tid = progress.add_task(f"[grey50]等待中: {aid}", total=100)
            tasks_map[aid] = tid
            
        # 流式获取元数据并启动处理
        async_tasks = []
        meta_queue = asyncio.Queue()
        
        def fetch_meta():
            try:
                for aid, meta in utils.get_arxiv_metadata_stream(arxiv_ids):
                    asyncio.run_coroutine_threadsafe(meta_queue.put((aid, meta)), loop)
            finally:
                # 放入结束标记
                asyncio.run_coroutine_threadsafe(meta_queue.put((None, None)), loop)
        
        loop = asyncio.get_running_loop()
        threading.Thread(target=fetch_meta, daemon=True).start()
        
        while True:
            aid, meta = await meta_queue.get()
            if aid is None: break
            
            tid = tasks_map[aid]
            t = asyncio.create_task(process_paper_async(
                aid, meta, local_dir, target_date, args.skip_deep_analysis, progress, tid, semaphores
            ))
            async_tasks.append(t)
            
        if async_tasks:
            results = await asyncio.gather(*async_tasks)
        else:
            results = []
        
    interested = [r for r in results if r and r.get('status') == 'interested']
    ignored = [r for r in results if r and r.get('status') == 'ignored']
    
    generate_daily_report(interested, ignored, target_date, local_dir)

if __name__ == "__main__":
    asyncio.run(main_async())
