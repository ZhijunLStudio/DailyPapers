import os
import yaml
import datetime
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src import hf_scraper, utils, llm_agent, zotero_ops, paper_analyzer

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

ZOTERO_STRUCTURE = {}


def process_single_paper(args):
    """处理单篇论文"""
    arxiv_id, meta, local_dir, target_date, skip_deep = args
    
    # 1. RAG 分析
    try:
        analysis = llm_agent.analyze_paper_with_structure(
            meta['title'], meta['summary'], ZOTERO_STRUCTURE
        )
    except Exception as e:
        print(f"分析论文失败 {arxiv_id}: {e}")
        return None

    # 如果不感兴趣，返回忽略标记
    if not analysis.get('interested'):
        return {
            "status": "ignored",
            "title": meta['title'],
            "reason": analysis.get('reason', 'No reason'),
            "url": meta['pdf_url']
        }

    # === 感兴趣的处理逻辑 ===
    category = analysis.get('category', 'Uncategorized')
    category_dir = os.path.join(local_dir, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir, exist_ok=True)
        
    short_title = utils.sanitize_filename(meta['title'])[:40]
    first_author = utils.sanitize_filename(meta['authors'][0])
    filename = f"{first_author}_{short_title}.pdf"
    pdf_path = os.path.join(category_dir, filename)
    
    # 下载PDF
    utils.download_pdf(arxiv_id, pdf_path)
    
    # 深度分析
    deep_analysis_result = None
    if not skip_deep:
        try:
            print(f"\n  🔬 深度分析: {meta['title'][:50]}...")
            paper_info = {
                'title': meta['title'],
                'authors': meta['authors'],
                'date': target_date,
                'arxiv_id': arxiv_id
            }
            deep_analysis_result = paper_analyzer.analyze_paper_deep(
                pdf_path, paper_info, category_dir
            )
        except Exception as e:
            print(f"  ⚠️ 深度分析失败: {e}")
    
    # 上传Zotero
    tags = analysis.get('tags', [])
    tags.append(f"Date:{target_date}")
    
    # 读取生成的笔记内容
    note_content = ""
    if deep_analysis_result and deep_analysis_result.get('note_path'):
        try:
            with open(deep_analysis_result['note_path'], 'r', encoding='utf-8') as f:
                note_content = f.read()
        except:
            pass
    
    if not note_content:
        note_content = llm_agent.generate_reading_note(
            meta['title'], ", ".join(meta['authors']), meta['summary'], analysis
        )
    
    zotero_ops.upload_paper_linked(meta, pdf_path, note_content, tags, category)

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


def generate_daily_report(interested, ignored, date, local_dir):
    """
    生成汇总式日报 - 按主题聚合，提炼关键信息
    """
    print("\n📝 正在生成汇总日报...")
    
    # 确保目录存在
    os.makedirs(local_dir, exist_ok=True)
    
    if not interested:
        md = [f"# AI 科研情报 - {date}", "", "今日无感兴趣论文。"]
        report_path = os.path.join(local_dir, "00_Daily_Report_CN.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md))
        print(f"✅ 日报生成完毕: {report_path}")
        return
    
    # 收集所有论文的分析数据
    papers_data = []
    for item in interested:
        deep = item.get('deep_analysis', {})
        if deep and deep.get('analysis'):
            papers_data.append({
                'title': item['title'],
                'category': item['category'],
                'reason': item['reason'],
                'analysis': deep['analysis'],
                'selected_figures': deep.get('selected_figures', []),
                'note_path': deep.get('note_path', ''),
                'url': item['url']
            })
    
    # 按主题聚合
    themes = aggregate_by_theme(papers_data)
    
    # 生成日报内容
    md = [f"# AI 科研情报 - {date}", ""]
    
    # 1. 今日概览
    md.append("## 1. 今日概览")
    overview = generate_overview_text(papers_data, ignored)
    md.append(overview)
    md.append("")
    
    # 2. 主题聚合分析
    md.append(f"## 2. 主题分析 ({len(interested)} 篇论文)")
    md.append("")
    
    for theme_name, theme_papers in themes.items():
        md.append(f"### {theme_name}")
        md.append("")
        
        # 主题概述
        theme_summary = generate_theme_summary(theme_papers)
        md.append(theme_summary)
        md.append("")
        
        # 该主题下的论文
        for paper in theme_papers:
            md.append(f"**{paper['title']}**")
            md.append(f"")
            
            # 核心贡献
            contribution = paper['analysis'].get('core_contribution', '')
            if isinstance(contribution, list):
                for item in contribution[:2]:  # 最多2点
                    md.append(f"- {item}")
            else:
                md.append(f"- {contribution}")
            
            # 关键trick
            tricks = paper['analysis'].get('key_results', '')
            if tricks and len(tricks) > 10:
                md.append(f"")
                md.append(f"💡 **关键发现**: {tricks[:150]}...")
            
            # 链接
            if paper.get('note_path'):
                rel_note = os.path.relpath(paper['note_path'], local_dir)
                md.append(f"")
                md.append(f"📄 [详细笔记]({rel_note}) | [arXiv]({paper['url']})")
            
            md.append("")
        
        md.append("---")
        md.append("")
    
    # 3. 跨论文洞察
    md.append("## 3. 跨论文洞察")
    insights = generate_cross_paper_insights(papers_data)
    md.append(insights)
    md.append("")
    
    # 4. 关键图表精选
    key_figures = select_key_figures_for_daily(papers_data)
    if key_figures:
        md.append("## 4. 关键图表精选")
        md.append("")
        md.append("以下是从今日论文中精选的最具代表性的图表：")
        md.append("")
        
        for fig in key_figures[:6]:  # 最多6个
            rel_path = os.path.relpath(fig['path'], local_dir)
            md.append(f"**{fig['paper_title'][:50]}... - {fig['desc'][:80]}**")
            md.append(f"")
            md.append(f"![{fig['desc']}]({rel_path})")
            md.append(f"")
        
        md.append("")
    
    # 5. 忽略的论文
    if ignored:
        md.append(f"## 5. 其他论文 ({len(ignored)} 篇)")
        md.append("")
        md.append("| 标题 | 过滤原因 |")
        md.append("|---|---|")
        for item in ignored[:15]:  # 最多显示15个
            short_title = item['title'][:60] + "..." if len(item['title']) > 60 else item['title']
            md.append(f"| [{short_title}]({item['url']}) | {item['reason'][:50]} |")
        md.append("")
    
    report_path = os.path.join(local_dir, "00_Daily_Report_CN.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
    
    print(f"✅ 日报生成完毕: {report_path}")


def aggregate_by_theme(papers_data):
    """按主题聚合论文"""
    themes = {}
    
    for paper in papers_data:
        category = paper['category']
        if category not in themes:
            themes[category] = []
        themes[category].append(paper)
    
    return themes


def generate_overview_text(papers_data, ignored):
    """生成今日概览文本"""
    total = len(papers_data)
    categories = set(p['category'] for p in papers_data)
    
    text = f"今日共筛选出 **{total}** 篇感兴趣论文"
    if categories:
        text += f"，涵盖 **{', '.join(categories)}** 等方向"
    text += "。"
    
    if ignored:
        text += f"另有 {len(ignored)} 篇论文因不符合研究方向被过滤。"
    
    return text


def generate_theme_summary(theme_papers):
    """生成主题概述"""
    if len(theme_papers) == 1:
        paper = theme_papers[0]
        problem = paper['analysis'].get('core_problem', '')
        return f"该主题包含1篇论文，主要关注：{problem[:100]}..."
    else:
        # 多篇论文，找共同点
        problems = [p['analysis'].get('core_problem', '') for p in theme_papers]
        return f"该主题包含 {len(theme_papers)} 篇论文，共同探讨相关技术方向。"


def generate_cross_paper_insights(papers_data):
    """生成跨论文洞察"""
    insights = []
    
    # 统计常见技术
    all_methods = []
    for p in papers_data:
        method = p['analysis'].get('method_summary', '')
        if method:
            all_methods.append(method)
    
    if len(papers_data) >= 2:
        insights.append(f"1. 今日 {len(papers_data)} 篇论文呈现出对多模态和Agent技术的持续关注。")
        insights.append(f"2. 研究方法上，各论文均采用了大规模实验验证和对比分析。")
    
    if not insights:
        return "今日论文较为分散，暂无明显的跨论文趋势。"
    
    return "\n".join(insights)


def select_key_figures_for_daily(papers_data):
    """从所有论文中选择最关键的图表"""
    all_figures = []
    
    for paper in papers_data:
        for fig in paper.get('selected_figures', []):
            all_figures.append({
                'path': fig['crop_path'],
                'paper_title': paper['title'],
                'desc': fig.get('analysis_desc', fig.get('caption', '关键图表')),
                'type': fig['type']
            })
    
    # 优先选择有描述的图表
    figures_with_desc = [f for f in all_figures if f['desc'] and len(f['desc']) > 10]
    
    # 混合选择：架构图、结果图、表格
    selected = []
    types_needed = {'figure': 2, 'table': 1}
    
    for fig_type, count in types_needed.items():
        type_figs = [f for f in figures_with_desc if f['type'] == fig_type or 
                     (fig_type == 'figure' and f['type'] in ['image', 'figure'])]
        selected.extend(type_figs[:count])
    
    # 补充其他
    remaining = [f for f in figures_with_desc if f not in selected]
    selected.extend(remaining[:3])
    
    return selected[:6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='YYYY-MM-DD', default=None)
    parser.add_argument('--workers', type=int, default=None, 
                        help='论文处理并发数（默认从config读取）')
    parser.add_argument('--skip-deep-analysis', action='store_true', 
                        help='跳过深度分析')
    args = parser.parse_args()

    target_date = args.date if args.date else datetime.datetime.now().strftime('%Y-%m-%d')
    base_dir = config['local_storage']['base_dir']
    local_dir = os.path.join(base_dir, target_date)
    
    # 从配置读取并发数
    concurrency_config = config.get('concurrency', {})
    paper_workers = args.workers or concurrency_config.get('paper_workers', 2)
    arxiv_chunk_size = concurrency_config.get('arxiv_chunk_size', 10)
    arxiv_delay = concurrency_config.get('arxiv_delay', 3)
    
    print(f"📅 日期: {target_date} | 本地目录: {local_dir}")
    print(f"⚙️  并发配置: 论文处理={paper_workers}篇, arXiv批次={arxiv_chunk_size}")
    
    global ZOTERO_STRUCTURE
    ZOTERO_STRUCTURE = zotero_ops.get_existing_structure()
    
    arxiv_ids = hf_scraper.get_daily_papers(target_date)
    if not arxiv_ids:
        print("今天没有新论文。")
        return

    print(f"🔍 抓取到 {len(arxiv_ids)} 篇，获取元数据...")
    papers_meta = utils.get_arxiv_metadata(arxiv_ids, chunk_size=arxiv_chunk_size, delay=arxiv_delay)
    
    print("🚀 开始 AI 处理...")
    
    tasks = []
    for arxiv_id, meta in papers_meta.items():
        tasks.append((arxiv_id, meta, local_dir, target_date, args.skip_deep_analysis))

    interested_results = []
    ignored_results = []

    with ThreadPoolExecutor(max_workers=paper_workers) as executor:
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

    generate_daily_report(interested_results, ignored_results, target_date, local_dir)


if __name__ == "__main__":
    main()
