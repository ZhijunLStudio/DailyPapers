from openai import OpenAI
import json
import yaml
import time

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

client = OpenAI(
    api_key=config['openai']['api_key'],
    base_url=config['openai']['base_url']
)

# 从配置读取LLM参数
openai_config = config.get('openai', {})
llm_timeout = openai_config.get('timeout', 60)
llm_max_retries = openai_config.get('max_retries', 3)
llm_retry_delay = openai_config.get('retry_delay', 5)



def analyze_paper_with_structure(title, abstract, zotero_structure):
    """
    RAG 模式分析：参考现有 Zotero 结构进行分类和打标
    """
    existing_cats = zotero_structure.get('collections', [])
    existing_tags = zotero_structure.get('tags', [])
    
    prompt = f"""
    Role: Senior AI Researcher.
    
    User Interests: 
    1. Language LLMs
    2. Reinforcement Learning (RL)
    3. Multimodal (Focus on Understanding, Grounding, RL-Agents)
    4. CV Foundation Models (YOLO, SAM)
    
    User Ignores: Pure entertainment generation (Music/Art) unless technically novel.

    **Reference Context (Existing Zotero Library):**
    - Existing Categories: {json.dumps(existing_cats)}
    - Frequent Tags: {json.dumps(existing_tags)}

    Task:
    1. **Interest Check**: Decide if the paper matches user interests.
    2. **Categorization**: 
       - PRIORITIZE using one of the "Existing Categories" if it fits well.
       - If NO fit, create a new concise category name (e.g., "Multimodal-Reasoning").
    3. **Tagging**: 
       - Generate 3-5 tags.
       - PRIORITIZE using "Frequent Tags" to maintain consistency.
       - Use English, lowercase.
    4. **Extraction**: 
       - 'summary_cn': One sentence summary in Chinese.
       - 'tricks_cn': Key tricks/findings in Chinese.

    Input:
    Title: {title}
    Abstract: {abstract}

    Return JSON strictly:
    {{
        "interested": true/false,
        "reason": "用中文简要说明为什么这篇论文符合或不符合用户兴趣，1-2句话",
        "category": "CategoryName",
        "tags": ["tag1", "tag2"],
        "summary_cn": "中文一句话总结",
        "tricks_cn": "关键技巧或结论(中文)"
    }}

    注意：reason 字段必须用中文回答，说明论文与用户兴趣的匹配程度。
    """
    
    for i in range(llm_max_retries):
        try:
            response = client.chat.completions.create(
                model=config['openai']['model'],
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=llm_timeout
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  LLM分析尝试 {i+1}/{llm_max_retries} 失败: {e}")
            if i == llm_max_retries - 1:
                print(f"  LLM Analyze Error: {e}")
                return {"interested": False}
            time.sleep(llm_retry_delay)

def generate_reading_note(title, authors, abstract, analysis_data):
    """
    生成详细中文笔记，使用用户指定的高质量模板
    """
    note_template = """
    ### 科研论文读书笔记

    **论文标题**: {Title}
    **作者**: {Authors}
    **年份/出处**: arXiv
    
    **标签**: {Tags}

    ---

    #### 1. 核心问题

    **研究背景**
    这篇论文是在什么样的大背景下出现的？当前领域存在哪些普遍的挑战或局限性？

    **具体问题**
    论文明确要解决的那个具体、未被满足的需求或技术空白是什么？

    ---

    #### 2. 核心思想与贡献

    **一句话总结**
    如果只能用一句话向别人介绍这篇论文，它是什么？

    **核心贡献**
    论文提出了哪些主要的、新颖的东西？分点列出：新方法、新理论、新数据集、实验发现。

    ---

    #### 3. 方法详述

    **整体架构**
    论文提出的方法由哪几个关键部分组成？

    **关键技术点**
    深入剖析最核心、最创新的技术细节。例如：模块是如何工作的？训练策略有什么特别之处？

    ---

    #### 4. 实验与结果

    **实验设置**
    数据集、评估指标、基线模型。

    **核心结果**
    主要指标表现如何？是否达到 SOTA？

    **消融实验**
    证明了哪个模块是有效的？

    ---

    #### 5. 个人思考与启发

    **亮点**
    思路巧妙？实验扎实？

    **局限性与未来方向**
    潜在问题或未来可以改进的方向。

    **启发**
    方法、思路能否应用到我自己的研究中？

    ---
    
    **写作要求**:
    1. 全中文书写，专有名词如 RLHF、Transformer 可保留英文。
    2. 拒绝翻译腔，直接写中文或英文术语，不要中英文混用括号，严禁出现英文原文段落。
    3. 深度总结，不要只翻译摘要，要根据摘要内容进行合理的逻辑推演和扩展。
    """
    
    user_input = f"""
    Title: {title}
    Authors: {authors}
    Abstract: {abstract}
    Tags: {analysis_data.get('tags')}
    
    Please generate the note following the template strictly.
    """
    
    try:
        response = client.chat.completions.create(
            model=config['openai']['model'],
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": f"Template:\n{note_template}\n\nTask:\n{user_input}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"笔记生成失败: {e}"


# ... (前面的代码保持不变) ...

def generate_daily_overview(interested_papers, ignored_papers):
    """
    生成日报的宏观综述部分
    """
    # 提取一些关键信息给 LLM
    interested_titles = [f"- {p['title']} (Category: {p['category']})" for p in interested_papers]
    ignored_titles = [f"- {p['title']} (Reason: {p['reason']})" for p in ignored_papers[:10]] # 只取前10个忽略的，防止太长

    prompt = f"""
    Role: Senior AI Research Lead.
    Task: Write an Executive Summary for today's Daily Papers Report (in Chinese).

    Data:
    - Interested Papers ({len(interested_papers)}):
    {chr(10).join(interested_titles)}
    
    - Ignored Papers (Sample):
    {chr(10).join(ignored_titles)}

    Requirements:
    1. **Overview**: Summarize the key trends today (e.g., "Today's focus is heavily on Multimodal Agents...").
    2. **Highlights**: Mention 1-2 most interesting papers briefly.
    3. **Brief Mention**: One sentence about what was ignored (e.g., "Mostly diffusion art generation").
    4. Style: Professional, concise, Chinese.
    """

    try:
        response = client.chat.completions.create(
            model=config['openai']['model'],
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return "今日综述生成失败。"


def summarize_papers_batch(papers_notes: list, batch_idx: int) -> dict:
    """
    对一批论文（最多10篇）进行小汇总

    Args:
        papers_notes: 论文笔记内容列表，每个元素是 dict 包含 note_content 和 metadata
        batch_idx: 批次编号

    Returns:
        dict: 包含 batch_summary（文本摘要）和 papers_highlights（每篇论文要点）
    """
    if not papers_notes:
        return {"batch_summary": "", "papers_highlights": []}

    # 构建输入文本
    papers_text = []
    for idx, paper in enumerate(papers_notes, 1):
        text = f"""
=== 论文 {idx} ===
标题: {paper.get('title', '')}
中文标题: {paper.get('title_cn', '')}
作者: {', '.join(paper.get('authors', [])[:3])}
类别: {paper.get('category', '')}

核心问题:
{paper.get('core_problem', '')}

核心贡献:
{chr(10).join(paper.get('core_contribution', [])) if isinstance(paper.get('core_contribution'), list) else paper.get('core_contribution', '')}

方法概述:
{paper.get('method_summary', '')[:500]}...

实验结果:
{paper.get('key_results', '')[:400]}...

亮点:
{chr(10).join(['- ' + p for p in paper.get('pros', [])[:2]])}
"""
        papers_text.append(text)

    prompt = f"""
Role: 资深AI研究分析师
Task: 对以下 {len(papers_notes)} 篇论文进行批次汇总分析（批次 #{batch_idx + 1}）

输入论文内容:
{chr(10).join(papers_text)}

请按以下JSON格式返回分析结果:
{{
    "batch_summary": "用3-5句话概括这批论文的整体研究趋势、共同主题和技术方向（中文）",
    "technical_trends": ["技术趋势1", "技术趋势2", "技术趋势3"],
    "papers_highlights": [
        {{
            "title": "论文1标题",
            "key_method": "该论文使用的核心方法（1句话）",
            "key_finding": "主要发现/贡献（1句话）",
            "result_highlight": "实验结果亮点（1句话，如有具体数据更佳）"
        }}
    ]
}}

要求:
1. batch_summary: 整体趋势分析，这批论文共同关注什么方向，有什么技术共性
2. technical_trends: 列出2-3个关键技术趋势或方法
3. papers_highlights: 为每篇论文提取方法、发现、结果三个要点
4. 所有输出必须是中文
"""

    for attempt in range(llm_max_retries):
        try:
            response = client.chat.completions.create(
                model=config['openai']['model'],
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=llm_timeout
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"  批次 {batch_idx + 1} 汇总尝试 {attempt + 1}/{llm_max_retries} 失败: {e}")
            if attempt == llm_max_retries - 1:
                return {
                    "batch_summary": f"批次 {batch_idx + 1} 汇总失败",
                    "technical_trends": [],
                    "papers_highlights": []
                }
            time.sleep(llm_retry_delay)


def generate_final_daily_report(batch_summaries: list, all_papers_count: int, date: str) -> dict:
    """
    基于所有小汇总结果生成最终的日报

    Args:
        batch_summaries: 所有批次汇总结果的列表
        all_papers_count: 论文总数
        date: 日期

    Returns:
        dict: 包含完整日报的各个部分
    """
    # 构建批次汇总文本
    batches_text = []
    for idx, batch in enumerate(batch_summaries, 1):
        summary = batch.get('batch_summary', '')
        trends = batch.get('technical_trends', [])
        batches_text.append(f"""
=== 批次 {idx} 汇总 ===
整体趋势: {summary}
技术趋势: {', '.join(trends)}
""")

    prompt = f"""
Role: 资深AI研究主管
Task: 基于以下各批次汇总，生成 {date} 的完整科研日报

总体情况: 今日共分析 {all_papers_count} 篇论文，分为 {len(batch_summaries)} 个批次处理

各批次汇总:
{chr(10).join(batches_text)}

请按以下JSON格式返回日报内容:
{{
    "daily_overview": "今日整体研究趋势概述（5-8句话，中文）",
    "key_insights": [
        "洞察1: 关于技术方向的深度分析",
        "洞察2: 关于研究热点的观察",
        "洞察3: 关于方法论的总结"
    ],
    "direction_summary": {{
        "Agent": "Agent方向的小结（如有该方向论文）",
        "Multimodal": "多模态方向的小结（如有）",
        "RL": "强化学习方向的小结（如有）"
    }},
    "notable_papers": [
        {{
            "title": "值得关注的论文标题",
            "why_notable": "为什么值得关注（1-2句话）"
        }}
    ],
    "future_trends": "对未来研究趋势的预测或建议（3-4句话）"
}}

要求:
1. daily_overview: 全面概括今日论文的整体特点、技术趋势、研究热点
2. key_insights: 提供3-5个深度洞察，不是简单罗列，而是分析性总结
3. direction_summary: 按研究方向分别小结，如果某方向没有则省略
4. notable_papers: 选出2-3篇最值得关注的论文并说明原因
5. future_trends: 基于今日论文预测未来可能的研究方向
6. 所有输出必须是中文，专业且流畅
"""

    for attempt in range(llm_max_retries):
        try:
            response = client.chat.completions.create(
                model=config['openai']['model'],
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=llm_timeout * 2  # 最终汇总给更多时间
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"  最终日报生成尝试 {attempt + 1}/{llm_max_retries} 失败: {e}")
            if attempt == llm_max_retries - 1:
                return {
                    "daily_overview": f"{date} 科研日报生成失败",
                    "key_insights": [],
                    "direction_summary": {},
                    "notable_papers": [],
                    "future_trends": ""
                }
            time.sleep(llm_retry_delay)


def parse_note_content(note_content: str) -> dict:
    """
    解析 note.md 内容，提取关键字段
    """
    result = {
        'title': '',
        'title_cn': '',
        'authors': [],
        'core_problem': '',
        'core_contribution': [],
        'method_summary': '',
        'key_results': '',
        'pros': [],
        'category': ''
    }

    lines = note_content.split('\n')
    current_section = None
    section_content = []

    for line in lines:
        line = line.strip()

        # 提取标题
        if line.startswith('# ') and not result['title']:
            result['title'] = line[2:].strip()

        # 提取中文标题
        if '**中文标题**:' in line or '**中文标题**：' in line:
            result['title_cn'] = line.split(':', 1)[-1].split('：', 1)[-1].strip()

        # 提取作者
        if '**作者**:' in line or '**作者**：' in line:
            authors_str = line.split(':', 1)[-1].split('：', 1)[-1].strip()
            result['authors'] = [a.strip() for a in authors_str.split(',')]

        # 识别章节
        if line.startswith('## '):
            # 保存上一个章节的内容
            if current_section == '核心问题' and section_content:
                result['core_problem'] = '\n'.join(section_content).strip()
            elif current_section == '核心贡献' and section_content:
                content = '\n'.join(section_content).strip()
                # 解析列表
                if content.startswith('- '):
                    result['core_contribution'] = [l[2:].strip() for l in section_content if l.strip().startswith('- ')]
                else:
                    result['core_contribution'] = [content]
            elif current_section == '方法概述' and section_content:
                result['method_summary'] = '\n'.join(section_content).strip()
            elif current_section == '实验结果' and section_content:
                result['key_results'] = '\n'.join(section_content).strip()
            elif current_section == '亮点' and section_content:
                result['pros'] = [l[2:].strip() for l in section_content if l.strip().startswith('- ')]

            current_section = line[3:].strip()
            section_content = []
        elif current_section and line and not line.startswith('!'):
            section_content.append(line)

    # 处理最后一个章节
    if current_section == '核心问题' and section_content:
        result['core_problem'] = '\n'.join(section_content).strip()
    elif current_section == '核心贡献' and section_content:
        content = '\n'.join(section_content).strip()
        if content.startswith('- '):
            result['core_contribution'] = [l[2:].strip() for l in section_content if l.strip().startswith('- ')]
        else:
            result['core_contribution'] = [content]
    elif current_section == '方法概述' and section_content:
        result['method_summary'] = '\n'.join(section_content).strip()
    elif current_section == '实验结果' and section_content:
        result['key_results'] = '\n'.join(section_content).strip()
    elif current_section == '亮点' and section_content:
        result['pros'] = [l[2:].strip() for l in section_content if l.strip().startswith('- ')]

    return result
