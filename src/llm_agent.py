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
        "reason": "Why matches interest",
        "category": "CategoryName",
        "tags": ["tag1", "tag2"],
        "summary_cn": "中文一句话总结",
        "tricks_cn": "关键技巧或结论(中文)"
    }}
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
    2. 拒绝翻译腔，直接写中文或英文术语，不要中英文混用括号。
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
