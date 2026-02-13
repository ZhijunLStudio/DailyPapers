"""
论文深度分析模块 - 对感兴趣的论文进行PDF转图片、OCR分析、图文报告生成
支持并发OCR处理
"""
import os
import re
import base64
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 获取配置
analysis_config = config.get('analysis', {})
concurrency_config = config.get('concurrency', {})
ocr_config = config.get('deepseek_ocr', {})
openai_config = config.get('openai', {})

# OpenAI客户端（用于内容分析）
client = OpenAI(
    api_key=config['openai']['api_key'],
    base_url=config['openai']['base_url']
)

# DeepSeek OCR客户端
if ocr_config.get('api_key') and ocr_config.get('base_url'):
    ocr_client = OpenAI(
        api_key=ocr_config['api_key'],
        base_url=ocr_config['base_url']
    )
else:
    ocr_client = client

# Token消耗记录
token_usage = {
    'ocr_calls': 0,
    'ocr_tokens': 0,
    'llm_calls': 0,
    'llm_tokens_input': 0,
    'llm_tokens_output': 0
}


def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = None) -> List[str]:
    """将PDF转换为图片，实时打印进度"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("错误: 请先安装 PyMuPDF: pip install PyMuPDF")
        return []

    dpi = dpi or analysis_config.get('pdf_dpi', 200)
    os.makedirs(output_dir, exist_ok=True)

    # 尝试禁用MuPDF的警告输出（兼容不同版本）
    try:
        fitz.set_messages_enabled(False)
    except AttributeError:
        pass  # 某些PyMuPDF版本不支持此方法

    doc = None
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        image_paths = []
        failed_pages = []

        for page_num in range(total_pages):
            try:
                page = doc[page_num]
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                image_path = os.path.join(output_dir, f"page_{page_num+1:03d}.png")
                pix.save(image_path)
                image_paths.append(image_path)
            except Exception as e:
                failed_pages.append(page_num + 1)
                print(f"   ⚠️  Page {page_num + 1} 转换失败: {str(e)[:50]}")
                continue

        if failed_pages:
            print(f"   ⚠️  PDF转换: {len(failed_pages)}/{total_pages} 页失败 (页码: {failed_pages[:5]}{'...' if len(failed_pages) > 5 else ''})")

        return image_paths
    except Exception as e:
        print(f"   ❌ PDF打开失败: {str(e)[:100]}")
        return []
    finally:
        if doc:
            try:
                doc.close()
            except:
                pass


def call_deepseek_ocr(image_path: str) -> Tuple[str, Dict]:
    """调用DeepSeek-OCR模型分析图片"""
    global token_usage
    
    timeout = ocr_config.get('timeout', 120)
    max_retries = ocr_config.get('max_retries', 3)
    retry_delay = ocr_config.get('retry_delay', 5)
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    "detail": "high"
                },
                {
                    "type": "text",
                    "text": "<|grounding|>Convert the document to markdown."
                }
            ]
        }
    ]
    
    for attempt in range(max_retries):
        try:
            ocr_model = ocr_config.get('model', 'deepseek-ocr')
            start_time = time.time()
            response = ocr_client.chat.completions.create(
                model=ocr_model,
                messages=messages,
                temperature=0.1,
                timeout=timeout
            )
            elapsed = time.time() - start_time
            
            token_usage['ocr_calls'] += 1
            usage = response.usage
            if usage:
                token_usage['ocr_tokens'] += usage.total_tokens
            
            return response.choices[0].message.content, {
                'model': ocr_model,
                'elapsed': elapsed,
                'tokens': usage.total_tokens if usage else 0
            }
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  OCR调用失败 (第{attempt + 1}次尝试): {error_msg}")
            if attempt < max_retries - 1:
                print(f"   等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"❌ OCR调用最终失败，已重试 {max_retries} 次")
                return None, {}


def process_single_page(args):
    """处理单页（用于并发）"""
    page_idx, image_path, ocr_dir, figures_dir, save_viz, save_cropped = args
    page_num = page_idx + 1
    
    # OCR识别
    ocr_text, ocr_token_info = call_deepseek_ocr(image_path)
    if not ocr_text:
        print(f"   📄 Page {page_num}: OCR识别失败，跳过该页")
        return None

    try:
        ocr_items = parse_ocr_response(ocr_text)
    except Exception as e:
        print(f"   📄 Page {page_num}: OCR结果解析失败 - {str(e)}")
        return None
    
    # 保存OCR文本
    ocr_txt_path = os.path.join(ocr_dir, f"page_{page_num:03d}.txt")
    with open(ocr_txt_path, 'w', encoding='utf-8') as f:
        f.write(ocr_text)
    
    # 可视化
    if save_viz:
        try:
            vis_path = os.path.join(ocr_dir, f"page_{page_num:03d}_vis.png")
            visualize_ocr_result(image_path, ocr_items, vis_path)
        except Exception as e:
            print(f"   📄 Page {page_num}: 可视化生成失败 - {str(e)}")

    # 提取关键图表
    page_figures = []
    if save_cropped:
        try:
            page_figures = extract_key_figures(ocr_items, image_path, figures_dir, page_num)
        except Exception as e:
            print(f"   📄 Page {page_num}: 图表提取失败 - {str(e)}")
    
    return {
        'page': page_num,
        'items': ocr_items,
        'raw_text': ocr_text,
        'figures': page_figures
    }


def parse_ocr_response(content: str) -> List[Dict[str, Any]]:
    """解析OCR响应，提取各个区域"""
    items = []
    tag_pattern = re.compile(r'(?P<type>\w+)\[\[(?P<rect>[\d,\s,]+)\]\]')
    matches = list(tag_pattern.finditer(content))
    
    for i, match in enumerate(matches):
        data = match.groupdict()
        label = data['type']
        rect_str = data['rect']
        
        try:
            bbox = [int(x) for x in re.split(r'[,\s]+', rect_str.strip()) if x]
        except ValueError:
            continue
            
        start_idx = match.end()
        if i < len(matches) - 1:
            end_idx = matches[i+1].start()
        else:
            end_idx = len(content)
            
        text_content = content[start_idx:end_idx].strip()
        
        items.append({
            "type": label,
            "bbox": bbox,
            "text": text_content
        })
    
    return items


def crop_region(image_path: str, bbox: List[int], output_path: str):
    """从图片中裁剪指定区域"""
    img = Image.open(image_path)
    width, height = img.size
    
    x1 = int(bbox[0] / 1000 * width)
    y1 = int(bbox[1] / 1000 * height)
    x2 = int(bbox[2] / 1000 * width)
    y2 = int(bbox[3] / 1000 * height)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    
    if x2 > x1 and y2 > y1:
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(output_path)


def visualize_ocr_result(image_path: str, ocr_items: List[Dict], output_path: str):
    """绘制OCR可视化结果 - 只在原图上画框和标签"""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    try:
        font_size = max(12, int(width / 80))
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()
    
    color_map = {
        "title": (255, 0, 0),
        "text": (0, 0, 0),
        "header": (0, 128, 0),
        "figure": (0, 0, 255),
        "image": (0, 0, 255),
        "image_caption": (255, 165, 0),
        "caption": (255, 165, 0),
        "table": (128, 0, 128),
        "table_caption": (255, 105, 180),
        "sub_title": (0, 128, 128),
        "author": (128, 128, 0),
        "abstract": (70, 130, 180),
        "reference": (105, 105, 105),
        "formula": (255, 20, 147),
        "code": (0, 100, 0),
    }
    
    for item in ocr_items:
        bbox = item['bbox']
        label = item['type']
        
        if len(bbox) == 4:
            x1 = int(bbox[0] / 1000 * width)
            y1 = int(bbox[1] / 1000 * height)
            x2 = int(bbox[2] / 1000 * width)
            y2 = int(bbox[3] / 1000 * height)
        else:
            continue
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        color = color_map.get(label, (100, 100, 100))
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text_w = font.getlength(label)
        draw.rectangle([x1, y1 - font_size - 2, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - font_size), label, fill=(255, 255, 255), font=font)
    
    img.save(output_path)


def extract_key_figures(ocr_items: List[Dict], image_path: str, 
                        figures_dir: str, page_num: int) -> List[Dict]:
    """提取关键图表"""
    os.makedirs(figures_dir, exist_ok=True)
    
    key_figures = []
    
    for i, item in enumerate(ocr_items):
        label = item['type']
        
        if label not in ['image', 'figure', 'table']:
            continue
        
        caption = ""
        for j in range(i+1, min(i+3, len(ocr_items))):
            if ocr_items[j]['type'] in ['caption', 'image_caption', 'table_caption']:
                caption = ocr_items[j]['text']
                break
        
        # 清理caption中的HTML标签
        import re
        caption = re.sub(r'<[^>]+>', '', caption)
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        ext = 'fig' if label in ['image', 'figure'] else 'table'
        crop_path = os.path.join(figures_dir, f"{ext}_p{page_num:03d}_{i+1:02d}.png")
        crop_region(image_path, item['bbox'], crop_path)
        
        key_figures.append({
            'type': label,
            'page': page_num,
            'index': i + 1,
            'caption': caption,
            'bbox': item['bbox'],
            'crop_path': crop_path,
            'text': item.get('text', '')
        })
    
    return key_figures


def analyze_paper_content(ocr_results: List[Dict]) -> Tuple[Dict, Dict]:
    """使用LLM分析论文OCR内容"""
    global token_usage
    
    timeout = openai_config.get('timeout', 60)
    max_retries = openai_config.get('max_retries', 3)
    retry_delay = openai_config.get('retry_delay', 5)
    max_length = analysis_config.get('max_ocr_text_length', 12000)
    
    # 合并文本
    all_text = ""
    for page in ocr_results:
        page_num = page['page']
        all_text += f"\n\n=== Page {page_num} ===\n\n"
        for item in page['items']:
            all_text += f"[{item['type']}] {item['text']}\n"
    
    if len(all_text) > max_length:
        all_text = all_text[:max_length] + "\n... (内容已截断)"
    
    prompt = f"""你是一个专业的学术论文分析助手。请仔细分析以下论文的OCR内容，提取关键信息。

论文内容:
{all_text}

请提取以下信息并以JSON格式返回:
{{
    "title": "论文标题",
    "title_cn": "论文中文标题或翻译",
    "authors": ["作者1", "作者2"],
    "abstract": "摘要内容",
    "core_problem": "核心问题描述，用1-2句话概括",
    "core_contribution": ["核心贡献1", "核心贡献2", "核心贡献3"],
    "method_summary": "方法概述（详细版，300-500字）：
        - 整体思路：论文解决问题的核心思路是什么
        - 关键技术：具体使用了哪些技术或方法
        - 创新点：相比现有方法有什么创新
        - 实现细节：关键的实现细节或算法流程",
    "key_figures_description": [
        "图1: 架构图/概览图 - 用中文描述这张图展示了什么框架或系统",
        "图2: 实验结果图 - 用中文描述这张图展示了什么实验结果或对比"
    ],
    "key_results": "主要实验结果（200-300字）：在哪些数据集上做了实验，性能如何提升，与基线方法的对比",
    "key_tables": ["表1: 描述表格内容和关键数据"],
    "conclusion": "结论（100-200字）：论文的主要贡献和意义",
    "pros": ["论文亮点1", "论文亮点2"],
    "cons": ["局限性1", "局限性2"],
    "inspirations": ["对未来研究或实践的启发1", "启发2"]
}}

注意：
1. 返回必须是有效的JSON格式
2. 所有描述必须使用纯中文，严禁夹杂任何英文原文（专有名词除外）
3. method_summary要详细，分点说明整体思路、关键技术、创新点、实现细节
4. 严禁出现任何英文原文段落。如果 OCR 结果是英文，你必须将其翻译成流畅的中文，而不是直接保留或简单替换。
5. key_figures_description 中的描述要用中文重新组织，不要包含任何如 "Figure 1 shows..." 这种英文表达，直接描述内容。
6. 对图表的描述要详细，说明其用途和展示的内容
7. pros, cons, inspirations 必须根据论文内容给出具体的分析，不要留空，且必须为中文。
"""
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=config['openai']['model'],
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=timeout
            )
            elapsed = time.time() - start_time
            
            token_usage['llm_calls'] += 1
            usage = response.usage
            if usage:
                token_usage['llm_tokens_input'] += usage.prompt_tokens
                token_usage['llm_tokens_output'] += usage.completion_tokens
            
            result = json.loads(response.choices[0].message.content)
            token_info = {
                'model': config['openai']['model'],
                'elapsed': elapsed,
                'tokens_input': usage.prompt_tokens if usage else 0,
                'tokens_output': usage.completion_tokens if usage else 0
            }
            return result, token_info
        except Exception as e:
            if attempt == max_retries - 1:
                return {}, {}
            time.sleep(retry_delay)


def select_key_figures_for_report(all_figures: List[Dict], analysis: Dict) -> List[Dict]:
    """选择最关键的图表 - 智能分类选择"""
    if not all_figures:
        return []
    
    max_figures = analysis_config.get('max_figures_per_paper', 4)
    selected = []
    
    # 分类：架构图、结果图、表格
    arch_figures = []
    result_figures = []
    tables = []
    
    for fig in all_figures:
        caption = fig.get('caption', '').lower()
        fig_type = fig['type']
        
        # 架构图关键词
        arch_keywords = ['arch', 'framework', 'overview', 'model', 'structure', 'pipeline', 'system', 'design']
        # 结果图关键词
        result_keywords = ['result', 'performance', 'comparison', 'ablation', 'accuracy', 'loss', 'curve', 'plot']
        
        if fig_type == 'table':
            tables.append(fig)
        elif any(kw in caption for kw in arch_keywords):
            arch_figures.append(fig)
        elif any(kw in caption for kw in result_keywords):
            result_figures.append(fig)
        else:
            # 其他图片，归入结果图
            result_figures.append(fig)
    
    # 选择：1-2张架构图，1-2张结果图，1张表格
    # 如果某类没有，从其他类补充
    selected.extend(arch_figures[:2])
    selected.extend(result_figures[:2])
    selected.extend(tables[:1])
    
    # 如果还不够，补充其他图
    remaining = [f for f in all_figures if f not in selected]
    selected.extend(remaining[:max_figures - len(selected)])
    
    # 添加LLM分析描述 - 按顺序匹配
    figure_descriptions = analysis.get('key_figures_description', [])
    for i, fig in enumerate(selected):
        if i < len(figure_descriptions):
            fig['analysis_desc'] = figure_descriptions[i]
    
    return selected[:max_figures]


def get_clean_caption(caption: str, fig_type: str) -> str: 
    """从原始 caption 中提取编号并转换为 '原文图/表 X' 格式，去除其余英文"""
    if not caption:
        return "原文图" if fig_type != 'table' else "原文表"
    
    import re
    # 匹配 Figure 1, Fig. 1, Table 1 等
    match = re.search(r'(?:Figure|Fig\.|Table|Tab\.)\s*(\d+[a-z]?)', caption, re.IGNORECASE)
    if match:
        num = match.group(1)
        prefix = "原文表" if "tab" in caption.lower() else "原文图"
        return f"{prefix}{num}"
    
    return "原文图" if fig_type != 'table' else "原文表"

def generate_paper_note(paper_info: Dict, analysis: Dict, selected_figures: List[Dict],
                        output_path: str, token_info: Dict):
    """生成单篇论文的详细笔记 - 图文并茂，图表融入内容"""
    title = analysis.get('title', paper_info['title'])
    title_cn = analysis.get('title_cn', '')
    
    md_content = f"# {title}\n\n"
    
    if title_cn:
        md_content += f"**中文标题**: {title_cn}\n\n"
    
    md_content += f"**作者**: {', '.join(analysis.get('authors', paper_info['authors']))}\n\n"
    md_content += f"**来源**: arXiv | **日期**: {paper_info.get('date', '')}\n\n"
    md_content += "---\n\n"
    
    md_content += "## 核心问题\n\n"
    md_content += f"{analysis.get('core_problem', '未提取')}\n\n"
    
    md_content += "## 核心贡献\n\n"
    contribution = analysis.get('core_contribution', '')
    if isinstance(contribution, list):
        for item in contribution:
            md_content += f"- {item}\n"
    else:
        md_content += f"{contribution}\n"
    md_content += "\n"
    
    md_content += "## 方法概述\n\n"
    method_summary = analysis.get('method_summary', '未提取')
    md_content += f"{method_summary}\n\n"
    
    # 记录已使用的图片，防止重复
    used_figures = set()
    
    # 融入架构图（在方法概述后）
    arch_figures = [f for f in selected_figures if f['type'] in ['image', 'figure'] and 
                   any(kw in f.get('caption', '').lower() for kw in ['arch', 'framework', 'overview', 'model', 'structure', 'pipeline', 'system'])]
    if arch_figures:
        md_content += "**架构图**\n\n"
        for fig in arch_figures[:2]:
            abs_path = os.path.abspath(fig['crop_path'])
            if abs_path in used_figures:
                continue
            
            desc = fig.get('analysis_desc', '')
            clean_cap = get_clean_caption(fig.get('caption', ''), fig['type'])
            
            if desc:
                md_content += f"{desc}\n\n"
            md_content += f"![{clean_cap}](file://{abs_path})\n\n"
            md_content += f"*{clean_cap}*\n\n"
            used_figures.add(abs_path)
    
    md_content += "---\n\n"
    
    md_content += "## 实验结果\n\n"
    md_content += f"{analysis.get('key_results', '未提取')}\n\n"
    
    # 融入结果图和表格
    result_figures = [f for f in selected_figures if 
                     (f['type'] in ['image', 'figure'] and any(kw in f.get('caption', '').lower() for kw in ['result', 'performance', 'comparison', 'ablation', 'accuracy', 'loss', 'curve', 'plot'])) or
                     f['type'] == 'table']
    
    if result_figures:
        md_content += "**实验数据**\n\n"
        for fig in result_figures[:3]:
            abs_path = os.path.abspath(fig['crop_path'])
            if abs_path in used_figures:
                continue
                
            desc = fig.get('analysis_desc', '')
            clean_cap = get_clean_caption(fig.get('caption', ''), fig['type'])
            
            if desc:
                md_content += f"{desc}\n\n"
            
            md_content += f"![{clean_cap}](file://{abs_path})\n\n"
            md_content += f"*{clean_cap}*\n\n"
            used_figures.add(abs_path)
    
    md_content += "---\n\n"
    md_content += "## 结论\n\n"
    md_content += f"{analysis.get('conclusion', '未提取')}\n\n"
    
    md_content += "---\n\n"
    md_content += "## 个人思考\n\n"
    md_content += "### 亮点\n\n"
    pros = analysis.get('pros', [])
    if isinstance(pros, list):
        for item in pros:
            md_content += f"- {item}\n"
    else:
        md_content += f"{pros}\n"
    md_content += "\n"
    
    md_content += "### 局限性\n\n"
    cons = analysis.get('cons', [])
    if isinstance(cons, list):
        for item in cons:
            md_content += f"- {item}\n"
    else:
        md_content += f"{cons}\n"
    md_content += "\n"
    
    md_content += "### 启发\n\n"
    inspirations = analysis.get('inspirations', [])
    if isinstance(inspirations, list):
        for item in inspirations:
            md_content += f"- {item}\n"
    else:
        md_content += f"{inspirations}\n"
    md_content += "\n"
    
    md_content += "---\n\n"
    md_content += "## 处理记录\n\n"
    md_content += f"- OCR模型: {token_info.get('ocr_model', 'unknown')}\n"
    md_content += f"- OCR调用次数: {token_info.get('ocr_calls', 0)}\n"
    md_content += f"- OCR总tokens: {token_info.get('ocr_tokens', 0)}\n"
    md_content += f"- LLM模型: {token_info.get('llm_model', 'unknown')}\n"
    md_content += f"- LLM调用次数: {token_info.get('llm_calls', 0)}\n"
    md_content += f"- LLM输入tokens: {token_info.get('llm_tokens_input', 0)}\n"
    md_content += f"- LLM输出tokens: {token_info.get('llm_tokens_output', 0)}\n"
    md_content += f"- 处理时间: {token_info.get('total_time', 0):.2f}秒\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return md_content


def analyze_paper_deep(pdf_path: str, paper_info: Dict, category_dir: str) -> Dict[str, Any]:
    """
    对论文进行深度分析的主函数 - 支持并发OCR
    
    目录结构:
    Category/
    └── Author_Title/
        ├── paper.pdf
        ├── note.md
        ├── ocr/
        │   ├── page_001.txt
        │   └── page_001_vis.png
        ├── figures/
        │   ├── fig_p001_01.png
        │   └── table_p001_02.png
        └── analysis.json
    """
    global token_usage
    token_usage = {
        'ocr_calls': 0,
        'ocr_tokens': 0,
        'llm_calls': 0,
        'llm_tokens_input': 0,
        'llm_tokens_output': 0
    }
    start_total = time.time()
    
    # 读取配置
    pdf_dpi = analysis_config.get('pdf_dpi', 200)
    max_pages = analysis_config.get('max_pages', 15)
    save_viz = analysis_config.get('save_visualization', True)
    save_cropped = analysis_config.get('save_cropped_figures', True)
    ocr_workers = concurrency_config.get('ocr_workers', 3)
    
    # 创建论文专属目录
    paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
    paper_dir = os.path.join(category_dir, paper_name)
    
    ocr_dir = os.path.join(paper_dir, "ocr")
    figures_dir = os.path.join(paper_dir, "figures")
    
    os.makedirs(ocr_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. PDF转图片（临时目录）
    import tempfile
    print(f"🔬 深度分析: {paper_name}")
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"   📑 正在转换PDF为图片 (DPI={pdf_dpi})...")
        image_paths = pdf_to_images(pdf_path, temp_dir, dpi=pdf_dpi)
        if not image_paths:
            print(f"   ❌ PDF转换失败，无法继续分析")
            return None
        print(f"   ✅ PDF转换完成，共 {len(image_paths)} 页")
        
        if max_pages is not None and len(image_paths) > max_pages:
            image_paths = image_paths[:max_pages]
        
        # 2. 并发OCR分析
        ocr_results = []
        all_key_figures = []

        print(f"   🔍 开始OCR分析 (并发数={ocr_workers})...")
        # 准备任务参数
        tasks = [(i, img_path, ocr_dir, figures_dir, save_viz, save_cropped)
                 for i, img_path in enumerate(image_paths)]

        # 并发执行OCR
        with ThreadPoolExecutor(max_workers=ocr_workers) as executor:
            futures = {executor.submit(process_single_page, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result and isinstance(result, dict):
                    ocr_results.append(result)
                    all_key_figures.extend(result.get('figures', []))

        # 按页码排序
        ocr_results.sort(key=lambda x: x['page'])
        print(f"   ✅ OCR完成: {len(ocr_results)}/{len(image_paths)} 页成功识别")

    # 3. LLM分析内容
    print(f"   🧠 正在LLM分析论文内容...")
    analysis, llm_token_info = analyze_paper_content(ocr_results)
    
    # 4. 选择关键图表
    selected_figures = select_key_figures_for_report(all_key_figures, analysis)
    
    # 5. 生成详细笔记
    note_path = os.path.join(paper_dir, "note.md")
    
    token_info_summary = {
        'ocr_model': ocr_config.get('model', 'deepseek-ocr'),
        'ocr_calls': token_usage['ocr_calls'],
        'ocr_tokens': token_usage['ocr_tokens'],
        'llm_model': config['openai']['model'],
        'llm_calls': token_usage['llm_calls'],
        'llm_tokens_input': token_usage['llm_tokens_input'],
        'llm_tokens_output': token_usage['llm_tokens_output'],
        'total_time': time.time() - start_total
    }
    
    generate_paper_note(paper_info, analysis, selected_figures, note_path, token_info_summary)
    print(f"   ✅ 笔记已生成: {note_path}")

    # 6. 保存分析数据
    analysis_data_path = os.path.join(paper_dir, "analysis.json")
    with open(analysis_data_path, 'w', encoding='utf-8') as f:
        json.dump({
            'paper_info': paper_info,
            'analysis': analysis,
            'selected_figures': selected_figures,
            'all_figures_count': len(all_key_figures),
            'token_usage': token_info_summary
        }, f, ensure_ascii=False, indent=2)

    print(f"   📊 分析完成: 提取 {len(selected_figures)} 张图表, 耗时 {token_info_summary['total_time']:.1f}秒")

    return {
        'paper_dir': paper_dir,
        'note_path': note_path,
        'analysis': analysis,
        'selected_figures': selected_figures,
        'token_usage': token_info_summary
    }


# ========== 解耦的OCR和LLM分析函数 ==========

# 全局变量用于存储OCR中间结果
_ocr_cache = {}

def extract_ocr_only(pdf_path: str, paper_info: Dict, category_dir: str) -> Dict[str, Any]:
    """
    仅执行OCR阶段，保存OCR结果供后续LLM分析使用

    Returns:
        包含OCR结果的字典，失败返回None
    """
    global token_usage
    token_usage = {
        'ocr_calls': 0,
        'ocr_tokens': 0,
        'llm_calls': 0,
        'llm_tokens_input': 0,
        'llm_tokens_output': 0
    }

    # 读取配置
    pdf_dpi = analysis_config.get('pdf_dpi', 200)
    max_pages = analysis_config.get('max_pages', 15)
    save_viz = analysis_config.get('save_visualization', True)
    save_cropped = analysis_config.get('save_cropped_figures', True)
    ocr_workers = concurrency_config.get('ocr_workers', 3)

    # 创建论文专属目录
    paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
    paper_dir = os.path.join(category_dir, paper_name)

    ocr_dir = os.path.join(paper_dir, "ocr")
    figures_dir = os.path.join(paper_dir, "figures")

    os.makedirs(ocr_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"   📄 OCR阶段: {paper_name}")

    # 1. PDF转图片（临时目录）
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"      📑 PDF转图片 (DPI={pdf_dpi})...")
        image_paths = pdf_to_images(pdf_path, temp_dir, dpi=pdf_dpi)
        if not image_paths:
            print(f"      ❌ PDF转换失败")
            return None
        print(f"      ✅ PDF转换完成: {len(image_paths)} 页")

        if max_pages is not None and len(image_paths) > max_pages:
            image_paths = image_paths[:max_pages]
            print(f"      ℹ️  限制处理前 {max_pages} 页")

        # 2. 并发OCR分析
        ocr_results = []
        all_key_figures = []

        print(f"      🔍 OCR识别中 (并发={ocr_workers})...")
        tasks = [(i, img_path, ocr_dir, figures_dir, save_viz, save_cropped)
                 for i, img_path in enumerate(image_paths)]

        with ThreadPoolExecutor(max_workers=ocr_workers) as executor:
            futures = {executor.submit(process_single_page, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result and isinstance(result, dict):
                    ocr_results.append(result)
                    all_key_figures.extend(result.get('figures', []))

        ocr_results.sort(key=lambda x: x['page'])
        print(f"      ✅ OCR完成: {len(ocr_results)}/{len(image_paths)} 页成功")

    # 保存OCR中间结果
    ocr_data = {
        'paper_name': paper_name,
        'paper_dir': paper_dir,
        'ocr_dir': ocr_dir,
        'figures_dir': figures_dir,
        'ocr_results': ocr_results,
        'all_key_figures': all_key_figures,
        'token_usage': {
            'ocr_calls': token_usage['ocr_calls'],
            'ocr_tokens': token_usage['ocr_tokens'],
        }
    }

    # 保存到缓存和文件
    _ocr_cache[pdf_path] = ocr_data
    ocr_cache_path = os.path.join(paper_dir, "ocr_cache.json")
    with open(ocr_cache_path, 'w', encoding='utf-8') as f:
        json.dump({
            'ocr_results': ocr_results,
            'all_key_figures_count': len(all_key_figures),
            'token_usage': ocr_data['token_usage']
        }, f, ensure_ascii=False, indent=2)

    return ocr_data


def analyze_with_llm(ocr_data: Dict[str, Any], paper_info: Dict, category_dir: str) -> Dict[str, Any]:
    """
    基于已有的OCR结果执行LLM分析

    Args:
        ocr_data: extract_ocr_only返回的结果
        paper_info: 论文信息
        category_dir: 分类目录

    Returns:
        完整的分析结果
    """
    global token_usage

    # 恢复token_usage
    token_usage = {
        'ocr_calls': ocr_data.get('token_usage', {}).get('ocr_calls', 0),
        'ocr_tokens': ocr_data.get('token_usage', {}).get('ocr_tokens', 0),
        'llm_calls': 0,
        'llm_tokens_input': 0,
        'llm_tokens_output': 0
    }
    start_total = time.time()

    paper_name = ocr_data['paper_name']
    paper_dir = ocr_data['paper_dir']
    ocr_results = ocr_data['ocr_results']
    all_key_figures = ocr_data['all_key_figures']

    print(f"   🧠 LLM阶段: {paper_name}")

    if not ocr_results:
        print(f"      ❌ 没有OCR结果可供分析")
        return None

    # 3. LLM分析内容
    print(f"      💭 正在调用LLM分析...")
    analysis, llm_token_info = analyze_paper_content(ocr_results)
    print(f"      ✅ LLM分析完成")

    # 4. 选择关键图表
    selected_figures = select_key_figures_for_report(all_key_figures, analysis)
    print(f"      📊 选择 {len(selected_figures)} 张关键图表")

    # 5. 生成详细笔记
    note_path = os.path.join(paper_dir, "note.md")

    token_info_summary = {
        'ocr_model': ocr_config.get('model', 'deepseek-ocr'),
        'ocr_calls': token_usage['ocr_calls'],
        'ocr_tokens': token_usage['ocr_tokens'],
        'llm_model': config['openai']['model'],
        'llm_calls': token_usage['llm_calls'],
        'llm_tokens_input': token_usage['llm_tokens_input'],
        'llm_tokens_output': token_usage['llm_tokens_output'],
        'total_time': time.time() - start_total
    }

    generate_paper_note(paper_info, analysis, selected_figures, note_path, token_info_summary)
    print(f"      📝 笔记已生成")

    # 6. 保存分析数据
    analysis_data_path = os.path.join(paper_dir, "analysis.json")
    with open(analysis_data_path, 'w', encoding='utf-8') as f:
        json.dump({
            'paper_info': paper_info,
            'analysis': analysis,
            'selected_figures': selected_figures,
            'all_figures_count': len(all_key_figures),
            'token_usage': token_info_summary
        }, f, ensure_ascii=False, indent=2)

    print(f"      ✅ LLM阶段完成 (耗时 {token_info_summary['total_time']:.1f}秒)")

    return {
        'paper_dir': paper_dir,
        'note_path': note_path,
        'analysis': analysis,
        'selected_figures': selected_figures,
        'token_usage': token_info_summary
    }
