from pyzotero import Zotero
import yaml
import os
import time

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

zot = Zotero(
    config['zotero']['library_id'],
    config['zotero']['library_type'],
    config['zotero']['api_key']
)

# 缓存
collection_cache = {} 

def get_existing_structure():
    """
    获取 Zotero 现有的分类和标签
    """
    print("🔄 正在扫描 Zotero 现有目录结构...")
    structure = {
        "collections": [],
        "tags": []
    }
    
    try:
        # 1. 获取所有 Collections
        colls = zot.collections()
        for c in colls:
            structure["collections"].append(c['data']['name'])
            collection_cache[c['data']['name']] = c['key']
            
        # 2. 获取标签
        # 【关键修改】：去掉了 sort='count'，因为 API 不支持。
        # limit=50 默认是按字母顺序获取前 50 个标签。
        tags = zot.tags(limit=50)
        structure["tags"] = [t for t in tags]
        
        print(f"✅ 扫描完成: {len(structure['collections'])} 个分类, {len(structure['tags'])} 个标签")
        return structure
    except Exception as e:
        print(f"⚠️ Zotero 扫描非致命错误 (不影响后续上传): {e}")
        return structure

def get_or_create_collection_id(category_name):
    # 1. 检查缓存
    if category_name in collection_cache:
        return collection_cache[category_name]
    
    # 2. 如果没缓存，说明是新分类，需要创建
    root_id = config['zotero'].get('collection_id')
    
    try:
        # 防止并发创建同名文件夹，简单查重
        # (注：严格来说这里应该加锁，但为了代码简单，依靠 Zotero 自身去重或接受重复)
        print(f"🔨 正在创建新分类: {category_name}")
        resp = zot.create_collections([{
            'name': category_name,
            'parentCollection': root_id if root_id else None
        }])
        if resp['success']:
            new_key = resp['success']['0']
            collection_cache[category_name] = new_key
            return new_key
        else:
            return root_id 
    except Exception as e:
        print(f"❌ 创建分类异常: {e}")
        return root_id

def upload_paper_linked(meta, pdf_path, note_content, tags, category):
    retries = 3
    for i in range(retries):
        try:
            return _upload_logic(meta, pdf_path, note_content, tags, category)
        except Exception as e:
            # 捕获网络超时
            if "handshake" in str(e).lower() or "timeout" in str(e).lower() or "connection" in str(e).lower():
                time.sleep(2 * (i+1))
                if i == retries - 1:
                    print(f"❌ Zotero 上传超时: {meta['title'][:15]}")
            # 捕获 400 错误（通常是参数问题，不重试）
            elif "400" in str(e):
                print(f"❌ Zotero 参数错误 (Code 400): {e}")
                break
            else:
                print(f"❌ Zotero 未知错误: {e}")
                break
    return None

def _upload_logic(meta, pdf_path, note_content, tags, category):
    # 1. 准备元数据
    template = zot.item_template('preprint') 
    template['title'] = meta['title']
    template['abstractNote'] = meta['summary']
    template['url'] = meta['pdf_url']
    
    if meta.get('authors'):
        creators = []
        for author in meta['authors']:
            parts = author.split(' ')
            creators.append({
                'creatorType': 'author',
                'firstName': parts[0],
                'lastName': ' '.join(parts[1:]) if len(parts)>1 else ''
            })
        template['creators'] = creators
        
    final_tags = list(set(tags + [category]))
    template['tags'] = [{'tag': t} for t in final_tags]
    
    col_id = get_or_create_collection_id(category)
    if col_id:
        template['collections'] = [col_id]

    # 创建条目
    resp = zot.create_items([template])
    if not resp['success']: return None
    parent_key = resp['success']['0']
    
    # 2. 链接本地文件
    if os.path.exists(pdf_path):
        try:
            # 使用正确的 pyzotero API 创建链接附件
            attachment_template = zot.item_template('attachment', 'linked_file')
            attachment_template['title'] = os.path.basename(pdf_path)
            attachment_template['path'] = pdf_path
            attachment_template['parentItem'] = parent_key
            zot.create_items([attachment_template])
        except Exception as e:
            print(f"⚠️ 链接文件失败: {e}")

    # 3. 上传笔记
    try:
        note_template = zot.item_template('note')
        html_note = f"<h1>{meta['title']}</h1><hr>{note_content.replace(chr(10), '<br>')}"
        note_template['note'] = html_note
        note_template['parentItem'] = parent_key
        zot.create_items([note_template])
    except Exception as e:
        print(f"⚠️ 笔记上传失败: {e}")
    
    return parent_key
