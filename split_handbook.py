import sqlite3
import re

def init_db(db_path='hust_handbook.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 预留 kimi, qwen, deepseek 三个字段存放合成的 JSONL 数据
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS handbook_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level1_title TEXT,
            level2_title TEXT,
            content TEXT,
            generate_num INTEGER,
            kimi_qa TEXT,
            qwen_qa TEXT,
            deepseek_qa TEXT
        )
    ''')
    conn.commit()
    return conn

def split_and_store(file_path, db_path='hust_handbook.db'):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    conn = init_db(db_path)
    cursor = conn.cursor()

    current_l1 = ""
    current_l2 = ""
    current_content = []

    def save_chunk(l1, l2, content_list):
        text = "".join(content_list).strip()
        if text:
            cursor.execute(
                "INSERT INTO handbook_chunks (level1_title, level2_title, content) VALUES (?, ?, ?)",
                (l1, l2, text)
            )

    for line in lines:
        # 匹配一级标题 #
        if re.match(r'^#\s', line):
            save_chunk(current_l1, current_l2, current_content)
            current_l1 = line.strip('# \n')
            current_l2 = "General" # 无二级标题时的默认标识
            current_content = []
        # 匹配二级标题 ##
        elif re.match(r'^##\s', line):
            save_chunk(current_l1, current_l2, current_content)
            current_l2 = line.strip('# \n')
            current_content = []
        else:
            current_content.append(line)

    # 保存最后一块
    save_chunk(current_l1, current_l2, current_content)
    conn.commit()
    conn.close()
    print("切分完成，数据已存入 SQLite。")


def update_generate_nums(db_path='hust_handbook.db', base_chars=400):
    """
    根据内容长度和章节重要性计算并填入 generate_num
    :param base_chars: 基础步长，即多少字符对应 1 条 QA
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 定义核心关键词（加权系数 1.5）
    core_keywords = ['学籍', '处分', '学位', '毕业', '奖学金', '申诉', '考试']
    # 2. 定义低价值关键词（加权系数 0.5，防止“附则”占太多坑）
    low_priority_keywords = ['附则', '校徽', '总则', '档案']

    # 获取所有记录
    cursor.execute("SELECT id, level1_title, level2_title, content FROM handbook_chunks")
    rows = cursor.fetchall()

    for row in rows:
        row_id, l1, l2, content = row
        char_count = len(content)

        # 基础采样数：向上取整，至少生成 1 条
        num = max(1, round(char_count / base_chars))

        # 计算权重系数
        weight = 1.0
        combined_title = (l1 + l2).lower()

        # 核心章节加权
        if any(key in combined_title for key in core_keywords):
            weight = 1.5
        # 低优先级章节降权
        elif any(key in combined_title for key in low_priority_keywords):
            weight = 0.5

        # 最终计算结果（四舍五入）
        final_num = int(round(num * weight))

        # 强制兜底：有内容至少出 1 条，长文不封顶但 LLM 有 Token 限制
        final_num = max(1, final_num)

        # 更新数据库
        cursor.execute(
            "UPDATE handbook_chunks SET generate_num = ? WHERE id = ?",
            (final_num, row_id)
        )

    conn.commit()
    conn.close()
    print(f"generate_num 计算更新完成。权重配置：基础步长 {base_chars} 字/条。")

# 执行
split_and_store('研究生手册.md')
update_generate_nums()