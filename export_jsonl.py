import sqlite3
import json


def export_to_alpaca_jsonl(db_path='hust_handbook.db', output_file='hust_handbook_alpaca.jsonl'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 提取所有合成好的数据列
    cursor.execute("SELECT kimi_qa, qwen_qa, deepseek_qa FROM handbook_chunks")
    rows = cursor.fetchall()

    total_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in rows:
            for col_data in row:
                if not col_data:
                    continue

                try:
                    # 数据库里存的是 JSON 字符串，先解析
                    qa_list = json.loads(col_data)

                    for qa in qa_list:
                        # 核心校验：确保 instruction 和 output 存在且不为空
                        if qa.get("instruction") and qa.get("output"):
                            # 构造 MiniMind 或一般微调框架通用的格式
                            entry = {
                                "instruction": "",
                                "input": qa["instruction"].strip(),
                                "output": qa["output"].strip()
                            }
                            # 写入文件，每行一个 JSON
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            total_count += 1
                except Exception as e:
                    print(f"解析失败跳过: {e}")

    conn.close()
    print(f"合并完成！共导出 {total_count} 条高质量问答数据至 {output_file}。")


def export_to_conversations_jsonl(db_path='hust_handbook.db', output_file='hust_handbook_sft.jsonl'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取三路模型合成的数据
    cursor.execute("SELECT kimi_qa, qwen_qa, deepseek_qa FROM handbook_chunks")
    rows = cursor.fetchall()

    total_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in rows:
            for col_data in row:
                if not col_data:
                    continue

                try:
                    qa_list = json.loads(col_data)
                    for qa in qa_list:
                        instruction = qa.get("instruction", "").strip()
                        output = qa.get("output", "").strip()

                        if instruction and output:
                            # 构造 conversations 格式
                            entry = {
                                "conversations": [
                                    {"role": "user", "content": instruction},
                                    {"role": "assistant", "content": output}
                                ]
                            }
                            # 写入 JSONL，每行一个独立对象
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            total_count += 1
                except Exception as e:
                    print(f"[*] 解析跳过: {e}")

    conn.close()
    print(f"导出完成！共计 {total_count} 条对话数据。")
    print(f"输出文件路径: {output_file}")

# 执行导出
export_to_alpaca_jsonl()
export_to_conversations_jsonl()