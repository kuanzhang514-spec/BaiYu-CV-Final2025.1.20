'''
读取 results1.json
调表：
CLIP置信度区间	样本数（有CLIP组）	正确样本数	正确率
≥0.25
0.20~0.25
<0.20

'''

import json
from collections import defaultdict


def analyze_clip_confidence_intervals(json_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化统计字典
    stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    # 遍历所有结果
    for result in data['results']:
        # 检查是否有CLIP置信度（排除clip_calls=0或没有clip_scores的样本）
        if result.get('clip_calls', 0) > 0 and result.get('confidence', 0) >= 0:
            confidence = result['confidence']

            # 确定置信度区间
            if confidence >= 0.25 and confidence < 1.0:
                interval = '≥0.25 && <1'
            elif confidence >= 0.20 and confidence < 0.25:
                interval = '0.20~0.25'
            elif confidence < 0.20:
                interval = '<0.20'
            else:
                # 处理其他情况（如confidence=1.0或异常值）
                continue

            # 更新统计
            stats[interval]['total'] += 1
            if result.get('is_correct', False):
                stats[interval]['correct'] += 1

    # 打印结果表格
    print("CLIP置信度区间\t样本数（有CLIP组）\t正确样本数\t正确率")
    print("-" * 60)

    # 定义区间顺序
    intervals_order = ['≥0.25 && <1', '0.20~0.25', '<0.20']

    for interval in intervals_order:
        if interval in stats:
            total = stats[interval]['total']
            correct = stats[interval]['correct']
            accuracy = correct / total if total > 0 else 0.0
            print(f"{interval}\t\t{total}\t\t{correct}\t\t{accuracy:.4f}")

    # 计算总计
    total_samples = sum(stats[interval]['total'] for interval in stats)
    total_correct = sum(stats[interval]['correct'] for interval in stats)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    print("-" * 60)
    print(f"总计\t\t{total_samples}\t\t{total_correct}\t\t{overall_accuracy:.4f}")

    return stats


# 使用函数分析JSON文件
if __name__ == "__main__":
    json_file_path = "results1.json"
    stats = analyze_clip_confidence_intervals(json_file_path)