# data_processing.py

import json

def load_data(dataset_name, is_dev=False):
    if dataset_name == "ARC-Challenge":
        file_path = 'data/data_arc_dev.jsonl' if is_dev else 'data/data_arc.jsonl'
    elif dataset_name == "CommonQA":
        file_path = 'data/data_common_qa_dev.jsonl' if is_dev else 'data/data_common_qa.jsonl' 
    else:
        raise ValueError("Unsupported dataset.")
    
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    return data

def get_examples(dev_data, k):
    examples = []
    for j in range(min(k, len(dev_data))):
        choice_texts = [c["text"] for c in dev_data[j]["question"]["choices"]]
        answer_label = dev_data[j]["answerKey"]
        answer_index = ord(answer_label) - ord('A')
        answer_text = choice_texts[answer_index]
        examples.append({
            "question": dev_data[j]["question"]["stem"],
            "choices": choice_texts,
            "answer": answer_label,
            "answer_text": answer_text
        })
    return examples
