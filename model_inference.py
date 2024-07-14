# model_inference.py

import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# inference
def run_inference(data, examples, model_name, folder_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(model_name)
    model.to(device)
    
    results = []

    for i, item in tqdm(enumerate(data), desc="Processing", total=len(data), unit="questions"):
        question_id = item["id"]
        question_stem = item["question"]["stem"]
        choices = item["question"]["choices"]
        true_answer = item["answerKey"]

        # CP
        cp_raw_choice, _ = cloze_prompting(model, tokenizer, question_stem, choices, examples, normalization='Raw')
        cp_ln_choice, _ = cloze_prompting(model, tokenizer, question_stem, choices, examples, normalization='LN')
        cp_un_choice, _ = cloze_prompting(model, tokenizer, question_stem, choices, examples, normalization='UN')

        # MCP
        mcp_choice, mcp_response = multiple_choice_prompting(model, tokenizer, question_stem, choices, examples)

        results.append({
            "QuestionID": question_id,
            "Answer": true_answer,
            "CP_Raw": cp_raw_choice,
            "CP_UN": cp_un_choice,
            "CP_LN": cp_ln_choice,
            "MCP_output": mcp_choice,
            "MCP_Answer": mcp_response
        })

    return results

# Zero-Shot and Few-Shot template
def cloze_prompting(model, tokenizer, question, choices, examples=[], normalization='Raw'):
    device = next(model.parameters()).device
    prompt = ""
    for ex in examples:
        prompt += f"Question: {ex['question']}\nAnswer: {ex['answer_text']}\n\n"
    scores = []
    for choice in choices:
        complete_prompt = prompt + f"Question: {question}\nAnswer: {choice['text']}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": complete_prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            input_ids = model_inputs['input_ids']
            
            # Compute the score as the sum of log probabilities for all tokens in the answer
            score = 0.0
            for i in range(len(input_ids[0]) - 1):
                score += log_probs[0, i, input_ids[0, i + 1]].item()
            
            score = np.exp(score)  # Convert log probabilities to actual probabilities

        if normalization == 'UN':
            uncond_prompt = f"Answer: {choice['text']}"
            uncond_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": uncond_prompt}
            ]
            uncond_text = tokenizer.apply_chat_template(uncond_messages, tokenize=False, add_generation_prompt=True)
            uncond_model_inputs = tokenizer([uncond_text], return_tensors="pt").to(device)

            with torch.no_grad():
                uncond_outputs = model(**uncond_model_inputs)
                uncond_logits = uncond_outputs.logits
                uncond_log_probs = torch.log_softmax(uncond_logits, dim=-1)
                uncond_input_ids = uncond_model_inputs['input_ids']

                uncond_score = 0.0
                for i in range(len(uncond_input_ids[0]) - 1):
                    uncond_score += uncond_log_probs[0, i, uncond_input_ids[0, i + 1]].item()
                
                uncond_score = np.exp(uncond_score)  # Convert log probabilities to actual probabilities

            score /= uncond_score

        elif normalization == 'LN':
            length = model_inputs.input_ids.size(1)
            score = score ** (1 / length)

        scores.append(score)
    
    best_choice = choices[scores.index(max(scores))]['label']
    return best_choice, scores

def multiple_choice_prompting(model, tokenizer, question, choices, examples=[]):
    device = next(model.parameters()).device
    prompt = ""
    for ex in examples:
        ex_prompt = f"Question: {ex['question']}\n"
        for idx, choice in enumerate(ex['choices']):
            ex_prompt += f"{chr(65 + idx)}. {choice}\n"
        ex_prompt += f"Answer: {ex['answer']}\n\n"
        prompt += ex_prompt

    prompt += f"Question: {question}\n"
    for idx, choice in enumerate(choices):
        prompt += f"{chr(65 + idx)}. {choice['text']}\n"
    prompt += "Answer:"

    messages = [
        {"role": "system", "content": "Below, I will give you a question. Please choose the correct answer from this question and output the correct option letter. Your answer should only contain one letter"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if len(response) >= 1 and response[0].isalpha():
        best_choice = response[0]
    else:
        best_choice = 'Invalid'

    return best_choice, response
