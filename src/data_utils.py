from datasets import load_dataset

def format_data(example, tokenizer, system_prompt, max_length=6144):
    """Định dạng dữ liệu cho Qwen chat template."""
    assistant_content = (
        f"<thinking>\n{example['deepseek_reasoning']}\n</thinking>\n"
        f"<|begin_of_solution|>\n{example['ground_truth_solution']}\n<|end_of_solution|>"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['problem']},
        {"role": "assistant", "content": assistant_content}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer(text, max_length=max_length, truncation=False, padding=False)

def get_processed_dataset(dataset_name, tokenizer, system_prompt):
    ds = load_dataset(dataset_name, "metadata", split="train[:15000]")
    # Lọc và map
    dataset = ds.filter(lambda x: x['domain'] == 'math')
    tokenized = dataset.map(lambda x: format_data(x, tokenizer, system_prompt), num_proc=4)
    return tokenized.filter(lambda x: len(x['input_ids']) <= 6144)