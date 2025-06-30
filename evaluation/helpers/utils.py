from transformers import AutoModelForCausalLM, AutoTokenizer


class TextGenerator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token


def generate_text(text_gen:TextGenerator, prompt, max_length=50):
    inputs = text_gen.tokenizer(prompt, return_tensors="pt", padding=True)
    output = text_gen.model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_length,
        num_return_sequences=1,
        pad_token_id=text_gen.tokenizer.pad_token_id
    )

    generated_text = text_gen.tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text.strip()


def get_answer_from_model_on_gsm8k(text_gen:TextGenerator, dataset, limit=10):
    results = []
    for example in dataset.select(range(limit)):
        question = example['question']
        
        # print(f"Question: {question}")
        generated_answer = generate_text(text_gen, question, max_length=50)
        # print(f"Answer: {generated_answer}")
        
        answer = example['answer']
        simple_answer = answer[answer.find("####") + len("####") + 1:]
        
        results.append({
            'question': question,
            'generated_answer': generated_answer,
            'correct_answer': simple_answer
        })
    return results


def evaluate_model_on_gsm8k(model:TextGenerator, dataset, limit=10):
    results = get_answer_from_model_on_gsm8k(model, dataset, limit=limit)
    n = len(results)
    correct_count = 0
    for answer in results:
        correct_count += 1 if answer['correct_answer'] in answer['generated_answer'] else 0
    
    accuracy = correct_count / n
    return accuracy
    