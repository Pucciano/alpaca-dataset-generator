import torch
from typing import List, Dict, Any
from collections import Counter
import random
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from utils import preprocess_text, generate_gpt2_output, generate_t5_output, extract_keywords, is_valid_output
from config import CONFIG

def generate_dataset(num_examples: int, input_texts: List[str], models: Dict, device: torch.device) -> List[Dict[str, Any]]:
    dataset = []
    instruction_type_counts = Counter()

    INSTRUCTION_TYPES = CONFIG['instruction_types']

    with tqdm(total=num_examples, desc="Generating examples", unit="example") as pbar:
        while len(dataset) < num_examples:
            instruction_type, instruction = random.choice(INSTRUCTION_TYPES)

            if instruction_type_counts[instruction_type] >= num_examples // len(INSTRUCTION_TYPES):
                continue

            input_text = random.choice(input_texts)
            sentences = sent_tokenize(input_text)
            num_sentences = min(len(sentences), random.randint(2, 4))
            text_sample = " ".join(sentences[:num_sentences])
            text_sample = preprocess_text(text_sample)

            example = generate_example(models, device, instruction_type, instruction, text_sample)
            if example:
                dataset.append(example)
                instruction_type_counts[instruction_type] += 1
                pbar.update(1)

    return dataset

def generate_example(models: Dict, device: torch.device, instruction_type: str, instruction: str, input_text: str) -> Dict[str, Any]:
    max_attempts = 5
    for _ in range(max_attempts):
        if instruction_type in ["concept_explanation", "provide_example", "analogy", "application"]:
            output = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], f"{instruction}\n\nConcept: {input_text}\n\nExplanation:", device, max_length=100)
        elif instruction_type in ["generate_question", "misconception"]:
            output = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], f"{instruction}\n\nConcept: {input_text}\n\nOutput:", device, max_length=50)
        elif instruction_type == "learning_path":
            output = generate_t5_output(models["t5_tokenizer"], models["t5_model"], "generate learning path", f"Concept: {input_text}", device, max_length=150)
        elif instruction_type == "quiz_generation":
            output = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], f"{instruction}\n\nConcept: {input_text}\n\nQuiz question:", device, max_length=200)
        elif instruction_type == "concept_relation":
            related_concept = extract_keywords(input_text, n=1)[0]
            output = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], f"{instruction}\n\nConcept 1: {input_text}\nConcept 2: {related_concept}\n\nRelation:", device, max_length=100)
        elif instruction_type == "difficulty_assessment":
            output = generate_t5_output(models["t5_tokenizer"], models["t5_model"], "assess difficulty", f"Concept: {input_text}", device, max_length=100)

        if is_valid_output(instruction_type, output, input_text, models["sentence_model"]):
            return {
                "instruction": instruction,
                "input": input_text,
                "output": output
            }
    return None
