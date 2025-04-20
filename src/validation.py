from typing import List, Dict, Any
from tqdm import tqdm
from utils import is_valid_output

def validate_dataset(dataset: List[Dict[str, Any]], sentence_model) -> List[Dict[str, Any]]:
    validated_dataset = []
    print(f"Starting validation of {len(dataset)} examples")
    
    with tqdm(total=len(dataset), desc="Validating examples", unit="example") as pbar:
        for i, example in enumerate(dataset):
            if is_valid_output(example["instruction_type"], example["output"], example["input"], sentence_model):
                validated_dataset.append(example)
            else:
                print(f"Example {i} failed validation")
            pbar.update(1)
    
    print(f"Validation complete. {len(validated_dataset)} out of {len(dataset)} examples passed validation")
    return validated_dataset
