import pandas as pd
import re
from typing import List, Tuple, Optional, Dict, Any
import random

# pytest tests/test_maths_catgen.py


MIN_TASK = "minimum"
MAX_TASK = "maximum"
AVG_TASK = "average"
SUM_TASK = "sum"
DIFF_TASK = "difference"
PROD_TASK = "product"
EXP_TASK = "exponential"
NUM_TASKS = 7
TASK_OVERFLOW = "overflow"


def get_maths_tasks():
    """
    These are mathematical tasks that we test the models on 
    """
    return [
        MIN_TASK,
        MAX_TASK, 
        AVG_TASK,
        SUM_TASK,
        DIFF_TASK,
        PROD_TASK,
        EXP_TASK
    ]

def get_prompt_template():
    """
    Returns the prompt template for a given mathematical task.
    """
    return "Answer minimally: Given the numbers {x} and {y} calculate the {task}"


def is_ground_truth_correct(answer: str, ground_truth: str) -> bool:
    """
    Returns True if the ground_truth appears as the final number in the answer, ignoring whitespace and punctuation.
    Accepts answers like '13', '13.', '13**', 'The answer is 13', '**13**', 'random text **13** random text', 'boxed{13}'.
    """

    # Remove trailing whitespace and punctuation
    answer_clean = answer.strip().rstrip('.!**')
    # Find all numbers in the answer (including negative numbers and those with commas)
    numbers = re.findall(r'-?[\d,]+', answer_clean)
    
    # Remove commas from the numbers for comparison
    numbers_clean = [num.replace(',', '') for num in numbers]

    answer_no_comma = " " + answer.replace(",", "") + " "

    return (
        ground_truth == answer_no_comma or
        "**"+ground_truth+"**" in answer or
        "boxed{"+ground_truth+"}" in answer or
        " "+ground_truth+" " in answer_no_comma  or
        " "+ground_truth+"\n" in answer_no_comma  or
        " "+ground_truth+")" in answer_no_comma  or
        " "+ground_truth+"." in answer_no_comma  or
        (
            numbers_clean and
            _is_last_number_close(numbers_clean[-1], ground_truth)
        )
    )

def _is_last_number_close(last_number: str, ground_truth: str) -> bool:
    try:
        return abs(float(last_number) - float(ground_truth)) < 0.001
    except (ValueError, TypeError):
        return False


def generate_number_pairs(n_examples: int = 200, 
                         min_val: int = 1, 
                         max_val: int = 99,
                         include_negatives: bool = False,
                         seed: int = 42) -> List[Tuple[int, int]]:
    """Generate diverse number pairs for testing"""
    random.seed(seed)
    pairs = []
    
    # Strategy: Mix of different number ranges for variety
    ranges = [
        (1, 9),      # Single digits
        (10, 99),    # Double digits
        (1, 99),     # Mixed
    ]
    
    if include_negatives:
        ranges.extend([
            (-99, -1),   # Negative numbers
            (-50, 50),   # Mixed positive/negative
        ])
    
    examples_per_range = n_examples // len(ranges)
    
    for min_r, max_r in ranges:
        for _ in range(examples_per_range):
            x = random.randint(min_r, max_r)
            y = random.randint(min_r, max_r)
            pairs.append((x, y))
    
    # Fill remaining with random pairs from full range
    while len(pairs) < n_examples:
        x = random.randint(min_val, max_val)
        y = random.randint(min_val, max_val)
        pairs.append((x, y))
    
    random.shuffle(pairs)
    return pairs[:n_examples]

def calculate_ground_truth(x: int, y: int, operation: str) -> str:
    """Calculate the correct answer for a given operation"""
    if operation == MIN_TASK:
        return str(min(x, y))
    elif operation == MAX_TASK:
        return str(max(x, y))
    elif operation == SUM_TASK:
        return str(x + y)
    elif operation == DIFF_TASK:
        return str(abs(x - y))  # Assuming absolute difference
    elif operation == PROD_TASK:
        return str(x * y)
    elif operation == AVG_TASK:
        return str((x + y) / 2)
    elif operation == EXP_TASK:
        # Limit exponential to prevent overflow
        try:
            result = x ** y
            # Cap at reasonable size
            if result > 10**15:
                return TASK_OVERFLOW
            return str(result)
        except:
            return TASK_OVERFLOW
    else:
        raise ValueError(f"Unknown operation: {operation}")

def generate_synthetic_data(tasks, prompt_template, n_examples_per_task: int = 200) -> pd.DataFrame:
    """Generate synthetic data for all tasks"""
    
    all_data = []
    
    for task in tasks:
        
        # For exponential, use smaller Y values to prevent overflow
        if task == EXP_TASK:
            pairs = generate_number_pairs(n_examples_per_task, min_val=2, max_val=15)
            # Limit Y further for exponential
            pairs = [(x, min(y, 10)) for x, y in pairs]
        else:
            pairs = generate_number_pairs(n_examples_per_task)
        
        for x, y in pairs:
            prompt = prompt_template.format(x=x, y=y, task=task)
            ground_truth = calculate_ground_truth(x, y, task)
            
            # Skip overflow cases
            if ground_truth == TASK_OVERFLOW:
                continue
                
            all_data.append({
                "task": task,
                "x": x,
                "y": y,
                "prompt": prompt,
                "ground_truth": ground_truth
            })
    
    df = pd.DataFrame(all_data)
    print(f"\nGenerated {len(df)} total examples across {len(tasks)} tasks")
    print(f"Examples per task: {df['task'].value_counts().to_dict()}")
    
    return df

def generate_synthetic_matrix(prompt_template, n_examples: int = 200, n_tasks: int = NUM_TASKS ) -> pd.DataFrame:
    """Generate synthetic matrix data for all tasks"""
    
    all_data = []
    
    pairs = generate_number_pairs(n_examples)
    use_exponential = n_tasks >= NUM_TASKS
    if use_exponential:
        # Given exponential, use smaller values to prevent overflow
        pairs = [(min(x,21), min(y, 12)) for x, y in pairs]

    for x, y in pairs:
        prompt = prompt_template.format(x=x, y=y, task="{task}")

        min_ground_truth = calculate_ground_truth(x, y, MIN_TASK)
        max_ground_truth = calculate_ground_truth(x, y, MAX_TASK) 
        avg_ground_truth = calculate_ground_truth(x, y, AVG_TASK)
        sum_ground_truth = calculate_ground_truth(x, y, SUM_TASK)
        diff_ground_truth = calculate_ground_truth(x, y, DIFF_TASK)
        prod_ground_truth = calculate_ground_truth(x, y, PROD_TASK)
        exp_ground_truth = calculate_ground_truth(x, y, EXP_TASK)

        task_ground_truths = [
            {"task": MIN_TASK, "ground_truth": min_ground_truth},
            {"task": MAX_TASK, "ground_truth": max_ground_truth},
            {"task": AVG_TASK, "ground_truth": avg_ground_truth},
            {"task": SUM_TASK, "ground_truth": sum_ground_truth},
            {"task": DIFF_TASK, "ground_truth": diff_ground_truth},
            {"task": PROD_TASK, "ground_truth": prod_ground_truth},
        ]
        if use_exponential:
            task_ground_truths += [
                {"task": EXP_TASK, "ground_truth": exp_ground_truth}
                ]
        
        all_data.append({
            "x": x,
            "y": y,
            "prompt": prompt,
            "task_ground_truths": task_ground_truths
        })
    
    df = pd.DataFrame(all_data)
    print(f"\nGenerated {len(df)} total examples across {n_tasks} tasks")
    
    return df
