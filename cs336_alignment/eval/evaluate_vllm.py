"""
Write a script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH. This script
should (1) load the MATH validation examples from /data/a5-alignment/MATH/validation.jsonl,
(2) format them as string prompts to the language model using the r1_zero prompt, and (3) gen-
erate outputs for each example. This script should also (4) calculate evaluation metrics and
(5) serialize the examples, model generations, and corresponding evaluation scores to disk for
analysis in subsequent problems.
Evaluate a language model on a list of prompts,
compute evaluation metrics, and serialize results to disk.
Deliverable: A script to evaluate baseline zero-shot MATH performance.
"""
import argparse
import json
import os
from typing import Callable, List, Dict, Any
from vllm import LLM, SamplingParams
import re

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def load_dataset_from_jsonl(file_path: str):
    """
    Load a dataset from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the dataset examples
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def load_dataset(dataset_path: str):
    """
    Load a dataset from a local JSONL file.
    Args:
        dataset_path: Path to the JSONL file
    Returns:
        List of dictionaries containing the dataset examples
    """
    return load_dataset_from_jsonl(dataset_path)


def load_r1_zero_prompt():
    """
    Load the r1_zero prompt template.
    
    Returns:
        String containing the prompt template
    """
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'r1_zero.prompt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def format_prompt_with_r1_zero(question: str) -> str:
    """
    Format a question using the r1_zero prompt template.
    
    Args:
        question: The question to format
        
    Returns:
        Formatted prompt string
    """
    prompt_template = load_r1_zero_prompt()
    return prompt_template.format(question=question)


def calculate_reward_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for the three reward categories:
    (1) correct with both format and answer reward 1
    (2) format reward 1 and answer reward 0  
    (3) format reward 0 and answer reward 0
    
    Args:
        results: List of evaluation results with reward dictionaries
        
    Returns:
        Dictionary containing counts and percentages for each category
    """
    total_examples = len(results)
    
    # Initialize counters
    category_1_count = 0  # format_reward=1, answer_reward=1
    category_2_count = 0  # format_reward=1, answer_reward=0
    category_3_count = 0  # format_reward=0, answer_reward=0
    
    for result in results:
        reward = result['reward']
        format_reward = reward['format_reward']
        answer_reward = reward['answer_reward']
        
        if format_reward == 1.0 and answer_reward == 1.0:
            category_1_count += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            category_2_count += 1
        elif format_reward == 0.0 and answer_reward == 0.0:
            category_3_count += 1
        else:
            # This shouldn't happen with r1_zero_reward_fn, but just in case
            print(f"Warning: Unexpected reward combination: format={format_reward}, answer={answer_reward}")
    
    # Calculate percentages
    category_1_percentage = (category_1_count / total_examples) * 100
    category_2_percentage = (category_2_count / total_examples) * 100
    category_3_percentage = (category_3_count / total_examples) * 100
    
    metrics = {
        'total_examples': total_examples,
        'category_1_correct_format_and_answer': {
            'count': category_1_count,
            'percentage': round(category_1_percentage, 2)
        },
        'category_2_correct_format_wrong_answer': {
            'count': category_2_count,
            'percentage': round(category_2_percentage, 2)
        },
        'category_3_wrong_format_and_answer': {
            'count': category_3_count,
            'percentage': round(category_3_percentage, 2)
        }
    }
    
    return metrics


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
) -> Dict[str, Any]:
    print(f"Generating responses for {len(prompts)} prompts...")
    
    # Generate responses
    outputs = vllm_model.generate(prompts=prompts, sampling_params=eval_sampling_params)
    
    results = []
    
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]
        
        reward = reward_fn(generated_text, ground_truth)
        
        result = {
            'prompt': prompt,
            'generated_text': generated_text,
            'reward': reward,
            'ground_truth': ground_truth,
        }
        results.append(result)
    
    # Calculate reward metrics
    metrics = calculate_reward_metrics(results)
    
    return {
        'results': results,
        'metrics': metrics,
    }


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save evaluation results to disk.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Path to save the results
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a language model on a dataset")
    parser.add_argument(
        "-d", "--dataset", 
        type=str, 
        default="data/gsm8k/test.jsonl",
        help="Dataset path (local JSONL file path, default: data/gsm8k/test.jsonl)"
    )
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Model to evaluate (default: Qwen/Qwen2.5-Math-1.5B)"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="cs336_alignment/eval/results/evaluation_results.json",
        help="Output file path for results (default: cs336_alignment/eval/results/evaluation_results.json)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)"
    )
    return parser.parse_args()


def main():
    """Main function to run the evaluation script."""
    args = parse_args()
    
    # Load the dataset
    print(f"Loading dataset from: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} examples")
    
    # Print sample data for debugging
    print("\n=== Sample dataset structure ===")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Sample data: {sample}")
        if len(dataset) > 1:
            print(f"Second sample: {dataset[1]}")
    print("=== End sample data ===\n")
    
    # Limit examples if specified
    if args.max_examples:
        dataset = dataset[:args.max_examples]
        print(f"Limited to {len(dataset)} examples")
    
    # Extract questions and answers from GSM8K-style dataset
    questions = [example['question'] for example in dataset]
    ground_truths = [example['answer'] for example in dataset]
    if "gsm8k" in args.dataset:
        ground_truths = [answer.split("####")[-1] for answer in ground_truths]
    
    # Format prompts using r1_zero template
    print("Formatting prompts...")
    prompts = [format_prompt_with_r1_zero(question) for question in questions]
    
    # Create the LLM
    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Use 0 temperature for deterministic evaluation
        top_p=1.0, 
        max_tokens=1024, 
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
    )
    
    # Display metrics
    print("\n=== Evaluation Metrics ===")
    metrics = results['metrics']
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Category 1 (correct format and answer): {metrics['category_1_correct_format_and_answer']['count']} ({metrics['category_1_correct_format_and_answer']['percentage']}%)")
    print(f"Category 2 (correct format, wrong answer): {metrics['category_2_correct_format_wrong_answer']['count']} ({metrics['category_2_correct_format_wrong_answer']['percentage']}%)")
    print(f"Category 3 (wrong format and answer): {metrics['category_3_wrong_format_and_answer']['count']} ({metrics['category_3_wrong_format_and_answer']['percentage']}%)")
    print("=== End Metrics ===\n")
    
    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()