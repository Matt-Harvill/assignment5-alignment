# Writeup

## 3 Measuring Zero-Shot MATH Performance

### Problem (math_baseline): 4 points
1. ✅
2. 
=== Evaluation Metrics ===
Total examples: 1319
Category 1 (correct format and answer): 233 (17.66%)
Category 2 (correct format, wrong answer): 432 (32.75%)
Category 3 (wrong format and answer): 654 (49.58%)

The parser actually feels wrong. It doesn't capture the answer formatted properly when there are extra newlines. In the cases where the format=1 but reward=0, there are no extra newlines.
Examples can be found in evaluation_results.json but for correctly formatted answers, sometimes the number was a $ amount, others it was purely the number, and even one example I viewed had the number in a sentence.

3. Overall, Qwen's 0-shot performance on GSM8k test is 17.66% so there's definitely room for improvement.

## 4 Supervised Finetuning for MATH

## 7 Group Relative Policy Optimization

### Problem (compute_group_normalized_rewards): Group normalization (2 points) ✅