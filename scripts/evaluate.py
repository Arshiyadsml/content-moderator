# Run evaluation on 100-example dataset

import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from src.config import DATA_DIR
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from anthropic import Anthropic

def load_eval_dataset():
    """Load ground-truth examples"""
    eval_file = DATA_DIR / "eval_dataset.jsonl"
    
    if not eval_file.exists():
        print(f"⚠️  Eval dataset not found at {eval_file}")
        print("Creating minimal dataset for testing...")
        
        examples = [
            {
                "id": "eval_001",
                "content": "I love this product!",
                "content_type": "text",
                "ground_truth_label": "approve",
                "ground_truth_reasoning": "Positive sentiment, no violations",
                "policy_category": "general"
            },
            {
                "id": "eval_002",
                "content": "All [group] should be eliminated",
                "content_type": "text",
                "ground_truth_label": "reject",
                "ground_truth_reasoning": "Clear incitement to violence",
                "policy_category": "violence"
            },
            {
                "id": "eval_003",
                "content": "Check out this amazing weight loss pill!",
                "content_type": "text",
                "ground_truth_label": "flag",
                "ground_truth_reasoning": "Unsubstantiated health claim",
                "policy_category": "misinformation"
            },
        ]
        
        return examples
    
    examples = []
    with open(eval_file) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    return examples

def evaluate_simple(examples):
    """
    Simple evaluation: use Claude to make decisions.
    In production, would use your full ModeratorGraph.
    """
    client = Anthropic()
    
    results = []
    predictions = []
    ground_truths = []
    
    print(f"Evaluating {len(examples)} examples...\n")
    
    for i, example in enumerate(examples):
        prompt = f"""Content: {example['content']}

Decide if this should be:
- APPROVE: Safe, no violations
- FLAG: Minor violation, needs warning
- QUARANTINE: Should be removed, user warned
- REJECT: Should be removed, user banned
- ESCALATE: Uncertain, needs human review

Respond with ONLY the decision (one word)."""
        
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            decision = response.content[0].text.strip().upper()
            
            # Normalize decision
            if decision not in ["APPROVE", "FLAG", "QUARANTINE", "REJECT", "ESCALATE"]:
                decision = "ESCALATE"
            
            ground_truth = example.get("ground_truth_label", "approve").upper()
            
            predictions.append(decision)
            ground_truths.append(ground_truth)
            
            correct = "✓" if decision == ground_truth else "✗"
            print(f"[{i+1}/{len(examples)}] {correct} Pred: {decision:10} | GT: {ground_truth:10} | {example['content'][:40]}")
        
        except Exception as e:
            print(f"[{i+1}/{len(examples)}] ✗ Error: {e}")
            predictions.append("ESCALATE")
            ground_truths.append(example.get("ground_truth_label", "approve").upper())
    
    # Compute metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    accuracy = accuracy_score(ground_truths, predictions)
    
    # Get unique labels
    labels = list(set(ground_truths + predictions))
    
    precision_macro = precision_score(ground_truths, predictions, labels=labels, average="macro", zero_division=0)
    recall_macro = recall_score(ground_truths, predictions, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(ground_truths, predictions, labels=labels, average="macro", zero_division=0)
    
    print(f"Accuracy:        {accuracy:.3f}")
    print(f"Precision (avg): {precision_macro:.3f}")
    print(f"Recall (avg):    {recall_macro:.3f}")
    print(f"F1 (avg):        {f1_macro:.3f}")
    
    # Save results
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = results_dir / f"results_{timestamp}.jsonl"
    with open(results_file, "w") as f:
        for pred, gt, ex in zip(predictions, ground_truths, examples):
            f.write(json.dumps({
                "example_id": ex.get("id"),
                "predicted": pred,
                "ground_truth": gt,
                "correct": pred == gt
            }) + "\n")
    
    metrics_file = results_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_examples": len(examples),
            "accuracy": float(accuracy),
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1": float(f1_macro)
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {results_dir}")

if __name__ == "__main__":
    examples = load_eval_dataset()
    evaluate_simple(examples)
