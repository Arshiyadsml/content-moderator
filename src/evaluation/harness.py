import json
import logging
from pathlib import Path
from datetime import datetime
from src.config import DATA_DIR
from src.models import EvalExample, EvalResult, MetricsReport
import pandas as pd

logger = logging.getLogger(__name__)

class EvaluationHarness:
    """Run system against ground-truth dataset"""
    
    def __init__(self, moderator_graph, db):
        self.graph = moderator_graph
        self.db = db
    
    def load_eval_dataset(self) -> list:
        """Load ground-truth examples"""
        eval_file = DATA_DIR / "eval_dataset.jsonl"
        examples = []
        
        with open(eval_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(EvalExample(**data))
        
        logger.info(f"Loaded {len(examples)} evaluation examples")
        return examples
    
    def run_eval(self, examples: list) -> list:
        """Run system on all examples"""
        results = []
        
        for i, example in enumerate(examples):
            logger.info(f"[{i+1}/{len(examples)}] Evaluating {example.id}...")
            
            # Run moderation
            request = {
                "request_id": f"eval_{example.id}",
                "content": example.content,
                "content_type": example.content_type,
                "user_id": "eval_user"
            }
            
            state = self.graph.run(request)
            
            # Compare to ground truth
            predicted = state["judge_decision"].get("decision")
            predicted_confidence = state["judge_decision"].get("confidence", 0.5)
            
            result = EvalResult(
                example_id=example.id,
                predicted_decision=predicted,
                predicted_confidence=predicted_confidence,
                ground_truth=example.ground_truth_label,
                correct=(predicted == example.ground_truth_label),
                match_confidence=predicted_confidence if predicted == example.ground_truth_label else 1.0 - predicted_confidence
            )
            
            results.append(result)
        
        return results
    
    def compute_metrics(self, results: list) -> MetricsReport:
        """Aggregate precision, recall, F1"""
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        predicted = [r.predicted_decision for r in results]
        ground_truth = [r.ground_truth for r in results]
        
        # Overall accuracy
        accuracy = accuracy_score(ground_truth, predicted)
        
        # Per-category metrics
        categories = set(ground_truth)
        precision_by_cat = {}
        recall_by_cat = {}
        f1_by_cat = {}
        
        for cat in categories:
            y_true = [1 if g == cat else 0 for g in ground_truth]
            y_pred = [1 if p == cat else 0 for p in predicted]
            
            precision_by_cat[cat] = precision_score(y_true, y_pred, zero_division=0)
            recall_by_cat[cat] = recall_score(y_true, y_pred, zero_division=0)
            f1_by_cat[cat] = f1_score(y_true, y_pred, zero_division=0)
        
        # Macro and weighted F1
        macro_f1 = sum(f1_by_cat.values()) / len(f1_by_cat) if f1_by_cat else 0
        
        report = MetricsReport(
            timestamp=datetime.now(),
            total_examples=len(results),
            accuracy=accuracy,
            precision=precision_by_cat,
            recall=recall_by_cat,
            f1=f1_by_cat,
            macro_f1=macro_f1,
            weighted_f1=sum(f1_by_cat.values()) / len(f1_by_cat)
        )
        
        return report
    
    def save_results(self, results: list, metrics: MetricsReport):
        """Persist results to file"""
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save per-example results
        results_file = results_dir / f"results_{timestamp}.jsonl"
        with open(results_file, "w") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")
        
        # Save metrics
        metrics_file = results_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics.model_dump(), f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_dir}")
        return results_file, metrics_file
