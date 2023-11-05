from torchmetrics.text import SacreBLEUScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import MetricCollection
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import evaluate
from ..visualization.visualize import plot_metrics_hist


def compute_metrics(preds, targets, toxicity):
    """
    Compute evaluation metrics for the given predictions and targets.

    Args:
        preds: List of predicted texts.
        targets: List of target texts.
        toxicity: The toxicity metric (loaded).

    Returns:
        List of dictionaries containing metrics for each prediction-target pair.
    """
    
    metric_collection = MetricCollection({
        'BLEU-2': SacreBLEUScore(2),
        'BLEU-4': SacreBLEUScore(),
        'ROUGE-1': ROUGEScore(rouge_keys=('rouge1')),
        'ROUGE-2': ROUGEScore(rouge_keys=('rouge2')),
    })
    
    all_metrics = []
    avg_metrics = {}
    
    N = len(preds)
    for i in range(N):
        metric = metric_collection([preds[i]], [[targets[i]]])
        metric = {k: v.item() for k, v in metric.items()}
        all_metrics.append(metric)
        for k, v in metric.items():
            if k not in avg_metrics:
                avg_metrics[k] = v
            else:
                avg_metrics[k] += v
                
    
    for k, v in avg_metrics.items():
        
        avg_metrics[k] = round(v / N, 4)
        
        # Do not print recall and precision since the f1 score is present
        if k not in ['rouge1_recall', 'rouge2_recall', 'rouge1_precision', 'rouge2_precision']:
            print(k, ':', avg_metrics[k])
            
    tox_metric = toxicity.compute(predictions=preds, aggregation="ratio")
    print('Toxicity ratio :', tox_metric['toxicity_ratio'])
    
    return all_metrics


def evaluate_on_test(approach_type, toxicity):
    """
    Evaluate the performance of a given approach type.
    
    Args:
        approach_type: The type of approach to evaluate (baseline/transformer/t5).
        toxicity: The toxicity metric (loaded).
    """
    
    prediction_col = approach_type + '_result'
    target_col = 'detox_reference'
    df = pd.read_csv(f'./data/interim/{approach_type}_results.csv')
    
    print('Average metrics:')
    all_metrics = compute_metrics(df[prediction_col].to_list(), df[target_col].to_list(), toxicity)
    
    preds_scores = toxicity.compute(predictions=df[prediction_col].to_list())
    preds_toxicity_scores = [round(score, 4) for score in preds_scores["toxicity"]]
    
    plot_metrics_hist(all_metrics, preds_toxicity_scores, approach_type)
    