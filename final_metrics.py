import json
import numpy as np
from scipy import stats
from collections import defaultdict

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

def analyze_data(data):
    # Initialize data structures
    config_counts_all = defaultdict(int)
    query_type_data = defaultdict(lambda: {'BGE': defaultdict(list), 'Roberta': defaultdict(list)})
    config_query_data = defaultdict(lambda: defaultdict(lambda: {'BGE': defaultdict(list), 'Roberta': defaultdict(list)}))
    best_configs = {}
    query_types = set()
    
    # Metric-specific storage and composite scoring
    metric_specific_data = {
        'P@1': defaultdict(lambda: defaultdict(list)),
        'MRR': defaultdict(lambda: defaultdict(list)),
        'NDCG@3': defaultdict(lambda: defaultdict(list))
    }
    config_composite_scores = defaultdict(list)
    
    # Configuration distribution tracking
    config_distribution = {
        'all': defaultdict(int),
        'by_query_type': defaultdict(lambda: defaultdict(int))
    }

    # First pass: Process all documents and configurations
    for doc_id, doc_data in data.items():
        best_score = -1
        best_config = None
        best_metrics = None
        query_type = None

        for config_key, config_data in doc_data.items():
            current_type = config_data.get('query_type', 'default').lower()
            query_type = current_type
            query_types.add(current_type)
            
            # Store metric-specific data
            roberta_metrics = config_data['Roberta']['metrics']
            for metric in ['P@1', 'MRR', 'NDCG@3']:
                if metric in roberta_metrics:
                    value = roberta_metrics[metric]
                    if isinstance(value, (np.float_, np.float32, np.float64)):
                        value = float(value)
                    elif isinstance(value, (np.int_, np.int32, np.int64)):
                        value = int(value)
                    elif isinstance(value, np.bool_):
                        value = bool(value)
                    metric_specific_data[metric][config_key][current_type].append(value)
            
            # Calculate weighted composite score (40% P@1, 30% MRR, 30% NDCG@3)
            composite_score = (
                0.4 * roberta_metrics.get('P@1', 0) +
                0.3 * roberta_metrics.get('MRR', 0) +
                0.3 * roberta_metrics.get('NDCG@3', 0)
            )
            config_composite_scores[config_key].append(composite_score)
            
            if composite_score > best_score:
                best_score = composite_score
                best_config = config_key
                best_metrics = config_data

        if best_config:
            # Update configuration distributions
            config_counts_all[best_config] += 1
            config_distribution['all'][best_config] += 1
            config_distribution['by_query_type'][query_type][best_config] += 1
            
            best_configs[doc_id] = {
                'query_type': query_type,
                'config': best_config,
                'BGE_metrics': best_metrics['BGE']['metrics'],
                'Roberta_metrics': best_metrics['Roberta']['metrics']
            }

    # Second pass: Aggregate metrics by query type
    for doc_id, doc in best_configs.items():
        qtype = doc['query_type']
        
        for model in ['BGE', 'Roberta']:
            for metric, value in doc[f'{model}_metrics'].items():
                query_type_data[qtype][model][metric].append(value)
                config_query_data[doc['config']][qtype][model][metric].append(value)

    # Statistics calculation functions
    def calculate_means(metrics_dict):
        return {metric: float(np.mean(values)) for metric, values in metrics_dict.items()}

    def calculate_pvalues(bge_metrics, roberta_metrics):
        pvals = {}
        common_metrics = set(bge_metrics.keys()) & set(roberta_metrics.keys())
        for metric in common_metrics:
            try:
                _, pval = stats.ttest_rel(bge_metrics[metric], roberta_metrics[metric])
                pvals[metric] = float(pval) if not np.isnan(pval) else None
            except:
                pvals[metric] = None
        return pvals

    # Prepare configuration distribution reports
    config_distribution_reports = {
        'config_distribution_all': dict(config_distribution['all'])
    }
    
    for qtype in query_types:
        config_distribution_reports[f'config_distribution_{qtype}'] = dict(config_distribution['by_query_type'][qtype])

    # Calculate composite scores for each configuration
    config_composite_means = {config: float(np.mean(scores)) for config, scores in config_composite_scores.items()}
    best_composite_config = max(config_composite_means.items(), key=lambda x: x[1]) if config_composite_means else (None, 0)

    # Query type-wise analysis (primary focus)
    query_stats = {}
    for qtype in sorted(query_types):
        if query_type_data[qtype]['BGE']:
            query_stats[qtype] = {
                'count': len(next(iter(query_type_data[qtype]['BGE'].values()))),
                'BGE': calculate_means(query_type_data[qtype]['BGE']),
                'Roberta': calculate_means(query_type_data[qtype]['Roberta']),
                'p_values': calculate_pvalues(query_type_data[qtype]['BGE'], query_type_data[qtype]['Roberta'])
            }

    # Combined overall analysis
    overall_bge = defaultdict(list)
    overall_roberta = defaultdict(list)
    for doc in best_configs.values():
        for metric, value in doc['BGE_metrics'].items():
            overall_bge[metric].append(value)
        for metric, value in doc['Roberta_metrics'].items():
            overall_roberta[metric].append(value)

    # Generate metric-specific reports
    metric_reports = {}
    for metric, metric_data in metric_specific_data.items():
        metric_stats = {}
        for qtype in query_types:
            qtype_data = {k: v[qtype] for k, v in metric_data.items() if qtype in v}
            if qtype_data:
                metric_stats[qtype] = {
                    'count': len(next(iter(qtype_data.values()))),
                    'config_means': {config: float(np.mean(vals)) for config, vals in qtype_data.items()},
                    'top_configs': sorted(
                        [(config, float(np.mean(vals))) for config, vals in qtype_data.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]  # Top 5 configs per metric per query type
                }
        
        metric_reports[metric] = {
            'overall_means': {config: float(np.mean(sum(vals.values(), []))) for config, vals in metric_data.items()},
            'by_query_type': metric_stats
        }

    # Main report with all data
    reports = {
        "query_type_analysis": query_stats,
        "combined_analysis": {
            "metadata": {
                "total_documents": len(best_configs),
                "best_configuration": best_composite_config[0],
                "best_configuration_score": best_composite_config[1],
                "config_composite_scores": config_composite_means,
                "docs_with_best_config": config_counts_all.get(best_composite_config[0], 0),
                "detected_query_types": sorted(query_types),
                "selection_criteria": "Weighted composite score (P@1:40%, MRR:30%, NDCG@3:30%)",
                **config_distribution_reports
            },
            "overall_comparison": {
                "BGE": calculate_means(overall_bge),
                "Roberta": calculate_means(overall_roberta),
                "p_values": calculate_pvalues(overall_bge, overall_roberta)
            }
        },
        "metric_specific_analysis": metric_reports
    }

    return reports

if __name__ == "__main__":
    # Load evaluation data
    with open('evaluation_results.json') as f:
        evaluation_data = json.load(f)
    
    # Generate analysis reports
    analysis_reports = analyze_data(evaluation_data)
    
    # Save single comprehensive report
    with open('enhanced_query_type_analysis.json', 'w') as f:
        json.dump(analysis_reports, f, indent=2, cls=NumpyEncoder)
    
    print("Analysis complete. Report saved to enhanced_query_type_analysis.json")