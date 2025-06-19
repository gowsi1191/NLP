import json
import numpy as np
from scipy import stats
from collections import defaultdict

def analyze_data(data):
    # Initialize data structures
    config_counts_all = defaultdict(int)
    query_type_data = defaultdict(lambda: {'BGE': defaultdict(list), 'Roberta': defaultdict(list)})
    config_query_data = defaultdict(lambda: defaultdict(lambda: {'BGE': defaultdict(list), 'Roberta': defaultdict(list)}))
    best_configs = {}
    query_types = set()
    
    # Configuration distribution tracking
    config_distribution = {
        'all': defaultdict(int),
        'by_query_type': defaultdict(lambda: defaultdict(int))
    }

    # First pass: Find best config for each document
    for doc_id, doc_data in data.items():
        best_ndcg = -1
        best_config = None
        best_metrics = None
        query_type = None

        for config_key, config_data in doc_data.items():
            current_type = config_data.get('query_type', 'default').lower()
            query_type = current_type
            query_types.add(query_type)
            
            current_ndcg = config_data['Roberta']['metrics']['NDCG@3']
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
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

    most_common_config = max(config_counts_all.items(), key=lambda x: x[1]) if config_counts_all else (None, 0)

    reports = {
        "query_type_analysis": query_stats,
        "combined_analysis": {
            "metadata": {
                "total_documents": len(best_configs),
                "best_configuration": most_common_config[0],
                "docs_with_best_config": most_common_config[1],
                "detected_query_types": sorted(query_types),
                **config_distribution_reports  # Include all config distribution reports
            },
            "overall_comparison": {
                "BGE": calculate_means(overall_bge),
                "Roberta": calculate_means(overall_roberta),
                "p_values": calculate_pvalues(overall_bge, overall_roberta)
            }
        }
    }

    return reports

if __name__ == "__main__":
    # Example usage
    with open('evaluation_results.json') as f:
        evaluation_data = json.load(f)
    
    analysis_reports = analyze_data(evaluation_data)
    
    with open('enhanced_query_type_analysis.json', 'w') as f:
        json.dump(analysis_reports, f, indent=2)
    
    print("Analysis complete. Reports saved to enhanced_query_type_analysis.json")