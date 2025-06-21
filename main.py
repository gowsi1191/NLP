from data_processing import load_data, prepare_data_for_metrics
from model_operations import ModelOperations
from metrics_calculation import MetricsCalculator
import json
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full paths to the JSON files
file_names = [
    os.path.join(script_dir, "json", "explicit.json"),
    # os.path.join(script_dir, "json","implicit_1.json"),
        # os.path.join(script_dir, "json","comparative.json"),
        #     os.path.join(script_dir, "json","scope.json"),
        #         os.path.join(script_dir, "json","prohibition.json")
]

def main():
    try:
        # Load and prepare data from multiple files
        combined_data = load_data(file_names)
        if not combined_data:
            raise ValueError("No valid data loaded from the provided files")
            
        queries, relevance_data = prepare_data_for_metrics(combined_data)
        
        # Initialize components
        model_ops = ModelOperations(nli_model_name="roberta-large-mnli")
        metrics_calc = MetricsCalculator()
        
        # Evaluate with different thresholds and steepness
        results = metrics_calc.evaluate_thresholds(queries, relevance_data, model_ops)
        
        # # Print results
        # for query_id, query_results in results.items():
        #     print(f"\nQuery: {query_id}")
        #     print("=" * 50)
        #     for config, scores in query_results.items():
        #         print(f"\nConfiguration: {config}")
        #         print("-" * 30)
        #         print("BGE Model:")
        #         print(f"Ranking: {scores['BGE']['rank']}")
        #         print("Metrics:")
        #         for metric, value in scores['BGE']['metrics'].items():
        #             print(f"{metric}: {value:.4f}")
                
        #         print("\nRoberta Model:")
        #         print(f"Ranking: {scores['Roberta']['rank']}")
        #         print("Metrics:")
        #         for metric, value in scores['Roberta']['metrics'].items():
        #             print(f"{metric}: {value:.4f}")
        #         print()

        # Save results to JSON file
        output_path = os.path.join(script_dir, 'evaluation_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()