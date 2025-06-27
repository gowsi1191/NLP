from data_processing import load_data, prepare_data_for_metrics
from model_operations import ModelOperations
from metrics_calculation import MetricsCalculator
import json
import os
import torch

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full paths to the JSON files
file_names = [
    os.path.join(script_dir, "json", "implicit.json"),
    # os.path.join(script_dir, "json","explicit.json"),
    # os.path.join(script_dir, "json","comparative_not.json"),
    # os.path.join(script_dir, "json","scope.json"),
    # os.path.join(script_dir, "json","prohibition.json")
]

def main():
    try:
        # Load and prepare data from multiple files
        combined_data = load_data(file_names)
        if not combined_data:
            raise ValueError("No valid data loaded from the provided files")
        
        queries, relevance_data = prepare_data_for_metrics(combined_data)
        metrics_calc = MetricsCalculator()

        # Hardcoded models
        AVAILABLE_MODELS = {
            "roberta-large-mnli": "RoBERTa-large (MNLI)",
            "microsoft/deberta-large-mnli": "DeBERTa-large (MNLI)",
            "prajjwal1/albert-base-v2-mnli": "ALBERT-base (MNLI)",
            "pritamdeka/PubMedBERT-MNLI-MedNLI": "Pub-MedBERT (MNLI → MedNLI)",
            "facebook/bart-large-mnli": "BART-large (MNLI)", 
            "cross-encoder/nli-deberta-base": "Cross-Encoder (DeBERTa-base NLI)",
            "typeform/distilbert-base-uncased-mnli": "DistilBERT-base (MNLI)"
        }

        for model_id in AVAILABLE_MODELS:
            print(f"Evaluating: {AVAILABLE_MODELS[model_id]}")
            model_ops = ModelOperations(nli_model_name=model_id)
            results = metrics_calc.evaluate_thresholds(queries, relevance_data, model_ops)

            suffix = model_id.replace("/", "-")
            output_path = os.path.join(script_dir, f"evaluation_results_{suffix}.json")

            def make_json_serializable(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.item() if obj.numel() == 1 else obj.tolist()
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(x) for x in obj]
                else:
                    return obj

            with open(output_path, 'w') as f:
                json.dump(make_json_serializable(results), f, indent=2)
            print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
