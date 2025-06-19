import json
from typing import Dict, List, Tuple, Any, Union
import os

def load_data(file_paths: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """Load and combine multiple JSON data files or a single file"""
    combined_data = []
    
    # Handle both single file (str) or multiple files (list)
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    combined_data.append(data)
        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in file - {file_path}")
    
    return combined_data

def prepare_data_for_metrics(data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare data structures for metrics calculation"""
    queries = {}
    relevance_data = {}
    
    for item in data:
        query_id = item['query_id']
        query_type = item['query_type']
        query_text = item['query_text']
        
        documents = []
        relevance_labels = []
        
        for doc in item['documents']:
            documents.append(doc['text'])
            relevance_labels.append(doc['relevance'])
        
        queries[query_id] = {
            'query_text': query_text,
            'query_type': query_type,
            'documents': documents
        }
        
        relevance_data[query_id] = relevance_labels
    
    return queries, relevance_data