from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
from typing import Tuple  # Added this import

class ModelOperations:
    def __init__(self):
        self.logic_bert_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.logic_bert = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        self.bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    
    def get_entailment_scores(self, query: str, document: str) -> Tuple[float, float, float]:
        """Get entailment, neutral, and contradiction scores"""
        inputs = self.logic_bert_tokenizer(document, query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.logic_bert(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()
            e, n, c = probs
        return e, n, c
    
    def get_semantic_similarity(self, query: str, document: str) -> float:
        """Get BGE semantic similarity score"""
        query_emb = self.bge_model.encode(query, convert_to_tensor=True)
        doc_emb = self.bge_model.encode(document, convert_to_tensor=True)
        return util.cos_sim(query_emb, doc_emb).item()
    
    def calculate_score(self, e: float, n: float, c: float, 
                       threshold: float = 0.14, steepness: float = 15.0) -> float:
        """Dynamic sigmoid-based scoring function with adjustable threshold and steepness"""
        e_tensor = torch.tensor(e) if not isinstance(e, torch.Tensor) else e
        c_tensor = torch.tensor(c) if not isinstance(c, torch.Tensor) else c
        n_tensor = torch.tensor(n) if not isinstance(n, torch.Tensor) else n
        
        MAX_SCORE = 1.5
        MIN_SCORE = -1.0
        C_WEIGHT = 1.4
        N_WEIGHT = 0.1
        
        penalty = 1.0 * torch.sigmoid(7 * (e_tensor - threshold))
        raw_score = (C_WEIGHT * torch.sigmoid(steepness * (c_tensor - threshold))) + (N_WEIGHT * n_tensor) - penalty
        return max(MIN_SCORE, min(MAX_SCORE, raw_score.item()))