from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
from typing import Tuple

class ModelOperations:
    def __init__(self, use_roberta: bool = True):
        self.use_roberta = use_roberta  # Now respects the input flag
        self.bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        
        if self.use_roberta:
            print("Using RoBERTa-large for NLI (roberta-large-mnli)")  # Debug print
            self.nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
            self.nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        else:
            self.nli_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
            self.nli_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
            print("[INFO] Using NLI model: microsoft/deberta-large-mnli")

    def get_entailment_scores(self, query: str, document: str) -> Tuple[float, float, float]:
        inputs = self.nli_tokenizer(document, query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()
            if not self.use_roberta:
                c, n, e = probs  # DeBERTa's order: contradiction, neutral, entailment
            else:
                e, n, c = probs  # RoBERTa's order
        return e, n, c
    
    def get_semantic_similarity(self, query: str, document: str) -> float:
        query_emb = self.bge_model.encode(query, convert_to_tensor=True)
        doc_emb = self.bge_model.encode(document, convert_to_tensor=True)
        return util.cos_sim(query_emb, doc_emb).item()
    
    def calculate_score(self, e: float, n: float, c: float, 
                       threshold: float = None, steepness: float = None) -> float:
        threshold = 0.14 if self.use_roberta else 0.12  # DeBERTa needs lower threshold
        steepness = 15.0 if self.use_roberta else 12.0  # Less steep for DeBERTa
        
        e_tensor = torch.tensor(e)
        c_tensor = torch.tensor(c)
        n_tensor = torch.tensor(n)
        
        C_WEIGHT = 1.4 if self.use_roberta else 1.4
        N_WEIGHT = 0.1
        MAX_SCORE = 1.5
        MIN_SCORE = -1.0
        
        penalty = 1.0 * torch.sigmoid(7 * (e_tensor - threshold))
        raw_score = (C_WEIGHT * torch.sigmoid(steepness * (c_tensor - threshold))) + (N_WEIGHT * n_tensor) - penalty
        return max(MIN_SCORE, min(MAX_SCORE, raw_score.item()))