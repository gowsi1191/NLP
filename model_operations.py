from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
from typing import Tuple, List

class ModelOperations:
    # Only models that natively support entailment/neutral/contradiction
    AVAILABLE_MODELS = {
        # General-purpose NLI models
        "roberta-large-mnli": "RoBERTa-large (MNLI)",
        "microsoft/deberta-large-mnli": "DeBERTa-large (MNLI)",
        "prajjwal1/albert-base-v2-mnli": "ALBERT-base (MNLI)",
        
        # Biomedical NLI model
        "pritamdeka/PubMedBERT-MNLI-MedNLI": "Pub-MedBERT (MNLI → MedNLI)",

        #gemini
            "facebook/bart-large-mnli": "BART-large (MNLI)", 
    "cross-encoder/nli-deberta-base": "Cross-Encoder (DeBERTa-base NLI)",
    "typeform/distilbert-base-uncased-mnli": "DistilBERT-base (MNLI)"
    }



    @staticmethod
    def list_available_models() -> List[Tuple[str, str]]:
        """
        Returns a list of tuples: (Hugging Face model ID, human-readable name)
        """
        return list(ModelOperations.AVAILABLE_MODELS.items())

    def __init__(self, nli_model_name: str = "roberta-large-mnli"):
        # Verify chosen model
        if nli_model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{nli_model_name}' is not supported. Choose from:\n"
                             + "\n".join(f"- {name} ({key})" 
                                         for key, name in self.list_available_models()))
        self.nli_model_name = nli_model_name

        # Load models
        self.bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

        # Determine label order mapping
        if "roberta" in nli_model_name:
            self.label_order = ("entailment", "neutral", "contradiction")
        elif "deberta" in nli_model_name:
            self.label_order = ("contradiction", "neutral", "entailment")
        else:
            self.label_order = ("entailment", "neutral", "contradiction")

        print(f"[INFO] Loaded NLI model '{nli_model_name}' ({self.AVAILABLE_MODELS[nli_model_name]}) "
              f"with label order {self.label_order}")

    def get_entailment_scores(self, query: str, document: str) -> Tuple[float, float, float]:
        inputs = self.nli_tokenizer(document, query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        if len(probs) != 3:
            raise ValueError(f"Model {self.nli_model_name} does not produce ENC labels (got {len(probs)} outputs)")

        label_mapping = dict(zip(self.label_order, probs))
        return label_mapping["entailment"], label_mapping["neutral"], label_mapping["contradiction"]

    def get_semantic_similarity(self, query: str, document: str) -> float:
        query_emb = self.bge_model.encode(query, convert_to_tensor=True)
        doc_emb = self.bge_model.encode(document, convert_to_tensor=True)
        return util.cos_sim(query_emb, doc_emb).item()
    
    def calculate_score(self, e: float, n: float, c: float, 
                        threshold: float = 0.12, steepness: float = 15.0, 
                        mode: str = "explicit") -> float:
        e_tensor = torch.tensor(e)
        c_tensor = torch.tensor(c)
        n_tensor = torch.tensor(n)

        MAX_SCORE = 1.5
        MIN_SCORE = -1.0

        model = self.nli_model_name

        if mode == "explicit_NOT":
            if model == "cross-encoder/nli-deberta-base":
                score = torch.sigmoid(15 * n_tensor) + c_tensor - torch.sigmoid(15 * e_tensor)
            elif model == "facebook/bart-large-mnli":
                score = torch.sigmoid(15 * n_tensor) + c_tensor - torch.sigmoid(15 * e_tensor)
            elif model == "microsoft/deberta-large-mnli":
                score = (n_tensor > 0.5).float() + c_tensor - (e_tensor > 0.5).float()
            elif model == "roberta-large-mnli":
                score = (n_tensor > 0.7).float() + c_tensor - (e_tensor > 0.7).float()

        elif mode == "implicit_NOT":
            if model == "cross-encoder/nli-deberta-base":
                score = (n_tensor > 0.5).float() + c_tensor - (e_tensor > 0.5).float()
            elif model == "facebook/bart-large-mnli":
                score = torch.sigmoid(15 * n_tensor) + c_tensor - torch.sigmoid(15 * e_tensor)
            elif model == "microsoft/deberta-large-mnli":
                score = torch.sigmoid(5 * n_tensor) + c_tensor - torch.sigmoid(5 * e_tensor)
            elif model == "roberta-large-mnli":
                score = (n_tensor > 0.3).float() + c_tensor - (e_tensor > 0.3).float()

        elif mode == "comparative_NOT":
            if model == "cross-encoder/nli-deberta-base":
                score = torch.sigmoid(15 * n_tensor) + c_tensor - torch.sigmoid(15 * e_tensor)
            elif model == "facebook/bart-large-mnli":
                score = (n_tensor > 0.3).float() + c_tensor - (e_tensor > 0.3).float()
            elif model == "microsoft/deberta-large-mnli":
                score = torch.sigmoid(5 * n_tensor) + c_tensor - torch.sigmoid(5 * e_tensor)
            elif model == "roberta-large-mnli":
                score = torch.sigmoid(15 * n_tensor) + c_tensor - torch.sigmoid(15 * e_tensor)

        if score is None:
            raise ValueError("Score not calculated: check model and mode logic")

        return max(MIN_SCORE, min(MAX_SCORE, score.item()))




