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



        if mode == "explicit_NOT":
            eps = 0.01  # for ratio stability
            # Option 1: Ratio formula (high Pearson)
            # return (n_tensor / (e_tensor + eps)) + c_tensor

            # Option 2: Sigmoid-based formula σ(15n) + c - σ(15e)
            return torch.sigmoid(15 * n_tensor) + c_tensor - torch.sigmoid(15 * e_tensor)

        #cross gives 41 facebook bart gives 44
        # elif mode == "implicit_NOT":
        # #     # Soft binary version of: (n > 0.3) + c - (e > 0.3)
        #     return torch.sigmoid(40 * (n_tensor - 0.3)) + c_tensor - torch.sigmoid(40 * (e_tensor - 0.3))
        
        elif mode == "implicit_NOT":
            # Smooth sigmoid-based version of: (n > 0.5) + c - (e > 0.5)
            return torch.sigmoid(40 * (n_tensor - 0.5)) + c_tensor - torch.sigmoid(40 * (e_tensor - 0.5))
        
        # cross gives 48 fb gives 31, prajwal gives 41
        elif mode == "implicit_NOT":
            # Smooth sigmoid-based formula: σ(10n) + c - σ(10e)
            return torch.sigmoid(10 * n_tensor) + c_tensor - torch.sigmoid(10 * e_tensor)






        elif mode == "comparative_NOT":
            # --- Entailment reward: scaled sigmoid rising after 0.03, near 1 by 0.1 ---
            return n_tensor + (n_tensor * c_tensor) - e_tensor


        elif mode == "prohibition_NOT":
            # Hardcoded values
            threshold = 0.02
            steepness = 30.0

            # Slight reward for high entailment
            e_reward = 0.5 * torch.sigmoid(25 * (e_tensor - 0.965))

            # Minimal penalty for neutral
            n_penalty = 0.2 * torch.sigmoid(40 * (n_tensor - 0.015))

            # Stronger penalty for contradiction
            c_penalty = 0.8 * torch.sigmoid(40 * (c_tensor - 0.02))

            raw_score = e_reward - n_penalty - c_penalty

        elif mode == "scope_NOT":
            threshold = 0.04
            steepness = 25.0

            # Penalty for high entailment
            e_penalty = 0.6 * torch.sigmoid(35 * (e_tensor - 0.90))

            # Mild penalty for neutral
            n_penalty = 0.3 * torch.sigmoid(30 * (n_tensor - 0.05))

            # Reward for contradiction
            c_reward = 1.2 * torch.sigmoid(steepness * (c_tensor - threshold))

            raw_score = c_reward - n_penalty - e_penalty

        else:
            # General mode: use sum of neutral and contradiction directly
            raw_score = n_tensor + c_tensor

        return max(MIN_SCORE, min(MAX_SCORE, raw_score.item()))


