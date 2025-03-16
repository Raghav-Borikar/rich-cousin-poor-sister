# src/preprocessing/tokenizer.py
import torch
from transformers import AutoTokenizer

class MultilingualTokenizer:
    """Tokenizer for Hindi-Chhattisgarhi language pair"""
    
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", src_lang="hin_Deva", tgt_lang="hne_Deva"):
        """
        Initialize the tokenizer
        
        Args:
            model_name: Model name from HuggingFace hub
            src_lang: Source language code (Hindi)
            tgt_lang: Target language code (Chhattisgarhi, using Hindi as proxy since it's not in NLLB)
        """
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def encode_src(self, text, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        """Encode source text (Hindi)"""
        return self.tokenizer(text, padding=padding, truncation=truncation, 
                             max_length=max_length, return_tensors=return_tensors)
    
    def encode_tgt(self, text, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        """Encode target text (Chhattisgarhi)"""
        return self.tokenizer(text, padding=padding, truncation=truncation, 
                             max_length=max_length, return_tensors=return_tensors)
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token ids to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(self, src_texts, tgt_texts, padding=True, truncation=True, max_length=128):
        """Encode a batch of source and target texts"""
        src_encodings = self.encode_src(src_texts, padding=padding, truncation=truncation, max_length=max_length)
        tgt_encodings = self.encode_tgt(tgt_texts, padding=padding, truncation=truncation, max_length=max_length)
        return src_encodings, tgt_encodings
