# src/models/base_model.py
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoConfig, EncoderDecoderCache

class EncoderDecoderModel(nn.Module):
    """Base encoder-decoder model for Hindi-Chhattisgarhi transfer learning"""
    
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", src_lang="hin_Deva", tgt_lang="hne_Deva"):
        """
        Initialize the encoder-decoder model
        
        Args:
            model_name: Pretrained model name from HuggingFace hub
            src_lang: Source language code (Hindi)
            tgt_lang: Target language code (Chhattisgarhi)
        """
        super(EncoderDecoderModel, self).__init__()
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Load configuration and model
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code = True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code = True)

        # Set language codes for the model if supported
        if hasattr(self.model, "config") and hasattr(self.model.config, "forced_bos_token_id"):
            self.model.config.forced_bos_token_id = None
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None):
        """
        Forward pass through the model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for inputs
            decoder_input_ids: Input IDs for the decoder
            labels: Target labels for computing loss
            
        Returns:
            Model outputs
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True
        )
    
    def generate(self, input_ids, attention_mask=None, max_length=128, num_beams=4):
        """
        Generate translations
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for inputs
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            
        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    def save_pretrained(self, output_dir):
        """Save model to directory"""
        self.model.save_pretrained(output_dir)
    
    @classmethod
    def from_pretrained(cls, model_path, src_lang="hin_Deva", tgt_lang="hne_Deva"):
        """Load a pretrained model from a directory"""
        instance = cls.__new__(cls)
        super(EncoderDecoderModel, instance).__init__()
        instance.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)   
        instance.config = instance.model.config
        instance.src_lang = src_lang
        instance.tgt_lang = tgt_lang
        return instance
