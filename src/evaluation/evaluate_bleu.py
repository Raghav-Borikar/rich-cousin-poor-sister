# src/evaluation/evaluate_bleu.py
import sacrebleu
from typing import List, Union

def calculate_bleu(references: Union[List[str], List[List[str]]], hypotheses: List[str]) -> float:
    """
    Calculate BLEU score for model evaluation
    
    Args:
        references: List of reference translations (or list of lists for multiple references)
        hypotheses: List of model-generated translations
        
    Returns:
        BLEU score
    """
    # Ensure references format is correct for sacrebleu
    if isinstance(references[0], str):
        references = [[ref] for ref in references]
    elif not isinstance(references[0], list):
        raise ValueError("References must be a list of strings or a list of lists of strings")
    
    # Transpose references to the format expected by sacrebleu
    refs_list = list(zip(*references))
    
    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, refs_list)
    
    return bleu.score
