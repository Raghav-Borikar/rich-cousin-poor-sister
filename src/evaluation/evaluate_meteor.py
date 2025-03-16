# src/evaluation/evaluate_meteor.py
import nltk
from nltk.translate import meteor_score
from typing import List, Union
import logging

# Ensure NLTK resources are downloaded
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logging.warning(f"Failed to download NLTK resources: {e}")

def calculate_meteor(references: Union[List[str], List[List[str]]], hypotheses: List[str]) -> float:
    """
    Calculate METEOR score for model evaluation
    
    Args:
        references: List of reference translations (or list of lists for multiple references)
        hypotheses: List of model-generated translations
        
    Returns:
        METEOR score
    """
    # Convert to the format expected by NLTK's METEOR implementation
    if isinstance(references[0], str):
        references = [[ref.split()] for ref in references]
    elif isinstance(references[0], list) and isinstance(references[0][0], str):
        references = [[ref.split() for ref in refs] for refs in references]
    
    # Convert hypotheses to tokens
    hypotheses = [hyp.split() for hyp in hypotheses]
    
    # Calculate METEOR score for each sentence pair
    meteor_scores = []
    for i, hyp in enumerate(hypotheses):
        score = meteor_score.meteor_score(references[i], hyp)
        meteor_scores.append(score)
    
    # Return average METEOR score
    return sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
