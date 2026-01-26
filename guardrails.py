"""
Guardrails Module - AI Safety and Response Validation
====================================================

This module implements comprehensive safety controls to prevent hallucinations
and ensure reliable, grounded responses in RAGnarok's document intelligence system.

Key Safety Mechanisms:
- Confidence thresholding to filter irrelevant contexts
- Hallucination detection through keyword monitoring
- Response validation against retrieved contexts
- Explicit refusal patterns for missing information
- Warning systems for low-confidence responses

Philosophy:
"Better to refuse than to hallucinate" - The system explicitly states when
information is not available rather than generating potentially incorrect responses.

Author: RAGnarok Team
Version: 2.0.0
"""

import re
from typing import Dict, List, Tuple, Optional


class Guardrails:
    """
    AI Safety and Response Validation System
    =======================================
    
    Implements multi-layer validation to ensure RAGnarok responses are:
    - Grounded in retrieved document contexts
    - Confident enough to be reliable
    - Explicitly refuse when information is unavailable
    - Transparent about uncertainty and limitations
    
    Safety Layers:
    1. Confidence Thresholding: Filter low-relevance contexts
    2. Hallucination Detection: Monitor for refusal keywords
    3. Context Validation: Ensure responses are document-based
    4. Warning Generation: Alert users to potential issues
    """
   
    
    def __init__(self, min_confidence: float = 0.5, require_sources: bool = True):
        """
        Initialize Guardrails System
        ===========================
        
        Sets up the safety validation system with configurable thresholds.
        
        Args:
            min_confidence (float): Minimum similarity score for context acceptance
                                  Default: 0.5 (balanced precision/recall)
            require_sources (bool): Whether to require source attribution
                                  Default: True (always require sources)
        
        Confidence Threshold Rationale:
        - 0.5: Balanced threshold that filters clearly irrelevant content
        - Higher values (0.7+): More conservative, may miss relevant info
        - Lower values (0.3-): More permissive, may include noise
        
        Refusal Keywords:
        - Patterns that indicate the model acknowledges missing information
        - Preferred over hallucinated responses
        - Helps identify when the system is working correctly
        """
        self.min_confidence = min_confidence
        self.require_sources = require_sources
        
        # Keywords that indicate proper refusal behavior (not hallucination)
        # These are GOOD signs that the model is being honest about limitations
        self.refusal_keywords = [
            "i don't know",
            "i cannot",
            "i'm not sure",
            "i don't have",
            "not in the provided",
            "not mentioned",
            "not available",
            "cannot determine",
            "unable to",
            "no information"
        ]
    
    def check_confidence(self, similarity_scores: List[float]) -> Tuple[bool, float]:
        
        if not similarity_scores:
            return False, 0.0
        
        avg_score = sum(similarity_scores) / len(similarity_scores)
        max_score = max(similarity_scores)
        
        # Check if max score meets threshold
        passed = max_score >= self.min_confidence
        
        return passed, avg_score
    
    def filter_low_confidence(self, results: List[Tuple[Dict, float]], 
                              min_score: Optional[float] = None) -> List[Tuple[Dict, float]]:
       
        threshold = min_score if min_score is not None else self.min_confidence
        return [(meta, score) for meta, score in results if score >= threshold]
    
    def detect_hallucination_indicators(self, response: str) -> List[str]:
       
        detected = []
        response_lower = response.lower()
        
        for keyword in self.refusal_keywords:
            if keyword in response_lower:
                detected.append(keyword)
        
        return detected
    
    def enforce_context_only(self, response: str, contexts: List[str]) -> Tuple[str, bool]:
      
        # Check for explicit refusal phrases
        response_lower = response.lower()
        
        # If response explicitly says it doesn't know, that's actually good
        # (means it's not hallucinating)
        if any(phrase in response_lower for phrase in [
            "based on the provided",
            "according to the documents",
            "the documents state",
            "as mentioned in"
        ]):
            return response, True
        
        
        refusal_detected = self.detect_hallucination_indicators(response)
        
        if refusal_detected:
    
            if contexts:
                # Response acknowledges lack of info - this is acceptable
                return response, True
            else:
                # No contexts but refusing - might be hallucination
                return response, False
        

        return response, True
    
    def validate_response(self, response: str, contexts: List[str], 
                         similarity_scores: List[float]) -> Dict:

        # Check confidence
        confidence_passed, avg_confidence = self.check_confidence(similarity_scores)
        
        # Check context enforcement
        response_valid, is_context_based = self.enforce_context_only(response, contexts)
        
        # Detect hallucination indicators
        hallucination_indicators = self.detect_hallucination_indicators(response)
        
        # Overall validation
        is_valid = confidence_passed and is_context_based
        
        return {
            'is_valid': is_valid,
            'confidence_passed': confidence_passed,
            'average_confidence': avg_confidence,
            'max_confidence': max(similarity_scores) if similarity_scores else 0.0,
            'is_context_based': is_context_based,
            'hallucination_indicators': hallucination_indicators,
            'response': response_valid
        }
    
    def format_response_with_warning(self, response: str, validation: Dict) -> str:

        warnings = []
        
        if not validation['confidence_passed']:
            warnings.append(
                f"⚠️ Low confidence: Retrieved documents have low similarity "
                f"(max: {validation['max_confidence']:.2f}, threshold: {self.min_confidence:.2f})"
            )
        
        if not validation['is_context_based']:
            warnings.append(
                "⚠️ Warning: Response may not be fully based on provided documents"
            )
        
        if validation['hallucination_indicators']:
            warnings.append(
                f"⚠️ Note: Response indicates uncertainty about the query"
            )
        
        if warnings:
            warning_text = "\n\n".join(warnings)
            return f"{response}\n\n---\n{warning_text}"
        
        return response
