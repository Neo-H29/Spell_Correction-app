from fastapi import FastAPI, HTTPException, Body
# Import Field directly from Pydantic along with other Pydantic types
from pydantic import BaseModel, Field, constr, conint, confloat
import pandas as pd # Keep for potential type hints if needed, though not directly used
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import Levenshtein
import re
import json
import nltk
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import time
import logging
import spacy
from spellchecker import SpellChecker
import os
import uvicorn
import hashlib

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Base Directory for Paths (less critical without templates, but good for consistency) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Global Store for Loaded Data/Models ---
LOADED_RESOURCES = {}

# --- Constants for the core logic (can be overridden by user) ---
DEFAULT_SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_SPACY_MODEL_NAME = "en_core_web_sm"
DEFAULT_TARGET_POS_TAGS = {"NOUN", "PROPN", "ADJ"}
PUNCTUATION_REGEX = re.compile(r'[!"#$%&\'()*+./:;<=>?@\[\\\]^_`{|}~]')
MAX_NGRAM_LENGTH_CONST = 20 # Max N-gram length used in hybrid matching logic


# --- NLTK Data Download ---
def download_nltk_data_startup():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK stopwords and punkt found.")
        return True
    except LookupError:
        try:
            logger.info("NLTK data not found. Downloading stopwords and punkt...")
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            logger.info("NLTK data downloaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed NLTK data download: {e}. Using basic fallback stopwords.", exc_info=True)
            return False

nltk_ready_global = False
ENGLISH_STOPWORDS_GLOBAL = set()

app = FastAPI(
    title="Term Normalization API",
    description="API for correcting and normalizing terms in sentences using configurable parameters and pre-processed dictionary data. Use the /docs endpoint for interactive testing.",
    version="1.0.2" # Incremented version
)

# --- Pydantic Models for Request Body/Params ---
class CorrectionRequest(BaseModel):
    data_path: str = Field("processed_dictionary_data", description="Path to the directory containing pre-processed dictionary artifacts (relative to where the API is run, or absolute).")
    sentence_to_correct: constr(min_length=1) = Field(..., description="The sentence to be corrected.")
    
    sentence_model_name: str = Field(DEFAULT_SENTENCE_MODEL_NAME, description="Sentence Transformer model name for embeddings.")
    spacy_model_name: str = Field(DEFAULT_SPACY_MODEL_NAME, description="SpaCy model name for NLP tasks.")
    
    use_spacy: bool = Field(True, description="Enable SpaCy for tokenization, POS tagging, lemmatization.")
    use_lemma: bool = Field(False, description="Match on lemmas (base form of words). Requires use_spacy.")
    use_pos_filter: bool = Field(True, description="Prioritize matches for target POS tags. Requires use_spacy.")
    use_ngram: bool = Field(True, description="Enable N-gram based matching.")
    use_ngram_spellcheck: bool = Field(True, description="Enable spell check for near-miss N-gram matches. Requires use_ngram and spell_checker.")
    use_token_spellcheck: bool = Field(True, description="Enable spell check assistance for individual tokens. Requires spell_checker.")
    use_embedding: bool = Field(True, description="Enable embedding-based matching.")
    use_levenshtein: bool = Field(True, description="Enable Levenshtein distance based matching.")

    min_ngram_len: conint(ge=1, le=5) = Field(2, description="Minimum length of N-grams to consider.")
    hybrid_ngram_sim: confloat(ge=0.0, le=1.0) = Field(0.95, description="Minimum embedding similarity for N-gram candidate matches. Also used as a threshold for overriding known words.")
    ngram_spellcheck_trigger_sim: confloat(ge=0.0, le=1.0) = Field(0.80, description="Similarity threshold below which N-gram spell check is attempted (must be < hybrid_ngram_sim).")
    min_cosine_sim: confloat(ge=0.0, le=1.0) = Field(0.80, description="Minimum cosine similarity for token embedding matches.")
    max_lev_distance: conint(ge=0, le=5) = Field(1, description="Maximum Levenshtein edit distance for token matching.")
    
    lev_weight: confloat(ge=0.0) = Field(1.0, description="Weight for Levenshtein scores in combined scoring.")
    emb_weight: confloat(ge=0.0) = Field(1.0, description="Weight for embedding scores in combined scoring.")
    pos_boost: confloat(ge=1.0) = Field(1.2, description="Boost factor for matches with target POS tags.")

    class Config:
        schema_extra = {
            "example": {
                "data_path": "processed_dictionary_data",
                "sentence_to_correct": "The patient has diabets melitus and a recent hart attack.",
                "sentence_model_name": "all-MiniLM-L6-v2",
                "spacy_model_name": "en_core_web_sm",
                "use_spacy": True,
                "use_lemma": False,
                "use_pos_filter": True,
                "use_ngram": True,
                "use_ngram_spellcheck": True,
                "use_token_spellcheck": True,
                "use_embedding": True,
                "use_levenshtein": True,
                "min_ngram_len": 2,
                "hybrid_ngram_sim": 0.95,
                "ngram_spellcheck_trigger_sim": 0.80,
                "min_cosine_sim": 0.80,
                "max_lev_distance": 1,
                "lev_weight": 1.0,
                "emb_weight": 1.0,
                "pos_boost": 1.2
            }
        }

class CorrectionResponse(BaseModel):
    original_sentence: str
    corrected_sentence: str
    replacements_map: Dict[str, str]
    details_map: Dict[str, Dict[str, Any]]
    processing_time_seconds: float
    parameters_used: CorrectionRequest # Echo back parameters for clarity

# --- Utility Functions (Span Tracking, Model/Data Loading - Adapted) ---
def span_overlaps(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    if span1[0] >= span1[1] or span2[0] >= span2[1]: return False # Invalid spans
    return max(span1[0], span2[0]) < min(span1[1], span2[1])

def is_span_covered(target_span: Tuple[int, int], covered_spans: List[Tuple[int, int]]) -> bool:
    if target_span[0] >= target_span[1]: return False # Invalid target span
    return any(covered[0] <= target_span[0] and covered[1] >= target_span[1]
               for covered in covered_spans if covered[0] < covered[1]) # Ensure covered spans are valid

def add_covered_span(new_span: Tuple[int, int], covered_spans: List[Tuple[int, int]]):
    if new_span[0] < new_span[1]: # Add only valid spans
        covered_spans.append(new_span)

def load_transformer_model(model_name: str) -> Optional[SentenceTransformer]:
    logger.info(f"Loading ST model: {model_name}")
    try: return SentenceTransformer(model_name)
    except Exception as e: logger.error(f"Error ST model {model_name}: {e}"); return None

def load_spacy_model(model_name: str) -> Optional[spacy.Language]:
    logger.info(f"Loading spaCy model: {model_name}")
    try: return spacy.load(model_name)
    except OSError:
        logger.warning(f"spaCy model '{model_name}' not found. Attempting download.")
        try: spacy.cli.download(model_name); return spacy.load(model_name)
        except Exception as e: logger.error(f"Failed download/load spaCy '{model_name}': {e}"); return None
    except Exception as e: logger.error(f"Error spaCy model '{model_name}': {e}"); return None

def get_spell_checker() -> Optional[SpellChecker]:
    logger.info("Loading SpellChecker")
    try: return SpellChecker()
    except Exception as e: logger.error(f"Error SpellChecker: {e}"); return None

def load_json_file(filepath: str) -> Optional[Any]:
    if not os.path.exists(filepath): logger.error(f"JSON missing: {filepath}"); return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        return data
    except Exception as e: logger.error(f"Error JSON {filepath}: {e}"); return None

def load_faiss_index_file(filepath: str) -> Optional[faiss.Index]:
    if not os.path.exists(filepath): logger.warning(f"FAISS missing: {filepath}"); return None
    try: return faiss.read_index(filepath)
    except Exception as e: logger.error(f"Error FAISS {filepath}: {e}"); return None


# --- Core Logic Functions ---
def find_best_levenshtein_match(token_lower: str, dictionary_list_lower: list, max_distance: int) -> Tuple[Optional[str], float]:
    best_match_lower = None; min_distance = float(max_distance + 1); score = 0.0
    len_token = len(token_lower)
    if max_distance < 0: return None, 0.0
    candidates = [d_lower for d_lower in dictionary_list_lower if abs(len_token - len(d_lower)) <= max_distance]
    if not candidates: return None, 0.0
    for dict_term_lower in candidates:
        distance = Levenshtein.distance(token_lower, dict_term_lower)
        if distance <= max_distance:
            if distance < min_distance:
                min_distance = distance; best_match_lower = dict_term_lower
                max_len = max(len_token, len(dict_term_lower))
                score = 1.0 - (distance / max_len) if max_len > 0 else 1.0
                if distance == 0: break
    if best_match_lower is None: return None, 0.0
    return best_match_lower, score

def find_best_embedding_match(token_embed_norm: np.ndarray, faiss_idx: Optional[faiss.Index], dict_list_orig: list, min_sim: float) -> Tuple[Optional[str], float]:
    if token_embed_norm is None or token_embed_norm.size == 0: return None, 0.0
    if faiss_idx is None or not hasattr(faiss_idx, 'ntotal') or faiss_idx.ntotal == 0: return None, 0.0
    try:
        query_embedding = token_embed_norm.reshape(1, -1).astype('float32')
        if query_embedding.shape[1] != faiss_idx.d: return None, 0.0
        distances, indices = faiss_idx.search(query_embedding, 1)
        if indices.size == 0 or indices[0].size == 0 or indices[0][0] < 0: return None, 0.0
        best_index = indices[0][0]
        similarity = max(0.0, min(float(distances[0][0]), 1.0)) 
        if similarity >= min_sim and 0 <= best_index < len(dict_list_orig):
             return dict_list_orig[best_index], similarity
        return None, 0.0
    except Exception: return None, 0.0

def get_token_match_scores(
    token_obj: Union[spacy.tokens.Token, Dict], token_embedding: Optional[np.ndarray],
    dictionary_map_lower_to_original: Dict[str, str], dictionary_list_original: List[str],
    dictionary_list_lower: List[str], faiss_index: Optional[faiss.Index],
    spell_checker: Optional[SpellChecker], params: CorrectionRequest 
) -> Tuple[Optional[str], Dict[str, Dict[str, Any]]]:
    potential_fuzzy_matches = {}
    is_spacy_token = isinstance(token_obj, spacy.tokens.Token)
    token_text = token_obj.text if is_spacy_token else token_obj['text']
    token_lower = token_obj.lower_ if is_spacy_token else token_obj['lower_']
    token_lemma = token_obj.lemma_ if is_spacy_token and params.use_lemma else token_lower
    token_pos = token_obj.pos_ if is_spacy_token else token_obj['pos_']
    is_stopword = (token_obj.is_stop or token_lower in ENGLISH_STOPWORDS_GLOBAL) if is_spacy_token else token_obj['is_stop']
    is_target_pos = token_pos in DEFAULT_TARGET_POS_TAGS
    effective_max_lev = params.max_lev_distance
    if params.use_pos_filter and not is_target_pos and is_spacy_token: effective_max_lev = min(effective_max_lev, 0)
    cleaned_token_lower = PUNCTUATION_REGEX.sub('', token_lower).strip()
    cleaned_token_lemma = PUNCTUATION_REGEX.sub('', token_lemma).strip()
    if not cleaned_token_lower and not cleaned_token_lemma: return None, {}
    
    # Exact match (case-insensitive) from dictionary is highest priority
    exact_match_original_case = dictionary_map_lower_to_original.get(cleaned_token_lower)
    if exact_match_original_case: 
        # If the exact match is found, we return it. The decision to use it (even if just case normalization)
        # happens in correct_sentence_hybrid based on `token_text.lower() != replacement_term.lower()`
        return exact_match_original_case, {} 
    
    if is_spacy_token and params.use_lemma and cleaned_token_lemma and cleaned_token_lemma != cleaned_token_lower:
         exact_lemma_match_original_case = dictionary_map_lower_to_original.get(cleaned_token_lemma)
         if exact_lemma_match_original_case: 
             return exact_lemma_match_original_case, {}

    # Fuzzy matching if no exact match
    if params.use_token_spellcheck and spell_checker and cleaned_token_lower:
        try:
            candidates = spell_checker.candidates(cleaned_token_lower)
            if candidates:
                # Prefer candidates that are in our custom dictionary
                best_dict_correction = next((c for c in candidates if c in dictionary_map_lower_to_original), None)
                if best_dict_correction:
                    corrected_original_case = dictionary_map_lower_to_original[best_dict_correction]
                    match_details = {"method": "spellcheck_token", "score": 0.99, "token": token_text, "confidence": "high", "matched_on_lemma": False, "pos_tag": token_pos, "is_target_pos": is_target_pos, "corrected_token": best_dict_correction}
                    if corrected_original_case not in potential_fuzzy_matches: potential_fuzzy_matches[corrected_original_case] = match_details
        except Exception: pass # spell_checker.candidates can sometimes fail

    match_term_for_fuzzy = cleaned_token_lemma if is_spacy_token and params.use_lemma and cleaned_token_lemma else cleaned_token_lower
    if params.use_levenshtein and match_term_for_fuzzy:
        current_max_lev = 0 if is_stopword and effective_max_lev > 0 else effective_max_lev
        if current_max_lev >= 0: # Allow max_lev_distance = 0 for exact fuzzy match after cleaning
            lev_match_lower, lev_score = find_best_levenshtein_match(match_term_for_fuzzy, dictionary_list_lower, current_max_lev)
            if lev_match_lower and lev_score > 0:
                lev_match_original = dictionary_map_lower_to_original.get(lev_match_lower)
                if lev_match_original:
                    confidence = "high" if lev_score > 0.9 else "medium" if lev_score >= 0.85 else "low"
                    match_details = {"method": "levenshtein", "score": lev_score, "token": token_text, "confidence": confidence, "matched_on_lemma": is_spacy_token and params.use_lemma and (match_term_for_fuzzy == cleaned_token_lemma), "pos_tag": token_pos, "is_target_pos": is_target_pos}
                    if lev_match_original not in potential_fuzzy_matches or potential_fuzzy_matches[lev_match_original]['score'] < lev_score: potential_fuzzy_matches[lev_match_original] = match_details
    
    if params.use_embedding and token_embedding is not None and faiss_index is not None and match_term_for_fuzzy: # Ensure faiss_index is not None
        emb_match_original, emb_similarity = find_best_embedding_match(token_embedding, faiss_index, dictionary_list_original, params.min_cosine_sim)
        if emb_match_original and emb_similarity >= params.min_cosine_sim:
            emb_similarity_float = float(emb_similarity)
            confidence = "high" if emb_similarity_float > 0.9 else "medium" if emb_similarity_float > 0.8 else "low"
            match_details = {"method": "embedding", "score": emb_similarity_float, "token": token_text, "confidence": confidence, "matched_on_lemma": False, "pos_tag": token_pos, "is_target_pos": is_target_pos}
            passes_pos_filter = not (is_spacy_token and params.use_pos_filter and not is_target_pos and emb_similarity_float < 0.95) # Less strict if not target POS
            if passes_pos_filter:
                if emb_match_original not in potential_fuzzy_matches or potential_fuzzy_matches[emb_match_original]['score'] < emb_similarity_float: potential_fuzzy_matches[emb_match_original] = match_details
    
    return None, potential_fuzzy_matches # Return None for exact_match_term as it's handled, and the fuzzy potentials

def select_best_correction(
    token_text: str, 
    potential_matches: Dict[str, Dict[str, Any]], 
    params: CorrectionRequest, 
    spell_checker_ref: Optional[SpellChecker], # Added spell_checker_ref
    is_stopword: bool = False 
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not potential_matches: return None
    scored_corrections = {}
    for dict_term, details in potential_matches.items():
        score = details['score']; method = details['method']; confidence = details['confidence']
        is_target_pos = details.get('is_target_pos', False)
        weighted_score = 0.0
        if method == "levenshtein": weighted_score = score * params.lev_weight
        elif method == "embedding": weighted_score = score * params.emb_weight
        elif method == "spellcheck_token": weighted_score = score * params.lev_weight 
        
        if is_target_pos and params.pos_boost > 1.0: 
            weighted_score *= params.pos_boost
            details['pos_boost_applied'] = True
        else: 
            details['pos_boost_applied'] = False
        
        passes_threshold = False
        min_req_score = 0.0
        if is_stopword:
             if confidence == "high":
                 if method == "spellcheck_token" and weighted_score > 0.98: passes_threshold = True
                 elif method == "levenshtein" and weighted_score > 0.95 and score == 1.0: passes_threshold = True 
                 elif method == "embedding" and weighted_score > 0.98: passes_threshold = True
        else:
            if confidence == "high": min_req_score = 0.65
            elif confidence == "medium": min_req_score = 0.78
            elif confidence == "low": min_req_score = 0.92
            
            if method == "spellcheck_token": 
                min_req_score = max(min_req_score, 0.90) 
            
            if weighted_score >= min_req_score: passes_threshold = True
        
        if passes_threshold: 
            details['weighted_score'] = weighted_score
            scored_corrections[dict_term] = details
            
    if not scored_corrections: return None
    
    best_dict_term = max(scored_corrections.keys(), key=lambda s: scored_corrections[s]['weighted_score'])
    
    if best_dict_term.lower() == token_text.lower(): 
        return None 

    # --- START: Precautionary check for known words ---
    if spell_checker_ref and token_text.isalpha() and len(token_text) > 2: 
        is_original_known_word = False
        try:
            known_status = spell_checker_ref.known([token_text.lower()])
            if token_text.lower() in known_status:
                is_original_known_word = True
        except Exception as e:
            logger.warning(f"Spellchecker 'known' check failed for '{token_text.lower()}': {e}")

        if is_original_known_word:
            details_of_best = scored_corrections[best_dict_term]
            if details_of_best['method'] != "spellcheck_token":
                 known_word_override_threshold = params.hybrid_ngram_sim 
                 current_weighted_score = details_of_best.get('weighted_score', 0.0)
                 if current_weighted_score < known_word_override_threshold:
                     logger.info(
                         f"Precaution: Original token '{token_text}' is a known word. "
                         f"Proposed fuzzy correction '{best_dict_term}' (method: {details_of_best['method']}, score: {current_weighted_score:.4f}) "
                         f"is below override threshold ({known_word_override_threshold:.4f}). Retaining original."
                     )
                     return None 
    # --- END: Precautionary check for known words ---

    return best_dict_term, scored_corrections[best_dict_term]

def hybrid_ngram_similarity_matching(original_sentence: str, spacy_doc: spacy.tokens.Doc, dictionary_list_original: List[str], ngram_index: Dict[str, Dict[str, List[str]]], model: SentenceTransformer, spell_checker: Optional[SpellChecker], params: CorrectionRequest) -> List[Dict[str, Any]]:
    if not original_sentence.strip() or not ngram_index or not model or spacy_doc is None: return []
    potential_matches: List[Dict[str, Any]] = []
    near_miss_candidates_for_spellcheck: List[Tuple[str, str, float, tuple, int]] = []
    
    tokens_with_spans = []
    try:
        tokens_with_spans = [(str(t.text), (int(t.idx), int(t.idx) + len(str(t.text)))) 
                             for t in spacy_doc 
                             if hasattr(t, 'is_punct') and hasattr(t, 'is_space') and not t.is_punct and not t.is_space]
    except Exception as e:
        logger.error(f"Error processing tokens for n-grams: {e}")
        return []
    if not tokens_with_spans: return []
    
    possible_n_values = sorted([int(n_str) for n_str in ngram_index.keys() if int(n_str) >= params.min_ngram_len and int(n_str) <= MAX_NGRAM_LENGTH_CONST], reverse=True)
    processed_spans_for_n_pass = defaultdict(list) 

    for n in possible_n_values:
        if n > len(tokens_with_spans): continue
        
        current_sentence_ngrams_with_spans: List[Dict[str, Any]] = []
        unique_sentence_ngrams_text_for_n: Set[str] = set()
        
        for i in range(len(tokens_with_spans) - n + 1):
            ngram_tokens_info = tokens_with_spans[i : i + n]
            ngram_text = " ".join([t[0] for t in ngram_tokens_info])
            start_char = ngram_tokens_info[0][1][0]
            end_char = ngram_tokens_info[-1][1][1]
            ngram_span = (start_char, end_char)
            
            covered_by_longer = any(is_span_covered(ngram_span, processed_spans_for_n_pass[h_n]) 
                                    for h_n in range(n + 1, max(possible_n_values, default=0) + 1) 
                                    if h_n in processed_spans_for_n_pass)
            if covered_by_longer: continue
            
            current_sentence_ngrams_with_spans.append({"text": ngram_text, "span": ngram_span})
            unique_sentence_ngrams_text_for_n.add(ngram_text)

        if not unique_sentence_ngrams_text_for_n: continue

        all_dict_ngrams_map_for_n = ngram_index.get(str(n), {})
        if not all_dict_ngrams_map_for_n: continue
        
        candidate_dict_terms_for_n_set: Set[str] = set()
        for original_casings_list in all_dict_ngrams_map_for_n.values():
            candidate_dict_terms_for_n_set.update(original_casings_list)
        
        if not candidate_dict_terms_for_n_set: continue

        unique_sentence_ngrams_list_for_n = sorted([s_ng for s_ng in unique_sentence_ngrams_text_for_n if s_ng and s_ng.strip()])
        candidate_dict_terms_list_for_n = sorted([d_ng for d_ng in candidate_dict_terms_for_n_set if d_ng and d_ng.strip()])

        if not unique_sentence_ngrams_list_for_n or not candidate_dict_terms_list_for_n: continue

        try:
            sentence_ngram_embeddings = model.encode(unique_sentence_ngrams_list_for_n, normalize_embeddings=True, show_progress_bar=False).astype('float32')
            candidate_term_embeddings = model.encode(candidate_dict_terms_list_for_n, normalize_embeddings=True, show_progress_bar=False).astype('float32')
            
            if sentence_ngram_embeddings.ndim != 2 or candidate_term_embeddings.ndim != 2 or \
               sentence_ngram_embeddings.shape[0] == 0 or candidate_term_embeddings.shape[0] == 0 or \
               sentence_ngram_embeddings.shape[1] != candidate_term_embeddings.shape[1]:
                logger.debug(f"N-gram embedding shape mismatch or empty for n={n}. Sentence shape: {sentence_ngram_embeddings.shape}, Dict shape: {candidate_term_embeddings.shape}")
                continue

            similarity_matrix = np.dot(sentence_ngram_embeddings, candidate_term_embeddings.T)
            sent_ngram_text_to_idx = {text: i for i, text in enumerate(unique_sentence_ngrams_list_for_n)}

        except Exception as e:
            logger.warning(f"N-gram embedding/similarity calculation failed for n={n}: {e}", exc_info=True)
            continue
        
        for item in current_sentence_ngrams_with_spans:
            sentence_ngram_text = item["text"]
            original_span = item["span"]

            if is_span_covered(original_span, processed_spans_for_n_pass[n]): 
                continue
            
            sent_ngram_matrix_idx = sent_ngram_text_to_idx.get(sentence_ngram_text)
            if sent_ngram_matrix_idx is None: continue

            best_candidate_dict_term_for_sent_ngram = None
            max_similarity_for_sent_ngram = -1.0

            for dict_term_candidate_idx, dict_term_candidate in enumerate(candidate_dict_terms_list_for_n):
                current_similarity = similarity_matrix[sent_ngram_matrix_idx, dict_term_candidate_idx]
                current_similarity = float(min(current_similarity, 1.0)) 

                if current_similarity > max_similarity_for_sent_ngram:
                    max_similarity_for_sent_ngram = current_similarity
                    best_candidate_dict_term_for_sent_ngram = dict_term_candidate
            
            if best_candidate_dict_term_for_sent_ngram and max_similarity_for_sent_ngram >= params.hybrid_ngram_sim:
                confidence = "high" if (sentence_ngram_text.lower() == best_candidate_dict_term_for_sent_ngram.lower()) or max_similarity_for_sent_ngram > 0.98 else "medium"
                match_data = {
                    "original_text": sentence_ngram_text, 
                    "replacement": best_candidate_dict_term_for_sent_ngram, 
                    "n_value": n, 
                    "similarity": max_similarity_for_sent_ngram,
                    "span": original_span, 
                    "confidence": confidence, 
                    "method": f"ngram-{n}"
                }
                potential_matches.append(match_data)
                add_covered_span(original_span, processed_spans_for_n_pass[n])
            
            elif params.use_ngram_spellcheck and spell_checker and best_candidate_dict_term_for_sent_ngram and \
                 params.ngram_spellcheck_trigger_sim <= max_similarity_for_sent_ngram < params.hybrid_ngram_sim:
                near_miss_candidates_for_spellcheck.append((
                    sentence_ngram_text, 
                    best_candidate_dict_term_for_sent_ngram,
                    max_similarity_for_sent_ngram, 
                    original_span, 
                    n
                ))

    spell_checked_potential_matches: List[Dict[str, Any]] = []
    all_direct_match_covered_spans = set()
    for n_val_covered in processed_spans_for_n_pass:
        for span_tuple_covered in processed_spans_for_n_pass[n_val_covered]:
            all_direct_match_covered_spans.add(tuple(span_tuple_covered))

    if params.use_ngram_spellcheck and spell_checker and near_miss_candidates_for_spellcheck:
        near_miss_candidates_for_spellcheck.sort(key=lambda x: (-x[4], -x[2]))
        spans_covered_by_ngram_spellcheck = set()

        for sent_ngram, _, orig_sim, span, n_val in near_miss_candidates_for_spellcheck:
            span_tuple = tuple(span)
            if span_tuple in all_direct_match_covered_spans or \
               any(span_overlaps(span, existing_spell_span) for existing_spell_span in spans_covered_by_ngram_spellcheck):
                continue
            
            try:
                words = sent_ngram.split(); corrected_words = []; changed = False
                for word in words:
                    correction_result = spell_checker.correction(word)
                    best_correction = correction_result if isinstance(correction_result, str) else word
                    if best_correction and best_correction.lower() != word.lower(): 
                        corrected_words.append(best_correction); changed = True
                    else:
                        corrected_words.append(word)
                
                if changed:
                    corrected_ngram_text = " ".join(corrected_words)
                    dict_terms_for_corrected = ngram_index.get(str(n_val), {}).get(corrected_ngram_text.lower())
                    if dict_terms_for_corrected:
                        replacement_via_spell = dict_terms_for_corrected[0]
                        spell_checked_potential_matches.append({
                            "original_text": sent_ngram, 
                            "corrected_text": corrected_ngram_text,
                            "replacement": replacement_via_spell, 
                            "n_value": n_val, 
                            "similarity": float(orig_sim), 
                            "span": span, 
                            "confidence": "medium",
                            "method": f"ngram_spellcheck-{n_val}"
                        })
                        spans_covered_by_ngram_spellcheck.add(span_tuple)
            except Exception as e:
                logger.warning(f"Error during n-gram spellcheck for '{sent_ngram}': {e}", exc_info=True)
                pass
    
    all_potential_matches = potential_matches + spell_checked_potential_matches
    all_potential_matches.sort(key=lambda x: (-x["n_value"], -x["similarity"], 0 if 'spellcheck' not in x["method"] else 1, x["span"][0]))
    
    final_ngram_matches = []; final_spans_covered_by_ngrams = set()
    for match in all_potential_matches:
        if not any(span_overlaps(match["span"], existing_span) for existing_span in final_spans_covered_by_ngrams):
            final_ngram_matches.append(match)
            final_spans_covered_by_ngrams.add(tuple(match["span"])) 
            
    final_ngram_matches.sort(key=lambda x: x["span"][0])
    return final_ngram_matches

def correct_sentence_hybrid(original_sentence: str, params: CorrectionRequest, resources: Dict[str, Any]) -> Tuple[str, Dict[str, str], Dict[str, Dict[str, Any]]]:
    if not isinstance(original_sentence, str): original_sentence = str(original_sentence)
    original_sentence = original_sentence.strip()
    if not original_sentence: return original_sentence, {}, {}

    spacy_nlp = resources["spacy_nlp"]
    dictionary_map_lower_to_original = resources["map_lower_to_original"]
    dictionary_list_original = resources["dictionary_list_original"]
    dictionary_list_lower = resources["dictionary_list_lower"]
    ngram_index = resources["ngram_index"]
    model_embed = resources["st_model"]
    faiss_index = resources["faiss_index"]
    spell_checker = resources["spell_checker"] # Main spell_checker instance from resources

    active_use_spacy = params.use_spacy and spacy_nlp is not None
    spacy_doc = None
    if active_use_spacy:
        try: spacy_doc = spacy_nlp(original_sentence)
        except Exception as e: logger.error(f"spaCy failed: {e}"); return original_sentence, {}, {}
    elif params.use_ngram: 
         pseudo_tokens = []
         for match in re.finditer(r'\w+|[^\w\s]', original_sentence): 
              token_text = match.group(0)
              is_punct_token = all(c in PUNCTUATION_REGEX.pattern for c in token_text) and not token_text.isalnum()
              pseudo_tokens.append(
                  type('obj', (object,), {
                      'text': token_text, 'idx': match.start(), 
                      'is_punct': is_punct_token, 'is_space': False, 
                      'pos_': 'X', 'lemma_': token_text.lower(), 
                      'is_stop': token_text.lower() in ENGLISH_STOPWORDS_GLOBAL 
                  })()
              )
         spacy_doc = type('obj', (object,), {'__iter__': lambda self: iter(pseudo_tokens)})()


    applied_corrections: List[Dict[str, Any]] = []; replacement_spans: List[Tuple[int, int]] = []
    active_use_ngram_eff = params.use_ngram and model_embed is not None and spacy_doc is not None and ngram_index is not None
    if active_use_ngram_eff:
        hybrid_matches = hybrid_ngram_similarity_matching(original_sentence, spacy_doc, dictionary_list_original, ngram_index, model_embed, spell_checker if params.use_ngram_spellcheck else None, params)
        for match in hybrid_matches:
            span, original_text_ngram, replacement_term = match["span"], match["original_text"], match["replacement"]
            # This condition is crucial: only add if there's an actual change (case-insensitive)
            if original_text_ngram.lower() != replacement_term.lower() and not any(span_overlaps(span, rep_span) for rep_span in replacement_spans):
                details = {"method": match["method"], "score": match["similarity"], "confidence": match["confidence"], "original_text_ngram": original_text_ngram, "weighted_score": match["similarity"], "n_value": match.get("n_value"), "original_corrected_text": match.get("corrected_text", None)}
                applied_corrections.append({"original_text": original_text_ngram, "replacement": replacement_term, "original_span": span, "details": details, "type": "ngram" if "spellcheck" not in match["method"] else "ngram_spellcheck"})
                add_covered_span(span, replacement_spans)

    tokens_to_process_objs = []
    if active_use_spacy and isinstance(spacy_doc, spacy.tokens.Doc): 
        for token in spacy_doc:
            if token.is_punct or token.is_space: continue
            token_span = (token.idx, token.idx + len(token.text))
            if not any(span_overlaps(token_span, rep_span) for rep_span in replacement_spans): tokens_to_process_objs.append(token)
    elif spacy_doc is not None: 
         try:
            for token_obj in spacy_doc: 
                if not all(hasattr(token_obj, attr) for attr in ['text', 'idx', 'is_punct', 'is_space', 'pos_', 'lemma_', 'is_stop']): continue
                if token_obj.is_punct or token_obj.is_space: continue
                token_text = token_obj.text
                token_span = (token_obj.idx, token_obj.idx + len(token_text))
                if not any(span_overlaps(token_span, rep_span) for rep_span in replacement_spans): 
                    tokens_to_process_objs.append({
                        "text": token_text, "lower_": token_text.lower(), 
                        "lemma_": token_obj.lemma_, "pos_": token_obj.pos_, 
                        "is_stop": token_obj.is_stop, "idx": token_obj.idx, 
                        "span": token_span 
                    })
         except Exception as e:
            logger.error(f"Error processing pseudo-tokens for token correction: {e}", exc_info=True)


    token_embeddings = None
    active_use_embedding_eff = params.use_embedding and model_embed is not None and faiss_index is not None
    if active_use_embedding_eff and tokens_to_process_objs:
        try:
            token_texts = [t.text if isinstance(t, spacy.tokens.Token) else t['text'] for t in tokens_to_process_objs]
            valid_token_indices = [i for i, txt in enumerate(token_texts) if txt and txt.strip()]
            valid_token_texts = [token_texts[i] for i in valid_token_indices]
            if valid_token_texts:
                token_embeddings_raw = model_embed.encode(valid_token_texts, normalize_embeddings=True, show_progress_bar=False).astype('float32')
                emb_dim = model_embed.get_sentence_embedding_dimension()
                if emb_dim is None: 
                    emb_dim = token_embeddings_raw.shape[1] if token_embeddings_raw.ndim == 2 and token_embeddings_raw.shape[0] > 0 else 384 
                
                token_embeddings = np.zeros((len(tokens_to_process_objs), emb_dim), dtype=np.float32)
                if token_embeddings_raw.ndim == 2 and token_embeddings_raw.shape[0] == len(valid_token_indices) and token_embeddings_raw.shape[1] == emb_dim:
                    for raw_idx, original_idx in enumerate(valid_token_indices):
                        token_embeddings[original_idx] = token_embeddings_raw[raw_idx]
                else: 
                    logger.warning(f"Token embedding shape mismatch or empty. Raw: {token_embeddings_raw.shape}, Valid indices: {len(valid_token_indices)}")
                    token_embeddings = None 
            else: token_embeddings = None
        except Exception as e:
            logger.error(f"Error generating token embeddings: {e}", exc_info=True)
            token_embeddings = None

    for i, token_obj_or_dict in enumerate(tokens_to_process_objs):
        is_spacy_token_loop = isinstance(token_obj_or_dict, spacy.tokens.Token)
        token_span = (token_obj_or_dict.idx, token_obj_or_dict.idx + len(token_obj_or_dict.text)) if is_spacy_token_loop else token_obj_or_dict.get('span')
        if token_span is None: continue 
        
        token_text = token_obj_or_dict.text if is_spacy_token_loop else token_obj_or_dict['text']
        is_stopword_flag = (token_obj_or_dict.is_stop or token_obj_or_dict.lower_ in ENGLISH_STOPWORDS_GLOBAL) if is_spacy_token_loop else token_obj_or_dict['is_stop']
        
        if any(span_overlaps(token_span, rep_span) for rep_span in replacement_spans): continue
        
        current_token_embedding = None
        if token_embeddings is not None and i < token_embeddings.shape[0]:
            current_token_embedding = token_embeddings[i]
            if np.all(current_token_embedding == 0): 
                current_token_embedding = None
        
        exact_match_term, potential_matches = get_token_match_scores(
            token_obj_or_dict, current_token_embedding, 
            dictionary_map_lower_to_original, dictionary_list_original, 
            dictionary_list_lower, faiss_index if active_use_embedding_eff else None, 
            spell_checker if params.use_token_spellcheck else None, # Pass main spell_checker for spellcheck_token method
            params
        )
        best_correction_result = None
        if exact_match_term: # This means an exact (case-insensitive) or exact lemma match was found in dictionary
            details = {
                "method": "exact_dictionary_match", # Renamed for clarity
                "score": 1.0, "confidence": "high", 
                "pos_tag": token_obj_or_dict.pos_ if is_spacy_token_loop else token_obj_or_dict.get('pos_', "X"), 
                "matched_on_lemma": (exact_match_term.lower() != token_text.lower() and 
                                     (token_obj_or_dict.lemma_.lower() if is_spacy_token_loop else token_obj_or_dict.get('lemma_',"").lower()) == exact_match_term.lower()),
                "weighted_score": 1.0 * max(params.lev_weight if params.use_levenshtein else 0, 
                                           params.emb_weight if active_use_embedding_eff else 0, 1.0)
            }
            # The actual replacement decision (if token_text.lower() != exact_match_term.lower()) happens below
            best_correction_result = (exact_match_term, details)
        else: # No exact match, so try fuzzy matching from potential_matches
            best_correction_result = select_best_correction(
                token_text, 
                potential_matches, 
                params, 
                spell_checker, # Pass main spell_checker for the known-word check
                is_stopword_flag
            )
        
        if best_correction_result:
            replacement_term, details = best_correction_result
            # Only apply if there's an actual change (case-insensitive)
            if token_text.lower() != replacement_term.lower() and not any(span_overlaps(token_span, rep_span) for rep_span in replacement_spans):
                 applied_corrections.append({"original_text": token_text, "replacement": replacement_term, "original_span": token_span, "details": details, "type": "token"})
                 add_covered_span(token_span, replacement_spans)
            # Case: Exact match found, but original text was different only by case. This is a case normalization.
            elif token_text != replacement_term and token_text.lower() == replacement_term.lower() and details.get("method") == "exact_dictionary_match":
                 applied_corrections.append({"original_text": token_text, "replacement": replacement_term, "original_span": token_span, "details": details, "type": "token_case_norm"})
                 add_covered_span(token_span, replacement_spans)


    applied_corrections.sort(key=lambda x: x["original_span"][0])
    final_sentence = original_sentence; offset = 0
    for correction in applied_corrections:
        original_text_corr, replacement_corr, original_span_corr = correction["original_text"], correction["replacement"], correction["original_span"]
        start = original_span_corr[0] + offset; end = original_span_corr[1] + offset
        if 0 <= start <= end <= len(final_sentence) and len(final_sentence[start:end]) == (original_span_corr[1] - original_span_corr[0]):
            final_sentence = final_sentence[:start] + replacement_corr + final_sentence[end:]
            offset += len(replacement_corr) - (original_span_corr[1] - original_span_corr[0])
    corrected_sentence_stage1 = re.sub(r'\b(\S+)(\s+\1)+\b', r'\1', final_sentence, flags=re.IGNORECASE)
    corrected_sentence = re.sub(r'\s+', ' ', corrected_sentence_stage1).strip()
    final_replacements: Dict[str, str] = {}; final_details: Dict[str, Dict[str, Any]] = {}; keys_used = defaultdict(int)
    for corr in applied_corrections:
        original_text_fin, replacement_fin, details_fin = corr['original_text'], corr['replacement'], corr['details']
        key_base = original_text_fin.strip() or f"span_{corr['original_span'][0]}_{corr['original_span'][1]}"
        count = keys_used[key_base]; keys_used[key_base] += 1
        final_key = f"{key_base}_{count}" if count > 0 else key_base
        final_replacements[final_key] = replacement_fin
        details_copy = {}
        for k, v in details_fin.items():
             if isinstance(v, (np.integer)): details_copy[k] = int(v)
             elif isinstance(v, (np.floating)): details_copy[k] = float(v)
             elif isinstance(v, (np.bool_)): details_copy[k] = bool(v)
             elif isinstance(v, (set, tuple, np.ndarray)): details_copy[k] = list(v)
             else:
                  try: json.dumps({k: v}); details_copy[k] = v
                  except TypeError: details_copy[k] = str(v)
        details_copy['original_span'] = list(corr['original_span']); details_copy['replacement_term'] = replacement_fin
        details_copy['correction_source'] = corr.get('type', 'unknown'); details_copy['match_method'] = details_fin.get('method', details_copy.get('method', 'unknown'))
        final_details[final_key] = details_copy
    return corrected_sentence, final_replacements, final_details
# --- End of Core Logic Functions ---


# --- Resource Loading and Management ---
def get_resource_hash(params: CorrectionRequest) -> str:
    critical_parts = (
        params.data_path,
        params.sentence_model_name,
        params.spacy_model_name,
    )
    m = hashlib.sha256()
    for part in critical_parts:
        m.update(str(part).encode('utf-8'))
    return m.hexdigest()

def load_resources_for_path(params: CorrectionRequest) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    resource_hash = get_resource_hash(params)

    if resource_hash in LOADED_RESOURCES and LOADED_RESOURCES[resource_hash]["is_loaded"]:
        logger.info(f"Using cached resources for hash: {resource_hash} (Path: {params.data_path})")
        current_resources = LOADED_RESOURCES[resource_hash]
        if (params.use_embedding or params.use_ngram) and current_resources.get("st_model") is None:
            current_resources["st_model"] = load_transformer_model(params.sentence_model_name)
            if current_resources["st_model"] is None: return None, f"Error: Failed to load ST model '{params.sentence_model_name}' for cached resources."
        if params.use_spacy and current_resources.get("spacy_nlp") is None: 
            current_resources["spacy_nlp"] = load_spacy_model(params.spacy_model_name)
        elif not params.use_spacy and current_resources.get("spacy_nlp") is not None: 
            pass 

        if (params.use_token_spellcheck or params.use_ngram_spellcheck) and current_resources.get("spell_checker") is None:
            current_resources["spell_checker"] = get_spell_checker()
        return current_resources, None

    logger.info(f"Loading new resources for hash: {resource_hash} (Path: {params.data_path})")
    
    actual_data_path = os.path.join(BASE_DIR, params.data_path) if not os.path.isabs(params.data_path) else params.data_path
    if not os.path.isdir(actual_data_path):
        return None, f"Error: Provided data_path '{actual_data_path}' (resolved from '{params.data_path}') is not a valid directory."

    resources = {"is_loaded": False, "data_path": actual_data_path, "resource_hash": resource_hash}

    resources["map_lower_to_original"] = load_json_file(os.path.join(actual_data_path, "map_lower_to_original.json"))
    resources["dictionary_list_original"] = load_json_file(os.path.join(actual_data_path, "dictionary_list_original.json"))
    resources["dictionary_list_lower"] = load_json_file(os.path.join(actual_data_path, "dictionary_list_lower.json"))
    resources["ngram_index"] = load_json_file(os.path.join(actual_data_path, "ngram_index.json"))
    
    essential_dict_artifacts = [resources["map_lower_to_original"], resources["dictionary_list_original"], resources["dictionary_list_lower"]]
    if not all(essential_dict_artifacts):
        missing = [name for name, val in zip(["map_lower_to_original.json", "dictionary_list_original.json", "dictionary_list_lower.json"], essential_dict_artifacts) if val is None]
        return None, f"Error: Essential dictionary artifacts missing from '{actual_data_path}': {missing}."

    resources["faiss_index"] = load_faiss_index_file(os.path.join(actual_data_path, "dictionary_faiss.index"))
    if resources["faiss_index"] is None and params.use_embedding: 
         logger.warning(f"FAISS index not found in '{actual_data_path}', embedding for tokens will be disabled if params.use_embedding is True.")
    if resources["ngram_index"] is None and params.use_ngram: 
         logger.warning(f"Ngram index not found in '{actual_data_path}', n-gram matching will be disabled if params.use_ngram is True.")

    resources["st_model"] = None
    if params.use_embedding or params.use_ngram: 
        resources["st_model"] = load_transformer_model(params.sentence_model_name)
        if resources["st_model"] is None: return None, f"Error: Failed to load ST model '{params.sentence_model_name}' which is required when use_embedding or use_ngram is True."
    
    resources["spacy_nlp"] = None
    if params.use_spacy:
        resources["spacy_nlp"] = load_spacy_model(params.spacy_model_name)

    resources["spell_checker"] = None
    if params.use_token_spellcheck or params.use_ngram_spellcheck or True: # Always load spell_checker if available for the known-word check
        resources["spell_checker"] = get_spell_checker()

    resources["is_loaded"] = True
    LOADED_RESOURCES[resource_hash] = resources
    logger.info(f"Successfully loaded resources for hash: {resource_hash}")
    return resources, None


# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    global nltk_ready_global, ENGLISH_STOPWORDS_GLOBAL
    nltk_ready_global = download_nltk_data_startup()
    if nltk_ready_global:
        from nltk.corpus import stopwords
        ENGLISH_STOPWORDS_GLOBAL = set(stopwords.words('english'))
    else: 
        ENGLISH_STOPWORDS_GLOBAL = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"} 
    logger.info("NLTK setup complete.")


# --- FastAPI Endpoints ---
@app.get("/", summary="API Welcome Message", response_model=Dict[str, str])
async def read_root():
    return {"message": "Welcome to the Term Normalization API. Use /docs to see available endpoints and test."}

@app.post("/correct/", summary="Correct a Single Sentence", response_model=CorrectionResponse)
async def correct_sentence_endpoint(
    correction_input: CorrectionRequest = Body(...)
):
    resources, error_msg = load_resources_for_path(correction_input)
    
    if error_msg or resources is None:
        raise HTTPException(status_code=400, detail=error_msg or "Failed to load necessary resources.")

    if correction_input.use_embedding and resources.get("faiss_index") is None:
        raise HTTPException(status_code=400, detail=f"FAISS index is required because use_embedding is True, but it was not found or failed to load from '{resources.get('data_path')}'.")
    if correction_input.use_ngram and resources.get("ngram_index") is None:
        raise HTTPException(status_code=400, detail=f"N-gram index is required because use_ngram is True, but it was not found or failed to load from '{resources.get('data_path')}'.")
    if (correction_input.use_embedding or correction_input.use_ngram) and resources.get("st_model") is None:
         raise HTTPException(status_code=400, detail=f"Sentence Transformer model '{correction_input.sentence_model_name}' is required for embeddings/n-grams but failed to load.")
    if correction_input.use_spacy and resources.get("spacy_nlp") is None:
        logger.warning(f"SpaCy model '{correction_input.spacy_model_name}' requested but not loaded. Proceeding with limited spaCy functionality (e.g. basic tokenization if n-grams enabled).")
    
    # Spell checker is now more broadly useful for the known-word check, so we only warn if it's specifically requested for active spellchecking features but failed to load.
    if (correction_input.use_token_spellcheck or correction_input.use_ngram_spellcheck) and resources.get("spell_checker") is None:
        logger.warning(f"Spell checker explicitly requested for token/n-gram spellcheck but not loaded. These features will be disabled.")
    elif resources.get("spell_checker") is None: # If not explicitly requested but still not loaded
        logger.info("Spell checker not loaded. The 'known word' precaution in token correction will be less effective.")


    try:
        start_time = time.time()
        corrected_sentence, final_replacements, final_details = correct_sentence_hybrid(
            correction_input.sentence_to_correct,
            correction_input, 
            resources
        )
        processing_time = time.time() - start_time

        return CorrectionResponse(
            original_sentence=correction_input.sentence_to_correct,
            corrected_sentence=corrected_sentence,
            replacements_map=final_replacements,
            details_map=final_details,
            processing_time_seconds=round(processing_time, 4),
            parameters_used=correction_input 
        )
    except Exception as e:
        logger.error(f"Error during sentence correction endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)
