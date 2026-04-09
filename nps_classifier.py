"""
NPS Feedback Categorization Pipeline
=====================================
High-precision classifier for FamApp NPS survey responses.
Prioritizes precision over recall — returns NULL when not confident.

Author: ML Pipeline
Product: FamApp (fintech for teenagers in India)
"""

import ssl
import httpx

# Fix macOS SSL certificate issues with HuggingFace Hub
ssl._create_default_https_context = ssl._create_unverified_context
_orig_client_init = httpx.Client.__init__
def _ssl_patched_client(self, *args, **kwargs):
    kwargs['verify'] = False
    _orig_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _ssl_patched_client
_orig_async_client_init = httpx.AsyncClient.__init__
def _ssl_patched_async_client(self, *args, **kwargs):
    kwargs['verify'] = False
    _orig_async_client_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = _ssl_patched_async_client

import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.78      # Minimum cosine similarity to assign a category
MARGIN_THRESHOLD     = 0.05      # Min gap between top-2 scores; if smaller → NULL
MIN_TOKENS           = 2         # Minimum word count after cleaning
TOP_K                = 7         # K nearest neighbors for voting
MAX_PER_CATEGORY     = 600       # Cap examples per category to reduce majority bias
MODEL_NAME           = "all-MiniLM-L6-v2"   # Fast, good for short texts

BASE_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────
# STEP 1 — UNIFIED CATEGORY TAXONOMY
# ─────────────────────────────────────────────────────────

# Maps raw labels from each training file → unified category
# Categories not listed here will be treated as NULL examples
CATEGORY_MAP = {
    # FEES_AND_CHARGES
    "Fees and Charges":                         "FEES_AND_CHARGES",
    "Fee & Charges":                            "FEES_AND_CHARGES",
    "Fee and Charges":                          "FEES_AND_CHARGES",

    # LIMITS
    "Deposit/Spend Limit":                      "LIMITS",
    "Limits ":                                  "LIMITS",
    "Limits":                                   "LIMITS",
    "Transaction Limit":                        "LIMITS",
    "Limits due to KYC ":                       "LIMITS",
    "Limits due to KYC":                        "LIMITS",

    # REWARDS_AND_CASHBACK
    "More rewards/cashbacks":                   "REWARDS_AND_CASHBACK",
    "Cashback & Rewards":                       "REWARDS_AND_CASHBACK",
    "Cashback and Rewards":                     "REWARDS_AND_CASHBACK",

    # PAYMENT_FAILURES
    "Failed Payments":                          "PAYMENT_FAILURES",
    "Money debited but credit not received":    "PAYMENT_FAILURES",

    # PAYMENT_UX_ISSUES  (issues navigating payments, not crashes)
    "Payment UX Issues":                        "PAYMENT_UX_ISSUES",
    "Payment experience issues":                "PAYMENT_UX_ISSUES",

    # PAYMENT_SPEED
    "Improvement in Payment Speed":             "PAYMENT_SPEED",
    "Payment Speed XP ":                        "PAYMENT_SPEED",
    "Payment Speed XP":                         "PAYMENT_SPEED",
    "Payment Speed Issues":                     "PAYMENT_SPEED",

    # BUGS_AND_LAGS
    "Bugs & Lags":                              "BUGS_AND_LAGS",
    "Bugs or lags":                             "BUGS_AND_LAGS",
    "App not working":                          "BUGS_AND_LAGS",

    # NETWORK_ISSUES
    "Low/No Internet Payments":                 "NETWORK_ISSUES",
    "Negative Network XP":                      "NETWORK_ISSUES",
    "Network Issues":                           "NETWORK_ISSUES",

    # RELIABILITY_ISSUES
    "Reliability Issues ":                      "RELIABILITY_ISSUES",
    "Reliability Issues":                       "RELIABILITY_ISSUES",
    "Server Downtimes":                         "RELIABILITY_ISSUES",
    "Server Downtime":                          "RELIABILITY_ISSUES",

    # AUTOPAY_ISSUES
    "Autopay ":                                 "AUTOPAY_ISSUES",
    "Autopay":                                  "AUTOPAY_ISSUES",
    "AutoPay XP":                               "AUTOPAY_ISSUES",
    "Autopay issues ":                          "AUTOPAY_ISSUES",
    "Autopay issues":                           "AUTOPAY_ISSUES",

    # STUCK_TRANSACTIONS_AND_REFUNDS
    "Refunds and Stuck Txn":                    "STUCK_TRANSACTIONS_AND_REFUNDS",
    "General Refund problems":                  "STUCK_TRANSACTIONS_AND_REFUNDS",
    "Refund for wrong debits":                  "STUCK_TRANSACTIONS_AND_REFUNDS",
    "Refund Issues ":                           "STUCK_TRANSACTIONS_AND_REFUNDS",
    "Refund Issues":                            "STUCK_TRANSACTIONS_AND_REFUNDS",

    # RECHARGE_ISSUES
    "Recharge UX Problems":                     "RECHARGE_ISSUES",
    "Recharge & Giftcard issue":                "RECHARGE_ISSUES",
    "Recharge Issues":                          "RECHARGE_ISSUES",

    # FAMCARD_ISSUES
    "FamCard":                                  "FAMCARD_ISSUES",
    "FamX card Issues":                         "FAMCARD_ISSUES",
    "Famx or Ultra issues":                     "FAMCARD_ISSUES",

    # KYC_AND_VERIFICATION
    "Limits due to KYC":                        "KYC_AND_VERIFICATION",
    "KYC Issues":                               "KYC_AND_VERIFICATION",
    "KYC & Verification":                       "KYC_AND_VERIFICATION",
    "KYC & Verification issues":                "KYC_AND_VERIFICATION",

    # SUPPORT_ISSUES
    "Support Issues":                           "SUPPORT_ISSUES",
    "Negative Customer Support":                "SUPPORT_ISSUES",
    "Customer Support":                         "SUPPORT_ISSUES",

    # ACCOUNT_BLOCKED
    "Account Blocked/inactive":                 "ACCOUNT_BLOCKED",
    "suspended or blocked":                     "ACCOUNT_BLOCKED",
    "Suspended":                                "ACCOUNT_BLOCKED",

    # LOGIN_ISSUES
    "Logout & Login XP":                        "LOGIN_ISSUES",
    "Logout issues":                            "LOGIN_ISSUES",
    "Login Issues":                             "LOGIN_ISSUES",

    # UPI_AND_TPAP_ISSUES
    "UPI Mapper Issues":                        "UPI_AND_TPAP_ISSUES",
    "TPAP/Linking Bank Account":                "UPI_AND_TPAP_ISSUES",
    "TPAP related issues":                      "UPI_AND_TPAP_ISSUES",
    "TPAP":                                     "UPI_AND_TPAP_ISSUES",
    "TPAP ":                                    "UPI_AND_TPAP_ISSUES",
    "SIM Binding/2FA  Issue":                   "UPI_AND_TPAP_ISSUES",
    "SIM Binding/2FA Issue":                    "UPI_AND_TPAP_ISSUES",
    "Friends or family using other TPAP":       "UPI_AND_TPAP_ISSUES",

    # QR_CODE_ISSUES
    "QR Code Issues":                           "QR_CODE_ISSUES",
    "Issues with QR":                           "QR_CODE_ISSUES",

    # MERCHANT_ISSUES
    "Declines from Merchant":                   "MERCHANT_ISSUES",
    "Not visible as a UPI App on merchant websites/apps": "MERCHANT_ISSUES",
    "App not supported by merchants":           "MERCHANT_ISSUES",

    # DIGIGOLD_ISSUES
    "DigiGold related issues":                  "DIGIGOLD_ISSUES",
    "Negative Digigold & keeper XP":            "DIGIGOLD_ISSUES",
    "DigiGold Charges":                         "DIGIGOLD_ISSUES",
    "Keeper issues":                            "DIGIGOLD_ISSUES",

    # GIFTCARD_ISSUES
    "Giftcards related feedback":               "GIFTCARD_ISSUES",
    "Giftcard issues":                          "GIFTCARD_ISSUES",

    # UX_IMPROVEMENT (general UI/UX feedback, not payment-specific)
    "General UX Issues":                        "UX_IMPROVEMENT",
    "UX Improvement":                           "UX_IMPROVEMENT",
    "Need Improvement in UI":                   "UX_IMPROVEMENT",
    "App looks very childish":                  "UX_IMPROVEMENT",
    "Needs more simple UX":                     "UX_IMPROVEMENT",
    "Needs App in Hindi":                       "UX_IMPROVEMENT",

    # PPI_PIN_ISSUES
    "Needs App level PIN for PPI Txn Authorization": "PPI_PIN_ISSUES",
    "PIN for PPI Txn Authorization":            "PPI_PIN_ISSUES",
    "PPI Txn Pin":                              "PPI_PIN_ISSUES",
    "PIN for seeing balance":                   "PPI_PIN_ISSUES",

    # METRO_ISSUES
    "Metro Payment Related Issues":             "METRO_ISSUES",
    "Issues in metro tickets":                  "METRO_ISSUES",

    # NEW_FEATURES_REQUEST
    "Wants newer features":                     "NEW_FEATURES_REQUEST",
    "New Feature suggestion":                   "NEW_FEATURES_REQUEST",
    "Wants Pay Later Options":                  "NEW_FEATURES_REQUEST",
    "Needs investment related features":        "NEW_FEATURES_REQUEST",
    "Likes Investment Features ":               "NEW_FEATURES_REQUEST",
    "Need investment features":                 "NEW_FEATURES_REQUEST",
    "Methods to earn money":                    "NEW_FEATURES_REQUEST",
    "want loan feature":                        "NEW_FEATURES_REQUEST",
    "Wants Collect Request option":             "NEW_FEATURES_REQUEST",
    "wants a bank acc":                         "NEW_FEATURES_REQUEST",
    "Bank Account Opening":                     "NEW_FEATURES_REQUEST",
    "Want cash deposit":                        "NEW_FEATURES_REQUEST",
    "Wants BBPS Categories":                    "NEW_FEATURES_REQUEST",
    "Wants International Payments using FamCard": "NEW_FEATURES_REQUEST",
    "No alert to parents for payments":         "NEW_FEATURES_REQUEST",
    "Parents notification":                     "NEW_FEATURES_REQUEST",
    "Option to block other users":              "NEW_FEATURES_REQUEST",

    # PROFILE_AND_ACCOUNT_MANAGEMENT
    "Name Change":                              "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Email Change":                             "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Change Email/Phone":                       "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Profile Customization":                    "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Like Customization inapp":                 "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "in-app Permissions":                       "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Permission Issues":                        "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Notification Related Issues":              "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Account Statement":                        "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Add/Delete Beneficiary Related Issues":    "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Finding Transaction History ":             "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Finding Transaction History":              "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Option to hide transactions from history": "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Access FamApp from Laptop":                "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Custom VPA Issues":                        "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Spend Analytics":                          "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "DM Experience":                            "PROFILE_AND_ACCOUNT_MANAGEMENT",

    # POSITIVE_FEEDBACK — training rows where users clearly expressed satisfaction
    # These teach the model what "positive but no specific complaint" looks like
    "Positive & Inconclusive":                  "POSITIVE_FEEDBACK",
    "Positive and Inconclusive ":               "POSITIVE_FEEDBACK",
    "Positive and Inconclusive":                "POSITIVE_FEEDBACK",
    "Positive experience ":                     "POSITIVE_FEEDBACK",
    "Positive experience":                      "POSITIVE_FEEDBACK",
    "Positive payment experience ":             "POSITIVE_FEEDBACK",
    "Positive payment experience":              "POSITIVE_FEEDBACK",
    "Positive UI experience":                   "POSITIVE_FEEDBACK",
    "Positive Teen App":                        "POSITIVE_FEEDBACK",
    "Positive Payment XP":                      "POSITIVE_FEEDBACK",
    "Positive Investment XP":                   "POSITIVE_FEEDBACK",
    "Positive UI/UX":                           "POSITIVE_FEEDBACK",
    "positive and doesnt need a bank acc":      "POSITIVE_FEEDBACK",
    "AVG & Looking forward for updates":        "POSITIVE_FEEDBACK",

    # INCONCLUSIVE — users expressed dissatisfaction but reason is unclear
    "Negative & Inconclusive":                  "INCONCLUSIVE",
    "Negative & Inconclusive ":                 "INCONCLUSIVE",
    "Negative":                                 "INCONCLUSIVE",
    "Positive/inconclusive/Negative":           "INCONCLUSIVE",

    # Map some previously-NULL labels to existing actionable categories
    "More Virtual Assets":                      "NEW_FEATURES_REQUEST",
    "Subscription ":                            "NEW_FEATURES_REQUEST",
    "Subscription":                             "NEW_FEATURES_REQUEST",
    "Does not like regular Updates":            "UX_IMPROVEMENT",

    # Previously unmapped labels discovered during audit
    "Likes Investment Features":                "POSITIVE_FEEDBACK",    # positive investment mentions
    "Likes Investment Features ":               "POSITIVE_FEEDBACK",
    "Negative Profile customization XP":        "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "Fam 2.0 Issues":                           "BUGS_AND_LAGS",
    "Fraud related concerns":                   "ACCOUNT_BLOCKED",
    "Free FamCard":                             "NEW_FEATURES_REQUEST",
    "Need new features":                        "NEW_FEATURES_REQUEST",
    "Option to change phone number for alert to parents": "PROFILE_AND_ACCOUNT_MANAGEMENT",
    "QR Scanner":                               "QR_CODE_ISSUES",
}

# These raw labels cannot be meaningfully categorized → treat as NULL training examples
# (Rows with these labels are excluded from KNN training entirely)
NULL_CATEGORIES = {
    "Gibberish",          # Random / unintelligible text — can't train on it reliably
    "Other",              # Catch-all with no actionable signal
    "Brand Positioning",  # Marketing/brand commentary, not a product issue
}


# ─────────────────────────────────────────────────────────
# STEP 2 — HINGLISH NORMALIZATION
# ─────────────────────────────────────────────────────────

HINGLISH_RULES = [
    # Payment failures / stuck
    (r'\b(payment|paise|paisa|amount|txn|transaction)\s*(nahi|nahin|ni|nhi|mat|fail|failed|stuck|pending|nai)\b', 'payment failed'),
    (r'\b(payment|transaction)\s*(ho\s*)?nahi\s*(ho\s*raha|ja\s*raha|hua|gaya)\b', 'payment failed'),
    (r'\bfail\s*ho\s*(gaya|gyi|gayi|gaya|gye)\b', 'failed'),
    (r'\bnahi\s*(ho\s*raha|ja\s*raha|chal\s*raha|work\s*kar\s*raha)\b', 'not working'),
    (r'\b(kyu|kyon|kyun|kyou)\s*(fail|nahi|band|slow)\b', 'why failed'),
    (r'\b(stuck|atka|atka\s*hua)\b', 'stuck'),
    (r'\bpaise\s*(wapas|refund|return)\b', 'refund'),
    (r'\bpaise\s*(nahi|nahin)\s*(aaya|aya|mila)\b', 'payment not received'),
    (r'\bpaise\s*(kat|kata|gaya|gye)\b', 'money deducted'),

    # App quality
    (r'\b(bahut|bohot|boht|bht|bhut)\s*(slow|lag|fast|accha|acha|bura|bekar|bakwas)\b',
     lambda m: f'very {m.group(2)}'),
    (r'\b(app|yapp)\s*(slow|hang|crash|band)\b', 'app slow'),
    (r'\b(app|yapp)\s*(acha|accha|mast|badhiya|badiya|sahi|theek|thik)\b', 'app good'),
    (r'\b(nahi|nahin|ni|nhi)\s*(chal|kaam|work)\s*(raha|rahi)\b', 'not working'),

    # Fees / charges
    (r'\b(charge|charges|katna|katta|kat\s*raha|kat\s*rahi|fee|fees)\b', 'charges fee'),
    (r'\b(monthly|mahina|mahine)\s*(charge|fee|cut|kat)\b', 'monthly charge'),
    (r'\b(paise|amount|rs|rupee|rupees)\s*(kat|kata|katta)\b', 'amount deducted'),

    # Limits
    (r'\b(limit|seema)\s*(badha|badhao|increase|kam|chhota)\b', 'limit increase'),
    (r'\b(spend|deposit|transaction)\s*limit\b', 'spending limit'),

    # Rewards / cashback
    (r'\b(cashback|cash\s*back|reward|rewards)\s*(nahi|nahin|chahiye|do|dena)\b', 'cashback needed'),
    (r'\b(zyada|jyada|more|adhik)\s*(cashback|reward)\b', 'more cashback'),

    # Support
    (r'\b(customer|support|helpline|help)\s*(nahi|nahin|bekar|kharab|bura)\b', 'support bad'),
    (r'\b(reply|response|jawab)\s*(nahi|nahin)\b', 'no response'),

    # Server / reliability — map explicitly to server downtime vocabulary
    (r'\bserver\s*(down|crash|issue|problem|nahi|band|fail|not\s*working)\b',
     'server downtime server crash server not working'),
    (r'\b(server|app)\s*(down|crash|hang)\b', 'server downtime server crash'),
    (r'\bserver\s*down\b', 'server downtime server crash'),

    # Limits (strengthen signal — avoid "amount" which bleeds into fees)
    (r'\b(payment|deposit|spend|spending)\s*limit\b', 'deposit spending limit'),
    (r'\blimit\s*(badha|increase|zyada|jyada|more|low|less|kam|chhota|nahi)\b', 'deposit spending limit increase'),
    (r'\b(limit|seema)\s*(nahi|nahin|kam|chhota|low|thoda)\b', 'deposit spending limit too low'),

    # Stuck / refund
    (r'\b(paise|amount|money|rs)\s*(nahi|nahin|nhi)\s*(aaya|aya|mila|received|credit)\b', 'payment not received refund'),
    (r'\b(money|paise|amount)\s*(stuck|hold|pending|not\s*received)\b', 'payment stuck refund'),

    # Praise (generic)
    (r'\b(mast|badhiya|badiya|zabardast|jhakaas|bindaas|sahi|theek|thik)\s*(app|hai|he|h)\b', 'good app'),
    (r'\b(acha|accha|aacha|achha)\s*(lag|lagta|hai)\b', 'feels good'),
    (r'\bbahut\s*(acha|accha)\b', 'very good'),
]

def normalize_hinglish(text: str) -> str:
    """Apply rule-based Hinglish → English normalization."""
    for pattern, replacement in HINGLISH_RULES:
        if callable(replacement):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        else:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ─────────────────────────────────────────────────────────
# STEP 3 — TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)

# Common meaningless short phrases that should always return NULL
NULL_PHRASES = {
    "ok", "okay", "k", "good", "nice", "great", "fine", "yes", "no",
    "thanks", "thank you", "ty", "thx", "👍", "👎", "😊", "🙂",
    "idk", "idc", "na", "nah", "hmm", "hm", "lol", "lmao",
    "nothing", "nil", "none", "no issues", "no problem", "no issue",
    "all good", "all fine", "its fine", "it fine", "its ok", "all ok",
    "best", "worst", "amazing", "awesome", "excellent",
    "very good", "very bad", "very nice",
    "no", "nope", "yep", "yeah", "yup",
    "n/a", "na",
    ".", "..", "...", "!", "?", "??", "???",
    "-", "--",
    "satisfied", "unsatisfied", "happy", "unhappy",
    "perfect", "pathetic",
    "superb", "good app", "nice app", "best app",
    "no comment", "no comments", "nothing to say",
    "don t know", "dont know", "i dont know", "i don t know",
    "idk what", "idk why", "no idea",
    "helping teenagers", "helping teens", "helping kids",
    "good for teens", "good for teenagers", "good for kids",
    "best for teenagers", "best for teens", "best for kids",
    "useful for teens", "useful for teenagers", "useful for kids",
    "helpful for teens", "helpful for teenagers", "helpful for kids",
    "bad experience", "good experience", "nice experience", "worst experience",
    "poor experience", "great experience", "amazing experience",
    "bad app", "worse app", "love the app", "love this app",
    # Ambiguous non-reasons
    "no reason", "no specific reason", "no particular reason",
    "not sure", "not sure why", "not sure about",
    "paise nhi hai", "paisa nhi hai", "paise nahi hai",
    "nothing specific", "nothing much", "nothing else",
}

# Positive-only words: if text contains ONLY these (no issue keywords), → NULL
POSITIVE_WORDS = {
    "good", "great", "nice", "amazing", "awesome", "excellent", "superb",
    "helpful", "useful", "easy", "simple", "smooth", "fast", "quick",
    "love", "like", "best", "perfect", "wonderful", "fantastic", "brilliant",
    "reliable", "convenient", "effective", "efficient", "safe", "secure",
    "happy", "satisfied", "cool", "fun", "enjoy", "enjoyed", "enjoying",
    "impressive", "incredible", "outstanding", "fabulous", "lovely",
    "friendly", "clean", "clear", "intuitive", "beautiful", "attractive",
    "works", "working", "well", "fine", "okay", "ok",
    "really", "very", "so", "too", "much", "more", "most",
    "app", "application", "fam", "famapp", "fampay", "it", "its", "this", "the",
    "is", "are", "was", "be", "been", "for", "to", "of", "and", "a", "an",
    "i", "my", "me", "use", "using", "used",
    # App-identity / audience words — not complaints
    "teen", "teens", "teenager", "teenagers", "kid", "kids",
    "student", "students", "child", "children", "minor", "minors", "youth",
    "under", "below", "age",
    # Praise amplifiers
    "highly", "absolutely", "totally", "completely", "truly", "really",
    "recommend", "recommended", "recommending", "suggest", "suggested",
    "friends", "family", "everyone", "people",
    # Generic positive outcome words
    "helpful", "helping", "help", "support", "service",
    "transaction", "transactions", "payment", "payments",
    "digital", "money", "finance", "financial",
    "anywhere", "anytime", "always",
    "have", "has", "had", "with", "in", "on", "at", "by", "from",
    "going", "goes", "went", "smoothly", "perfectly", "properly",
    "quickly", "swiftly", "easily", "normally", "correctly",
    "well", "less", "time", "takes", "take", "faster", "faster",
}

# Issue signal words: presence of any of these means text is NOT purely positive
ISSUE_WORDS = {
    "not", "no", "can't", "cant", "cannot", "won't", "wont", "doesn't",
    "didn't", "don't", "isn't", "wasn't", "aren't", "haven't",
    "fail", "failed", "failing", "failure", "error", "bug", "crash", "crashed",
    "down", "outage",
    "slow", "lag", "stuck", "pending", "issue", "problem", "trouble",
    "bad", "worst", "terrible", "horrible", "pathetic", "useless",
    "charge", "fee", "deduct", "deducted", "lost",
    "limit", "block", "blocked", "suspend", "suspended",
    "refund", "return", "declined", "reject", "rejected",
    "nahi", "nahin", "nhi", "mat", "band", "kharab", "bekar",
}

def is_purely_positive(text: str) -> bool:
    """Return True if text appears to be generic positive praise with no issue signal."""
    tokens = set(text.lower().split())
    if tokens & ISSUE_WORDS:
        return False
    non_positive = tokens - POSITIVE_WORDS
    # Allow at most 3 unknown tokens (names, proper nouns, filler)
    return len(non_positive) <= 3

def score_to_threshold(score) -> float:
    """
    Adjust confidence threshold based on NPS score.
    High scorers are promoters — their feedback needs much stronger signal
    to be classified as an issue.
    Score 9-10 (Promoters):  very strict → 0.90
    Score 7-8  (Passives):   strict      → 0.83
    Score 4-6  (Detractors): standard    → 0.78
    Score 0-3  (Strong det): lenient     → 0.76
    Unknown / NaN:           standard    → 0.80
    """
    try:
        s = int(float(score))
    except (ValueError, TypeError):
        return CONFIDENCE_THRESHOLD
    if s >= 9:
        return 0.87
    if s >= 7:
        return 0.81
    if s >= 4:
        return 0.76
    return 0.70   # score 0-3: lowered from 0.74 to catch more clear detractor complaints

def preprocess_text(text: str) -> tuple[str, bool, str | None]:
    """
    Clean and normalize text.
    Returns (cleaned_text, is_valid, invalid_type).
      is_valid=True  → proceed to KNN classification
      is_valid=False → assign invalid_type directly as predicted_category:
          "EMPTY"     — blank, whitespace-only, or meaningless filler phrases
          "GIBBERISH" — random digits, consonant clusters, too-short / unreadable text
    """
    if not isinstance(text, str) or not text.strip():
        return "", False, "EMPTY"

    original = text.strip()

    # Remove emojis
    text = EMOJI_PATTERN.sub(" ", text)

    # Lowercase
    text = text.lower().strip()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # Remove special characters but keep letters, digits, spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Normalize repeated characters: "goooood" → "good"
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Pure digits only (e.g. "10", "9", "100") → GIBBERISH
    if re.match(r'^\d+$', text):
        return text, False, "GIBBERISH"

    # Check for pure-emoji / empty after stripping
    if not text:
        return "", False, "EMPTY"

    # PRE-normalization NULL check (uninformative filler phrases → EMPTY)
    if text in NULL_PHRASES:
        return text, False, "EMPTY"

    # Apply Hinglish normalization
    text = normalize_hinglish(text)
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return "", False, "EMPTY"

    if text in NULL_PHRASES:
        return text, False, "EMPTY"

    # Non-Latin script detection (Bengali, Devanagari, Tamil, Telugu, etc.)
    # These are real comments but in a script the model can't handle — label separately
    latin_chars = len(re.findall(r'[a-z]', text))
    total_alpha  = len(re.findall(r'[^\W\d_]', text, re.UNICODE))
    if total_alpha > 3 and latin_chars / max(total_alpha, 1) < 0.3:
        return text, False, "NON_ENGLISH"

    # Minimum token check: single token — check if it's a real word or true gibberish
    tokens = text.split()
    if len(tokens) < MIN_TOKENS:
        token = tokens[0] if tokens else ""
        # Single letters / pure digit strings / random consonant clusters → GIBBERISH
        if len(token) <= 2 or re.match(r'^[bcdfghjklmnpqrstvwxyz]{4,}$', token) or re.match(r'^\d+$', token):
            return text, False, "GIBBERISH"
        # Real single words (e.g. "super", "convenience", "fast") → EMPTY
        # They're valid English but too short to classify into a specific category
        return text, False, "EMPTY"

    # Gibberish detection: majority of tokens are random consonant clusters / single chars
    non_word_count = sum(
        1 for t in tokens
        if len(t) <= 1 or re.match(r'^[bcdfghjklmnpqrstvwxyz]{4,}$', t)
    )
    if len(tokens) > 0 and non_word_count / len(tokens) > 0.6:
        return text, False, "GIBBERISH"

    # Pass all other text to KNN — positive text will be classified as POSITIVE_FEEDBACK
    return text, True, None


# ─────────────────────────────────────────────────────────
# STEP 4 — LOAD TRAINING DATA
# ─────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    """Load all training files and return a unified DataFrame."""
    training_dir = BASE_DIR / "Training Datasets"
    frames = []

    file_configs = [
        {
            "path": training_dir / "August NPS.csv",
            "text_col": "reason",
            "label_col": "category",
            "score_col": None,
        },
        {
            "path": training_dir / "Dec NPS.csv",
            "text_col": "reason",
            "label_col": "Category",
            "score_col": None,
        },
        {
            "path": training_dir / "Dec Payment issues.csv",
            "text_col": "User input",
            "label_col": "Reason",
            "score_col": None,
        },
        {
            "path": training_dir / "Feb NPS.csv",
            "text_col": "reason",
            "label_col": "category",
            "score_col": None,
        },
        {
            "path": training_dir / "Jan NPS.csv",
            "text_col": "User input",
            "label_col": "Reason",
            "score_col": "score",   # use NPS score to label unlabeled rows
        },
        {
            "path": training_dir / "Jan Payment issues.csv",
            "text_col": "User input",
            "label_col": "Reason",
            "score_col": None,
        },
    ]

    for cfg in file_configs:
        df = pd.read_csv(cfg["path"])
        cols = [cfg["text_col"], cfg["label_col"]]
        if cfg["score_col"] and cfg["score_col"] in df.columns:
            cols.append(cfg["score_col"])
            sub = df[cols].copy()
            sub.columns = ["feedback_text", "raw_category", "nps_score"]
            # For rows with no label, assign proxy label from NPS score:
            #   score >= 9 → POSITIVE_FEEDBACK (promoters who didn't specify a complaint)
            #   score <= 5 → INCONCLUSIVE (unhappy but reason unknown)
            #   score 6-8  → leave as NaN (too ambiguous to proxy-label)
            mask_unlabeled = sub["raw_category"].isna()
            def proxy_label(score):
                try:
                    s = int(float(score))
                    if s >= 9:
                        return "POSITIVE_FEEDBACK_proxy"
                    if s <= 5:
                        return "INCONCLUSIVE_proxy"
                except (ValueError, TypeError):
                    pass
                return None
            sub.loc[mask_unlabeled, "raw_category"] = sub.loc[mask_unlabeled, "nps_score"].apply(proxy_label)
            sub = sub[["feedback_text", "raw_category"]].copy()
        else:
            sub = df[[cfg["text_col"], cfg["label_col"]]].copy()
            sub.columns = ["feedback_text", "raw_category"]
        sub["source"] = cfg["path"].name
        frames.append(sub)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["feedback_text"])
    combined["feedback_text"] = combined["feedback_text"].astype(str).str.strip()
    combined = combined[combined["feedback_text"] != ""]

    # Map to unified category
    combined["category"] = combined["raw_category"].str.strip().map(CATEGORY_MAP)

    # Handle proxy labels (not in CATEGORY_MAP — map them directly)
    proxy_map = {
        "POSITIVE_FEEDBACK_proxy": "POSITIVE_FEEDBACK",
        "INCONCLUSIVE_proxy":      "INCONCLUSIVE",
    }
    for proxy, cat in proxy_map.items():
        mask = combined["raw_category"].str.strip() == proxy
        combined.loc[mask, "category"] = cat

    # Mark NULL categories
    combined.loc[
        combined["raw_category"].str.strip().isin(NULL_CATEGORIES),
        "category"
    ] = "NULL"

    print(f"\n[Data] Loaded {len(combined):,} total training rows")
    print(f"[Data] Mapped categories: {combined['category'].notna().sum():,} rows")
    print(f"[Data] NULL/ambiguous: {(combined['category'] == 'NULL').sum():,} rows")
    print(f"[Data] Unmapped (excluded): {combined['category'].isna().sum():,} rows")

    # Keep only rows with a category (including NULL)
    combined = combined.dropna(subset=["category"])

    return combined


# ─────────────────────────────────────────────────────────
# STEP 5 — BUILD EMBEDDING INDEX
# ─────────────────────────────────────────────────────────

def build_embedding_index(df: pd.DataFrame, model: SentenceTransformer):
    """
    Build embedding matrix for training examples.
    Only use rows with actionable categories (not NULL).
    """
    # Only train on actionable categories
    train_df = df[df["category"] != "NULL"].copy()

    # Preprocess
    results = [preprocess_text(t) for t in train_df["feedback_text"]]
    train_df["cleaned"] = [r[0] for r in results]
    train_df["valid"] = [r[1] for r in results]
    train_df = train_df[train_df["valid"]].copy()
    train_df = train_df[train_df["cleaned"].str.strip() != ""]

    # Balance: cap each category to prevent majority-class bias
    balanced_frames = []
    for _, grp in train_df.groupby("category"):
        n = min(len(grp), MAX_PER_CATEGORY)
        balanced_frames.append(grp.sample(n=n, random_state=42))
    train_df = pd.concat(balanced_frames).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n[Embeddings] Building index from {len(train_df):,} training examples (balanced)")
    print(f"[Embeddings] Categories: {train_df['category'].nunique()}")

    texts = train_df["cleaned"].tolist()
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True,
                              normalize_embeddings=True)

    return train_df.reset_index(drop=True), embeddings


# ─────────────────────────────────────────────────────────
# KEYWORD OVERRIDES — high-specificity pattern → category
# Applied AFTER Hinglish normalization; bypasses KNN when
# a pattern unambiguously identifies the category.
# ─────────────────────────────────────────────────────────

KEYWORD_OVERRIDES: list[tuple] = [
    # Server downtime (any mention of "server down" or "server crash" → RELIABILITY)
    (re.compile(r'\bserver\s*(down|downtime|crash|not\s*working|band|fail)\b', re.I),
     "RELIABILITY_ISSUES"),
    # Payment limit specifically (not fee-related)
    (re.compile(r'\b(deposit|spend|spending)\s*limit\b', re.I), "LIMITS"),
    (re.compile(r'\blimit\s*(increase|badha|kam|too\s*low|low|less)\b', re.I), "LIMITS"),
    # Autopay specific
    (re.compile(r'\bautopay\b', re.I), "AUTOPAY_ISSUES"),
    # KYC / PAN specific — require issue verb nearby (not just mentioning kyc in passing)
    (re.compile(r'\b(pan\s*card|kyc)\s*\w*\s*(not|fail|issue|problem|error|reject|pending|required|need|done|complete|unable|can\s*t|cant)\b', re.I), "KYC_AND_VERIFICATION"),
    (re.compile(r'\b(not|fail|issue|problem|error|reject|pending|unable|can\s*t|cant)\s*\w*\s*(pan\s*card|kyc)\b', re.I), "KYC_AND_VERIFICATION"),
    (re.compile(r'\b(pan\s*card|kyc)\s*(verification|verify)\s*(fail|not|issue|problem|error|reject|required)\b', re.I), "KYC_AND_VERIFICATION"),
    (re.compile(r'\b(asking|ask|require|required|need|needed)\s*(pan|pan\s*card|kyc)\b', re.I), "KYC_AND_VERIFICATION"),
    # Refund specific
    (re.compile(r'\b(refund|wapas|money\s*back|return\s*money)\b', re.I),
     "STUCK_TRANSACTIONS_AND_REFUNDS"),
    # Account blocked
    (re.compile(r'\b(account|fampay)\s*(blocked|suspended|freeze|band|locked)\b', re.I),
     "ACCOUNT_BLOCKED"),
    (re.compile(r'\b(blocked|suspended|freeze|locked)\s*(account|fampay)\b', re.I),
     "ACCOUNT_BLOCKED"),
    # QR code
    (re.compile(r'\b(qr|scanner|qr\s*code)\b', re.I), "QR_CODE_ISSUES"),
    # Digi gold — ONLY "digi gold" or "digold", not bare "gold"
    (re.compile(r'\b(digi\s*gold|digold|digi-gold)\b', re.I), "DIGIGOLD_ISSUES"),
    # Metro
    (re.compile(r'\b(metro|dmrc)\b', re.I), "METRO_ISSUES"),
    # FamCard / FamX
    (re.compile(r'\b(famcard|fam\s*card|famx|fam\s*x|debit\s*card)\b', re.I),
     "FAMCARD_ISSUES"),
    # Recharge
    (re.compile(r'\b(recharge|mobile\s*recharge)\b', re.I), "RECHARGE_ISSUES"),
]

def apply_keyword_overrides(cleaned_text: str) -> str | None:
    """
    Returns a category string if a high-specificity keyword override fires,
    else returns None (proceed with KNN).
    Requires at least one ISSUE_WORDS token to be present — prevents
    overriding on positive mentions of the same keyword (e.g. "invest in gold").
    """
    tokens = set(re.findall(r'\w+', cleaned_text.lower()))
    if not (tokens & ISSUE_WORDS):
        return None  # No issue signal → skip overrides, let KNN handle
    for pattern, category in KEYWORD_OVERRIDES:
        if pattern.search(cleaned_text):
            return category
    return None


# ─────────────────────────────────────────────────────────
# STEP 6 — CLASSIFY SINGLE FEEDBACK
# ─────────────────────────────────────────────────────────

def classify_feedback(
    text: str,
    train_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    model: SentenceTransformer,
    threshold: float = CONFIDENCE_THRESHOLD,
    margin: float = MARGIN_THRESHOLD,
    top_k: int = TOP_K,
    nps_score=None,
) -> dict:
    """
    Classify a single feedback string.
    Returns dict with predicted_category and confidence_score.
    """
    cleaned, is_valid, invalid_type = preprocess_text(text)

    if not is_valid:
        # invalid_type is "EMPTY" or "GIBBERISH" — use directly as the predicted category
        cat = invalid_type if invalid_type else "NULL"
        return {"predicted_category": cat, "confidence_score": 0.0,
                "cleaned_text": cleaned, "null_reason": "invalid_text"}

    # Keyword override: if an unambiguous pattern fires, skip KNN
    override_cat = apply_keyword_overrides(cleaned)
    if override_cat is not None:
        return {
            "predicted_category": override_cat,
            "confidence_score": 1.0,
            "cleaned_text": cleaned,
            "null_reason": None,
        }

    # Encode query
    query_emb = model.encode([cleaned], normalize_embeddings=True)

    # Cosine similarity with all training examples
    sims = cosine_similarity(query_emb, train_embeddings)[0]

    # Get top-k indices
    top_indices = np.argsort(sims)[::-1][:top_k]
    top_scores  = sims[top_indices]
    top_cats    = train_df["category"].iloc[top_indices].tolist()

    # Weighted voting: weight = similarity score
    cat_weights: dict = {}
    for cat, score in zip(top_cats, top_scores):
        cat_weights[cat] = cat_weights.get(cat, 0.0) + float(score)

    if not cat_weights:
        return {"predicted_category": "NULL", "confidence_score": 0.0,
                "cleaned_text": cleaned, "null_reason": "no_neighbors"}

    # Sort by weight
    sorted_cats = sorted(cat_weights.items(), key=lambda x: x[1], reverse=True)
    best_cat, best_weight = sorted_cats[0]

    # Normalize confidence to a 0-1 score using max similarity
    confidence = float(top_scores[0])

    # Parse NPS score once
    nps_int = None
    try:
        nps_int = int(float(nps_score))
    except (ValueError, TypeError):
        pass

    # --- Guard 1: Confidence threshold ---
    # POSITIVE_FEEDBACK for NPS > 6 (promoters/passives): confident positive feedback
    # from satisfied users — 0.80 keeps precision high while capturing clear positives.
    # POSITIVE_FEEDBACK for NPS <= 6: stricter (0.84) — detractors shouldn't be classified
    # as positive unless the signal is very clear.
    # INCONCLUSIVE: same as high-score issue threshold (0.81) — only classify when clear.
    # All actionable issue categories: NPS-score-adjusted high thresholds (precision-first).
    if best_cat == "POSITIVE_FEEDBACK" and nps_int is not None and nps_int > 6:
        effective_threshold = 0.80
    elif best_cat == "POSITIVE_FEEDBACK":
        effective_threshold = 0.84
    elif best_cat == "INCONCLUSIVE":
        effective_threshold = 0.81
    else:
        effective_threshold = score_to_threshold(nps_score) if nps_score is not None else threshold

    if confidence < effective_threshold:
        return {"predicted_category": "NULL", "confidence_score": round(confidence, 4),
                "cleaned_text": cleaned, "null_reason": "low_confidence"}

    # --- Guard 2: Category margin check ---
    # Skip margin check when: POSITIVE_FEEDBACK predicted AND NPS >= 7.
    # A promoter/passive saying "very easy to use" (conf=1.0) is unambiguously positive
    # even if INCONCLUSIVE scores closely — the NPS score breaks the tie.
    skip_margin = (best_cat == "POSITIVE_FEEDBACK" and nps_int is not None and nps_int >= 7)
    if not skip_margin and len(sorted_cats) > 1:
        second_cat, second_weight = sorted_cats[1]
        total = best_weight + second_weight
        if total > 0:
            margin_score = (best_weight - second_weight) / total
            if margin_score < margin:
                return {"predicted_category": "NULL", "confidence_score": round(confidence, 4),
                        "cleaned_text": cleaned, "null_reason": "ambiguous_categories"}

    return {
        "predicted_category": best_cat,
        "confidence_score": round(confidence, 4),
        "cleaned_text": cleaned,
        "null_reason": None,
    }


def apply_confidence_threshold(results: list[dict]) -> pd.DataFrame:
    """Convert list of result dicts to DataFrame."""
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────
# STEP 7 — VALIDATION (internal split)
# ─────────────────────────────────────────────────────────

def validate_model(train_df, train_embeddings, model, sample_size=1500):
    """
    Evaluate on a held-out sample.
    Reports precision, recall, F1, null_rate.
    """
    print("\n[Validation] Running internal validation ...")

    # Sample across categories for balanced eval
    val_frames = []
    for cat, grp in train_df.groupby("category"):
        n = min(len(grp), max(10, sample_size // train_df["category"].nunique()))
        val_frames.append(grp.sample(n=n, random_state=42))

    val_df = pd.concat(val_frames).sample(frac=1, random_state=42)

    # Remove val samples from training embeddings
    # (approximate: we use leave-out by filtering by index)
    val_indices = set(val_df.index.tolist())
    mask_bool = np.array([i not in val_indices for i in train_df.index])
    reduced_train = train_df[mask_bool]
    reduced_embeddings = train_embeddings[mask_bool]

    y_true, y_pred = [], []
    for _, row in val_df.iterrows():
        result = classify_feedback(
            row["feedback_text"], reduced_train, reduced_embeddings, model
        )
        true_label = row["category"]
        pred_label = result["predicted_category"]
        y_true.append(true_label)
        y_pred.append(pred_label)

    # Compute metrics (treating NULL as one class)
    null_rate = sum(1 for p in y_pred if p == "NULL") / len(y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p and p != "NULL")
    classified = sum(1 for p in y_pred if p != "NULL")
    true_classified_correct = sum(
        1 for t, p in zip(y_true, y_pred)
        if p != "NULL" and t == p
    )
    true_positive_pred = classified  # all non-null predictions

    precision_approx = true_classified_correct / classified if classified > 0 else 0
    recall_approx = correct / len(y_true)

    print(f"  Validation samples: {len(val_df)}")
    print(f"  NULL rate:          {null_rate:.1%}")
    print(f"  Classified:         {classified} ({100-null_rate*100:.1f}%)")
    print(f"  Precision (approx): {precision_approx:.1%}")
    print(f"  Recall (approx):    {recall_approx:.1%}")

    return {
        "null_rate": null_rate,
        "precision": precision_approx,
        "recall": recall_approx,
    }


# ─────────────────────────────────────────────────────────
# STEP 8 — GENERATE OUTPUT
# ─────────────────────────────────────────────────────────

def generate_output(test_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """Merge results with original test data and write output CSV."""
    output = test_df.copy()
    output["cleaned_text"]       = results_df["cleaned_text"].values
    output["predicted_category"] = results_df["predicted_category"].values
    output["confidence_score"]   = results_df["confidence_score"].values
    output["null_reason"]        = results_df["null_reason"].values

    out_path = BASE_DIR / "output_classified.csv"
    output.to_csv(out_path, index=False)
    print(f"\n[Output] Saved → {out_path}")
    return output


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("NPS Feedback Categorization Pipeline")
    print("=" * 60)

    # ── Load model ──────────────────────────────────────────
    print(f"\n[Model] Loading sentence-transformer: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # ── Load training data ──────────────────────────────────
    train_raw = load_training_data()

    # ── Build embedding index ────────────────────────────────
    train_df, train_embeddings = build_embedding_index(train_raw, model)

    # ── Show category distribution ───────────────────────────
    print("\n[Training] Category distribution:")
    cat_counts = train_df["category"].value_counts()
    for cat, cnt in cat_counts.items():
        print(f"  {cnt:5d}  {cat}")

    # ── Validate ─────────────────────────────────────────────
    val_metrics = validate_model(train_df, train_embeddings, model)

    # ── Load test data ───────────────────────────────────────
    test_path = BASE_DIR / "Test" / "TEST.csv"
    test_df   = pd.read_csv(test_path)
    print(f"\n[Test] Loaded {len(test_df):,} test rows")

    # Use "reason" column as feedback text
    texts = test_df["reason"].astype(str).tolist()

    # ── Classify ─────────────────────────────────────────────
    scores = test_df["score"].tolist()
    print(f"\n[Classify] Running classification on {len(texts):,} records ...")
    results = []
    for i, (text, nps_score) in enumerate(zip(texts, scores)):
        if (i + 1) % 1000 == 0:
            print(f"  ... {i+1:,}/{len(texts):,}", flush=True)
        res = classify_feedback(text, train_df, train_embeddings, model, nps_score=nps_score)
        results.append(res)

    results_df = apply_confidence_threshold(results)

    # ── Generate output ──────────────────────────────────────
    output = generate_output(test_df, results_df)

    # ── Summary statistics ───────────────────────────────────
    NOISE_CATS    = {"GIBBERISH", "EMPTY", "NON_ENGLISH"}
    META_CATS     = {"POSITIVE_FEEDBACK", "INCONCLUSIVE"}
    UNRESOLVED    = {"NULL"}

    total         = len(output)
    actionable    = (~output["predicted_category"].isin(NOISE_CATS | META_CATS | UNRESOLVED)).sum()
    meta_count    = output["predicted_category"].isin(META_CATS).sum()
    noise_count   = output["predicted_category"].isin(NOISE_CATS).sum()
    null_count    = (output["predicted_category"] == "NULL").sum()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total test records:        {total:,}")
    print(f"  Actionable issues:         {actionable:,}  ({actionable/total*100:.1f}%)")
    print(f"  Sentiment (pos/neg/incl):  {meta_count:,}  ({meta_count/total*100:.1f}%)")
    print(f"  Noise (empty/gibberish):   {noise_count:,}  ({noise_count/total*100:.1f}%)")
    print(f"  NULL (low confidence):     {null_count:,}  ({null_count/total*100:.1f}%)")

    print("\n[Distribution] Predicted category counts:")
    pred_dist = output["predicted_category"].value_counts()
    for cat, cnt in pred_dist.items():
        pct = cnt / total * 100
        print(f"  {cnt:5d}  {pct:5.1f}%  {cat}")

    print("\n[Reasons for NULL]:")
    null_reasons = output[output["predicted_category"] == "NULL"]["null_reason"].value_counts()
    for reason, cnt in null_reasons.items():
        print(f"  {cnt:5d}  {reason}")

    print(f"\n[Validation Metrics]")
    print(f"  Precision: {val_metrics['precision']:.1%}")
    print(f"  Recall:    {val_metrics['recall']:.1%}")
    print(f"  NULL rate: {val_metrics['null_rate']:.1%}")

    # ── Clean output CSV ─────────────────────────────────────
    # Sort order: actionable issues → sentiment → noise (GIBBERISH/EMPTY) → NULL
    if "distinct_id" in output.columns:
        clean = output[["distinct_id", "score", "reason", "predicted_category", "confidence_score"]].copy()
        clean.columns = ["user_id", "nps_score", "feedback_text", "predicted_category", "confidence_score"]
    else:
        clean = output[["score", "reason", "predicted_category", "confidence_score"]].copy()
        clean.columns = ["nps_score", "feedback_text", "predicted_category", "confidence_score"]

    def sort_priority(cat):
        if cat in NOISE_CATS:   return 3
        if cat in UNRESOLVED:   return 4
        if cat in META_CATS:    return 2
        return 1  # actionable issue

    clean["_sort"] = clean["predicted_category"].map(sort_priority)
    clean_sorted = clean.sort_values(
        ["_sort", "predicted_category", "confidence_score"],
        ascending=[True, True, False]
    ).drop(columns=["_sort"])
    clean_path = BASE_DIR / "output_classified_clean.csv"
    clean_sorted.to_csv(clean_path, index=False)
    print(f"[Output] Clean CSV → {clean_path}")

    print("\n[Done] Pipeline complete.")
    return output


if __name__ == "__main__":
    output = main()
