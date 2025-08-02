"""
Qualitative evaluation helpers.
"""

import random
from typing import List, Dict


def analyze_errors(results: List[Dict], top_n: int = 10) -> List[Dict]:
  
    errors = [r for r in results if r.get("prediction") != r.get("reference")]
    return errors[:top_n]


def sample_failure_cases(results: List[Dict], num_samples: int = 5) -> List[Dict]:

    errors = [r for r in results if r.get("prediction") != r.get("reference")]
    return random.sample(errors, min(num_samples, len(errors)))
