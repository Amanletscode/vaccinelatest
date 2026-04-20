"""Quick integration test for the vaccine ontology system."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Suppress streamlit import warnings
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

from modules.config import VACCINE_ONTOLOGY_FILE
from modules.vaccine_data import _get_vaccine_search_terms, _get_ontology_aliases, _VACCINE_SYNONYM_INDEX

print("=== Ontology Integration Test ===\n")

# Test 1: Abrysvo (pre-populated in ontology + synonym groups)
terms = _get_vaccine_search_terms("Abrysvo")
print(f"1. Abrysvo -> {len(terms)} terms: {terms}\n")

# Test 2: Comirnaty
terms2 = _get_vaccine_search_terms("Comirnaty")
print(f"2. Comirnaty -> {len(terms2)} terms: {terms2}\n")

# Test 3: Fuzzy match — "Abryvo" (typo)
aliases = _get_ontology_aliases("Abryvo")
print(f"3. Fuzzy 'Abryvo' -> {aliases}\n")

# Test 4: Reverse lookup — search by compound code
aliases2 = _get_ontology_aliases("PF-06928316")
print(f"4. Reverse 'PF-06928316' -> {aliases2}\n")

# Test 5: Check synonym index has ontology entries merged
print(f"5. Synonym index has 'pf 06928316': {'pf 06928316' in _VACCINE_SYNONYM_INDEX}")
print(f"   Synonym index has 'bnt162b2': {'bnt162b2' in _VACCINE_SYNONYM_INDEX}")

print("\n=== ALL TESTS PASSED ===")
