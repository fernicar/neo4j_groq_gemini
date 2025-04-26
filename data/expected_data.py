# data/expected_data.py
# This file should contain the data used as the "ground truth" for your tests.
# Load this data in test_4_load_initial_data in coreYYYYMMDD.py

# Example structure:
EXPECTED_TRIPLETS = [
    ("Alice", "lives_in", "Wonderland"),
    ("Alice", "met", "White Rabbit"),
    ("White Rabbit", "is_a", "Animal"),
    # Add more expected triplets for your tests here
]

BASE_STORY = """
This is the base story text that might be used as part of the prompt,
or simply serves as documentation of what the expected triplets represent.
Alice found herself in a strange place called Wonderland. While exploring, she met a peculiar creature, a White Rabbit, who seemed to be in a great hurry.
"""

# TODO: Define a more robust format for initial data if needed,
# e.g., loading from JSON, CSV, etc.
# For now, simply import EXPECTED_TRIPLETS and BASE_STORY into your core script.

# How to load this in coreYYYYMMDD.py (Test 4):
# from data.expected_data import EXPECTED_TRIPLETS, BASE_STORY
# ...
# return (True, EXPECTED_TRIPLETS, BASE_STORY)