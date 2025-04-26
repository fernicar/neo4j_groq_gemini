# core20250424.py
# Core logic for Knowledge Graph Consistency PoC - Dated 2025-04-24
import os
import sys
import time
import configparser
from neo4j import GraphDatabase
import requests
import spacy
from collections import deque
import datetime
import json # Might be useful for parsing initial data or LLM output
import warnings
from pathlib import Path

# --- Configuration and Setup ---

# Load environment variables from .env file
# Using python-dotenv requires 'pip install python-dotenv'
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Cannot load .env file.")
    print("Please install it: pip install python-dotenv")

# --- Neo4j Configuration (from .env) ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

# --- LLM API Configuration (from .env) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Global Resources ---
neo4j_driver = None
config = None
nlp = None # spaCy model
nlp_large = None # Larger spaCy model for advanced analysis
coref_nlp = None # spaCy model with coreference resolution
google_api_call_timestamps = deque(maxlen=5) # For Google rate limiting

# --- Model Configuration ---
SPACY_MODEL_SMALL = "en_core_web_sm"  # Default small model
SPACY_MODEL_LARGE = "en_core_web_lg"  # Larger model with word vectors
USE_COREF = False  # Whether to use coreference resolution (requires neuralcoref)

# --- Placeholder for Initial Data ---
# In a real scenario, this would be loaded from data/expected_data.py or another file
# For this PoC code generation, we'll define it directly as a placeholder.
# TODO: Implement loading this from data/expected_data.py or a similar file based on PoC needs.
def load_initial_data():
    """Loads the expected triplets and base story for testing."""
    # This is placeholder data. Replace with actual loading logic.
    print("Loading placeholder initial data...")
    expected_triplets = [
        ("Alice", "lives_in", "Wonderland"),
        ("Alice", "met", "White Rabbit"),
        ("White Rabbit", "is_a", "Animal")
    ]
    base_story = "Alice was in Wonderland and met a White Rabbit."
    print(f"Loaded {len(expected_triplets)} expected triplets.")
    return expected_triplets, base_story

# --- Logging ---
LOG_FILE = "history.log"

def log_message(message):
    """Appends a timestamped message to the log file."""
    timestamp = datetime.datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"Log: {message}") # Also print to console for immediate feedback

# --- Test Functions ---

def test_1_check_neo4j_connection():
    """Test 1: Checks if the script can connect to Neo4j."""
    global neo4j_driver
    print("\n--- Running Test 1: Checking Neo4j Connection ---")
    log_message("Running Test 1: Checking Neo4j Connection")
    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        print("Test 1 Passed: Successfully connected to Neo4j.")
        log_message("Test 1 Passed: Successfully connected to Neo4j.")
        return True
    except Exception as e:
        print(f"Test 1 Failed: Could not connect to Neo4j at {NEO4J_URI}. Error: {e}")
        print("Please ensure Neo4j Desktop is running and credentials in .env are correct.")
        log_message(f"Test 1 Failed: Could not connect to Neo4j. Error: {e}")
        return False

def test_2_clean_neo4j_database():
    """Test 2: Ensures the script can clean the Neo4j database."""
    print("\n--- Running Test 2: Cleaning Neo4j Database ---")
    log_message("Running Test 2: Cleaning Neo4j Database")
    if neo4j_driver is None:
        print("Test 2 Skipped: Neo4j driver not initialized (Test 1 failed?).")
        log_message("Test 2 Skipped: Neo4j driver not initialized.")
        return False # Cannot proceed without driver

    try:
        with neo4j_driver.session() as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
        print("Test 2 Passed: Database cleaned.")
        log_message("Test 2 Passed: Database cleaned.")
        return True
    except Exception as e:
        print(f"Test 2 Failed: Could not clean database. Error: {e}")
        log_message(f"Test 2 Failed: Could not clean database. Error: {e}")
        return False

def test_3_load_configuration(config_path):
    """Test 3: Loads and parses the specified .ini configuration file."""
    global config
    print(f"\n--- Running Test 3: Loading Configuration from {config_path} ---")
    log_message(f"Running Test 3: Loading Configuration from {config_path}")
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        print(f"Test 3 Failed: Config file not found at {config_path}")
        log_message(f"Test 3 Failed: Config file not found at {config_path}")
        return False

    try:
        config.read(config_path)
        print(f"Test 3 Passed: Configuration loaded from {config_path}.")
        # Basic validation of essential sections/keys
        if 'api' not in config or 'prompts' not in config or 'format' not in config:
             print("Test 3 Warning: Config file missing essential sections ([api], [prompts], [format]).")
             log_message("Test 3 Warning: Config file missing essential sections.")
        log_message(f"Test 3 Passed: Configuration loaded.")
        return True
    except Exception as e:
        print(f"Test 3 Failed: Could not parse config file {config_path}. Error: {e}")
        log_message(f"Test 3 Failed: Could not parse config file. Error: {e}")
        return False

def test_4_load_initial_data():
    """Test 4: Loads the initial expected triplets and base story."""
    print("\n--- Running Test 4: Loading Initial Test Data ---")
    log_message("Running Test 4: Loading Initial Test Data")
    try:
        # Call the placeholder loading function
        expected_triplets, base_story = load_initial_data()
        if not expected_triplets:
            print("Test 4 Warning: No expected triplets loaded. Core PoC tests may not be meaningful.")
            log_message("Test 4 Warning: No expected triplets loaded.")
        print("Test 4 Passed: Initial data loaded.")
        log_message("Test 4 Passed: Initial data loaded.")
        return (True, expected_triplets, base_story) # Return data needed for later tests
    except Exception as e:
        print(f"Test 4 Failed: Could not load initial data. Error: {e}")
        log_message(f"Test 4 Failed: Could not load initial data. Error: {e}")
        return (False, None, None)

def call_llm_api(api_name, model_name, system_prompt, query_prompt, model_params, use_file_response=False, file_path="Raw Response Text.txt"):
    """Calls the specified LLM API with the given prompts and parameters.

    If use_file_response is True, reads the response from the specified file instead of making an API call.
    """
    if use_file_response:
        print(f"Using file-based response from {file_path} instead of calling LLM API")
        log_message(f"Using file-based response from {file_path} instead of calling LLM API")

        try:
            # Read the raw response from the file
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_response_text = f.read()

            # Create a mock response JSON structure based on the API type
            if api_name.lower() == "groq":
                # Mimic Groq/OpenAI response structure
                mock_response = {
                    "choices": [
                        {
                            "message": {
                                "content": raw_response_text
                            }
                        }
                    ]
                }
            elif api_name.lower() == "google":
                # Mimic Google Gemini response structure
                mock_response = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": raw_response_text}
                                ]
                            }
                        }
                    ]
                }
            else:
                print(f"Error: Unknown API name specified for file-based response: {api_name}")
                log_message(f"Error: Unknown API name specified for file-based response: {api_name}")
                return None

            log_message(f"Successfully loaded file-based response")
            return mock_response

        except Exception as e:
            print(f"Error loading file-based response from {file_path}: {e}")
            log_message(f"Error loading file-based response: {e}")
            return None

    # If not using file-based response, proceed with normal API call
    print(f"Calling LLM API: {api_name} with model {model_name}")
    log_message(f"Calling LLM API: {api_name} with model {model_name}")

    headers = {}
    json_data = {}
    api_url = ""

    if api_name.lower() == "groq":
        api_key = GROQ_API_KEY
        if not api_key:
            print("Error: GROQ_API_KEY not found in environment variables.")
            log_message("Error: GROQ_API_KEY not found.")
            return None
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        json_data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_prompt}
            ],
            "model": model_name,
            **model_params # Include model parameters from config
        }
        # Groq has higher limits, not implementing custom rate limit here for simplicity,
        # but API errors will be caught.

    elif api_name.lower() == "google":
        api_key = GOOGLE_API_KEY
        if not api_key:
            print("Error: GOOGLE_API_KEY not found in environment variables.")
            log_message("Error: GOOGLE_API_KEY not found.")
            return None

        # --- Google Rate Limiting Logic (5 calls per minute) ---
        current_time = time.time()
        # Remove timestamps older than 60 seconds
        while google_api_call_timestamps and google_api_call_timestamps[0] <= current_time - 60:
            google_api_call_timestamps.popleft()

        # Check if we've made 5 calls in the last 60 seconds
        if len(google_api_call_timestamps) >= 5:
            time_to_wait = 60 - (current_time - google_api_call_timestamps[0]) + 1 # Wait a bit extra
            print(f"Google API rate limit reached (5 calls/min). Waiting {time_to_wait:.2f} seconds...")
            log_message(f"Google API rate limit reached. Waiting {time_to_wait:.2f} seconds.")
            time.sleep(time_to_wait)
            current_time = time.time() # Update current time after waiting
            # Re-check and remove old timestamps after waiting
            while google_api_call_timestamps and google_api_call_timestamps[0] <= current_time - 60:
                 google_api_call_timestamps.popleft()

        # Proceed with the call
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        # Google API expects contents, not roles like OpenAI/Groq, for simple calls
        # It also has a different structure for parameters
        # This is a simplified mapping. Need to check Google API docs for exact parameter names.
        google_model_params = {}
        if 'temperature' in model_params: google_model_params['temperature'] = model_params['temperature']
        if 'max_tokens' in model_params: google_model_params['maxOutputTokens'] = model_params['max_tokens']
        # Map other potential params like top_p, etc. based on Google API spec

        json_data = {
            "contents": [
                {"parts": [{"text": system_prompt + "\n" + query_prompt}]}
            ],
             **google_model_params
        }
        headers = {"Content-Type": "application/json"} # API key is in URL for this endpoint format

    else:
        print(f"Error: Unknown API name specified in config: {api_name}")
        log_message(f"Error: Unknown API name specified: {api_name}")
        return None

    # Perform the API call
    try:
        response = requests.post(api_url, headers=headers, json=json_data)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        if api_name.lower() == "google":
             google_api_call_timestamps.append(time.time()) # Log timestamp for rate limiting

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error calling {api_name} API: {e}")
        log_message(f"Error calling {api_name} API: {e}")
        if api_name.lower() == "google":
            print("For Google API quotas, check: https://console.cloud.google.com/iam-admin/quotas")
            log_message("Check Google API quotas at https://console.cloud.google.com/iam-admin/quotas")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        log_message(f"Unexpected error during API call: {e}")
        return None


def parse_llm_response(llm_response_json, config):
    """Parses the LLM response based on the format defined in the config."""
    print("Parsing LLM response...")
    log_message("Parsing LLM response")

    if not llm_response_json:
        print("Parsing failed: Empty LLM response.")
        log_message("Parsing failed: Empty LLM response.")
        return None, None

    # Extract text content from response (this structure depends on the API)
    # This is a common structure, but might need adjustment based on actual API responses
    try:
        # Groq/OpenAI structure
        if 'choices' in llm_response_json and len(llm_response_json['choices']) > 0:
            response_text = llm_response_json['choices'][0]['message']['content']
        # Google Gemini structure might be different
        elif 'candidates' in llm_response_json and len(llm_response_json['candidates']) > 0:
             response_text = llm_response_json['candidates'][0]['content']['parts'][0]['text']
        else:
             print("Parsing failed: Could not find text content in LLM response JSON.")
             log_message("Parsing failed: Could not find text content in LLM response JSON.")
             return None, None
    except (KeyError, TypeError) as e:
        print(f"Parsing failed: Unexpected LLM response JSON structure. Error: {e}")
        log_message(f"Parsing failed: Unexpected LLM response JSON structure. Error: {e}")
        log_message(f"Raw LLM Response: {json.dumps(llm_response_json, indent=2)}") # Log the raw response
        return None, None


    log_message(f"Raw Response Text:\n---\n{response_text}\n---")

    # Handle <think> sections in the response by removing them
    # Some LLMs include thinking steps enclosed in <think> tags
    if "<think>" in response_text and "</think>" in response_text:
        think_start = response_text.find("<think>")
        think_end = response_text.find("</think>") + len("</think>")

        # Remove the think section
        print("Detected <think> section in response. Removing it before parsing...")
        log_message("Detected <think> section in response. Removing it before parsing.")

        # Extract the parts before and after the think section
        before_think = response_text[:think_start].strip()
        after_think = response_text[think_end:].strip()

        # Combine the parts, with a space if both parts exist
        if before_think and after_think:
            response_text = before_think + "\n\n" + after_think
        else:
            response_text = before_think + after_think

        log_message(f"Response after removing <think> section:\n---\n{response_text}\n---")

    # Get format details from config
    try:
        separator = config['format']['separator']
        triplet_start_marker = config['format']['triplet_start_marker']
        triplet_end_marker = config['format']['triplet_end_marker']
        story_start_marker = config['format']['story_start_marker']
        story_end_marker = config['format']['story_end_marker']
    except KeyError as e:
        print(f"Parsing failed: Missing format key in config. Error: {e}")
        log_message(f"Parsing failed: Missing format key in config. Error: {e}")
        return None, None

    # Split response by separator
    parts = response_text.split(separator, 1)
    if len(parts) != 2:
        print(f"Parsing Warning: Response did not split into two parts using separator '{separator}'.")
        log_message(f"Parsing Warning: Response did not split into two parts using separator '{separator}'.")
        # Attempt to find parts anyway based on markers
        triplets_section = response_text
        story_section = "" # Assume story might be missing if separator is gone
    else:
        triplets_section = parts[0]
        story_section = parts[1]

    # Extract triplets part
    triplets_content = None
    triplet_start_index = triplets_section.find(triplet_start_marker)
    if triplet_start_index != -1:
        content_start = triplet_start_index + len(triplet_start_marker)
        triplet_end_index = triplets_section.find(triplet_end_marker, content_start)
        if triplet_end_index != -1:
            triplets_content = triplets_section[content_start:triplet_end_index].strip()
        else:
            print(f"Parsing Warning: Triplet end marker '{triplet_end_marker}' not found after start marker.")
            log_message(f"Parsing Warning: Triplet end marker '{triplet_end_marker}' not found after start marker.")
            triplets_content = triplets_section[content_start:].strip() # Take rest of the section

    # Extract story part
    story_content = None
    story_start_index = story_section.find(story_start_marker)
    if story_start_index != -1:
        content_start = story_start_index + len(story_start_marker)
        story_end_index = story_section.find(story_end_marker, content_start)
        if story_end_index != -1:
            story_content = story_section[content_start:story_end_index].strip()
        else:
            print(f"Parsing Warning: Story end marker '{story_end_marker}' not found after start marker.")
            log_message(f"Parsing Warning: Story end marker '{story_end_marker}' not found after start marker.")
            story_content = story_section[content_start:].strip() # Take rest of the section

    # --- Further Parsing of Triplet Content ---
    # The `triplets_content` string needs to be parsed into a list of (S, P, O) tuples.
    # This logic is highly dependent on the exact format the LLM produces within the markers.
    # Example: Assuming `(Subject, predicate, Object)` on each line.
    parsed_triplets = []
    if triplets_content:
        print("Attempting to parse triplets content...")
        log_message(f"Attempting to parse triplets content: {triplets_content}")
        # This is a very basic parsing example. It will fail on complex cases.
        # TODO: Refine this parsing logic based on the LLM's actual output format.
        for line in triplets_content.split('\n'):
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                try:
                    # Remove parentheses and split by comma, then strip whitespace
                    parts = [part.strip() for part in line[1:-1].split(',')]
                    if len(parts) == 3:
                        parsed_triplets.append(tuple(parts))
                        log_message(f"Parsed triple: {tuple(parts)}")
                    else:
                        print(f"Parsing Warning: Skipping malformed triplet line: {line} (Incorrect number of parts)")
                        log_message(f"Parsing Warning: Skipping malformed triplet line: {line} (Incorrect number of parts)")
                except Exception as e:
                    print(f"Parsing Warning: Skipping line due to error: {line} - {e}")
                    log_message(f"Parsing Warning: Skipping line due to error: {line} - {e}")
            elif line: # Log non-empty lines that don't match expected format
                 print(f"Parsing Warning: Skipping line, does not match expected triplet format: {line}")
                 log_message(f"Parsing Warning: Skipping line, does not match expected triplet format: {line}")


    if parsed_triplets:
        print(f"Successfully parsed {len(parsed_triplets)} triplets.")
        log_message(f"Successfully parsed {len(parsed_triplets)} triplets.")
    else:
        print("No triplets parsed from LLM response.")
        log_message("No triplets parsed from LLM response.")


    if story_content:
        print("Successfully extracted narrative content.")
        log_message("Successfully extracted narrative content.")
    else:
        print("No narrative content extracted from LLM response.")
        log_message("No narrative content extracted from LLM response.")


    return parsed_triplets, story_content

def update_neo4j_with_triplets(neo4j_driver, triplets_to_add):
    """Adds parsed triplets to the Neo4j database."""
    print(f"Updating Neo4j with {len(triplets_to_add)} triplets...")
    log_message(f"Updating Neo4j with {len(triplets_to_add)} triplets")

    if not triplets_to_add:
        print("No triplets to add to Neo4j.")
        log_message("No triplets to add to Neo4j.")
        return True # Success, nothing to do

    try:
        with neo4j_driver.session() as session:
            # Use a transaction function for robustness
            def create_triplet_tx(tx, s, p, o):
                # Cypher relationship types cannot be parameterized directly in standard ways.
                # We'll format the query string with the predicate as the type.
                # WARNING: This is a simplification for PoC. In production, sanitize 'p'
                # or use a mapping if predicates are user-defined/uncontrolled.
                # Assumes predicate 'p' is safe to use as a relationship type identifier.
                # Replace spaces or non-alphanumeric chars if necessary (e.g., 'is a' -> 'IS_A')
                # For this PoC, let's simple replace spaces with underscores and upper case.
                relation_type = p.replace(" ", "_").upper()
                if not relation_type.isalnum() and "_" not in relation_type:
                    # Very basic check, improve for real use
                    print(f"Warning: Predicate '{p}' might not be a valid Cypher relationship type. Using '{relation_type}'.")
                    log_message(f"Warning: Predicate '{p}' might not be a valid Cypher relationship type. Using '{relation_type}'.")


                query = (
                    f"MERGE (a:Entity {{name: $s_name}}) "
                    f"MERGE (b:Entity {{name: $o_name}}) "
                    f"MERGE (a)-[:{relation_type}]->(b)"
                )
                tx.run(query, s_name=s, o_name=o)

            for s, p, o in triplets_to_add:
                session.execute_write(create_triplet_tx, s, p, o)
                log_message(f"Added triple to Neo4j (attempted): ({s})-[:{p.replace(' ', '_').upper()}]->({o})")

        print("Neo4j update successful.")
        log_message("Neo4j update successful.")
        return True
    except Exception as e:
        print(f"Error updating Neo4j: {e}")
        log_message(f"Error updating Neo4j: {e}")
        return False

def verify_neo4j_content(neo4j_driver, expected_triplets):
    """Verifies if the expected triplets exist in the Neo4j database."""
    print(f"Verifying Neo4j content. Checking for {len(expected_triplets)} expected triplets...")
    log_message(f"Verifying Neo4j content. Checking for {len(expected_triplets)} expected triplets")

    if not expected_triplets:
        print("No expected triplets to verify.")
        log_message("No expected triplets to verify.")
        return {"status": "skipped", "found_count": 0, "missing": []}

    found_count = 0
    missing_triplets = []

    try:
        with neo4j_driver.session() as session:
             def check_triple_exists_tx(tx, s, p, o):
                relation_type = p.replace(" ", "_").upper() # Match the type used in update_neo4j
                query = (
                    f"MATCH (a:Entity {{name: $s_name}})-[r:{relation_type}]->(b:Entity {{name: $o_name}}) "
                    "RETURN count(r) > 0 AS exists"
                )
                result = tx.run(query, s_name=s, o_name=o).single()
                return result["exists"] if result else False

             for s, p, o in expected_triplets:
                 exists = session.execute_read(check_triple_exists_tx, s, p, o)
                 if exists:
                     found_count += 1
                     log_message(f"Verified: ({s})-[:{p}]->({o}) found in KG.")
                 else:
                     missing_triplets.append((s, p, o))
                     log_message(f"Verified: ({s})-[:{p}]->({o}) NOT found in KG.")


        print(f"Neo4j Verification Results: Found {found_count}/{len(expected_triplets)} expected triplets.")
        if missing_triplets:
            print("Missing triplets:")
            for t in missing_triplets:
                print(f"  {t}")
        log_message(f"Neo4j Verification Results: Found {found_count}/{len(expected_triplets)}. Missing: {missing_triplets}")
        return {"status": "completed", "found_count": found_count, "missing": missing_triplets}

    except Exception as e:
        print(f"Error verifying Neo4j content: {e}")
        log_message(f"Error verifying Neo4j content: {e}")
        return {"status": "error", "error": str(e)}

def load_spacy_model():
    """Loads the spaCy language models."""
    global nlp, nlp_large, coref_nlp

    # Load the small model (always required)
    print(f"\nLoading spaCy model '{SPACY_MODEL_SMALL}'...")
    log_message(f"Loading spaCy model '{SPACY_MODEL_SMALL}'")
    try:
        nlp = spacy.load(SPACY_MODEL_SMALL)
        print(f"spaCy model {SPACY_MODEL_SMALL} loaded.")
        log_message(f"spaCy model {SPACY_MODEL_SMALL} loaded.")
    except Exception as e:
        print(f"Error loading spaCy model {SPACY_MODEL_SMALL}. Have you run 'python -m spacy download {SPACY_MODEL_SMALL}'? Error: {e}")
        log_message(f"Error loading spaCy model {SPACY_MODEL_SMALL}. Error: {e}")
        return False

    # Try to load the large model for advanced analysis
    try:
        print(f"\nLoading larger spaCy model '{SPACY_MODEL_LARGE}'...")
        log_message(f"Loading larger spaCy model '{SPACY_MODEL_LARGE}'")
        nlp_large = spacy.load(SPACY_MODEL_LARGE)
        print(f"Larger spaCy model {SPACY_MODEL_LARGE} loaded.")
        log_message(f"Larger spaCy model {SPACY_MODEL_LARGE} loaded.")
    except Exception as e:
        print(f"Warning: Could not load larger spaCy model {SPACY_MODEL_LARGE}. Using {SPACY_MODEL_SMALL} for all analyses.")
        print(f"To use advanced features, install the larger model: python -m spacy download {SPACY_MODEL_LARGE}")
        log_message(f"Warning: Could not load larger spaCy model {SPACY_MODEL_LARGE}. Error: {e}")
        nlp_large = nlp  # Fall back to small model

    # Try to set up coreference resolution if requested
    if USE_COREF:
        try:
            print("\nSetting up coreference resolution...")
            log_message("Setting up coreference resolution")

            # Try to import neuralcoref
            try:
                import neuralcoref
                # Create a new spaCy model instance for coreference
                coref_nlp = spacy.load(SPACY_MODEL_SMALL)
                # Add neuralcoref to the pipeline
                neuralcoref.add_to_pipe(coref_nlp)
                print("Coreference resolution set up successfully.")
                log_message("Coreference resolution set up successfully.")
            except ImportError:
                print("Warning: neuralcoref not installed. Coreference resolution will not be available.")
                print("To use coreference resolution, python 3.7 required: pip install neuralcoref")
                log_message("Warning: neuralcoref not installed. Coreference resolution will not be available.")
                coref_nlp = nlp  # Fall back to regular model
        except Exception as e:
            print(f"Warning: Could not set up coreference resolution. Error: {e}")
            log_message(f"Warning: Could not set up coreference resolution. Error: {e}")
            coref_nlp = nlp  # Fall back to regular model
    else:
        coref_nlp = nlp  # Use regular model if coreference not requested

    return True

def analyze_narrative_with_spacy(narrative_text, triplets_in_kg):
    """Analyzes the narrative with spaCy and compares entities/relations with triplets in KG.

    This is the original basic implementation that only uses named entity recognition.
    """
    print("\nAnalyzing narrative with spaCy (Basic Method)...")
    log_message("Analyzing narrative with spaCy (Basic Method)")

    if nlp is None:
        print("SpaCy model not loaded. Skipping narrative analysis.")
        log_message("SpaCy model not loaded. Skipping narrative analysis.")
        return {"status": "skipped", "match_percentage": 0, "kg_entity_count": 0, "narrative_entity_count": 0, "kg_entities_in_narrative": 0}

    if not narrative_text:
        print("No narrative text to analyze.")
        log_message("No narrative text to analyze.")
        return {"status": "skipped", "match_percentage": 0, "kg_entity_count": 0, "narrative_entity_count": 0, "kg_entities_in_narrative": 0}

    if not triplets_in_kg:
         print("No triplets found in KG to compare against spaCy analysis.")
         log_message("No triplets found in KG to compare against spaCy analysis.")
         # Can still analyze narrative, but match percentage will be 0/skipped relevant part
         kg_entities = set()
    else:
        # Extract unique entities (subjects and objects) from the triplets successfully added to KG
        kg_entities = set()
        for s, p, o in triplets_in_kg:
            kg_entities.add(s)
            kg_entities.add(o)
        print(f"KG contains {len(kg_entities)} unique entities from added triplets.")
        log_message(f"KG contains {len(kg_entities)} unique entities from added triplets.")

    try:
        doc = nlp(narrative_text)

        # Basic Entity Extraction comparison
        narrative_entities = set()
        for ent in doc.ents:
            # Convert entities to lowercase for case-insensitive matching
            narrative_entities.add(ent.text.lower())
            log_message(f"SpaCy entity found: {ent.text} ({ent.label_})")

        # Calculate narrative coverage rate (what percentage of words were recognized as entities)
        total_words = len([token for token in doc if not token.is_punct and not token.is_space])
        entity_words = sum(len(ent.text.split()) for ent in doc.ents)
        narrative_coverage = (entity_words / total_words) * 100 if total_words > 0 else 0

        # Compare entities found by spaCy with entities in the KG triplets
        kg_entities_lower = {ent.lower() for ent in kg_entities}
        kg_entities_in_narrative = 0
        for kg_ent_lower in kg_entities_lower:
            if kg_ent_lower in narrative_entities:
                kg_entities_in_narrative += 1
                log_message(f"KG entity '{kg_ent_lower}' found in narrative.")

        match_percentage = (kg_entities_in_narrative / len(kg_entities)) * 100 if kg_entities else 0

        print(f"SpaCy Analysis Results (Basic Method):")
        print(f"  Entities in Narrative (found by spaCy): {len(narrative_entities)}")
        print(f"  Entities in KG (from added triplets): {len(kg_entities)}")
        print(f"  KG Entities found in Narrative: {kg_entities_in_narrative}")
        print(f"  Entity Presence Match Percentage: {match_percentage:.2f}%")
        print(f"  Narrative Coverage Rate: {narrative_coverage:.2f}%")

        log_message(f"SpaCy Analysis Results (Basic): Narrative entities={len(narrative_entities)}, KG entities={len(kg_entities)}, KG entities in narrative={kg_entities_in_narrative}, Match Percentage={match_percentage:.2f}%, Coverage={narrative_coverage:.2f}%")

        return {
            "status": "completed",
            "match_percentage": match_percentage,
            "kg_entity_count": len(kg_entities),
            "narrative_entity_count": len(narrative_entities),
            "kg_entities_in_narrative": kg_entities_in_narrative,
            "narrative_coverage": narrative_coverage
            }

    except Exception as e:
        print(f"Error during spaCy analysis: {e}")
        log_message(f"Error during spaCy analysis: {e}")
        return {"status": "error", "error": str(e)}

def analyze_narrative_enhanced(narrative_text, triplets_in_kg):
    """Enhanced analysis of narrative with improved entity matching.

    This implementation adds noun chunks and partial matching to improve entity detection.
    """
    print("\nAnalyzing narrative with Enhanced Entity Matching...")
    log_message("Analyzing narrative with Enhanced Entity Matching")

    if nlp is None:
        print("SpaCy model not loaded. Skipping narrative analysis.")
        log_message("SpaCy model not loaded. Skipping narrative analysis.")
        return {"status": "skipped", "match_percentage": 0}

    if not narrative_text:
        print("No narrative text to analyze.")
        log_message("No narrative text to analyze.")
        return {"status": "skipped", "match_percentage": 0}

    if not triplets_in_kg:
        print("No triplets found in KG to compare against analysis.")
        log_message("No triplets found in KG to compare against analysis.")
        kg_entities = set()
    else:
        # Extract unique entities from KG triplets
        kg_entities = set()
        for s, p, o in triplets_in_kg:
            kg_entities.add(s)
            kg_entities.add(o)
        print(f"KG contains {len(kg_entities)} unique entities from added triplets.")
        log_message(f"KG contains {len(kg_entities)} unique entities from added triplets.")

    try:
        # Process the narrative
        doc = nlp(narrative_text)

        # 1. Extract named entities
        named_entities = set()
        for ent in doc.ents:
            named_entities.add(ent.text.lower())
            log_message(f"SpaCy named entity found: {ent.text} ({ent.label_})")

        # 2. Extract noun chunks (helps with compound names)
        noun_chunks = set()
        for chunk in doc.noun_chunks:
            noun_chunks.add(chunk.text.lower())
            log_message(f"SpaCy noun chunk found: {chunk.text}")

        # 3. Extract individual nouns
        individual_nouns = set()
        for token in doc:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                individual_nouns.add(token.text.lower())
                log_message(f"SpaCy noun found: {token.text} ({token.pos_})")

        # Combine all potential entity mentions
        all_potential_entities = named_entities.union(noun_chunks).union(individual_nouns)

        # Calculate narrative coverage rate
        total_words = len([token for token in doc if not token.is_punct and not token.is_space])
        entity_words = sum(len(entity.split()) for entity in all_potential_entities)
        narrative_coverage = (entity_words / total_words) * 100 if total_words > 0 else 0

        # Normalize KG entities
        kg_entities_lower = {ent.lower() for ent in kg_entities}

        # Match entities using multiple strategies
        matched_entities = set()
        match_details = {}

        for kg_ent in kg_entities_lower:
            # Strategy 1: Direct matching
            if kg_ent in all_potential_entities:
                matched_entities.add(kg_ent)
                match_details[kg_ent] = "direct_match"
                log_message(f"KG entity '{kg_ent}' found in narrative (direct match).")
                continue

            # Strategy 2: Partial matching for multi-word entities
            for pot_ent in all_potential_entities:
                # Check if KG entity is contained within a longer entity or vice versa
                if (len(kg_ent.split()) > 1 or len(pot_ent.split()) > 1) and \
                   (kg_ent in pot_ent or pot_ent in kg_ent):
                    matched_entities.add(kg_ent)
                    match_details[kg_ent] = f"partial_match_with_{pot_ent}"
                    log_message(f"KG entity '{kg_ent}' found in narrative (partial match with '{pot_ent}').")
                    break

        # Calculate match percentage
        match_percentage = (len(matched_entities) / len(kg_entities)) * 100 if kg_entities else 0

        # Log results
        print(f"Enhanced Entity Matching Results:")
        print(f"  All potential entities in narrative: {len(all_potential_entities)}")
        print(f"  Entities in KG: {len(kg_entities)}")
        print(f"  KG Entities matched in narrative: {len(matched_entities)}")
        print(f"  Entity Match Percentage: {match_percentage:.2f}%")
        print(f"  Narrative Coverage Rate: {narrative_coverage:.2f}%")

        for kg_ent in kg_entities_lower:
            status = "✓ MATCHED" if kg_ent in matched_entities else "✗ NOT MATCHED"
            details = match_details.get(kg_ent, "")
            print(f"  {kg_ent}: {status} {details}")

        log_message(f"Enhanced Entity Matching Results: Potential entities={len(all_potential_entities)}, KG entities={len(kg_entities)}, Matched={len(matched_entities)}, Match Percentage={match_percentage:.2f}%, Coverage={narrative_coverage:.2f}%")

        return {
            "status": "completed",
            "match_percentage": match_percentage,
            "kg_entity_count": len(kg_entities),
            "narrative_entity_count": len(all_potential_entities),
            "kg_entities_matched": len(matched_entities),
            "matched_entities": list(matched_entities),
            "match_details": match_details,
            "narrative_coverage": narrative_coverage
        }

    except Exception as e:
        print(f"Error during enhanced narrative analysis: {e}")
        log_message(f"Error during enhanced narrative analysis: {e}")
        return {"status": "error", "error": str(e)}

def analyze_with_semantic_similarity(narrative_text, triplets_in_kg):
    """Analyzes narrative using semantic similarity with word vectors.

    This implementation uses the larger spaCy model with word vectors to find
    semantically similar entities even when they don't match exactly.
    """
    print("\nAnalyzing narrative with Semantic Similarity Matching...")
    log_message("Analyzing narrative with Semantic Similarity Matching")

    if nlp_large is None:
        print("Larger spaCy model not loaded. Skipping semantic similarity analysis.")
        log_message("Larger spaCy model not loaded. Skipping semantic similarity analysis.")
        return {"status": "skipped", "match_percentage": 0}

    if not narrative_text:
        print("No narrative text to analyze.")
        log_message("No narrative text to analyze.")
        return {"status": "skipped", "match_percentage": 0}

    if not triplets_in_kg:
        print("No triplets found in KG to compare against analysis.")
        log_message("No triplets found in KG to compare against analysis.")
        kg_entities = set()
    else:
        # Extract unique entities from KG triplets
        kg_entities = set()
        for s, p, o in triplets_in_kg:
            kg_entities.add(s)
            kg_entities.add(o)
        print(f"KG contains {len(kg_entities)} unique entities from added triplets.")
        log_message(f"KG contains {len(kg_entities)} unique entities from added triplets.")

    try:
        # Process the narrative with the larger model that has word vectors
        doc = nlp_large(narrative_text)

        # Extract potential entities (named entities and noun chunks)
        named_entities = list(doc.ents)
        noun_chunks = list(doc.noun_chunks)

        # Combine all potential entity mentions
        potential_entities = named_entities + noun_chunks

        # Log the potential entities
        for ent in named_entities:
            log_message(f"Named entity for similarity matching: {ent.text} ({ent.label_})")
        for chunk in noun_chunks:
            log_message(f"Noun chunk for similarity matching: {chunk.text}")

        # Calculate narrative coverage
        total_words = len([token for token in doc if not token.is_punct and not token.is_space])
        entity_words = sum(len(ent.text.split()) for ent in potential_entities)
        narrative_coverage = (entity_words / total_words) * 100 if total_words > 0 else 0

        # Normalize KG entities
        kg_entities_lower = {ent.lower() for ent in kg_entities}

        # Track matches with similarity scores
        matched_entities = set()
        similarity_scores = {}

        # Similarity threshold (adjust as needed)
        SIMILARITY_THRESHOLD = 0.6

        # Compare each KG entity with each potential entity in the narrative
        for kg_ent in kg_entities_lower:
            # Create a Doc object for the KG entity
            kg_ent_doc = nlp_large(kg_ent)

            best_match = None
            best_score = 0.0

            # Check each potential entity for similarity
            for pot_ent in potential_entities:
                # Skip if the potential entity has no vector (rare but possible)
                if not pot_ent.vector_norm or not kg_ent_doc.vector_norm:
                    continue

                # Calculate similarity score
                similarity = kg_ent_doc.similarity(pot_ent)

                # Update if this is the best match so far
                if similarity > best_score and similarity > SIMILARITY_THRESHOLD:
                    best_score = similarity
                    best_match = pot_ent.text

            # If we found a good match, record it
            if best_match:
                matched_entities.add(kg_ent)
                similarity_scores[kg_ent] = (best_match, best_score)
                log_message(f"KG entity '{kg_ent}' semantically similar to '{best_match}' (score: {best_score:.2f})")

        # Calculate match percentage
        match_percentage = (len(matched_entities) / len(kg_entities)) * 100 if kg_entities else 0

        # Log results
        print(f"Semantic Similarity Matching Results:")
        print(f"  Potential entities in narrative: {len(potential_entities)}")
        print(f"  Entities in KG: {len(kg_entities)}")
        print(f"  KG Entities matched by similarity: {len(matched_entities)}")
        print(f"  Entity Match Percentage: {match_percentage:.2f}%")
        print(f"  Narrative Coverage Rate: {narrative_coverage:.2f}%")
        print(f"  Similarity Threshold: {SIMILARITY_THRESHOLD}")

        for kg_ent in kg_entities_lower:
            if kg_ent in matched_entities:
                match_text, score = similarity_scores[kg_ent]
                print(f"  {kg_ent}: ✓ MATCHED with '{match_text}' (score: {score:.2f})")
            else:
                print(f"  {kg_ent}: ✗ NOT MATCHED")

        log_message(f"Semantic Similarity Results: Potential entities={len(potential_entities)}, KG entities={len(kg_entities)}, Matched={len(matched_entities)}, Match Percentage={match_percentage:.2f}%, Coverage={narrative_coverage:.2f}%")

        return {
            "status": "completed",
            "match_percentage": match_percentage,
            "kg_entity_count": len(kg_entities),
            "narrative_entity_count": len(potential_entities),
            "kg_entities_matched": len(matched_entities),
            "matched_entities": list(matched_entities),
            "similarity_scores": similarity_scores,
            "narrative_coverage": narrative_coverage
        }

    except Exception as e:
        print(f"Error during semantic similarity analysis: {e}")
        log_message(f"Error during semantic similarity analysis: {e}")
        return {"status": "error", "error": str(e)}

def analyze_with_coreference(narrative_text, triplets_in_kg):
    """Analyzes narrative with coreference resolution to link pronouns to their referents.

    This implementation uses the spaCy model with coreference resolution to improve
    entity detection by resolving pronouns to their referents.
    """
    print("\nAnalyzing narrative with Coreference Resolution...")
    log_message("Analyzing narrative with Coreference Resolution")

    if coref_nlp is None:
        print("Coreference resolution not available. Skipping coreference analysis.")
        log_message("Coreference resolution not available. Skipping coreference analysis.")
        return {"status": "skipped", "match_percentage": 0}

    if not narrative_text:
        print("No narrative text to analyze.")
        log_message("No narrative text to analyze.")
        return {"status": "skipped", "match_percentage": 0}

    if not triplets_in_kg:
        print("No triplets found in KG to compare against analysis.")
        log_message("No triplets found in KG to compare against analysis.")
        kg_entities = set()
    else:
        # Extract unique entities from KG triplets
        kg_entities = set()
        for s, p, o in triplets_in_kg:
            kg_entities.add(s)
            kg_entities.add(o)
        print(f"KG contains {len(kg_entities)} unique entities from added triplets.")
        log_message(f"KG contains {len(kg_entities)} unique entities from added triplets.")

    try:
        # Process the narrative with coreference resolution
        doc = coref_nlp(narrative_text)

        # Check if coreference resolution is available
        if not hasattr(doc, '_.coref_clusters'):
            print("Coreference resolution not properly set up. Skipping coreference analysis.")
            log_message("Coreference resolution not properly set up. Skipping coreference analysis.")
            return {"status": "skipped", "reason": "Coreference resolution not properly set up"}

        # Get coreference clusters
        coref_clusters = doc._.coref_clusters

        # Create a resolved text by replacing pronouns with their referents
        resolved_text = narrative_text

        # Track coreference resolutions
        coref_resolutions = []

        # Replace pronouns with their referents in the text
        if coref_clusters:
            for cluster in coref_clusters:
                # Get the main mention (usually the first non-pronoun mention)
                main_mention = cluster.main

                # Replace each mention with the main mention
                for mention in cluster.mentions:
                    if mention.text.lower() != main_mention.text.lower():
                        coref_resolutions.append((mention.text, main_mention.text))
                        log_message(f"Coreference resolution: '{mention.text}' -> '{main_mention.text}'")

        # Log the resolved text
        if coref_resolutions:
            print("Coreference resolutions found:")
            for mention, referent in coref_resolutions:
                print(f"  '{mention}' refers to '{referent}'")

            # Create a simple resolved text for demonstration
            resolved_text = narrative_text
            for mention, referent in coref_resolutions:
                resolved_text = resolved_text.replace(mention, f"{mention}[={referent}]")

            print("\nText with coreference annotations:")
            print(resolved_text)
        else:
            print("No coreference resolutions found in the text.")

        # Now analyze the resolved text using the enhanced entity matching
        # Process the narrative with the resolved text
        doc = nlp(resolved_text)

        # Extract entities as in the enhanced method
        named_entities = set()
        for ent in doc.ents:
            named_entities.add(ent.text.lower())
            log_message(f"Named entity after coreference: {ent.text} ({ent.label_})")

        noun_chunks = set()
        for chunk in doc.noun_chunks:
            noun_chunks.add(chunk.text.lower())
            log_message(f"Noun chunk after coreference: {chunk.text}")

        # Add resolved entities from coreference
        resolved_entities = set()
        for mention, referent in coref_resolutions:
            resolved_entities.add(referent.lower())
            log_message(f"Resolved entity from coreference: {referent}")

        # Combine all potential entity mentions
        all_potential_entities = named_entities.union(noun_chunks).union(resolved_entities)

        # Calculate narrative coverage
        total_words = len([token for token in doc if not token.is_punct and not token.is_space])
        entity_words = sum(len(entity.split()) for entity in all_potential_entities)
        narrative_coverage = (entity_words / total_words) * 100 if total_words > 0 else 0

        # Normalize KG entities
        kg_entities_lower = {ent.lower() for ent in kg_entities}

        # Match entities using multiple strategies
        matched_entities = set()
        match_details = {}

        for kg_ent in kg_entities_lower:
            # Strategy 1: Direct matching
            if kg_ent in all_potential_entities:
                matched_entities.add(kg_ent)
                match_details[kg_ent] = "direct_match"
                log_message(f"KG entity '{kg_ent}' found after coreference (direct match).")
                continue

            # Strategy 2: Partial matching for multi-word entities
            for pot_ent in all_potential_entities:
                # Check if KG entity is contained within a longer entity or vice versa
                if (len(kg_ent.split()) > 1 or len(pot_ent.split()) > 1) and \
                   (kg_ent in pot_ent or pot_ent in kg_ent):
                    matched_entities.add(kg_ent)
                    match_details[kg_ent] = f"partial_match_with_{pot_ent}"
                    log_message(f"KG entity '{kg_ent}' found after coreference (partial match with '{pot_ent}').")
                    break

        # Calculate match percentage
        match_percentage = (len(matched_entities) / len(kg_entities)) * 100 if kg_entities else 0

        # Log results
        print(f"Coreference Resolution Results:")
        print(f"  Coreference resolutions found: {len(coref_resolutions)}")
        print(f"  All potential entities after resolution: {len(all_potential_entities)}")
        print(f"  Entities in KG: {len(kg_entities)}")
        print(f"  KG Entities matched after resolution: {len(matched_entities)}")
        print(f"  Entity Match Percentage: {match_percentage:.2f}%")
        print(f"  Narrative Coverage Rate: {narrative_coverage:.2f}%")

        for kg_ent in kg_entities_lower:
            status = "✓ MATCHED" if kg_ent in matched_entities else "✗ NOT MATCHED"
            details = match_details.get(kg_ent, "")
            print(f"  {kg_ent}: {status} {details}")

        log_message(f"Coreference Resolution Results: Resolutions={len(coref_resolutions)}, Potential entities={len(all_potential_entities)}, KG entities={len(kg_entities)}, Matched={len(matched_entities)}, Match Percentage={match_percentage:.2f}%, Coverage={narrative_coverage:.2f}%")

        return {
            "status": "completed",
            "match_percentage": match_percentage,
            "kg_entity_count": len(kg_entities),
            "narrative_entity_count": len(all_potential_entities),
            "kg_entities_matched": len(matched_entities),
            "matched_entities": list(matched_entities),
            "match_details": match_details,
            "coref_resolutions": coref_resolutions,
            "narrative_coverage": narrative_coverage
        }

    except Exception as e:
        print(f"Error during coreference analysis: {e}")
        log_message(f"Error during coreference analysis: {e}")
        return {"status": "error", "error": str(e)}

def extract_potential_triplets(narrative_text):
    """Extract potential new triplets from the narrative using dependency parsing.

    This function identifies subject-verb-object patterns in the text that could
    represent new knowledge to add to the KG.
    """
    print("\nExtracting potential new triplets from narrative...")
    log_message("Extracting potential new triplets from narrative")

    if nlp is None or not narrative_text:
        return {"status": "skipped", "potential_triplets": []}

    try:
        # Process the narrative
        doc = nlp(narrative_text)

        # Store potential triplets
        potential_triplets = []

        # Analyze each sentence
        for sent in doc.sents:
            # Find verbs that might be predicates
            for token in sent:
                if token.pos_ == "VERB":
                    # Find subjects
                    subjects = []
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            # Get the full noun phrase
                            subject_span = get_span_for_token(child)
                            subjects.append(subject_span.text)

                    # Find objects
                    objects = []
                    for child in token.children:
                        if child.dep_ in ("dobj", "pobj", "attr"):
                            # Get the full noun phrase
                            object_span = get_span_for_token(child)
                            objects.append(object_span.text)

                    # Create triplets from all subject-object combinations
                    for subj in subjects:
                        for obj in objects:
                            if subj and obj:  # Ensure both subject and object exist
                                triplet = (subj, token.lemma_, obj)
                                potential_triplets.append(triplet)
                                log_message(f"Potential triplet found: {triplet}")

        # Log results
        if potential_triplets:
            print(f"Found {len(potential_triplets)} potential new triplets:")
            for s, p, o in potential_triplets:
                print(f"  ({s}, {p}, {o})")
        else:
            print("No potential new triplets found in the narrative.")

        log_message(f"Potential triplet extraction completed. Found {len(potential_triplets)} triplets.")

        return {
            "status": "completed",
            "potential_triplets": potential_triplets
        }

    except Exception as e:
        print(f"Error during potential triplet extraction: {e}")
        log_message(f"Error during potential triplet extraction: {e}")
        return {"status": "error", "error": str(e), "potential_triplets": []}

def extract_enhanced_triplets(narrative_text):
    """Extract potential new triplets with enhanced relation extraction.

    This function uses more sophisticated patterns and the larger model
    to extract higher-quality triplets from the narrative.
    """
    print("\nExtracting enhanced triplets from narrative...")
    log_message("Extracting enhanced triplets from narrative")

    if nlp_large is None or not narrative_text:
        print("Larger spaCy model not loaded. Falling back to basic model.")
        log_message("Larger spaCy model not loaded. Falling back to basic model.")
        # Fall back to the basic model if the larger one isn't available
        return extract_potential_triplets(narrative_text)

    try:
        # Process the narrative with the larger model
        doc = nlp_large(narrative_text)

        # Try to use coreference resolution if available
        resolved_text = narrative_text
        if coref_nlp is not None and USE_COREF:
            try:
                coref_doc = coref_nlp(narrative_text)
                if hasattr(coref_doc, '_.coref_clusters'):
                    # Get coreference clusters
                    coref_clusters = coref_doc._.coref_clusters

                    # Replace pronouns with their referents
                    if coref_clusters:
                        # Create a simple resolved text
                        for cluster in coref_clusters:
                            main_mention = cluster.main
                            for mention in cluster.mentions:
                                if mention.text.lower() != main_mention.text.lower():
                                    # Replace the mention with the main mention
                                    resolved_text = resolved_text.replace(mention.text, main_mention.text)

                        print("Applied coreference resolution for triplet extraction.")
                        log_message("Applied coreference resolution for triplet extraction.")

                        # Re-process with the resolved text
                        doc = nlp_large(resolved_text)
            except Exception as e:
                print(f"Warning: Could not apply coreference resolution: {e}")
                log_message(f"Warning: Could not apply coreference resolution: {e}")

        # Store potential triplets
        potential_triplets = []

        # Track entities for better triplet extraction
        entities = {}
        for ent in doc.ents:
            entities[ent.text] = ent.label_

        # Enhanced patterns for triplet extraction
        # 1. Subject-Verb-Object pattern (basic)
        # 2. Subject-Verb-Preposition-Object pattern (e.g., "lives in Wonderland")
        # 3. Subject-Verb-Adjective pattern (e.g., "is curious")
        # 4. Possessive pattern (e.g., "Alice's rabbit")

        # Analyze each sentence
        for sent in doc.sents:
            # Pattern 1 & 2: Find verbs and their arguments
            for token in sent:
                if token.pos_ == "VERB":
                    # Find subjects
                    subjects = []
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            # Get the full noun phrase
                            subject_span = get_span_for_token(child)
                            subjects.append(subject_span.text)

                    # Find direct objects
                    direct_objects = []
                    for child in token.children:
                        if child.dep_ == "dobj":
                            # Get the full noun phrase
                            object_span = get_span_for_token(child)
                            direct_objects.append(object_span.text)

                    # Find prepositional objects
                    prep_objects = []
                    for child in token.children:
                        if child.dep_ == "prep":
                            for prep_child in child.children:
                                if prep_child.dep_ == "pobj":
                                    # Create a predicate from verb + preposition
                                    predicate = f"{token.lemma_}_{child.text}"
                                    # Get the full noun phrase
                                    object_span = get_span_for_token(prep_child)
                                    prep_objects.append((predicate, object_span.text))

                    # Find attributes (for "is a" relationships)
                    attributes = []
                    for child in token.children:
                        if child.dep_ == "attr":
                            # Get the full noun phrase
                            attr_span = get_span_for_token(child)
                            attributes.append(attr_span.text)

                    # Create triplets from subject-verb-object
                    for subj in subjects:
                        # Direct objects
                        for obj in direct_objects:
                            if subj and obj:
                                triplet = (subj, token.lemma_, obj)
                                potential_triplets.append(triplet)
                                log_message(f"Enhanced triplet (SVO): {triplet}")

                        # Prepositional objects
                        for pred, obj in prep_objects:
                            if subj and obj:
                                triplet = (subj, pred, obj)
                                potential_triplets.append(triplet)
                                log_message(f"Enhanced triplet (SVPO): {triplet}")

                        # Attributes (especially for "is a" relationships)
                        for attr in attributes:
                            if subj and attr and token.lemma_ in ["be", "become", "remain"]:
                                # Check if this might be an "is_a" relationship
                                if any(det.dep_ == "det" and det.text.lower() in ["a", "an"] for det in token.children):
                                    triplet = (subj, "is_a", attr)
                                else:
                                    triplet = (subj, token.lemma_, attr)
                                potential_triplets.append(triplet)
                                log_message(f"Enhanced triplet (SVA): {triplet}")

            # Pattern 3: Possessive relationships
            for token in sent:
                if token.dep_ == "poss":
                    # Get the possessor (e.g., "Alice" in "Alice's rabbit")
                    possessor = token.text

                    # Get the possessed (e.g., "rabbit" in "Alice's rabbit")
                    if token.head:
                        possessed = get_span_for_token(token.head).text

                        # Create a "has" or "owns" relationship
                        triplet = (possessor, "has", possessed)
                        potential_triplets.append(triplet)
                        log_message(f"Enhanced triplet (possessive): {triplet}")

        # Filter out duplicate triplets
        unique_triplets = []
        seen = set()
        for s, p, o in potential_triplets:
            triplet_key = (s.lower(), p.lower(), o.lower())
            if triplet_key not in seen:
                seen.add(triplet_key)
                unique_triplets.append((s, p, o))

        # Log results
        if unique_triplets:
            print(f"Found {len(unique_triplets)} potential new triplets (enhanced):")
            for s, p, o in unique_triplets:
                print(f"  ({s}, {p}, {o})")
        else:
            print("No potential new triplets found in the narrative (enhanced).")

        log_message(f"Enhanced triplet extraction completed. Found {len(unique_triplets)} unique triplets.")

        # Save the enhanced triplets to the report file
        try:
            with open("autoUpdateKGfromResponse.txt", "a", encoding="utf-8") as f:
                f.write("\n\n# Enhanced Triplet Extraction Results\n\n")
                f.write(f"Date: {datetime.datetime.now().isoformat()}\n\n")
                f.write("The following triplets were extracted using enhanced relation extraction techniques:\n\n")
                for s, p, o in unique_triplets:
                    f.write(f"* ({s}, {p}, {o})\n")
                f.write("\nThese triplets represent potential new knowledge that could be added to the Knowledge Graph.\n")
            print(f"Enhanced triplets saved to autoUpdateKGfromResponse.txt")
            log_message(f"Enhanced triplets saved to autoUpdateKGfromResponse.txt")
        except Exception as e:
            print(f"Error saving enhanced triplets to file: {e}")
            log_message(f"Error saving enhanced triplets to file: {e}")

        return {
            "status": "completed",
            "potential_triplets": unique_triplets
        }

    except Exception as e:
        print(f"Error during enhanced triplet extraction: {e}")
        log_message(f"Error during enhanced triplet extraction: {e}")
        return {"status": "error", "error": str(e), "potential_triplets": []}

def get_span_for_token(token):
    """Helper function to get the full noun phrase span for a token."""
    # If token is part of a noun chunk, return the whole chunk
    for chunk in token.doc.noun_chunks:
        if token in chunk:
            return chunk

    # Otherwise, just return the token itself as a span
    return token


# --- Main Test Runner Function ---

def run_tests(start_test_number=1, config_path=None, use_file_response=False, file_response_path="Raw Response Text.txt"):
    """Runs the sequence of tests starting from start_test_number.

    If use_file_response is True, uses the content from file_response_path instead of making an API call.
    """
    print(f"\n--- Starting PoC Test Sequence from Test {start_test_number} ---")
    log_message(f"--- Starting PoC Test Sequence from Test {start_test_number} ---")
    log_message(f"Using config file: {config_path}")

    if use_file_response:
        print(f"Using file-based response from {file_response_path} instead of making API calls")
        log_message(f"Using file-based response from {file_response_path}")

    test_results = {} # Dictionary to store results of each test
    shared_data = {
        'use_file_response': use_file_response,
        'file_response_path': file_response_path
    } # Dictionary to pass data between tests (e.g., loaded config, initial data)

    # Define the ordered list of tests
    # Each tuple: (test_number, test_function, list_of_required_shared_data_keys)
    all_tests = [
        (1, test_1_check_neo4j_connection, []),
        (2, test_2_clean_neo4j_database, []), # Requires driver from test 1
        (3, test_3_load_configuration, [config_path]), # Takes config_path as direct arg
        (4, test_4_load_initial_data, []),
        (5, lambda: test_5_core_poc_logic(
            config=shared_data.get('config'),
            neo4j_driver=neo4j_driver, # global driver
            initial_data=shared_data.get('initial_data'), # (expected_triplets, base_story) tuple
            use_file_response=shared_data.get('use_file_response', False),
            file_path=shared_data.get('file_response_path', "Raw Response Text.txt")
            ), ['config', 'initial_data']), # Requires config from test 3, data from test 4

        # Add more tests here as needed for further validation or analysis steps
        (6, lambda: test_6_verify_neo4j_against_expected(
            neo4j_driver=neo4j_driver,
            initial_data=shared_data.get('initial_data') # (expected_triplets, base_story) tuple
            ), ['initial_data']), # Requires initial data (expected triplets)

        (7, lambda: test_7_spacy_analysis(
             narrative_text=shared_data.get('llm_narrative'), # from test 5
             triplets_in_kg=shared_data.get('llm_parsed_triplets') # from test 5 (triplets added to KG)
             ), ['llm_narrative', 'llm_parsed_triplets']), # Requires LLM output from test 5

        (8, lambda: test_8_enhanced_entity_matching(
             narrative_text=shared_data.get('llm_narrative'), # from test 5
             triplets_in_kg=shared_data.get('llm_parsed_triplets') # from test 5 (triplets added to KG)
             ), ['llm_narrative', 'llm_parsed_triplets']), # Requires LLM output from test 5

        (9, lambda: test_9_potential_triplet_extraction(
             narrative_text=shared_data.get('llm_narrative') # from test 5
             ), ['llm_narrative']), # Requires narrative from test 5

        (10, lambda: test_10_semantic_similarity(
             narrative_text=shared_data.get('llm_narrative'), # from test 5
             triplets_in_kg=shared_data.get('llm_parsed_triplets') # from test 5 (triplets added to KG)
             ), ['llm_narrative', 'llm_parsed_triplets']), # Requires LLM output from test 5

        (11, lambda: test_11_coreference_resolution(
             narrative_text=shared_data.get('llm_narrative'), # from test 5
             triplets_in_kg=shared_data.get('llm_parsed_triplets') # from test 5 (triplets added to KG)
             ), ['llm_narrative', 'llm_parsed_triplets']), # Requires LLM output from test 5

        (12, lambda: test_12_enhanced_triplet_extraction(
             narrative_text=shared_data.get('llm_narrative') # from test 5
             ), ['llm_narrative']), # Requires narrative from test 5

        # Add tests for cleaning up, generating reports, etc.
    ]

    # Ensure spaCy model is loaded early if needed by any test we plan to run
    # Tests 7-12 need it, so load it if start_test_number is 1 or less than 12.
    if start_test_number <= 12:
         load_spacy_model()


    for test_number, test_func, required_data_keys in all_tests:
        if test_number < start_test_number:
            print(f"\n--- Skipping Test {test_number} as requested ---")
            log_message(f"Skipping Test {test_number} as requested.")
            continue

        # Check if required shared data is available (only for tests that need it)
        can_run_test = True
        if required_data_keys:
             for key in required_data_keys:
                 # Special handling for config_path which is a direct arg, not in shared_data
                 if key == config_path: continue # Skip if the key is just the config path argument name

                 if key not in shared_data or shared_data[key] is None:
                     print(f"\n--- Skipping Test {test_number}: Requires data '{key}' from previous test which is missing or failed. ---")
                     log_message(f"Skipping Test {test_number}: Requires data '{key}' which is missing.")
                     can_run_test = False
                     test_results[test_number] = {"status": "skipped", "reason": f"Missing dependency: {key}"}
                     break # Stop checking dependencies for this test

        if not can_run_test:
             continue # Skip to the next test in the loop

        # Execute the test function
        # Pass config_path directly to test 3, use shared_data for others via lambda
        if test_number == 3:
             success = test_func(config_path)
             if success: shared_data['config'] = config # Store the loaded config
             test_results[test_number] = {"status": "passed" if success else "failed"}
        elif test_number == 4:
             success, expected_triplets, base_story = test_func()
             if success:
                 shared_data['initial_data'] = (expected_triplets, base_story)
             test_results[test_number] = {"status": "passed" if success else "failed"}
        elif test_number == 5:
             # Test 5 returns (success, parsed_triplets, story_content)
             success, parsed_triplets, story_content = test_func()
             if success:
                 shared_data['llm_parsed_triplets'] = parsed_triplets
                 shared_data['llm_narrative'] = story_content
             test_results[test_number] = {"status": "passed" if success else "failed"}
        elif test_number == 6:
            # Test 6 returns the results dictionary
            results = test_func()
            test_results[test_number] = results
            success = results.get('status') == 'completed' and results.get('found_count', 0) == len(shared_data.get('initial_data', ([],))[0]) # Check if all expected were found
        elif test_number in [7, 8, 9, 10, 11, 12]:
            # Tests 7-12 return results dictionaries
            results = test_func()
            test_results[test_number] = results
            success = results.get('status') == 'completed'

            # For triplet extraction tests, we don't want to stop the test sequence if no potential triplets are found
            if test_number in [9, 12] and success and not results.get('potential_triplets'):
                print("Note: No potential triplets found, but test is considered successful.")
                log_message("Note: No potential triplets found, but test is considered successful.")

            # For semantic similarity and coreference tests, we don't want to stop if the model isn't available
            if test_number in [10, 11] and results.get('status') == 'skipped':
                print(f"Note: Test {test_number} was skipped, but continuing test sequence.")
                log_message(f"Note: Test {test_number} was skipped, but continuing test sequence.")
                success = True
        else:
             # For other tests, assume they return boolean success
             success = test_func()
             test_results[test_number] = {"status": "passed" if success else "failed"}


        # Decide whether to stop based on failure
        if not success:
            print(f"\n--- Test {test_number} Failed ---")
            log_message(f"--- Test {test_number} Failed ---")
            # Stop if crucial setup tests (e.g., < 5) fail
            if test_number < 5:
                print("Setup test failed. Aborting sequence.")
                log_message("Setup test failed. Aborting sequence.")
                break
            # For core PoC tests (>= 5), log failure but potentially continue?
            # Or stop as the core PoC validation failed? Let's stop for now.
            print("PoC test failed. Aborting sequence.")
            log_message("PoC test failed. Aborting sequence.")
            break # Stop loop on first core test failure

        print(f"\n--- Test {test_number} Passed ---")
        log_message(f"--- Test {test_number} Passed ---")


    print("\n--- Test Sequence Finished ---")
    log_message("--- Test Sequence Finished ---")

    # TODO: Generate final summary report based on test_results

# --- Core PoC Logic (Test 5) ---
def test_5_core_poc_logic(config, neo4j_driver, initial_data, use_file_response=False, file_path="Raw Response Text.txt"):
    """Test 5: Executes the core PoC logic: call LLM, parse, update Neo4j.

    If use_file_response is True, uses the content from file_path instead of making an API call.
    """
    print("\n--- Running Test 5: Core PoC Logic (LLM Call, Parse, Neo4j Update) ---")
    log_message("Running Test 5: Core PoC Logic")

    if use_file_response:
        print(f"Note: Using file-based response from {file_path} instead of making an API call")
        log_message(f"Note: Using file-based response from {file_path}")

    if not config or not initial_data or not neo4j_driver:
        print("Test 5 Failed: Missing configuration, initial data, or Neo4j driver.")
        log_message("Test 5 Failed: Missing config, initial data, or Neo4j driver.")
        return (False, None, None) # Return failure status and no data

    expected_triplets, base_story = initial_data

    # --- 5.1 Construct Prompt ---
    try:
        system_prompt_template = config['prompts']['system']
        query_prompt_template = config['prompts']['query']
        api_name = config['api']['api_name']
        model_name = config['api']['model_name']
        # Extract model parameters, exclude known non-parameter keys like api_name, model_name
        model_params = {k: float(v) if '.' in v else int(v) if v.isdigit() else v
                        for k, v in config['api'].items() if k not in ['api_name', 'model_name']}

        separator = config['format']['separator']
        triplet_start_marker = config['format']['triplet_start_marker']
        triplet_end_marker = config['format']['triplet_end_marker']
        story_start_marker = config['format']['story_start_marker']
        story_end_marker = config['format']['story_end_marker']
        triplet_format_desc = config['format']['triplet_format_desc']

    except KeyError as e:
        print(f"Test 5 Failed: Missing key in config file. Error: {e}")
        log_message(f"Test 5 Failed: Missing key in config. Error: {e}")
        return (False, None, None)


    # Format the input triplets for the prompt (simple string format)
    # TODO: Refine triplet formatting for prompt based on experimentation
    triplets_input_formatted = "\n".join([f"({s}, {p}, {o})" for s, p, o in expected_triplets])

    # Fill placeholders in the query prompt
    query_prompt = query_prompt_template.format(
        triplets_input=triplets_input_formatted,
        separator=separator,
        triplet_start_marker=triplet_start_marker,
        triplet_end_marker=triplet_end_marker,
        story_start_marker=story_start_marker,
        story_end_marker=story_end_marker,
        triplet_format_desc=triplet_format_desc
        # Add other format keys as needed
    )

    log_message(f"System Prompt used:\n---\n{system_prompt_template}\n---")
    log_message(f"Query Prompt used:\n---\n{query_prompt}\n---")
    log_message(f"Model parameters used: {model_params}")


    # --- 5.2 Call LLM API ---
    llm_response_json = call_llm_api(
        api_name,
        model_name,
        system_prompt_template,
        query_prompt,
        model_params,
        use_file_response=use_file_response,
        file_path=file_path
    )

    if llm_response_json is None:
        print("Test 5 Failed: LLM API call failed.")
        log_message("Test 5 Failed: LLM API call failed.")
        return (False, None, None)


    # --- 5.3 Parse LLM Response ---
    parsed_triplets, story_content = parse_llm_response(llm_response_json, config)

    if parsed_triplets is None and story_content is None:
        print("Test 5 Failed: LLM response parsing failed.")
        log_message("Test 5 Failed: LLM response parsing failed.")
        # Decide if parsing failure should stop the test or if we try to proceed
        # with partial data? For PoC validation, probably stop.
        return (False, None, None)

    # --- 5.4 Update Neo4j ---
    # Update Neo4j with the triplets *parsed from the LLM's response*
    # Note: This is different from the 'expected_triplets'.
    update_success = update_neo4j_with_triplets(neo4j_driver, parsed_triplets)

    if not update_success:
        print("Test 5 Failed: Neo4j update failed.")
        log_message("Test 5 Failed: Neo4j update failed.")
        # Decide if DB update failure should stop the test. Probably yes.
        return (False, parsed_triplets, story_content) # Return parsed data even on DB failure


    print("Test 5 Passed: Core PoC Logic executed (LLM called, parsed, Neo4j updated).")
    log_message("Test 5 Passed: Core PoC Logic executed.")

    # Return the parsed data for subsequent tests (like verification and spaCy analysis)
    return (True, parsed_triplets, story_content)

# --- Verification Test (Test 6) ---
def test_6_verify_neo4j_against_expected(neo4j_driver, initial_data):
    """Test 6: Verifies if the original expected triplets are in the updated KG."""
    print("\n--- Running Test 6: Verifying Neo4j Content Against Expected Triplets ---")
    log_message("Running Test 6: Verifying Neo4j Content Against Expected Triplets")

    if not initial_data or not neo4j_driver:
         print("Test 6 Failed: Missing initial data or Neo4j driver.")
         log_message("Test 6 Failed: Missing initial data or Neo4j driver.")
         return {"status": "failed", "reason": "Missing dependencies"}

    expected_triplets, _ = initial_data # We only need the expected triplets

    return verify_neo4j_content(neo4j_driver, expected_triplets)

# --- SpaCy Analysis Test (Test 7) ---
def test_7_spacy_analysis(narrative_text, triplets_in_kg):
    """Test 7: Analyzes LLM narrative with spaCy and compares to triplets actually added to KG.

    This test uses the basic entity recognition method.
    """
    print("\n--- Running Test 7: Basic SpaCy Analysis of Narrative ---")
    log_message("Running Test 7: Basic SpaCy Analysis of Narrative")

    # Note: load_spacy_model is called early in run_tests if needed.
    if nlp is None:
         print("Test 7 Failed: SpaCy model not loaded.")
         log_message("Test 7 Failed: SpaCy model not loaded.")
         return {"status": "failed", "reason": "SpaCy model not loaded"}

    if narrative_text is None or triplets_in_kg is None:
        print("Test 7 Failed: Missing narrative text or triplets added to KG.")
        log_message("Test 7 Failed: Missing narrative text or triplets added to KG.")
        return {"status": "failed", "reason": "Missing narrative text or triplets_in_kg"}

    # Pass the parsed triplets (which were used to update KG) to spaCy analysis
    # so it can compare narrative against what's *actually* in the DB from the LLM call.
    return analyze_narrative_with_spacy(narrative_text, triplets_in_kg)

def test_8_enhanced_entity_matching(narrative_text, triplets_in_kg):
    """Test 8: Analyzes LLM narrative with enhanced entity matching techniques.

    This test uses noun chunks and partial matching to improve entity detection.
    """
    print("\n--- Running Test 8: Enhanced Entity Matching Analysis ---")
    log_message("Running Test 8: Enhanced Entity Matching Analysis")

    if nlp is None:
         print("Test 8 Failed: SpaCy model not loaded.")
         log_message("Test 8 Failed: SpaCy model not loaded.")
         return {"status": "failed", "reason": "SpaCy model not loaded"}

    if narrative_text is None or triplets_in_kg is None:
        print("Test 8 Failed: Missing narrative text or triplets added to KG.")
        log_message("Test 8 Failed: Missing narrative text or triplets added to KG.")
        return {"status": "failed", "reason": "Missing narrative text or triplets_in_kg"}

    # Use the enhanced entity matching analysis
    return analyze_narrative_enhanced(narrative_text, triplets_in_kg)

def test_9_potential_triplet_extraction(narrative_text):
    """Test 9: Extracts potential new triplets from the narrative.

    This test identifies subject-verb-object patterns that could represent
    new knowledge to add to the KG.
    """
    print("\n--- Running Test 9: Potential Triplet Extraction ---")
    log_message("Running Test 9: Potential Triplet Extraction")

    if nlp is None:
         print("Test 9 Failed: SpaCy model not loaded.")
         log_message("Test 9 Failed: SpaCy model not loaded.")
         return {"status": "failed", "reason": "SpaCy model not loaded"}

    if narrative_text is None:
        print("Test 9 Failed: Missing narrative text.")
        log_message("Test 9 Failed: Missing narrative text.")
        return {"status": "failed", "reason": "Missing narrative text"}

    # Extract potential new triplets from the narrative
    results = extract_potential_triplets(narrative_text)

    # Save the potential triplets to a file for review
    if results["status"] == "completed" and results["potential_triplets"]:
        try:
            with open("autoUpdateKGfromResponse.txt", "a", encoding="utf-8") as f:
                f.write("\n\n# Potential New Triplets Extracted from Narrative\n\n")
                f.write(f"Date: {datetime.datetime.now().isoformat()}\n\n")
                f.write("The following triplets were automatically extracted from the narrative and could potentially be added to the Knowledge Graph:\n\n")
                for s, p, o in results["potential_triplets"]:
                    f.write(f"* ({s}, {p}, {o})\n")
                f.write("\nThese triplets are provided for review and are not automatically added to the Knowledge Graph.\n")
            print(f"Potential triplets saved to autoUpdateKGfromResponse.txt")
            log_message(f"Potential triplets saved to autoUpdateKGfromResponse.txt")
        except Exception as e:
            print(f"Error saving potential triplets to file: {e}")
            log_message(f"Error saving potential triplets to file: {e}")

    return results

def test_10_semantic_similarity(narrative_text, triplets_in_kg):
    """Test 10: Analyzes LLM narrative using semantic similarity with word vectors.

    This test uses the larger spaCy model with word vectors to find semantically
    similar entities even when they don't match exactly.
    """
    print("\n--- Running Test 10: Semantic Similarity Analysis ---")
    log_message("Running Test 10: Semantic Similarity Analysis")

    if nlp_large is None:
         print("Test 10 Failed: Larger spaCy model not loaded.")
         log_message("Test 10 Failed: Larger spaCy model not loaded.")
         return {"status": "failed", "reason": "Larger spaCy model not loaded"}

    if narrative_text is None or triplets_in_kg is None:
        print("Test 10 Failed: Missing narrative text or triplets added to KG.")
        log_message("Test 10 Failed: Missing narrative text or triplets added to KG.")
        return {"status": "failed", "reason": "Missing narrative text or triplets_in_kg"}

    # Use semantic similarity matching
    return analyze_with_semantic_similarity(narrative_text, triplets_in_kg)

def test_11_coreference_resolution(narrative_text, triplets_in_kg):
    """Test 11: Analyzes LLM narrative with coreference resolution.

    This test uses coreference resolution to link pronouns to their referents
    and improve entity detection.
    """
    print("\n--- Running Test 11: Coreference Resolution Analysis ---")
    log_message("Running Test 11: Coreference Resolution Analysis")

    if coref_nlp is None:
         print("Test 11 Failed: Coreference resolution not available.")
         log_message("Test 11 Failed: Coreference resolution not available.")
         return {"status": "failed", "reason": "Coreference resolution not available"}

    if narrative_text is None or triplets_in_kg is None:
        print("Test 11 Failed: Missing narrative text or triplets added to KG.")
        log_message("Test 11 Failed: Missing narrative text or triplets added to KG.")
        return {"status": "failed", "reason": "Missing narrative text or triplets_in_kg"}

    # Use coreference resolution analysis
    return analyze_with_coreference(narrative_text, triplets_in_kg)

def test_12_enhanced_triplet_extraction(narrative_text):
    """Test 12: Extracts potential new triplets with enhanced relation extraction.

    This test uses more sophisticated patterns and the larger model to extract
    higher-quality triplets from the narrative.
    """
    print("\n--- Running Test 12: Enhanced Triplet Extraction ---")
    log_message("Running Test 12: Enhanced Triplet Extraction")

    if narrative_text is None:
        print("Test 12 Failed: Missing narrative text.")
        log_message("Test 12 Failed: Missing narrative text.")
        return {"status": "failed", "reason": "Missing narrative text"}

    # Extract enhanced triplets from the narrative
    return extract_enhanced_triplets(narrative_text)


# --- Close Resources ---
def close_resources():
    """Closes connections and releases resources."""
    global neo4j_driver
    if neo4j_driver:
        print("\nClosing Neo4j driver...")
        log_message("Closing Neo4j driver...")
        try:
            neo4j_driver.close()
            print("Neo4j driver closed.")
            log_message("Neo4j driver closed.")
        except Exception as e:
             print(f"Error closing Neo4j driver: {e}")
             log_message(f"Error closing Neo4j driver: {e}")
        neo4j_driver = None

    # SpaCy model doesn't typically need explicit closing if loaded via spacy.load

    print("Resources closed.")
    log_message("Resources closed.")


# Ensure resources are closed on script exit
import atexit
atexit.register(close_resources)

# Example of how run_tests might be called from main.py
# if __name__ == "__main__":
#     # This __main__ block is typically handled by main.py
#     # This is just for demonstration if you wanted to run core directly
#     if len(sys.argv) > 1:
#         start_num = int(sys.argv[1])
#         config_file = sys.argv[2] if len(sys.argv) > 2 else "approach/default.ini"
#     else:
#         start_num = 1
#         config_file = "approach/default.ini" # Default config if none specified
#
#     run_tests(start_test_number=start_num, config_path=config_file)