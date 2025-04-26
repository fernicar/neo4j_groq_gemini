Okay, actuando como un LLM que interpreta tu README "Zero Source" para el PoC de consistencia de grafo de conocimiento. Basado en la especificación y funcionalidades descritas, aquí tienes el código Python y los archivos de soporte para iniciar tu proyecto.

Generaré los siguientes archivos:

1.  `main.py`: El punto de entrada que carga el core más reciente y la configuración.
2.  `coreYYYYMMDD.py`: El archivo principal con la lógica del PoC (la fecha será la de hoy).
3.  `requirements.txt`: Lista de dependencias de Python.
4.  `approach/default.ini`: Un archivo `.ini` de configuración de ejemplo.
5.  `.env.example`: Un archivo de ejemplo para configurar las variables de entorno.
6.  `README.md`: (El contenido que me proporcionaste, para que lo tengas en el repositorio).
7.  `LICENSE`: (Texto estándar de la licencia MIT).
8.  `history.log`: Un archivo vacío inicial para el log.
9.  `data/expected_data.py`: Un archivo placeholder simple para los datos iniciales.

**Importante:** Este código es una *implementación inicial* basada directamente en tu README. Refleja la estructura y la lógica descritas, pero necesitará refinamiento, manejo de errores más robusto y la implementación detallada de la lógica de parsing y análisis con spaCy, que son las partes complejas y experimentales de tu PoC. Los tests iniciales de setup están más completos, y las partes del LLM/SpaCy/Neo4j tienen la estructura básica lista para que construyas sobre ella.

---

```python
# main.py
import argparse
import os
import glob
import importlib.util
import sys
import time

def find_latest_core_module(directory="."):
    """Finds the latest coreYYYYMMDD.py file."""
    core_files = sorted(glob.glob(os.path.join(directory, "core*.py")))
    if not core_files:
        return None
    # Simple sort by name should work for YYYYMMDD format
    latest_core_path = core_files[-1]
    return latest_core_path

def find_latest_ini_file(directory="approach"):
    """Finds the latest .ini file in the specified directory."""
    if not os.path.exists(directory):
        return None
    ini_files = glob.glob(os.path.join(directory, "*.ini"))
    if not ini_files:
        return None
    # Sort by modification time
    latest_ini_path = max(ini_files, key=os.path.getmtime)
    return latest_ini_path

def main():
    parser = argparse.ArgumentParser(description="Run Knowledge Graph Consistency PoC tests.")
    parser.add_argument("--test", type=int, default=1,
                        help="Start running tests from this number.")
    parser.add_argument("--approach", type=str, default=None,
                        help="Specify the configuration .ini file (e.g., default.ini).")
    args = parser.parse_args()

    # 1. Find the latest core module
    latest_core_path = find_latest_core_module()
    if not latest_core_path:
        print("Error: No coreYYYYMMDD.py file found.")
        sys.exit(1)

    print(f"Using core logic from: {latest_core_path}")

    # 2. Determine the configuration file
    config_path = args.approach
    if config_path is None:
        latest_ini_path = find_latest_ini_file()
        if latest_ini_path:
            config_path = latest_ini_path
            print(f"No --approach specified. Using latest config: {config_path}")
        else:
            print("Error: No --approach specified and no .ini files found in ./approach/.")
            sys.exit(1)
    else:
        config_path = os.path.join("approach", config_path)
        if not os.path.exists(config_path):
            print(f"Error: Specified config file not found: {config_path}")
            sys.exit(1)
        print(f"Using specified config: {config_path}")

    # 3. Load the core module dynamically
    module_name = os.path.basename(latest_core_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, latest_core_path)
    if spec is None:
         print(f"Error: Could not load module spec for {latest_core_path}")
         sys.exit(1)
    core_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = core_module
    try:
        spec.loader.exec_module(core_module)
    except Exception as e:
        print(f"Error loading core module {latest_core_path}: {e}")
        sys.exit(1)


    # 4. Run the tests from the core module
    if hasattr(core_module, 'run_tests'):
        core_module.run_tests(start_test_number=args.test, config_path=config_path)
    else:
        print(f"Error: Core module {latest_core_path} does not have a 'run_tests' function.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

```python
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
google_api_call_timestamps = deque(maxlen=5) # For Google rate limiting

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

def call_llm_api(api_name, model_name, system_prompt, query_prompt, model_params):
    """Calls the specified LLM API with the given prompts and parameters."""
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
    """Loads the spaCy language model."""
    global nlp
    print("\nLoading spaCy model 'en_core_web_sm'...")
    log_message("Loading spaCy model 'en_core_web_sm'")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded.")
        log_message("spaCy model loaded.")
        return True
    except Exception as e:
        print(f"Error loading spaCy model. Have you run 'python -m spacy download en_core_web_sm'? Error: {e}")
        log_message(f"Error loading spaCy model. Error: {e}")
        return False

def analyze_narrative_with_spacy(narrative_text, triplets_in_kg):
    """Analyzes the narrative with spaCy and compares entities/relations with triplets in KG."""
    print("\nAnalyzing narrative with spaCy...")
    log_message("Analyzing narrative with spaCy")

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


        # Compare entities found by spaCy with entities in the KG triplets
        kg_entities_lower = {ent.lower() for ent in kg_entities}
        kg_entities_in_narrative = 0
        for kg_ent_lower in kg_entities_lower:
            if kg_ent_lower in narrative_entities:
                kg_entities_in_narrative += 1
                log_message(f"KG entity '{kg_ent_lower}' found in narrative.")

        # TODO: More advanced spaCy analysis (e.g., dependency parsing to find relationships)
        # This is the complex part dependent on your PoC goals for spaCy's role.
        # For now, we focus on entity presence as a simple metric.

        match_percentage = (kg_entities_in_narrative / len(kg_entities)) * 100 if kg_entities else 0

        print(f"SpaCy Analysis Results:")
        print(f"  Entities in Narrative (found by spaCy): {len(narrative_entities)}")
        print(f"  Entities in KG (from added triplets): {len(kg_entities)}")
        print(f"  KG Entities found in Narrative: {kg_entities_in_narrative}")
        print(f"  Entity Presence Match Percentage: {match_percentage:.2f}%")

        log_message(f"SpaCy Analysis Results: Narrative entities={len(narrative_entities)}, KG entities={len(kg_entities)}, KG entities in narrative={kg_entities_in_narrative}, Match Percentage={match_percentage:.2f}%")


        return {
            "status": "completed",
            "match_percentage": match_percentage,
            "kg_entity_count": len(kg_entities),
            "narrative_entity_count": len(narrative_entities),
            "kg_entities_in_narrative": kg_entities_in_narrative
            }

    except Exception as e:
        print(f"Error during spaCy analysis: {e}")
        log_message(f"Error during spaCy analysis: {e}")
        return {"status": "error", "error": str(e)}


# --- Main Test Runner Function ---

def run_tests(start_test_number=1, config_path=None):
    """Runs the sequence of tests starting from start_test_number."""

    print(f"\n--- Starting PoC Test Sequence from Test {start_test_number} ---")
    log_message(f"--- Starting PoC Test Sequence from Test {start_test_number} ---")
    log_message(f"Using config file: {config_path}")

    test_results = {} # Dictionary to store results of each test
    shared_data = {} # Dictionary to pass data between tests (e.g., loaded config, initial data)

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
            initial_data=shared_data.get('initial_data') # (expected_triplets, base_story) tuple
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

        # Add tests for cleaning up, generating reports, etc.
    ]

    # Ensure spaCy model is loaded early if needed by any test we plan to run
    # Test 7 needs it, so load it if start_test_number is 1 or less than 7.
    if start_test_number <= 7:
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
        elif test_number == 7:
            # Test 7 returns the results dictionary
            results = test_func()
            test_results[test_number] = results
            success = results.get('status') == 'completed'
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
def test_5_core_poc_logic(config, neo4j_driver, initial_data):
    """Test 5: Executes the core PoC logic: call LLM, parse, update Neo4j."""
    print("\n--- Running Test 5: Core PoC Logic (LLM Call, Parse, Neo4j Update) ---")
    log_message("Running Test 5: Core PoC Logic")

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
    llm_response_json = call_llm_api(api_name, model_name, system_prompt_template, query_prompt, model_params)

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
    """Test 7: Analyzes LLM narrative with spaCy and compares to triplets actually added to KG."""
    print("\n--- Running Test 7: SpaCy Analysis of Narrative ---")
    log_message("Running Test 7: SpaCy Analysis of Narrative")

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

```

---

```python
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

```
*(You'll need to create a directory named `data` and place this file inside)*

---

```ini
# approach/default.ini
# Default configuration for KG Consistency PoC

[api]
api_name = google
model_name = gemini-1.5-pro-latest
# Add other model parameters as needed based on API docs
# Groq parameters example: temperature = 0.7, max_tokens = 500, top_p = 0.9
# Google parameters example: temperature = 0.7, max_output_tokens = 500, top_p = 0.9
temperature = 0.7
max_tokens = 500 # Using max_tokens as a generic name, map to specific API param in code

[format]
# Define the markers and separator based on the expected LLM output
# TripleBackquote is represented as ```
separator = ###END_TRIPLETS###
triplet_start_marker = ```markdown
triplet_end_marker = ```
story_start_marker = ```text
story_end_marker = ```
# Description of how triplets should look within the markers (used in prompt)
triplet_format_desc = Each triplet should be on a new line in the format: (Subject, predicate, Object)

[prompts]
# System prompt defines the LLM's persona and overall instructions
system = You are a highly accurate knowledge graph generator and storyteller. Your task is to take a list of factual triplets provided to you. You must re-state these exact facts in the specified output format and then write a natural language story that fully incorporates these facts. Ensure the output strictly follows the specified format, including markers and separator.

# Query prompt defines the specific task for this turn, including dynamic data
# Use placeholders like {triplets_input}, {separator}, {triplet_format_desc}, etc.
query = """
Process the following list of factual triplets.

First, output these same facts in the specified triplet output format, enclosed in the triplet start and end markers.
Second, output a natural language story based ONLY on these facts, enclosed in the story start and end markers.
Separate the triplets section from the story section with the specified separator string.

---
REQUIRED OUTPUT FORMAT:
{triplet_start_marker}
[LIST OF TRIPLETS IN SPECIFIED FORMAT]
{triplet_end_marker}
{separator}
{story_start_marker}
[NATURAL LANGUAGE STORY BASED ON TRIPLETS]
{story_end_marker}
---
Separator String: {separator}
Triplet Output Format Description: {triplet_format_desc}
Triplet Start Marker: {triplet_start_marker}
Triplet End Marker: {triplet_end_marker}
Story Start Marker: {story_start_marker}
Story End Marker: {story_end_marker}

---
Input Triplets (Facts to use):
{triplets_input}
---
Generate the output now.
"""

```
*(You'll need to create a directory named `approach` and place this file inside)*

---

```text
# requirements.txt
neo4j==5.21.0 # Or the latest compatible version with your Neo4j Desktop
spacy==3.7.5 # Or the latest compatible version
requests==2.31.0 # Or the latest version
python-dotenv==1.0.0 # Or the latest version
# configparser and argparse are built-in
```

---

```text
# .env.example
# Environment variables for KG Consistency PoC
# Copy this file to .env and fill in your actual credentials

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password_here

GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

```
*(Copy this file to `.env` and fill in your details. Do NOT commit your `.env` file to Git)*

---

```text
# history.log
# This file will store the results of each test run.
# Entries will be appended with timestamps.

```
*(Create an empty file named `history.log`)*

---

```text
# LICENSE
MIT License

Copyright (c) YYYY Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.```
*(Replace YYYY with the current year and Your Name with your name.)*

---

```markdown
# README.md
<!-- Zero Source Specification v1.0 -->
<!-- ZS:COMPLEXITY:HIGH -->
<!-- ZS:PRIORITY:HIGH -->
<!-- ZS:PLATFORM:PYTHON -->
<!-- ZS:LANGUAGE:PYTHON -->

# Zero Source Knowledge Graph Consistency PoC

## Description

This project is a Proof of Concept (PoC) designed to test and validate the consistency between Large Language Model (LLM) outputs (specifically, structured data representing knowledge graph triplets and corresponding natural language narrative) and a Neo4j knowledge graph. The core idea is to verify if an LLM can reliably generate both update instructions for a knowledge graph and a narrative that accurately reflects those updates in a single API call, and if this can be programmatically verified.

The PoC is implemented as a Python command-line tool that performs a series of incremental tests, including setting up the environment, interacting with LLM APIs, updating and querying a Neo4j database, analyzing generated narrative using spaCy, and logging the results.

## Functionality

### Core Features

The tool's primary function is to execute a defined test sequence. For each test run (after initial setup):
1.  Load a pre-defined set of "expected triplets" and potentially a "base story" from configuration or data files.
2.  Initialize the Neo4j knowledge graph to an empty state.
3.  Construct a prompt for the selected LLM API based on configuration, including the expected triplets and instructions for structured output.
4.  Call the chosen LLM API (Groq or Google Gemini).
5.  Parse the LLM's response into two distinct parts: knowledge graph update instructions (triplets) and the generated narrative, using markers defined in the configuration.
6.  Update the Neo4j knowledge graph with the triplets parsed from the LLM's response.
7.  Verify that the pre-defined "expected triplets" are now present in the updated Neo4j graph and summarize any differences.
8.  Process the generated narrative using spaCy to identify entities and relationships.
9.  Compare the entities/relationships found by spaCy in the narrative against the triplets that were added to the Neo4j graph (from the LLM's response), reporting a percentage of match.
10. Log all relevant details of the test run (configuration, prompts, raw LLM response, parsing results, Neo4j validation results, spaCy analysis results, timestamps) to a history file.

### Command-Line Interface

The tool should be executable via a simple Python command, supporting arguments:
-   `python main.py`: Runs the full test sequence from the beginning.
-   `python main.py --test N`: Starts the test sequence from test number `N`. Useful for skipping initial setup tests during development.
-   `python main.py --approach config_file.ini`: Uses the specified `.ini` file from the `approach/` directory for configuration. If not specified, the tool should find and use the most recently modified `.ini` file in `approach/`.

### Configuration Management

Configuration for each test run (LLM API, model, parameters, prompt structure, output parsing format, markers) is defined in `.ini` files located in the `approach/` directory.
API keys for LLMs (Groq, Google) are read from environment variables, preferably loaded from a `.env` file in the project root for local development.

### Test Execution Flow

Tests are executed incrementally in a defined, numbered sequence.
-   Initial tests (e.g., Test 1: Check Neo4j connection, Test 2: Clean DB, Test 3: Load Configuration, Test 4: Load Initial Data) verify the environment setup.
-   Core PoC tests follow the initial setup, performing the steps outlined in "Core Features".
-   If any test step fails (especially during initial setup), the script should report the failure clearly and may halt execution, depending on the nature of the failure.

### Rate Limiting

Implement a rate-limiting mechanism for Google Gemini API calls to prevent exceeding typical free tier limits (e.g., 5 calls per minute). Before making a Gemini API call, the script should check recent call history and pause execution if necessary to stay within limits. If a rate limit error is still received from the API, log the error and inform the user where they can check their usage quota (e.g., Google Cloud Console).

### Logging

All test runs, including configuration used, prompts sent, raw LLM response, parsing results, Neo4j interactions (updates, queries, validation results), spaCy analysis output, and any errors, must be appended to a history log file (e.g., `history.log`) with timestamps.

## Technical Implementation

### Architecture

The project will consist of:
-   A simple `main.py` script to handle command-line arguments and load the core logic module.
-   A versioned core logic module (`coreYYYYMMDD.py`) containing the main test execution logic.
-   A `approach/` directory holding `.ini` configuration files for different test approaches.
-   External dependencies: Neo4j database, LLM APIs (Groq/Google), Python libraries (`neo4j`, `spacy`, `requests`, `python-dotenv`).

### Data Model

-   **Expected Triplets:** Represented internally as a list of tuples or objects, conceptually `(Subject: string, Predicate: string, Object: string)`. Format in initial data file TBD, but easily parsable by Python.
-   **LLM Input Prompt:** A formatted string built dynamically using templates from the `.ini` file, incorporating the expected triplets and format instructions.
-   **LLM Output:** A single string containing two sections (triplets and narrative) separated by a defined marker.
    -   **Output Triplets:** Parsed from the LLM output string based on start/end markers and internal structure defined in the `.ini`. Conceptual structure matches Expected Triplets.
    -   **Generated Narrative:** Parsed text string from the LLM output based on start/end markers.
-   **Neo4j Data:**
    -   Nodes: Primarily representing entities (Subjects and Objects from triplets), likely with a `label` like `Entity` and a `name` property storing the string identifier. `(:Entity {name: 'Subject Name'})`.
    -   Relationships: Representing predicates, with the predicate string as the relationship *type*. `(SubjectEntity)-[:PREDICATE_TYPE]->(ObjectEntity)`.
-   **SpaCy Analysis Data:** SpaCy `Doc` object, identified `Entities` (Named Entities), and potentially dependency parse or custom extracted relationships.
-   **Log Data:** Structured entries (could be JSON or a custom delimited format) containing all relevant test run details.

### Algorithms

-   **Test Runner:** Iterate through a list of test functions, executing from the specified starting number. Pass necessary shared resources (Neo4j driver, config, initial data) to test functions. Halt or continue based on test success/failure.
-   **Neo4j Interaction:**
    -   Connection: Use `neo4j.GraphDatabase.driver`.
    -   Cleaning: Execute Cypher `MATCH (n) DETACH DELETE n` within a session/transaction.
    -   Adding Triplets: For each parsed triplet `(s, p, o)`, execute Cypher `MERGE (a:Entity {{name: $s_name}}) MERGE (b:Entity {{name: $o_name}}) MERGE (a)-[:{relation_type}]->(b)` where `:RELATION_TYPE` is the predicate `p` (sanitized for use as a relationship type) and `$s_name`, `$o_name` are parameters for `s` and `o`.
    -   Validation: For each expected triplet `(es, ep, eo)`, execute Cypher `MATCH (a:Entity {{name: $es_name}})-[r:{relation_type}]->(b:Entity {{name: $eo_name}}) RETURN count(r) > 0 AS exists`. Count how many expected triplets are found.
-   **LLM Interaction:** Use the `requests` library to make POST requests to the chosen API endpoint (Groq or Google), including the system prompt, query prompt constructed from the template and dynamic data, and model parameters based on the active `.ini` configuration.
-   **Response Parsing:** Use standard Python string manipulation (`.find()`, `.split()`) based on the `separator` and section `start`/`end` markers read from the `.ini` file.
-   **SpaCy Analysis:** Load a spaCy model (`en_core_web_sm` initially). Process the generated narrative text (`nlp(narrative_text)`). Extract entities (`doc.ents`). Attempt to relate entities based on dependency parsing or proximity to identify potential relationships. Compare the extracted information (entities, relationships) to the entities and relationships *successfully added to Neo4j* in this test run to calculate the percentage match.
-   **Rate Limiting:** Maintain a list/deque of timestamps. Before calling the Gemini API, check if the required interval has passed since the 5th most recent call. Use `time.sleep()` if necessary.

### External Integrations

-   **Neo4j Database:** Local instance (e.g., via Neo4j Desktop), accessed via the `neo4j` Python driver using the Bolt protocol.
-   **Groq API:** Accessed via HTTP requests using an API key from `.env`.
-   **Google Generative AI API (Gemini):** Accessed via HTTP requests using an API key from `.env`. Requires custom rate-limiting logic in the script.
-   **spaCy Library:** Used locally within the Python script for NLP processing. Requires downloading a language model (`en_core_web_sm`).

## Testing Scenarios

The numbered test functions will cover:
1.  Verify connection to Neo4j.
2.  Verify ability to clean the Neo4j database.
3.  Verify ability to load and parse a specified `.ini` configuration file.
4.  Verify ability to load initial "expected triplets" and "base story" data.
5.  Execute core PoC logic: LLM call, parsing, Neo4j update.
6.  Validate Neo4j content against expected triplets, report differences.
7.  Execute spaCy analysis on generated narrative and report match percentage with updated KG content.
8.  Verify successful logging of test results.

## Logging

Test results will be appended to a file (e.g., `history.log`). Each entry should include a timestamp and clearly delineate:
-   Configuration file used.
-   LLM API and model used.
-   System and query prompts sent (including filled placeholders).
-   Raw LLM response.
-   Parsed triplets and narrative.
-   Status and details of the Neo4j update.
-   Result of Neo4j validation (expected vs. actual triplets in KG).
-   Result of spaCy analysis (extracted info vs. KG content, match percentage).
-   Any errors encountered.

## Getting Started

1.  Ensure you have Python 3.12.9 installed.
2.  Clone this repository.
3.  Navigate to the project directory in your terminal.
4.  Create a Python virtual environment: `python -m venv .venv`
5.  Activate the virtual environment:
    -   On macOS/Linux: `source .venv/bin/activate`
    -   On Windows: `.\\.venv\\Scripts\\activate`
6.  Install dependencies: `pip install -r requirements.txt`
7.  Download the necessary spaCy language model: `python -m spacy download en_core_web_sm`
8.  Install and run Neo4j Desktop. Create a new database instance. Note down the connection details (URI, username, password - default is often `bolt://localhost:7687`, `neo4j`, `neo4j`).
9.  Create a `.env` file in the project root with your API keys and Neo4j credentials (see `.env.example`). **Do NOT commit your `.env` file to Git.**
10. Prepare your initial data (expected triplets, base story). See `data/expected_data.py` for a placeholder structure.
11. Create or modify a configuration file in `approach/` (e.g., `approach/default.ini`) defining the LLM, model, parameters, and expected output format markers.
12. Run the test script: `python main.py [--test N] [--approach config_file.ini]`

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contributing

Contributions to refine the README specification for this PoC, improve the testing methodology, or suggest better prompting strategies are welcome via pull requests.

## Implementation Notes (For the LLM generating code)

- The `main.py` script should find the latest `coreYYYYMMDD.py` numerically/chronologically.
- Handle potential file not found errors gracefully for `.ini` and initial data files.
- The rate limiting for Gemini should use `time.sleep()` and print informative messages.
- Error handling for API calls and Neo4j operations is critical.
- The parsing logic should be robust enough to handle minor variations in LLM output formatting if possible, based on the `.ini` definitions.
- The spaCy analysis should focus on verifying the *presence* of entities and the *relationships* implied by the narrative text, mapping them back to the entities and relationships actually added to the Neo4j graph. The percentage calculation should reflect how well the narrative covers the factual information stored in the KG for that test run.

```

---

**Instrucciones Adicionales:**

1.  Crea las carpetas `approach` y `data` en el directorio de tu proyecto.
2.  Guarda el contenido de cada bloque de código en el archivo correspondiente.
3.  Asegúrate de renombrar `coreYYYYMMDD.py` a `core20250424.py` (o la fecha actual si prefieres que sea la de hoy).
4.  Crea el archivo `.env` copiando `.env.example` y llenando tus credenciales.
5.  Sigue los pasos en la sección "Getting Started" del `README.md` generado para configurar tu entorno e instalar dependencias.

Ahora tienes la estructura base del proyecto lista para empezar a experimentar con las configuraciones `.ini`, refinar la lógica de parsing y análisis de spaCy, y probar la consistencia entre los LLMs y Neo4j. ¡Mucha suerte con esta interesante prueba de concepto!