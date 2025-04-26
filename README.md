<!-- Zero Source Specification v1.0 -->
<!-- ZS:COMPLEXITY:HIGH -->
<!-- ZS:PRIORITY:HIGH -->
<!-- ZS:PLATFORM:PYTHON -->
<!-- ZS:LANGUAGE:PYTHON -->

# Knowledge Graph Consistency PoC ([TINS Edition](https://ThereIsNoSource.com))

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