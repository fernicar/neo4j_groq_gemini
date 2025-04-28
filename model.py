# model.py
import os
import json
import time
import logging
import datetime
from pathlib import Path
from collections import deque
from typing import Dict, List, Any, Tuple, Optional
import sys # Import sys for atexit handler

# External Libraries
from neo4j import GraphDatabase, basic_auth, Driver # Import Driver for type hinting
from neo4j.exceptions import ServiceUnavailable # Import specific Neo4j exceptions
import requests
from lxml import etree # Recommended for robust XML parsing
from dotenv import load_dotenv # Requires python-dotenv

# --- Configuration and Setup ---

# Load environment variables from .env file
# This should be done once at the application startup
load_dotenv()

# --- Logging Setup ---
LOG_FILE = "history.log"
# Ensure log directory exists if LOG_FILE has a path
log_dir = os.path.dirname(LOG_FILE)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)

# --- Paths Setup ---
# Define the directory for prompt templates
PROMPTS_DIR = Path("prompts")

# Basic logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(LOG_FILE, encoding='utf-8'),
    logging.StreamHandler(sys.stdout) # Also log to console
])
# Silence some chatty loggers if necessary
logging.getLogger('neo4j').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


def log_info(message):
    """Logs an info message."""
    logging.info(message)

def log_warning(message):
    """Logs a warning message."""
    logging.warning(message)

def log_error(message, exc_info=False):
    """Logs an error message."""
    logging.error(message, exc_info=exc_info)

# --- Data Structures (Matching conceptual XML/KG Model) ---
# These represent the parsed data or data structures for KG interaction
# Note: The actual Neo4j interaction uses Cypher and dictionaries,
# but these types help structure the data flow.

class ParsedEntity:
    def __init__(self, xml_id: str, text_span: str, canonical: Optional[str] = None,
                 entity_type: Optional[str] = None, status: Optional[str] = 'Pending',
                 provenance: Optional[List[str]] = None, attributes: Optional[Dict[str, Any]] = None):
        self.xml_id = xml_id # ID from the XML tag
        self.text_span = text_span # The actual text wrapped by the tag in the narrative
        self.canonical = canonical if canonical is not None else text_span.strip() # Default canonical to stripped text_span
        self.entity_type = entity_type
        self.status = status # Suggested status from LLM or default
        self.provenance = provenance if provenance is not None else ['LLM_XML_Generated']
        self.attributes = attributes if attributes is not None else {} # Store any other XML attributes

    def to_dict(self):
         # Return a dictionary representation for display/logging
         return self.__dict__

    def __repr__(self):
        return f"Entity(id='{self.xml_id}', text='{self.text_span}', canonical='{self.canonical}', type='{self.entity_type}', status='{self.status}')"


class ParsedRelation:
    def __init__(self, xml_id: str, text_span: str, relation_type: str, subj_id: str, obj_id: str,
                 status: Optional[str] = 'Pending', provenance: Optional[List[str]] = None,
                 attributes: Optional[Dict[str, Any]] = None):
        self.xml_id = xml_id # ID from the XML tag (optional, can be generated)
        self.text_span = text_span # The actual text wrapped by the tag
        self.relation_type = relation_type # Suggested type for KG edge
        self.subj_id = subj_id # Refers to xml_id of ParsedEntity
        self.obj_id = obj_id # Refers to xml_id of ParsedEntity
        self.status = status # Suggested status
        self.provenance = provenance if provenance is not None else ['LLM_XML_Generated']
        self.attributes = attributes if attributes is not None else {} # Store any other XML attributes

    def to_dict(self):
         return self.__dict__

    def __repr__(self):
        return f"Relation(id='{self.xml_id}', text='{self.text_span}', type='{self.relation_type}', subj='{self.subj_id}', obj='{self.obj_id}', status='{self.status}')"


class ParsedQuery:
    def __init__(self, xml_id: str, purpose: str, query_string: str):
        self.xml_id = xml_id # ID from the XML tag (optional)
        self.purpose = purpose # Purpose from XML attribute
        self.query_string = query_string.strip() # Cypher query text

    def to_dict(self):
         return self.__dict__

    def __repr__(self):
        return f"Query(id='{self.xml_id}', purpose='{self.purpose}', query='{self.query_string}')"


class LLMResponseParsed:
    def __init__(self, raw_xml: str, narrative_xml_element: Optional[etree.Element],
                 entities: List[ParsedEntity], relations: List[ParsedRelation],
                 queries: List[ParsedQuery], raw_response_json: Dict[str, Any]):
        self.raw_xml = raw_xml # The full raw XML string from LLM
        self.narrative_xml_element = narrative_xml_element # The lxml element for the narrative section
        self.entities = entities # List of parsed entities
        self.relations = relations # List of parsed relations
        self.queries = queries # List of parsed queries
        self.raw_response_json = raw_response_json # Full original JSON response


    def to_dict(self):
         # Return a serializable dictionary
         return {
             "raw_xml": self.raw_xml,
             # Note: narrative_xml_element is not serializable, handle in UI logic
             "entities": [e.to_dict() for e in self.entities],
             "relations": [r.to_dict() for r in self.relations],
             "queries": [q.to_dict() for q in self.queries],
             "raw_response_json": self.raw_response_json
         }


# --- Model Class ---

class CurationModel:
    def __init__(self, config_path: str = "approach/default.json"):
        self.config_path = config_path
        self.config: Optional[Dict[str, Any]] = None
        self.neo4j_driver: Optional[Driver] = None
        self._google_api_call_timestamps = deque(maxlen=5) # For Google rate limiting

        log_info("Initializing CurationModel...")
        self.load_config(self.config_path) # Load config on init

        # Placeholder for loaded prompt content
        self._system_prompt_template: str = ""
        self._query_prompt_template: str = ""
        self._load_prompt_templates() # Load prompts based on initial config

        # Placeholder for conversation history (managed as XML turns)
        # In a real app, this would be loaded/saved from a file
        self._conversation_history_turns: List[Tuple[str, str]] = [] # List of (user_prompt_xml, llm_response_xml) pairs

        log_info("CurationModel initialized.")

    def load_config(self, config_path: str) -> bool:
        """Loads configuration from the specified JSON file and reloads prompts."""
        self.config_path = config_path # Update path
        log_info(f"Loading configuration from {self.config_path}")
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            log_info("Configuration loaded successfully.")
            # Basic validation (check for essential keys)
            if not all(k in self.config for k in ['api', 'format', 'prompts']):
                 log_warning("Config file missing essential top-level keys ('api', 'format', 'prompts').")
            # TODO: More detailed config validation based on expected structure

            self._load_prompt_templates() # Reload prompts based on the new config

            return True
        except FileNotFoundError:
            log_error(f"Configuration file not found: {self.config_path}")
            self.config = None
            return False
        except json.JSONDecodeError as e:
            log_error(f"Error decoding JSON configuration file: {self.config_path}. Error: {e}")
            self.config = None
            return False
        except Exception as e:
            log_error(f"An unexpected error occurred loading config: {e}", exc_info=True)
            self.config = None
            return False

    def get_config(self) -> Optional[Dict[str, Any]]:
        """Returns the currently loaded configuration."""
        return self.config

    def _load_prompt_templates(self):
        """Loads prompt content from text files specified in config."""
        if not self.config or 'prompts' not in self.config:
            log_warning("Config or 'prompts' section not loaded. Cannot load prompt templates.")
            self._system_prompt_template = ""
            self._query_prompt_template = ""
            return

        prompts_config = self.config.get('prompts', {})
        system_file = prompts_config.get('system_prompt_file')
        query_file = prompts_config.get('query_prompt_file')

        self._system_prompt_template = "" # Clear previous templates
        self._query_prompt_template = ""

        if system_file:
            # Handle the case where the path already includes 'prompts\'
            if system_file.startswith('prompts\\') or system_file.startswith('prompts/'):
                system_file_path = Path(system_file)
            else:
                system_file_path = Path(system_file)
                if not system_file_path.is_absolute():
                    system_file_path = PROMPTS_DIR / system_file # Assume relative to PROMPTS_DIR

            log_info(f"Loading system prompt from {system_file_path}")
            try:
                with open(system_file_path, 'r', encoding='utf-8') as f:
                    self._system_prompt_template = f.read()
                log_info("System prompt template loaded.")
            except FileNotFoundError:
                 log_error(f"System prompt file not found: {system_file_path}")
            except Exception as e:
                log_error(f"Error loading system prompt template: {e}", exc_info=True)

        else:
             log_warning("'system_prompt_file' not specified in config.")


        if query_file:
            # Handle the case where the path already includes 'prompts\'
            if query_file.startswith('prompts\\') or query_file.startswith('prompts/'):
                query_file_path = Path(query_file)
            else:
                query_file_path = Path(query_file)
                if not query_file_path.is_absolute():
                    query_file_path = PROMPTS_DIR / query_file # Assume relative to PROMPTS_DIR

            log_info(f"Loading query prompt from {query_file_path}")
            try:
                with open(query_file_path, 'r', encoding='utf-8') as f:
                    self._query_prompt_template = f.read()
                log_info("Query prompt template loaded.")
            except FileNotFoundError:
                 log_error(f"Query prompt file not found: {query_file_path}")
            except Exception as e:
                log_error(f"Error loading query prompt template: {e}", exc_info=True)
        else:
             log_warning("'query_prompt_file' not specified in config.")


    def connect_neo4j(self) -> bool:
        """Establishes connection to Neo4j."""
        if self.neo4j_driver:
            try:
                # Verify connectivity without closing the existing driver
                self.neo4j_driver.verify_connectivity()
                # If verify_connectivity passes, the driver is likely fine
                log_info("Already connected to Neo4j.")
                return True
            except ServiceUnavailable:
                log_warning("Existing Neo4j driver is not connected or service unavailable. Re-initializing.")
                self.close_neo4j() # Close the old one
            except Exception as e:
                 log_warning(f"Unexpected error verifying Neo4j connectivity: {e}. Re-initializing.", exc_info=True)
                 self.close_neo4j()


        log_info(f"Connecting to Neo4j at {os.getenv('NEO4J_URI')}")
        try:
            uri = os.getenv("NEO4J_URI")
            user = os.getenv("NEO4J_USER")
            password = os.getenv("NEO4J_PASSWORD")
            if not uri or not user or not password:
                 log_error("Neo4j credentials missing in environment variables (.env file?).")
                 return False

            # Use a connection timeout
            self.neo4j_driver = GraphDatabase.driver(uri, auth=basic_auth(user, password), connection_timeout=10.0)
            # Check connectivity by running a simple query or verifying connection
            # driver.verify_connectivity() is good, or a quick query:
            with self.neo4j_driver.session() as session:
                 session.run("RETURN 1").consume() # Simple query to check connection

            log_info("Neo4j connection successful.")
            return True
        except ServiceUnavailable as e:
            log_error(f"Neo4j service unavailable at {uri}: {e}", exc_info=True)
            self.neo4j_driver = None
            return False
        except Exception as e:
            log_error(f"Failed to connect to Neo4j: {e}", exc_info=True)
            self.neo4j_driver = None
            return False

    def close_neo4j(self):
        """Closes the Neo4j connection."""
        if self.neo4j_driver:
            log_info("Closing Neo4j driver...")
            try:
                self.neo4j_driver.close()
                log_info("Neo4j driver closed.")
            except Exception as e:
                 log_error(f"Error closing Neo4j driver: {e}", exc_info=True)
            self.neo4j_driver = None

    def run_cypher_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        """Runs a Cypher read query and returns results."""
        if not self.neo4j_driver or not self.connect_neo4j(): # Ensure connection is active
            log_error("Cannot run Cypher query: Not connected to Neo4j.")
            return None
        log_info(f"Running Cypher query: {query} with params: {parameters}")
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query, parameters)
                return result.data() # Return list of dictionaries
        except Exception as e:
            log_error(f"Error running Cypher query: {e}", exc_info=True)
            return None

    def run_cypher_transaction(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        """Runs a Cypher write transaction and returns summary."""
        if not self.neo4j_driver or not self.connect_neo4j(): # Ensure connection is active
            log_error("Cannot run Cypher transaction: Not connected to Neo4j.")
            return None
        # log_info(f"Running Cypher transaction: {query} with params: {parameters}") # Log query only on success or error to save space
        try:
            with self.neo4j_driver.session() as session:
                summary = session.execute_write(lambda tx: tx.run(query, parameters).consume())
                log_info(f"Transaction successful. Query: {query[:100]}... Summary: {summary.counters}") # Log summary
                return summary
        except Exception as e:
            log_error(f"Error running Cypher transaction. Query: {query[:100]}... Error: {e}", exc_info=True)
            return None

    # TODO: Implement robust conversation history management in XML format
    def get_conversation_history_xml(self) -> str:
        """Retrieves the full conversation history in XML format."""
        # This is a placeholder. In a real app, load from a file or internal structure.
        # The format should match what build_llm_prompt expects for history.
        log_info("Retrieving placeholder conversation history XML.")
        # Example structure: <history><turn><user_prompt>...</user_prompt><llm_response>...</llm_response></turn>...</history>
        history_xml_content = "".join([
            f"<turn><user_prompt>{user_xml}</user_prompt><llm_response>{llm_xml}</llm_response></turn>"
            for user_xml, llm_xml in self._conversation_history_turns
        ])
        return f"<conversation_history>{history_xml_content}</conversation_history>"

    # TODO: Implement saving conversation turn
    def save_conversation_turn(self, user_prompt_xml: str, llm_response_xml: str):
        """Saves a single turn (prompt and response XML) to the history."""
        # This is a placeholder. In a real app, append to an XML history file.
        log_info("Saving conversation turn (placeholder).")
        self._conversation_history_turns.append((user_prompt_xml, llm_response_xml))
        log_info(f"Conversation history now has {len(self._conversation_history_turns)} turns.")
        # TODO: Implement persistence (e.g., save to a file).

    # TODO: Implement KG Context retrieval (potentially based on user selection or automated)
    def retrieve_kg_context_for_prompt(self, user_instruction: str, last_llm_response_xml: Optional[str] = None) -> str:
        """Retrieves relevant KG facts and formats as <kg> XML for the prompt."""
        # This is a placeholder. The logic here is complex:
        # - Analyze user_instruction to find mentioned entities/topics.
        # - Optionally analyze last_llm_response_xml (or its parsed data) for entities.
        # - Query Neo4j for facts (entities, relations, properties) related to these entities/topics.
        # - Format the query results into a structured <kg> XML tag.
        log_info("Retrieving placeholder KG context XML.")

        # Example: Retrieve a few random facts or specific nodes
        # Example query: MATCH (e:Entity) RETURN e LIMIT 5
        # Example result formatting: <kg><entity name="...">...</entity>...</kg>

        # For now, return an empty <kg> tag
        return "<kg></kg>"

    def build_llm_prompt(self, user_instruction: str) -> List[Dict[str, str]]:
        """Constructs the full list of messages for the LLM call."""
        if not self._system_prompt_template or not self._query_prompt_template:
            log_error("Prompt templates not loaded. Cannot build LLM prompt.")
            return []

        # 1. Get conversation history XML
        conversation_history_xml = self.get_conversation_history_xml() # Placeholder

        # 2. Get relevant KG context XML
        # TODO: Logic to decide *what* context to retrieve. Maybe based on entities in instruction?
        kg_context_xml = self.retrieve_kg_context_for_prompt(user_instruction) # Placeholder

        # 3. Format the current user prompt including history and context
        # This relies on the structure defined in your query_prompt_file.txt
        # Placeholders {user_instruction}, {conversation_history}, {kg_context}
        full_query_content = self._query_prompt_template.format(
            user_instruction=user_instruction,
            conversation_history=conversation_history_xml, # Full history XML
            kg_context=kg_context_xml # KG context XML
            # Add other placeholders if needed, e.g., current_date, user_persona etc.
        )

        # 4. Construct the messages list for the API call
        # API formats (like OpenAI/Groq/Google) typically use a list of message objects with roles.
        # The system prompt is one message, the formatted query content is another.
        messages = [
            {"role": "system", "content": self._system_prompt_template},
            {"role": "user", "content": full_query_content}
        ]

        log_info(f"Built LLM prompt with System ({len(self._system_prompt_template)} chars) and User ({len(full_query_content)} chars).")
        # log_info(f"Full prompt messages: {messages}") # Log full messages for debug

        return messages


    def call_llm_and_parse(self, user_instruction: str) -> Optional[LLMResponseParsed]:
        """Handles the full workflow: build prompt, call LLM, parse response."""
        log_info(f"Starting LLM call and parse workflow for instruction: '{user_instruction[:50]}...'")

        # 1. Build prompt (includes history and KG context placeholders)
        # The user_instruction is passed to build_llm_prompt to potentially influence context retrieval
        messages = self.build_llm_prompt(user_instruction)
        if not messages:
            log_error("Failed to build LLM prompt.")
            return None

        # 2. Call LLM API
        llm_response_json = self.call_llm_api(messages)
        if llm_response_json is None:
            log_error("LLM API call failed.")
            return None

        # 3. Parse XML response
        parsed_data = self.parse_llm_response_xml(llm_response_json)
        if parsed_data is None:
            log_error("Failed to parse LLM response XML.")
            # Handle parsing failure: UI should show raw text maybe?
            # Return a special object indicating parsing error but containing raw text/JSON?
            # For now, return None.
            return None

        # 4. Save conversation turn (placeholder)
        # This needs the full user prompt *sent* and the full LLM raw XML *received*.
        # Reconstructing the exact user prompt XML sent is a TODO in build_llm_prompt.
        # For now, just save the raw LLM response XML.
        # user_prompt_xml_sent = "..." # Need the actual XML sent as user message
        # self.save_conversation_turn(user_prompt_xml_sent, parsed_data.raw_xml) # TODO: Implement saving

        log_info("LLM call and parse workflow completed successfully.")
        return parsed_data


    def parse_llm_response_xml(self, llm_response_json: Dict[str, Any]) -> Optional[LLMResponseParsed]:
        """Parses the LLM response (expected to contain XML) into structured data."""
        log_info("Attempting to parse LLM response as XML...")

        if not self.config or 'format' not in self.config:
             log_error("Config or 'format' section not loaded. Cannot parse LLM response.")
             return None

        try:
            # 1. Extract text content from response JSON
            # This structure depends on the API (Groq/OpenAI vs Google)
            raw_xml_text = None
            try:
                if 'choices' in llm_response_json and len(llm_response_json['choices']) > 0:
                    # Groq/OpenAI structure
                    raw_xml_text = llm_response_json['choices'][0]['message'].get('content')
                elif 'candidates' in llm_response_json and len(llm_response_json['candidates']) > 0:
                    # Google Gemini structure
                    # Check if content[0].parts[0].text exists
                    parts = llm_response_json['candidates'][0].get('content', {}).get('parts', [])
                    if parts and len(parts) > 0:
                        raw_xml_text = parts[0].get('text')

                if not raw_xml_text:
                     log_error("Parsing failed: Could not extract text content from LLM response JSON.")
                     log_info(f"Raw LLM Response JSON: {json.dumps(llm_response_json, indent=2)}")
                     return None

            except (KeyError, TypeError) as e:
                log_error(f"Parsing failed: Unexpected LLM response JSON structure when extracting text. Error: {e}", exc_info=True)
                log_info(f"Raw LLM Response JSON: {json.dumps(llm_response_json, indent=2)}")
                return None

            log_info(f"Extracted text content from LLM response (first 500 chars):\n---\n{raw_xml_text[:500]}...\n---")

            # 2. Find the root XML tag defined in config and parse the XML
            # The LLM might include text before/after the XML, or generate incomplete XML.
            # Need to find the <response_root_tag> (e.g., <response>)
            format_config = self.config.get('format', {})
            response_root_tag = format_config.get('response_root_tag') # e.g., "response"

            if not response_root_tag:
                 log_error("Config missing 'format.response_root_tag'. Cannot parse XML.")
                 return None

            # Find the start and end of the root tag in the raw text
            start_marker = f"<{response_root_tag}>"
            end_marker = f"</{response_root_tag}>"
            start_index = raw_xml_text.find(start_marker)
            end_index = raw_xml_text.rfind(end_marker) # Use rfind for the last occurrence

            if start_index == -1 or end_index == -1 or end_index < start_index:
                 log_warning(f"Could not find complete root tag <{response_root_tag}>...<{response_root_tag}> in response.")
                 # Attempt to parse the whole text anyway, lxml recover=True might help
                 xml_content_to_parse = raw_xml_text
            else:
                 # Extract only the content within and including the root tags
                 xml_content_to_parse = raw_xml_text[start_index : end_index + len(end_marker)]
                 log_info(f"Extracted XML content based on root tag (first 500 chars):\n---\n{xml_content_to_parse[:500]}...\n---")


            # Use lxml's parser with recovery for robustness
            parser = etree.XMLParser(recover=True, encoding='utf-8') # recover=True to handle malformed XML
            try:
                # etree.fromstring expects bytes, ensure encoding
                root = etree.fromstring(xml_content_to_parse.encode('utf-8'), parser=parser)

                if root is None and parser.error_log:
                    log_error(f"lxml failed to parse XML, even with recovery. Errors: {parser.error_log}")
                    log_info(f"XML content that failed parsing:\n---\n{xml_content_to_parse}\n---")
                    return None
                elif parser.error_log:
                    log_warning(f"lxml parsed XML with recovery, but errors were present: {parser.error_log}")
                    log_info(f"XML content parsed with errors:\n---\n{xml_content_to_parse}\n---")

                log_info("XML parsed successfully using lxml.")

            except Exception as e:
                 log_error(f"Unexpected error during lxml parsing: {e}", exc_info=True)
                 log_info(f"XML content that caused error:\n---\n{xml_content_to_parse}\n---")
                 return None


            # 3. Extract Narrative Element, Entities, Relations, Queries from the parsed XML tree
            narrative_element = root.find('.//narrative') # Find narrative tag anywhere

            if narrative_element is None:
                 log_warning("Could not find <narrative> tag in parsed XML.")
                 # Continue parsing other elements if possible, but narrative will be missing.


            parsed_entities: List[ParsedEntity] = []
            parsed_relations: List[List[ParsedRelation]] = [] # Relations might be grouped by sentence/paragraph? Or just a flat list. Assume flat list for now.
            parsed_queries: List[ParsedQuery] = []

            # Define expected tags (can be from config if needed)
            ENTITY_TAG = 'entity'
            RELATION_TAG = 'relation'
            QUERY_TAG = 'query'

            # Recursive function to traverse the XML tree and extract data
            def extract_data_from_element(element: etree.Element):
                 extracted_entities: List[ParsedEntity] = []
                 extracted_relations: List[ParsedRelation] = []
                 extracted_queries: List[ParsedQuery] = []

                 # Process the current element if it's one of the tags we care about
                 if element.tag == ENTITY_TAG:
                     try:
                         entity_id = element.get('id')
                         # Get text content *only within this tag* (excluding children's text)
                         text_span = etree.tostring(element, encoding='unicode', method='text').strip()

                         if not entity_id:
                              log_warning(f"<{ENTITY_TAG}> tag found without 'id' attribute. Skipping extraction for this tag: {etree.tostring(element, encoding='unicode', with_tail=False)[:200]}...")
                         else:
                             extracted_entities.append(ParsedEntity(
                                 xml_id=entity_id,
                                 text_span=text_span,
                                 canonical=element.get('canonical'),
                                 entity_type=element.get('type'), # Or get('entityType') depending on your schema
                                 status=element.get('status', 'Pending'),
                                 # Parse provenance string (e.g., "LLM,User") into a list
                                 provenance=[p.strip() for p in element.get('provenance', 'LLM_XML_Generated').split(',') if p.strip()],
                                 attributes=dict(element.attrib) # Store all attributes
                             ))
                     except Exception as e:
                         log_error(f"Error parsing <{ENTITY_TAG}> tag: {etree.tostring(element, encoding='unicode', with_tail=False)[:200]}... Error: {e}", exc_info=True)

                 elif element.tag == RELATION_TAG:
                     try:
                         relation_id = element.get('id', f"rel_{len(parsed_relations) + len(extracted_relations)}") # Generate ID if missing
                         text_span = etree.tostring(element, encoding='unicode', method='text').strip()
                         relation_type = element.get('type') # Or get('relationType')
                         subj_id = element.get('subj')
                         obj_id = element.get('obj')

                         if not relation_type or not subj_id or not obj_id:
                              log_warning(f"<{RELATION_TAG}> tag found without essential attributes (type, subj, obj). Skipping extraction for this tag: {etree.tostring(element, encoding='unicode', with_tail=False)[:200]}...")
                         else:
                             extracted_relations.append(ParsedRelation(
                                 xml_id=relation_id,
                                 text_span=text_span,
                                 relation_type=relation_type,
                                 subj_id=subj_id,
                                 obj_id=obj_id,
                                 status=element.get('status', 'Pending'),
                                 provenance=[p.strip() for p in element.get('provenance', 'LLM_XML_Generated').split(',') if p.strip()],
                                 attributes=dict(element.attrib) # Store all attributes
                             ))
                     except Exception as e:
                         log_error(f"Error parsing <{RELATION_TAG}> tag: {etree.tostring(element, encoding='unicode', with_tail=False)[:200]}... Error: {e}", exc_info=True)

                 elif element.tag == QUERY_TAG:
                     try:
                         query_id = element.get('id', f"q_{len(parsed_queries) + len(extracted_queries)}") # Generate ID if missing
                         purpose = element.get('purpose', 'No purpose specified')
                         query_string = etree.tostring(element, encoding='unicode', method='text').strip()

                         if not query_string:
                             log_warning(f"<{QUERY_TAG}> tag found with empty content. Skipping extraction for this tag: {etree.tostring(element, encoding='unicode', with_tail=False)[:200]}...")
                         else:
                             extracted_queries.append(ParsedQuery(
                                 xml_id=query_id,
                                 purpose=purpose,
                                 query_string=query_string
                             ))
                     except Exception as e:
                         log_error(f"Error parsing <{QUERY_TAG}> tag: {etree.tostring(element, encoding='unicode', with_tail=False)[:200]}... Error: {e}", exc_info=True)

                 # Recursively call for children
                 for child in element:
                     child_entities, child_relations, child_queries = extract_data_from_element(child)
                     extracted_entities.extend(child_entities)
                     extracted_relations.extend(child_relations)
                     extracted_queries.extend(child_queries)

                 return extracted_entities, extracted_relations, extracted_queries

            # Start extraction from the root element
            all_extracted_entities, all_extracted_relations, all_extracted_queries = extract_data_from_element(root)

            # Store the extracted data
            parsed_entities = all_extracted_entities
            parsed_relations = all_extracted_relations
            parsed_queries = all_extracted_queries


            log_info(f"Parsed {len(parsed_entities)} entities, {len(parsed_relations)} relations, {len(parsed_queries)} queries from XML.")
            log_info(f"Parsed Entities: {[repr(e) for e in parsed_entities]}")
            log_info(f"Parsed Relations: {[repr(r) for r in parsed_relations]}")
            log_info(f"Parsed Queries: {[repr(q) for q in parsed_queries]}")


            return LLMResponseParsed(
                raw_xml=raw_xml_text, # Store the text *before* extracting root, for full raw log
                narrative_xml_element=narrative_element, # Pass the lxml element for narrative rendering
                entities=parsed_entities,
                relations=parsed_relations,
                queries=parsed_queries,
                raw_response_json=llm_response_json # Store original for full log
            )

        except etree.XMLSyntaxError as e:
             log_error(f"XML Syntax Error during lxml parsing: {e}", exc_info=True)
             log_info(f"XML content that failed parsing:\n---\n{xml_content_to_parse}\n---") # Log the content that caused error
             return None
        except Exception as e:
            log_error(f"An unexpected error occurred during XML parsing: {e}", exc_info=True)
            return None


    def update_knowledge_graph_from_parsed_data(self, parsed_data: LLMResponseParsed):
        """Updates Neo4j KG based on parsed data (as suggested by LLM XML).

        This is the initial update *before* user curation. Items are likely 'Pending'.
        """
        if not self.neo4j_driver or not self.connect_neo4j():
            log_error("Cannot update KG: Not connected to Neo4j.")
            return False

        log_info(f"Initiating KG update transaction from parsed LLM data...")
        log_info(f"Entities to process: {len(parsed_data.entities)}, Relations to process: {len(parsed_data.relations)}")


        try:
            with self.neo4j_driver.session() as session:

                # --- Create or Merge Entities ---
                # Use MERGE based on 'name' (canonical).
                # Set properties on CREATE and ON MATCH (for provenance, xml_ref, etc.)
                # Status from parsed data, defaults to 'Pending'. Provenance gets 'LLM_XML_Generated'.
                # Need to handle canonical name vs text_span.

                # Ensure entity data is prepared
                entities_data = []
                for entity in parsed_data.entities:
                    # Simple relation type sanitization (replace non-alphanum/underscore with underscore)
                    # Apply sanitization to entity type and other relevant string attributes if needed
                    safe_entity_type = ''.join(c if c.isalnum() or c == '_' else '_' for c in (entity.entity_type or 'UnknownType'))

                    entities_data.append({
                        "xml_id": entity.xml_id,
                        "canonical_name": entity.canonical,
                        "text_span": entity.text_span, # Store original text
                        "entity_type": safe_entity_type,
                        "status": entity.status, # Use status from parsed data (default Pending)
                        "provenance": entity.provenance, # Use provenance from parsed data (default LLM_XML_Generated)
                        "attributes": entity.attributes # Store original attributes
                    })

                if entities_data:
                    log_info(f"Merging {len(entities_data)} entities...")
                    # Use UNWIND to process the list of entities in a single query
                    entity_merge_query = """
                    UNWIND $entities_data AS entity_data
                    MERGE (e:Entity {name: entity_data.canonical_name})
                    ON CREATE SET
                        e.xml_ref = entity_data.xml_id,
                        e.entityType = entity_data.entity_type,
                        e.status = entity_data.status,
                        e.provenance = entity_data.provenance, // Provenance is a list
                        e.attributes = entity_data.attributes,
                        e.text_spans = [entity_data.text_span] // Store text spans as a list
                    ON MATCH SET
                        // Add provenance source if not already present
                        e.provenance = apoc.coll.toSet(e.provenance + entity_data.provenance),
                        // Only update status if the incoming status is 'Pending' or if current is 'Pending'
                        // This prevents LLM overwriting a 'Canon' status
                        e.status = CASE
                                     WHEN e.status = 'Pending' THEN entity_data.status // If current is Pending, use incoming
                                     WHEN entity_data.status = 'Pending' THEN e.status // If incoming is Pending, keep current
                                     ELSE entity_data.status // Otherwise (e.g. incoming is Canon from curated XML), use incoming
                                   END,
                        e.attributes = apoc.map.merge(e.attributes, entity_data.attributes),
                        e.text_spans = apoc.coll.toSet(e.text_spans + [entity_data.text_span]) // Add text span if new
                    """
                    # Requires APOC library installed in Neo4j for apoc.coll.toSet and apoc.map.merge
                    # Make sure APOC is enabled in Neo4j Desktop settings.
                    try:
                         session.run(entity_merge_query, parameters={"entities_data": entities_data}).consume()
                         log_info("Entity merge completed.")
                    except Exception as e:
                         log_error(f"Cypher error merging entities: {e}", exc_info=True)
                         # Decide on error handling: abort or continue? Abort transaction might be safer.
                         raise # Re-raise to abort the transaction

                # --- Create or Merge Relations ---
                # Use MERGE based on source entity, target entity, and relation type.
                # Need to map xml_subj_id and xml_obj_id to Neo4j nodes (using canonical name).
                # Need to sanitize relation type for Cypher.
                # Set properties on CREATE and ON MATCH.

                # Need canonical names mapped from xml_ids within this dataset
                # Create a map from the processed entities data
                entity_id_to_canonical_map = {e['xml_id']: e['canonical_name'] for e in entities_data}
                # Also need canonical names for entities that might *already* exist in the DB
                # but weren't in this LLM response. This requires querying the DB first.
                # This adds complexity. For now, assume relations only connect entities present
                # in the *current* parsed_data for simplicity, or rely on Neo4j MERGE to find them.
                # Let's rely on MERGE finding existing entities by name, but we need to ensure
                # those entities *are* merged/created first (handled by entity merge above).

                relations_data = []
                for relation in parsed_data.relations:
                    subj_canonical_name = entity_id_to_canonical_map.get(relation.subj_id)
                    obj_canonical_name = entity_id_to_canonical_map.get(relation.obj_id)

                    # Skip relation if endpoints were not found/processed in this batch
                    if not subj_canonical_name or not obj_canonical_name:
                        log_warning(f"Skipping relation {relation.xml_id}: Could not find canonical name for subject XML ID {relation.subj_id} or object XML ID {relation.obj_id} in current parsed batch.")
                        continue

                    # Relation type sanitization
                    relation_type = relation.relation_type.replace(" ", "_").upper()
                    if not relation_type:
                         log_warning(f"Skipping relation {relation.xml_id}: Empty relation type.")
                         continue
                    # Very basic check, Cypher identifiers are complex
                    if not all(c.isalnum() or c == '_' for c in relation_type):
                         log_warning(f"Relation type '{relation.relation_type}' sanitized to '{relation_type}' might be unsafe Cypher.")


                    relations_data.append({
                        "xml_id": relation.xml_id,
                        "subj_name": subj_canonical_name,
                        "obj_name": obj_canonical_name,
                        "relation_type": relation_type, # Sanitized type
                        "text_span": relation.text_span,
                        "status": relation.status, # Default Pending
                        "provenance": relation.provenance, # Default LLM_XML_Generated
                        "attributes": relation.attributes
                    })

                if relations_data:
                    log_info(f"Merging {len(relations_data)} relations...")
                    # Use UNWIND to process the list of relations in a single query
                    # Need to find existing entities to connect them
                    relation_merge_query = """
                    UNWIND $relations_data AS relation_data
                    MATCH (a:Entity {name: relation_data.subj_name})
                    MATCH (b:Entity {name: relation_data.obj_name})
                    MERGE (a)-[r:DYNAMIC_REL_TYPE]->(b) // DYNAMIC_REL_TYPE is a placeholder, type is dynamic
                    ON CREATE SET
                        r.xml_ref = relation_data.xml_id,
                        r.status = relation_data.status,
                        r.provenance = relation_data.provenance, // Provenance is a list
                        r.attributes = relation_data.attributes,
                        r.text_spans = [relation_data.text_span] // Store text spans as a list
                    ON MATCH SET
                        r.provenance = apoc.coll.toSet(r.provenance + relation_data.provenance),
                         r.status = CASE
                                     WHEN r.status = 'Pending' THEN relation_data.status // If current is Pending, use incoming
                                     WHEN relation_data.status = 'Pending' THEN r.status // If incoming is Pending, keep current
                                     ELSE relation_data.status // Otherwise, use incoming
                                   END,
                        r.attributes = apoc.map.merge(r.attributes, relation_data.attributes),
                        r.text_spans = apoc.coll.toSet(r.text_spans + [relation_data.text_span]) // Add text span if new
                    """
                    # Cypher does not allow dynamic relationship types in MERGE/CREATE directly like :$type.
                    # This requires constructing the query string dynamically *per relation type*
                    # or using a procedure like apoc.create.relationship.
                    # For simplicity, let's build queries per relation type or use APOC.
                    # Using APOC is cleaner if available.
                    # Example using APOC (assuming APOC is installed and enabled):
                    relation_merge_apoc_query = """
                    UNWIND $relations_data AS relation_data
                    MATCH (a:Entity {name: relation_data.subj_name})
                    MATCH (b:Entity {name: relation_data.obj_name})
                    CALL apoc.merge.relationship(a, relation_data.relation_type, {}, {}, b, {}) YIELD rel
                    ON CREATE SET
                        rel.xml_ref = relation_data.xml_id,
                        rel.status = relation_data.status,
                        rel.provenance = relation_data.provenance,
                        rel.attributes = relation_data.attributes,
                        rel.text_spans = [relation_data.text_span]
                    ON MATCH SET
                        rel.provenance = apoc.coll.toSet(rel.provenance + relation_data.provenance),
                         rel.status = CASE
                                     WHEN rel.status = 'Pending' THEN relation_data.status
                                     WHEN relation_data.status = 'Pending' THEN rel.status
                                     ELSE relation_data.status
                                   END,
                        rel.attributes = apoc.map.merge(rel.attributes, relation_data.attributes),
                        rel.text_spans = apoc.coll.toSet(rel.text_spans + [relation_data.text_span])
                    RETURN rel
                    """

                    # Let's use the APOC version as it handles dynamic types cleanly.
                    # Need to ensure APOC is installed in Neo4j.
                    try:
                         session.run(relation_merge_apoc_query, parameters={"relations_data": relations_data}).consume()
                         log_info("Relation merge completed (using APOC).")
                    except Exception as e:
                         log_error(f"Cypher error merging relations (using APOC): {e}", exc_info=True)
                         log_error("Ensure APOC library is installed and enabled in your Neo4j instance.")
                         raise # Re-raise to abort the transaction

                # Commit the entire transaction (handled by 'with self.neo4j_driver.session() as session:')
                log_info("Knowledge Graph update transaction from parsed data committed.")
                return True

        except Exception as e:
            log_error(f"KG update transaction failed: {e}", exc_info=True)
            return False


    # TODO: Implement methods for UI to trigger specific actions after user curation
    # These methods would be called by the PySide6 UI after the user interacts
    # with the Quick Curation Panel and approves/edits data.
    # They will update the KG by changing `status`, `canonical_ref`, `provenance`, `attributes`
    # of existing nodes/relations in Neo4j, identified perhaps by their `xml_ref` or canonical name.

    def apply_curation_updates(self, curated_entities_data: List[Dict], curated_relations_data: List[Dict]):
        """Applies user-curated updates (status, canonical, etc.) to the KG."""
        log_info(f"Applying user curation updates: {len(curated_entities_data)} entities, {len(curated_relations_data)} relations.")
        if not self.neo4j_driver or not self.connect_neo4j():
             log_error("Cannot apply curation updates: Not connected to Neo4j.")
             return False

        # This requires a new Cypher transaction that looks up nodes/relations
        # by their canonical name or xml_ref and sets their properties
        # based on the curated data from the UI.

        try:
             with self.neo4j_driver.session() as session:
                 # Example: Update status and add 'User_Curated' provenance for entities
                 if curated_entities_data:
                     entity_curation_query = """
                     UNWIND $curated_entities AS entity_data
                     MATCH (e:Entity {name: entity_data.canonical_name})
                     SET
                         e.status = entity_data.status, // Set status approved by user
                         e.provenance = apoc.coll.toSet(e.provenance + ['User_Curated']), // Add User_Curated provenance
                         e.entityType = COALESCE(entity_data.entity_type, e.entityType), // Update type if provided
                         e.attributes = apoc.map.merge(e.attributes, entity_data.attributes) // Merge attributes
                     """
                     # Note: This assumes curated_entities_data contains canonical_name, status, etc.
                     # The UI needs to structure this data correctly.
                     session.run(entity_curation_query, parameters={"curated_entities": curated_entities_data}).consume()
                     log_info("Entity curation updates applied.")

                 # Example: Update status and add 'User_Curated' provenance for relations
                 if curated_relations_data:
                      # Needs subj/obj names and relation type to match the relation
                      relation_curation_query = """
                      UNWIND $curated_relations AS relation_data
                      MATCH (a:Entity {name: relation_data.subj_name})
                      MATCH (b:Entity {name: relation_data.obj_name})
                      MATCH (a)-[r]->(b)
                      WHERE type(r) = relation_data.relation_type // Match by sanitized type
                      SET
                         r.status = relation_data.status, // Set status approved by user
                         r.provenance = apoc.coll.toSet(r.provenance + ['User_Curated']), // Add User_Curated provenance
                         r.attributes = apoc.map.merge(r.attributes, relation_data.attributes) // Merge attributes
                      """
                      # Note: This assumes curated_relations_data contains subj_name, obj_name, relation_type, status, etc.
                      # Need to handle potential multiple relations of the same type between two nodes if that's possible.
                      # Using xml_ref might be more precise if stored and passed back from UI.
                      session.run(relation_curation_query, parameters={"curated_relations": curated_relations_data}).consume()
                      log_info("Relation curation updates applied.")

             log_info("User curation updates applied successfully.")
             return True
        except Exception as e:
             log_error(f"Error applying user curation updates: {e}", exc_info=True)
             return False


    # TODO: Implement methods for managing Glosario (Canonical/Alias) in KG via UI interaction
    # These methods will interact with the KG model for Glosario (e.g., :Canonical, :Alias labels, :HAS_ALIAS rels)
    # They might be called from the Quick Curation Panel when the user maps an alias or creates a new canonical.

    def suggest_canonical(self, alias_name: str) -> Optional[str]:
        """Suggests a canonical name from KG/Glosario for a given alias."""
        log_info(f"Suggesting canonical for alias: {alias_name}")
        if not self.neo4j_driver or not self.connect_neo4j():
             log_error("Cannot suggest canonical: Not connected to Neo4j.")
             return None
        # Example: Look for existing aliases that match or are similar
        # Example query: MATCH (:Alias {name: $alias_name})-[:HAS_ALIAS]->(c:Canonical) RETURN c.name LIMIT 1
        # Or search for similar names in Canonical entities

        # For now, a simple placeholder
        log_warning("Suggest canonical not fully implemented.")
        return None # Or return a list of suggestions


    def create_alias_mapping(self, alias_name: str, canonical_name: str):
        """Creates a new alias mapping in the KG Glosario."""
        log_info(f"Creating alias mapping: '{alias_name}' -> '{canonical_name}'")
        if not self.neo4j_driver or not self.connect_neo4j():
             log_error("Cannot create alias mapping: Not connected to Neo4j.")
             return False
        # Example Cypher:
        # MERGE (a:Alias {name: $alias_name}) SET a.provenance = apoc.coll.toSet(COALESCE(a.provenance, []) + ['User_Curated_Alias'])
        # MERGE (c:Canonical:Entity {name: $canonical_name}) // Ensure canonical exists and is marked as Canonical/Entity
        # MERGE (a)-[:HAS_CANONICAL]->(c) // Or [:HAS_ALIAS] from canonical to alias

        log_warning("Create alias mapping not fully implemented.")
        return False # Indicate failure

    # TODO: Implement method to execute LLM suggested query
    def execute_suggested_query(self, query_string: str) -> Optional[List[Dict]]:
         """Executes a user-selected query suggested by LLM."""
         log_info(f"Executing suggested query: {query_string}")
         # This is just a wrapper around run_cypher_query
         if not query_string.strip():
              log_warning("Attempted to execute empty query string.")
              return None

         # Decide if it should be a read or write query based on the query string
         # Simple check: if query starts with MATCH, it's likely a read. MERGE/CREATE/DELETE are writes.
         is_write_query = query_string.strip().upper().startswith(("CREATE", "MERGE", "SET", "DELETE", "REMOVE", "CALL ")) # Basic check

         if is_write_query:
             log_warning(f"Executing a potential write query: {query_string[:100]}...")
             # Maybe add a confirmation dialog in the UI before calling this for writes
             return self.run_cypher_transaction(query_string)
         else:
             return self.run_cypher_query(query_string)


# --- Ensure Neo4j connection is closed on script exit ---
# This handler is useful if running model.py as a standalone script for testing.
# In a PySide6 app, the app's closeEvent handler should explicitly call model.close_neo4j().
# Registering atexit here might interfere with PySide's event loop termination.
# It's generally safer to handle resource cleanup explicitly in the GUI app's shutdown.

# import atexit
# atexit.register(lambda: logging.info("atexit called, attempting cleanup.")) # Placeholder
# atexit.register(lambda: model_instance.close_neo4j() if 'model_instance' in locals() and model_instance else None) # Needs model instance to be global or accessible
# ^ This atexit approach is tricky with class instances. Explicit closing in app is better.


# Note: Direct calls to CurationModel methods should typically be done from the main PySide application
# loop or event handlers, not directly in this file's top level or a __main__ block,
# as this file is intended to be imported. The main application file (e.g., main.py)
# is responsible for creating a CurationModel instance and managing its lifecycle.