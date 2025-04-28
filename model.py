# model.py
import os
import sys
import json
import time
import logging
import datetime
import uuid  # For generating unique IDs if needed
from collections import deque
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import webbrowser  # Used for opening Neo4j browser

# External Libraries
from neo4j import GraphDatabase, basic_auth, Driver, exceptions as neo4j_exceptions
import requests
from lxml import etree  # Recommended for robust XML parsing
from dotenv import load_dotenv  # Requires python-dotenv
from groq import Groq  # Groq SDK
import google.genai as genai  # Google Gemini SDK

# --- Configuration and Setup ---

# Load environment variables from .env file
load_dotenv()

# --- Logging Setup ---
LOG_FILE = "history.log"  # TINS specified log file name
# Configure logging to both file and console as per TINS
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
# Silence some chatty loggers if necessary
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.INFO)  # Keep Groq info logs for now
logging.getLogger("google").setLevel(
    logging.INFO
)  # Keep Google info logs for now


# Custom Logging functions for clarity, as used in main.py stub
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
    def __init__(
        self,
        xml_id: str,
        text_span: str,
        canonical: Optional[str] = None,
        entity_type: Optional[str] = None,
        status: Optional[str] = "Pending",
        provenance: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.xml_id = xml_id  # Original ID from XML tag
        self.text_span = (
            text_span  # The exact text from the narrative tag content
        )
        self.canonical = (
            canonical
            if canonical is not None and canonical.strip()
            else text_span
        )  # Default canonical to text_span, handle empty string
        self.entity_type = entity_type
        self.status = (
            status if status in ["Pending", "Canon", "Ignored"] else "Pending"
        )  # Validate status
        self.provenance = (
            provenance if provenance is not None else ["LLM_XML_Generated"]
        )
        self.attributes = (
            attributes if attributes is not None else {}
        )  # Store any other XML attributes

    def to_dict(self):
        return self.__dict__


class ParsedRelation:
    def __init__(
        self,
        xml_id: str,
        text_span: str,
        relation_type: str,
        subj_id: str,
        obj_id: str,
        status: Optional[str] = "Pending",
        provenance: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.xml_id = (
            xml_id  # ID from XML tag (optional, auto-generated if not present)
        )
        self.text_span = (
            text_span  # The exact text from the narrative tag content
        )
        self.relation_type = (
            relation_type.replace(" ", "_").upper()
            if relation_type
            else "RELATED_TO"
        )  # Sanitize and provide default relation type
        self.subj_id = subj_id  # Refers to xml_id of ParsedEntity
        self.obj_id = obj_id  # Refers to xml_id of ParsedEntity
        self.status = (
            status if status in ["Pending", "Canon", "Ignored"] else "Pending"
        )  # Validate status
        self.provenance = (
            provenance if provenance is not None else ["LLM_XML_Generated"]
        )
        self.attributes = attributes if attributes is not None else {}

    def to_dict(self):
        return self.__dict__


class ParsedQuery:
    def __init__(self, xml_id: str, purpose: str, query_string: str):
        self.xml_id = (
            xml_id  # ID from XML tag (optional, auto-generated if not present)
        )
        self.purpose = purpose
        self.query_string = query_string

    def to_dict(self):
        return self.__dict__


class LLMResponseParsed:
    """Represents the fully parsed data from an LLM response."""

    def __init__(
        self,
        raw_xml: str,
        narrative_xml_element: etree.Element,
        entities: List[ParsedEntity],
        relations: List[ParsedRelation],
        queries: List[ParsedQuery],
        raw_response_json: Dict[str, Any],
    ):
        self.raw_xml = raw_xml
        self.narrative_xml_element = (
            narrative_xml_element  # Keep the lxml element for UI rendering
        )
        self.entities = entities
        self.relations = relations
        self.queries = queries
        self.raw_response_json = (
            raw_response_json  # Keep original json for full log
        )

    def to_dict(self):
        # Note: narrative_xml_element is not easily serializable to dict
        # raw_response_json can be large, might exclude in some UI contexts
        return {
            "raw_xml": self.raw_xml,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "queries": [q.to_dict() for q in self.queries],
            # "raw_response_json": self.raw_response_json # Exclude by default for cleaner view
        }


class ConversationTurn:
    """Represents a single turn in the conversation history."""

    def __init__(
        self,
        user_prompt_text: str,
        llm_response_raw_xml: str,
        timestamp: Optional[datetime.datetime] = None,
    ):
        self.user_prompt_text = user_prompt_text
        self.llm_response_raw_xml = llm_response_raw_xml
        self.timestamp = (
            timestamp if timestamp is not None else datetime.datetime.now()
        )

    def to_xml_element(self) -> etree.Element:
        """Converts the turn to an XML element for session saving."""
        turn_elem = etree.Element("turn", timestamp=self.timestamp.isoformat())
        user_prompt_elem = etree.SubElement(turn_elem, "user_prompt")
        user_prompt_elem.text = self.user_prompt_text
        llm_response_elem = etree.SubElement(turn_elem, "llm_response_raw_xml")
        llm_response_elem.text = self.llm_response_raw_xml
        return turn_elem

    @staticmethod
    def from_xml_element(
        turn_elem: etree.Element,
    ) -> Optional["ConversationTurn"]:
        """Creates a ConversationTurn from an XML element loaded from a session file."""
        timestamp_str = turn_elem.get("timestamp")
        timestamp = None
        if timestamp_str:
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
            except ValueError:
                log_warning(
                    f"Could not parse timestamp '{timestamp_str}' in history."
                )

        user_prompt_elem = turn_elem.find("user_prompt")
        llm_response_elem = turn_elem.find("llm_response_raw_xml")

        if user_prompt_elem is not None and llm_response_elem is not None:
            user_prompt_text = "".join(
                user_prompt_elem.xpath(".//text()")
            )  # Get all text content
            llm_response_raw_xml = "".join(
                llm_response_elem.xpath(".//text()")
            )  # Get all text content
            return ConversationTurn(
                user_prompt_text, llm_response_raw_xml, timestamp
            )
        else:
            log_warning(
                f"Skipping malformed turn element in history: {etree.tostring(turn_elem, encoding='unicode')}"
            )
            return None


# --- Model Class ---


class CurationModel:
    def __init__(self, config_path: str = "approach/default.json"):
        self.config_path = Path(config_path)
        self.config: Optional[Dict[str, Any]] = None
        self.neo4j_driver: Optional[Driver] = None
        self._google_api_call_timestamps = deque(
            maxlen=5
        )  # For Google rate limiting (5 calls/min)
        self._conversation_turns: List[ConversationTurn] = (
            []
        )  # Structured history
        self._current_session_file: Optional[Path] = (
            None  # Track the current session file
        )
        self._emulator_response_index = 1  # Start with response_1.xml
        self._glossary_xml_content: str = ""  # Store raw glossary XML content

        log_info("Initializing CurationModel...")
        self._load_config()
        self._load_environment_variables()  # Ensure env vars are accessed
        # Neo4j driver is initialized lazily or on first connect test
        self._load_prompt_templates()  # Load initial prompts based on config

        # Configure Google Generative AI SDK (requires API key in env)
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            try:
                genai.configure(api_key=google_api_key)
                log_info("Google Generative AI SDK configured.")
            except Exception as e:
                log_error(f"Failed to configure Google Generative AI SDK: {e}")
        else:
            log_warning(
                "GOOGLE_API_KEY not found. Google LLM provider will be disabled."
            )

        log_info("CurationModel initialized.")

    def _load_environment_variables(self):
        """Ensure required environment variables are loaded/available."""
        # Access using os.getenv directly in methods that need them (e.g., connect_neo4j, call_llm_api)
        pass  # load_dotenv() call at the top handles the actual loading

    def _load_config(self):
        """Loads configuration from the specified JSON file."""
        log_info(f"Loading configuration from {self.config_path}")
        try:
            if not self.config_path.exists():
                log_warning(
                    f"Config file not found at {self.config_path}. Using default structure if possible."
                )
                # Optionally create a default config file here if it doesn't exist, similar to main.py
                # For robustness, let's create a basic default if missing
                default_config_data = {
                    "api": {
                        "api_name": "emulator",  # Default to emulator
                        "model_name": "default-model",
                        "temperature": 0.7,
                        "max_tokens": 1024,
                    },
                    "format": {
                        "separator": "###END_XML_RESPONSE###",
                        "response_root_tag": "response",
                    },
                    "prompts": {
                        "system_prompt_file": "prompts/default_system.txt",
                        "query_prompt_file": "prompts/default_query.txt",
                    },
                }
                try:
                    self.config_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.config_path, "w", encoding="utf-8") as f:
                        json.dump(default_config_data, f, indent=2)
                    log_info(
                        f"Created default config file: {self.config_path}"
                    )
                except Exception as e:
                    log_error(f"Failed to create default config file: {e}")

            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            log_info("Configuration loaded successfully.")
            # Basic validation (check for essential keys)
            if not all(k in self.config for k in ["api", "format", "prompts"]):
                log_warning(
                    "Config file missing essential top-level keys ('api', 'format', 'prompts')."
                )
            # TODO: More detailed config validation based on expected structure
        except json.JSONDecodeError as e:
            log_error(
                f"Error decoding JSON configuration file {self.config_path}: {e}"
            )
            self.config = None  # Ensure config is None on error
        except Exception as e:
            log_error(
                f"An unexpected error occurred loading config {self.config_path}: {e}"
            )
            self.config = None  # Ensure config is None on error

    def _load_prompt_templates(self):
        """Loads prompt content from text files specified in config."""
        self._system_prompt_template = ""  # Clear previous
        self._query_prompt_template = ""  # Clear previous

        if not self.config or "prompts" not in self.config:
            log_warning(
                "Config or 'prompts' section not loaded. Cannot load prompt templates."
            )
            return

        try:
            system_file_path = self.config["prompts"].get("system_prompt_file")
            query_file_path = self.config["prompts"].get("query_prompt_file")

            if system_file_path:
                abs_system_path = Path(system_file_path)
                if not abs_system_path.is_absolute():
                    # Assume relative to current working directory or a known prompts dir
                    # Using PROMPTS_DIR from main.py context is better, but model should be self-contained
                    # Let's assume paths are relative to the script location or are absolute
                    # For simplicity, let's assume relative paths are from project root where script is run
                    abs_system_path = Path(
                        system_file_path
                    )  # Assuming relative path works from execution dir

                if abs_system_path.exists():
                    log_info(f"Loading system prompt from {abs_system_path}")
                    with open(abs_system_path, "r", encoding="utf-8") as f:
                        self._system_prompt_template = f.read()
                else:
                    log_warning(
                        f"System prompt file not found: {abs_system_path}"
                    )

            else:
                log_warning("'system_prompt_file' not specified in config.")

            if query_file_path:
                abs_query_path = Path(query_file_path)
                if not abs_query_path.is_absolute():
                    abs_query_path = Path(
                        query_file_path
                    )  # Assuming relative path works

                if abs_query_path.exists():
                    log_info(f"Loading query prompt from {abs_query_path}")
                    with open(abs_query_path, "r", encoding="utf-8") as f:
                        self._query_prompt_template = f.read()
                else:
                    log_warning(
                        f"Query prompt file not found: {abs_query_path}"
                    )

            else:
                log_warning("'query_prompt_file' not specified in config.")

        except Exception as e:
            log_error(
                f"An unexpected error occurred loading prompt templates: {e}"
            )

    def connect_neo4j(self) -> bool:
        """Establishes connection to Neo4j."""
        if self.neo4j_driver:
            try:
                self.neo4j_driver.verify_connectivity()
                # log_info("Already connected to Neo4j.") # Too chatty on recurring calls
                return True
            except Exception:
                log_warning(
                    "Existing Neo4j driver is not connected. Re-initializing."
                )
                self.close_neo4j()  # Close the old one

        log_info(f"Connecting to Neo4j...")  # Avoid logging sensitive URI/user
        try:
            uri = os.getenv("NEO4J_URI")
            user = os.getenv("NEO4J_USER")
            password = os.getenv("NEO4J_PASSWORD")
            if not uri or not user or not password:
                log_error(
                    "Neo4j credentials (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) missing in environment variables (.env file?)."
                )
                return False

            self.neo4j_driver = GraphDatabase.driver(
                uri, auth=basic_auth(user, password)
            )
            self.neo4j_driver.verify_connectivity()  # Check if connection is successful
            log_info("Neo4j connection successful.")
            return True
        except Exception as e:
            log_error(
                f"Failed to connect to Neo4j at {uri}: {e}", exc_info=True
            )
            self.neo4j_driver = None
            return False

    def close_neo4j(self):
        """Closes the Neo4j connection."""
        if self.neo4j_driver:
            log_info("Closing Neo4j connection...")
            try:
                self.neo4j_driver.close()
                log_info("Neo4j connection closed.")
            except Exception as e:
                log_error(f"Error closing Neo4j driver: {e}")
            self.neo4j_driver = None
    def _open_neo4j_browser(self):
        """Opens Neo4j browser using the configured URI."""
        uri = os.getenv("NEO4J_URI", "http://localhost:7474")
        # Strip authentication and path components if present, keep base URL
        base_url = uri.split("://")[1].split("@")[-1].split("/")[0]
        browser_url = f"http://{base_url}:7474"
        webbrowser.open(browser_url)

    def run_cypher_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ):
        """Runs a Cypher read query and returns results."""
        # log_info(f"Running Cypher query: {query} with params: {parameters}") # Too chatty
        if (
            not self.neo4j_driver or not self.connect_neo4j()
        ):  # Ensure connection is active
            # log_error("Cannot run Cypher query: Not connected to Neo4j.") # Too chatty
            return None
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query, parameters)
                return result.data()  # Return list of dictionaries
        except Exception as e:
            log_error(
                f"Error running Cypher query '{query[:100]}...': {e}",
                exc_info=True,
            )
            return None

    def run_cypher_transaction(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ):
        """Runs a Cypher write transaction and returns summary."""
        # log_info(f"Running Cypher transaction: {query} with params: {parameters}") # Too chatty
        if (
            not self.neo4j_driver or not self.connect_neo4j()
        ):  # Ensure connection is active
            # log_error("Cannot run Cypher transaction: Not connected to Neo4j.") # Too chatty
            return None
        try:
            with self.neo4j_driver.session() as session:
                # Use execute_write for explicit transaction boundary
                summary = session.execute_write(
                    lambda tx: tx.run(query, parameters).consume()
                )
                # log_info(f"Transaction successful. Summary: {summary.counters}") # Too chatty
                return summary
        except Exception as e:
            log_error(
                f"Error running Cypher transaction '{query[:100]}...': {e}",
                exc_info=True,
            )
            return None

    def build_llm_prompt(
        self,
        user_instruction: str,
        conversation_history_xml: str,
        kg_context_xml: str,
        glossary_xml_content: str,
    ) -> List[Dict[str, str]]:
        """Constructs the full prompt messages list for the LLM."""
        if not self._system_prompt_template or not self._query_prompt_template:
            log_error("Prompt templates not loaded. Cannot build LLM prompt.")
            return []

        # --- Construct the XML for the prompt ---
        # Based on TINS, the full prompt structure includes history, KG context, and glossary context.
        # The conversation_history_xml is already formatted as <conversation_history>...</conversation_history>
        # The kg_context_xml is already formatted as <kg_context>...</kg_context>
        # The glossary_xml_content is the raw content of the glossary file.

        # Build the current turn's user prompt content
        current_user_prompt_content = f"<user_instruction>{etree.CDATA(user_instruction)}</user_instruction>"  # Use CDATA for raw text

        # Add KG context if available
        if kg_context_xml:
            current_user_prompt_content += f"\n{kg_context_xml}"

        # Add Glossary context if available
        if glossary_xml_content:
            # Wrap raw glossary content in a tag for LLM
            # Use CDATA to ensure internal XML is treated as text by prompt parser
            current_user_prompt_content += f"\n<glossary_context>{etree.CDATA(glossary_xml_content)}</glossary_context>"

        # Inject history and current user content into the query template
        # The query template is expected to have placeholders like {conversation_history}, {current_user_content}
        # Let's use a simpler approach assuming the query template *is* the structure around these parts
        # and re-format the template to be the full XML structure
        full_prompt_xml_content = self._query_prompt_template.format(
            conversation_history=conversation_history_xml,
            current_user_content=current_user_prompt_content,  # Include user instruction, kg, glossary here
        )
        # Need to decide if query template contains the root <session> or just the content for <current_turn>
        # Let's assume the query template is just the core instruction/context part and build the full XML here

        # Simpler approach for common API formats: Send system message + user message.
        # Combine all context XML into the user message.
        user_message_content = f"""
<prompt>
  {conversation_history_xml}
  <current_turn>
    {current_user_prompt_content}
  </current_turn>
</prompt>
"""
        # Clean up excessive whitespace that formatting might add
        user_message_content = "\n".join(
            [
                line.strip()
                for line in user_message_content.splitlines()
                if line.strip()
            ]
        )

        messages = [
            {"role": "system", "content": self._system_prompt_template},
            {"role": "user", "content": user_message_content},
        ]

        log_info(
            f"Built LLM Prompt (user message content):\n---\n{user_message_content}\n---"
        )

        return messages

    def call_llm_api(
        self, messages: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Calls the configured LLM API or emulator."""
        if not self.config or "api" not in self.config:
            log_error("LLM API configuration not loaded.")
            return None

        try:
            api_config = self.config.get("api", {})
            api_name = api_config.get("api_name", "").lower()
            model_name = api_config.get("model_name")
            # Extract model parameters, excluding known non-parameter keys
            model_params = {
                k: v
                for k, v in api_config.items()
                if k not in ["api_name", "model_name"]
            }

            log_info(f"Calling LLM API: {api_name} with model {model_name}")

            if api_name == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    log_error(
                        "GROQ_API_KEY not found in environment variables."
                    )
                    return None
                if not model_name:
                    model_name = "llama3-70b-8192"  # Default Groq model

                try:
                    # Use Groq SDK
                    client = Groq(api_key=api_key)
                    response = client.chat.completions.create(
                        messages=messages,
                        model=model_name,
                        **model_params,  # Include model parameters from config
                    )
                    response_json = (
                        response.model_dump()
                    )  # Convert response object to dict

                except Exception as e:
                    log_error(f"Error calling Groq API: {e}", exc_info=True)
                    return None

            elif api_name == "google":
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    log_error(
                        "GOOGLE_API_KEY not found in environment variables."
                    )
                    return None
                if not model_name:
                    model_name = "gemini-pro"  # Default Google model

                # --- Google Rate Limiting Logic (5 calls per minute) ---
                current_time = time.time()
                # Remove timestamps older than 60 seconds
                while (
                    self._google_api_call_timestamps
                    and self._google_api_call_timestamps[0]
                    <= current_time - 60
                ):
                    self._google_api_call_timestamps.popleft()

                # Check if we've made 5 calls in the last 60 seconds
                if len(self._google_api_call_timestamps) >= 5:
                    time_to_wait = (
                        60
                        - (current_time - self._google_api_call_timestamps[0])
                        + 1
                    )  # Wait a bit extra
                    log_warning(
                        f"Google API internal rate limit reached (5 calls/min). Waiting {time_to_wait:.2f} seconds..."
                    )
                    time.sleep(time_to_wait)
                    current_time = (
                        time.time()
                    )  # Update current time after waiting
                    # Re-check and remove old timestamps after waiting
                    while (
                        self._google_api_call_timestamps
                        and self._google_api_call_timestamps[0]
                        <= current_time - 60
                    ):
                        self._google_api_call_timestamps.popleft()

                # Adapt messages format for Google
                # Google expects a list of Contents, each with a list of Parts
                # Simplistic mapping: combine system/user messages into parts within a single content block
                content_parts = []
                for msg in messages:
                    # Google roles can be 'user' or 'model'. Map 'system' to 'user' for simplicity here.
                    role = (
                        "user"
                        if msg.get("role") == "system"
                        else msg.get("role", "user")
                    )
                    if "content" in msg:
                        content_parts.append(
                            {"text": msg["content"]}
                        )  # Assumes text content

                google_model_params = {}
                # Map generic model_params keys to Google's specific names
                if "temperature" in model_params:
                    google_model_params["temperature"] = model_params[
                        "temperature"
                    ]
                if "max_tokens" in model_params:
                    google_model_params["maxOutputTokens"] = model_params[
                        "max_tokens"
                    ]
                # TODO: Map other potential params like top_p etc. based on Google API spec and add safe settings

                try:
                    # Use Google GenAI SDK
                    model = genai.GenerativeModel(model_name=model_name)
                    response = model.generate_content(
                        contents=[
                            {"parts": content_parts}
                        ],  # Structure as list of contents with parts
                        generation_config=google_model_params,  # Pass model params
                    )
                    # Access text via response.text
                    # Access raw response details via response.candidates, etc.
                    # Create a dict structure that parse_llm_response_xml can handle
                    response_json = {
                        "candidates": [
                            {"content": {"parts": [{"text": response.text}]}}
                        ]
                    }

                    self._google_api_call_timestamps.append(
                        time.time()
                    )  # Log timestamp for rate limiting

                except genai.exceptions.BlockedPromptException as e:
                    log_error(f"Google API Blocked Prompt: {e}")
                    return None  # Indicate failure
                except genai.exceptions.ResourceExhausted as e:
                    # Specific Google quota error
                    log_error(f"Google API Quota Exceeded: {e}")
                    # TODO: Signal to UI for message box
                    return None  # Indicate failure
                except Exception as e:
                    log_error(f"Error calling Google API: {e}", exc_info=True)
                    return None

            elif api_name == "emulator":
                emulator_responses_dir = Path("emulator_responses")
                emulator_responses_dir.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure dir exists

                # Find the next response file
                file_path = (
                    emulator_responses_dir
                    / f"response_{self._emulator_response_index}.xml"
                )

                if not file_path.exists():
                    log_warning(
                        f"Emulator response file not found: {file_path}. Resetting emulator index."
                    )
                    self._emulator_response_index = 1  # Reset index
                    file_path = (
                        emulator_responses_dir
                        / f"response_{self._emulator_response_index}.xml"
                    )
                    if not file_path.exists():
                        log_error(
                            f"Emulator response file not found even after resetting index: {file_path}. Cannot simulate response."
                        )
                        return None

                log_info(f"Reading emulator response from: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_xml_text = f.read()

                    # Create a mock response dictionary that mimics a real API response structure
                    # This allows parse_llm_response_xml to process it consistently.
                    # Mimic OpenAI/Groq structure for simplicity
                    response_json = {
                        "choices": [
                            {
                                "message": {
                                    "content": raw_xml_text,
                                    "role": "assistant",  # Or 'model'
                                },
                                "finish_reason": "stop",
                                "index": 0,
                            }
                        ],
                        "created": int(time.time()),
                        "model": model_name,
                        "object": "chat.completion",
                        "system_fingerprint": "emulator",
                        "usage": {
                            "completion_tokens": len(
                                raw_xml_text.split()
                            ),  # Mock token count
                            "prompt_tokens": sum(
                                len(msg["content"].split()) for msg in messages
                            ),
                            "total_tokens": sum(
                                len(msg["content"].split()) for msg in messages
                            )
                            + len(raw_xml_text.split()),
                        },
                    }

                    self._emulator_response_index += (
                        1  # Increment for the next call
                    )

                except Exception as e:
                    log_error(
                        f"Error reading emulator response file {file_path}: {e}",
                        exc_info=True,
                    )
                    return None

            else:
                log_error(
                    f"Unknown API name specified in config: {api_name}. Supported: groq, google, emulator."
                )
                # TODO: Signal to UI for message box
                return None

            # Log and return the raw response JSON (or mock JSON for emulator)
            # log_info(f"Raw LLM API Response (or mock): {json.dumps(response_json, indent=2)}") # Too verbose
            log_info(f"LLM API call ({api_name}) successful.")
            return response_json

        except Exception as e:
            log_error(
                f"An unexpected error occurred during API call setup or execution: {e}",
                exc_info=True,
            )
            return None

    def parse_llm_response_xml(
        self, llm_response_json: Dict[str, Any]
    ) -> Optional[LLMResponseParsed]:
        """Parses the LLM response (expected to contain XML) into structured data."""
        log_info("Attempting to parse LLM response as XML...")

        if not llm_response_json:
            log_error("Parsing failed: Empty LLM response JSON.")
            return None

        # Extract text content from response JSON (structure depends on the API)
        raw_xml_text = ""
        try:
            # Try different common API response structures
            # Groq/OpenAI structure
            if (
                "choices" in llm_response_json
                and len(llm_response_json["choices"]) > 0
            ):
                raw_xml_text = llm_response_json["choices"][0]["message"][
                    "content"
                ]
            # Google Gemini structure (from mock or SDK)
            elif (
                "candidates" in llm_response_json
                and len(llm_response_json["candidates"]) > 0
            ):
                # Need to handle candidates[0].content.parts structure
                content = llm_response_json["candidates"][0].get("content")
                if (
                    content
                    and "parts" in content
                    and len(content["parts"]) > 0
                ):
                    # Concatenate text from all parts
                    raw_xml_text = "".join(
                        part.get("text", "") for part in content["parts"]
                    )
            # Add other potential API structures here if needed

            if not raw_xml_text:
                log_error(
                    "Parsing failed: Could not find text content in LLM response JSON."
                )
                log_info(
                    f"Raw LLM Response JSON: {json.dumps(llm_response_json, indent=2)}"
                )
                return None
        except (KeyError, TypeError) as e:
            log_error(
                f"Parsing failed: Unexpected LLM response JSON structure. Error: {e}",
                exc_info=True,
            )
            log_info(
                f"Raw LLM Response JSON: {json.dumps(llm_response_json, indent=2)}"
            )
            return None

        log_info(
            f"Raw XML Text received for parsing:\n---\n{raw_xml_text[:500]}...\n---"
        )  # Log first 500 chars

        # --- XML Parsing using lxml ---
        root_element = None
        narrative_element = None
        parsed_entities: List[ParsedEntity] = []
        parsed_relations: List[List[ParsedRelation]] = (
            []
        )  # Store relations per root for entity mapping
        parsed_queries: List[ParsedQuery] = []

        xml_string_to_parse = raw_xml_text.strip()

        # Try to find the root tag specified in config
        root_tag_name = self.config.get("format", {}).get("response_root_tag")
        if not root_tag_name:
            log_warning(
                "Response root tag name not specified in config. Attempting generic XML parse."
            )

        roots = []
        # Use a robust parser with recover=True
        parser = etree.XMLParser(recover=True)
        try:
            # Try parsing the whole string first
            root = etree.fromstring(
                xml_string_to_parse.encode("utf-8"), parser=parser
            )
            if root is not None:
                roots.append(root)
                log_info("Parsed XML successfully with etree.fromstring.")
            elif parser.error_log:
                log_warning(
                    f"etree.fromstring encountered errors: {parser.error_log}"
                )

        except etree.XMLSyntaxError as e:
            log_error(
                f"etree.fromstring XML Syntax Error: {e}. Attempting multi-root or wrapped parse.",
                exc_info=True,
            )
        except Exception as e:
            log_error(
                f"Unexpected error during initial XML parse: {e}",
                exc_info=True,
            )

        # If initial parse failed or multiple roots might exist, try finding roots explicitly
        if not roots:
            try:
                # Wrap in a dummy root to handle multiple top-level elements or text before root
                wrapped_xml = f"<dummy_root>{xml_string_to_parse}</dummy_root>"
                dummy_root = etree.fromstring(
                    wrapped_xml.encode("utf-8"), parser=parser
                )
                if dummy_root is not None:
                    # Find potential roots within the dummy root
                    if root_tag_name:
                        roots = dummy_root.findall(f".//{root_tag_name}")
                        if not roots:
                            log_warning(
                                f"Could not find expected root tag <{root_tag_name}> inside dummy root. Searching for all elements."
                            )
                            # Fallback: collect all direct children of dummy_root that are elements
                            roots = [
                                elem
                                for elem in dummy_root
                                if isinstance(elem.tag, str)
                            ]  # Ensure it's an element
                        else:
                            log_info(
                                f"Found {len(roots)} root elements with tag <{root_tag_name}> inside dummy root."
                            )
                    else:
                        # No root tag name specified, just take elements under dummy root
                        roots = [
                            elem
                            for elem in dummy_root
                            if isinstance(elem.tag, str)
                        ]
                        log_info(
                            f"No root tag name specified. Found {len(roots)} potential root elements under dummy root."
                        )

                if parser.error_log:
                    log_warning(
                        f"XML parsing errors encountered during wrapped parse: {parser.error_log}"
                    )

            except Exception as e:
                log_error(
                    f"An error occurred during wrapped XML parsing attempt: {e}",
                    exc_info=True,
                )

        if not roots:
            log_error(
                "Parsing failed: Could not find any potential root XML element in the response."
            )
            return None

        # Assume the first root found is the primary one, or process all roots?
        # TINS implies a single primary response XML. Let's process the first root found.
        root_element = roots[0]
        log_info(f"Processing primary root element: <{root_element.tag}>")

        # Extract Narrative Element (assuming it's the main part we want to render)
        # Find narrative tag anywhere within the primary root
        narrative_element = root_element.find(".//narrative")

        if narrative_element is None:
            log_warning("Could not find <narrative> tag in LLM XML response.")
            # Create an empty narrative element so UI doesn't fail
            narrative_element = etree.Element("narrative")

        # Extract Entities, Relations, Queries from all found roots/elements
        # Collect entities globally first to build mapping for relations
        entity_map: Dict[str, ParsedEntity] = {}
        all_parsed_relations: List[ParsedRelation] = []
        all_parsed_queries: List[ParsedQuery] = []

        # Iterate through all elements in all found roots to be comprehensive
        for current_root in roots:
            # Find all entity tags within this root
            for entity_elem in current_root.findall(".//entity"):
                try:
                    entity_id = entity_elem.get("id")
                    text_span = "".join(
                        entity_elem.xpath(".//text()")
                    ).strip()  # Get all text content and strip
                    # Attributes exclude 'id' and 'canonical' which are handled separately
                    attributes = dict(entity_elem.attrib)
                    attributes.pop("id", None)
                    attributes.pop("canonical", None)
                    attributes.pop("type", None)
                    attributes.pop("status", None)
                    attributes.pop("provenance", None)

                    if not entity_id:
                        # Generate a unique ID if missing
                        entity_id = f"gen_ent_{uuid.uuid4().hex[:8]}"
                        log_warning(
                            f"Entity tag found without 'id' attribute. Generating ID: {entity_id}. Tag: {etree.tostring(entity_elem, encoding='unicode')[:200]}..."
                        )

                    # Ensure canonical is set
                    canonical_name = entity_elem.get("canonical")
                    if not canonical_name or not canonical_name.strip():
                        canonical_name = (
                            text_span
                            if text_span
                            else f"UnknownEntity_{entity_id}"
                        )
                        log_warning(
                            f"Entity tag {entity_id} has no canonical name. Using text span '{canonical_name}' or generated."
                        )

                    parsed_entity = ParsedEntity(
                        xml_id=entity_id,
                        text_span=text_span,
                        canonical=canonical_name,
                        entity_type=entity_elem.get("type"),
                        status=entity_elem.get(
                            "status", "Pending"
                        ),  # Default status
                        provenance=entity_elem.get(
                            "provenance", "LLM_XML_Generated"
                        ).split(
                            ","
                        ),  # Assume comma-separated
                        attributes=attributes,  # Store all other attributes
                    )
                    parsed_entities.append(parsed_entity)
                    # Store entity by xml_id for relation mapping
                    entity_map[entity_id] = (
                        parsed_entity  # Map ID to the ParsedEntity object
                    )

                except Exception as e:
                    log_error(
                        f"Error parsing <entity> tag: {etree.tostring(entity_elem, encoding='unicode')[:200]}.... Error: {e}"
                    )

        # Now parse relations, using the entity_map
        for current_root in roots:
            for relation_elem in current_root.findall(".//relation"):
                try:
                    relation_id = relation_elem.get(
                        "id", f"gen_rel_{uuid.uuid4().hex[:8]}"
                    )  # Generate ID if missing
                    text_span = "".join(
                        relation_elem.xpath(".//text()")
                    ).strip()
                    relation_type_str = relation_elem.get("type")
                    subj_id = relation_elem.get("subj")
                    obj_id = relation_elem.get("obj")

                    attributes = dict(relation_elem.attrib)
                    attributes.pop("id", None)
                    attributes.pop("type", None)
                    attributes.pop("subj", None)
                    attributes.pop("obj", None)
                    attributes.pop("status", None)
                    attributes.pop("provenance", None)

                    if not relation_type_str or not subj_id or not obj_id:
                        log_warning(
                            f"Relation tag found without essential attributes (type, subj, obj). Skipping: {etree.tostring(relation_elem, encoding='unicode')[:200]}..."
                        )
                        continue

                    # Check if subject and object IDs map to parsed entities
                    if subj_id not in entity_map or obj_id not in entity_map:
                        log_warning(
                            f"Relation tag {relation_id} references unknown entity IDs ({subj_id}, {obj_id}). Skipping: {etree.tostring(relation_elem, encoding='unicode')[:200]}..."
                        )
                        continue

                    all_parsed_relations.append(
                        ParsedRelation(
                            xml_id=relation_id,
                            text_span=text_span,
                            relation_type=relation_type_str,  # Relation type sanitization done in __init__
                            subj_id=subj_id,  # xml_id ref
                            obj_id=obj_id,  # xml_id ref
                            status=relation_elem.get("status", "Pending"),
                            provenance=relation_elem.get(
                                "provenance", "LLM_XML_Generated"
                            ).split(","),
                            attributes=attributes,
                        )
                    )
                except Exception as e:
                    log_error(
                        f"Error parsing <relation> tag: {etree.tostring(relation_elem, encoding='unicode')[:200]}.... Error: {e}"
                    )

        # Find all query tags
        for current_root in roots:
            for query_elem in current_root.findall(".//query"):
                try:
                    query_id = query_elem.get(
                        "id", f"gen_q_{uuid.uuid4().hex[:8]}"
                    )  # Generate ID if missing
                    purpose = query_elem.get("purpose", "No purpose specified")
                    query_string = "".join(
                        query_elem.xpath(".//text()")
                    ).strip()

                    if not query_string:
                        log_warning(
                            f"Query tag found with empty content. Skipping: {etree.tostring(query_elem, encoding='unicode')[:200]}..."
                        )
                        continue

                    all_parsed_queries.append(
                        ParsedQuery(
                            xml_id=query_id,
                            purpose=purpose,
                            query_string=query_string,
                        )
                    )
                except Exception as e:
                    log_error(
                        f"Error parsing <query> tag: {etree.tostring(query_elem, encoding='unicode')[:200]}.... Error: {e}"
                    )

        log_info(
            f"Parsing complete. Found {len(parsed_entities)} entities, {len(all_parsed_relations)} relations, {len(all_parsed_queries)} queries."
        )

        return LLMResponseParsed(
            raw_xml=raw_xml_text,
            narrative_xml_element=narrative_element,  # Pass the lxml element for the found narrative
            entities=parsed_entities,  # List of all parsed entities from all roots
            relations=all_parsed_relations,  # List of all parsed relations from all roots
            queries=all_parsed_queries,  # List of all parsed queries from all roots
            raw_response_json=llm_response_json,  # Store original for log
        )

    def update_knowledge_graph(self, parsed_data: LLMResponseParsed):
        """Updates Neo4j KG based on parsed data (entities, relations, queries), applying creation/merge/status logic."""
        if not self.neo4j_driver or not self.connect_neo4j():
            log_error("Cannot update KG: Not connected to Neo4j.")
            return False

        if not parsed_data:
            log_warning("No parsed data provided for KG update.")
            return True  # Considered successful if nothing to update

        log_info(
            f"Updating KG with {len(parsed_data.entities)} entities and {len(parsed_data.relations)} relations from parsed data..."
        )

        # Need a mapping from XML ID to Canonical Name for creating relations
        entity_xml_id_to_canonical = {
            entity.xml_id: entity.canonical for entity in parsed_data.entities
        }

        # Use UNWIND for batching MERGE operations
        entity_data_for_unwind = [e.to_dict() for e in parsed_data.entities]
        relation_data_for_unwind = [r.to_dict() for r in parsed_data.relations]

        try:
            # Execute the update operations within a session (tx functions handle individual writes)
            with self.neo4j_driver.session() as session:
                # --- Batch Merge Entities ---
                if entity_data_for_unwind:
                    entity_merge_query = """
                    UNWIND $entities AS entity_data
                    MERGE (e:Entity {name: entity_data.canonical})
                    ON CREATE SET
                        e.xml_ref = entity_data.xml_id,
                        e.entityType = entity_data.entity_type,
                        e.status = entity_data.status,
                        e.provenance = entity_data.provenance,
                        e.attributes = entity_data.attributes,
                        e.text_spans = [entity_data.text_span] // Start with text_span from XML
                    ON MATCH SET
                        e.provenance = CASE
                                         WHEN NOT entity_data.provenance[0] IN e.provenance THEN e.provenance + entity_data.provenance
                                         ELSE e.provenance
                                       END,
                        e.status = CASE
                                     WHEN e.status = 'Pending' THEN entity_data.status // Only update status if currently pending
                                     ELSE e.status // Otherwise keep existing status (Canon, Ignored)
                                   END,
                        e.attributes = apoc.map.merge(e.attributes, entity_data.attributes), // Merge attributes using APOC
                        e.text_spans = CASE
                                         WHEN NOT entity_data.text_span IN e.text_spans THEN e.text_spans + entity_data.text_span
                                         ELSE e.text_spans
                                       END // Add text_span if not already present
                    RETURN count(e) AS entities_processed
                    """
                    try:
                        # Ensure status and provenance are lists in entity_data_for_unwind
                        for ed in entity_data_for_unwind:
                            if not isinstance(ed.get("provenance"), list):
                                ed["provenance"] = (
                                    [ed.get("provenance")]
                                    if ed.get("provenance")
                                    else []
                                )
                            # Sanitize entity type for label/property
                            ed["entity_type"] = "".join(
                                c if c.isalnum() or c == "_" else "_"
                                for c in (
                                    ed.get("entity_type") or "UnknownType"
                                )
                            )

                        summary = session.execute_write(
                            lambda tx: tx.run(
                                entity_merge_query,
                                entities=entity_data_for_unwind,
                            ).consume()
                        )
                        log_info(
                            f"Batch merged entities. Summary: {summary.counters}"
                        )
                    except Exception as e:
                        log_error(
                            f"Failed during batch entity merge: {e}",
                            exc_info=True,
                        )
                        # Decide whether to raise or continue... continue for now.

                # --- Batch Merge Relations ---
                if relation_data_for_unwind:
                    # Need to map subj_id and obj_id (XML IDs) to canonical names
                    # This requires an intermediate step or a JOIN in Cypher (more complex)
                    # Simplest is to look up canonical names in Python before UNWIND
                    relation_data_with_canonical = []
                    for rel_data in relation_data_for_unwind:
                        subj_canonical = entity_xml_id_to_canonical.get(
                            rel_data.get("subj_id")
                        )
                        obj_canonical = entity_xml_id_to_canonical.get(
                            rel_data.get("obj_id")
                        )

                        if not subj_canonical or not obj_canonical:
                            log_warning(
                                f"Skipping relation {rel_data.get('xml_id')}: Could not resolve canonical names for {rel_data.get('subj_id')} or {rel_data.get('obj_id')}"
                            )
                            continue

                        # Sanitize relation type for dynamic query part
                        relation_type_safe = (
                            rel_data.get("relation_type", "RELATED_TO")
                            .replace(" ", "_")
                            .upper()
                        )
                        if not relation_type_safe or not all(
                            c.isalnum() or c == "_" for c in relation_type_safe
                        ):
                            log_warning(
                                f"Relation type '{rel_data.get('relation_type')}' sanitized to '{relation_type_safe}' might be unsafe. Skipping."
                            )
                            continue

                        rel_data_copy = rel_data.copy()
                        rel_data_copy["subj_canonical"] = subj_canonical
                        rel_data_copy["obj_canonical"] = obj_canonical
                        rel_data_copy["relation_type_safe"] = (
                            relation_type_safe  # Add safe type for query
                        )
                        relation_data_with_canonical.append(rel_data_copy)

                    # Use APOC for dynamic relation type in MERGE if available and desired, or format string carefully
                    # Using format string here for simplicity, but be cautious.
                    # APOC approach: CALL apoc.merge.relationship(a, type, props, ON_MATCH_props, ON_CREATE_props, b)
                    # With UNWIND, the type must be the same for all relations in the batch, or use FOREACH+CALL
                    # Let's process relations individually within a single transaction function for safety/dynamic types
                    # Reverting to individual transaction functions for relations for dynamic types

                    def merge_single_relation_tx(tx, rel_data):
                        query = f"""
                         MATCH (a:Entity {{name: $subj_canonical}})
                         MATCH (b:Entity {{name: $obj_canonical}})
                         MERGE (a)-[r:{rel_data['relation_type_safe']}]->(b)
                         ON CREATE SET
                             r.xml_ref = $xml_id,
                             r.status = $status,
                             r.provenance = $provenance,
                             r.attributes = $attributes,
                             r.text_spans = [$text_span]
                         ON MATCH SET
                             r.provenance = CASE
                                              WHEN NOT $provenance[0] IN r.provenance THEN r.provenance + $provenance
                                              ELSE r.provenance
                                            END,
                             r.status = CASE
                                          WHEN r.status = 'Pending' THEN $status // Only update status if currently pending
                                          ELSE r.status // Otherwise keep existing status
                                        END,
                             r.attributes = apoc.map.merge(r.attributes, $attributes), // Merge attributes
                             r.text_spans = CASE
                                              WHEN NOT $text_span IN r.text_spans THEN r.text_spans + $text_span
                                              ELSE r.text_spans
                                            END // Add text_span if not already present
                         """
                        # Ensure provenance is a list
                        provenance_list = rel_data.get("provenance")
                        if not isinstance(provenance_list, list):
                            provenance_list = (
                                [provenance_list] if provenance_list else []
                            )

                        tx.run(
                            query,
                            subj_canonical=rel_data["subj_canonical"],
                            obj_canonical=rel_data["obj_canonical"],
                            xml_id=rel_data.get("xml_id"),
                            status=rel_data.get("status"),
                            provenance=provenance_list,
                            attributes=rel_data.get("attributes", {}),
                            text_span=rel_data.get("text_span"),
                        )

                    # Execute relation merges individually within the session
                    for rel_data in relation_data_with_canonical:
                        try:
                            session.execute_write(
                                merge_single_relation_tx, rel_data
                            )
                            # log_info(f"Merged relation: ({rel_data['subj_canonical']})-[:{rel_data['relation_type_safe']}]->({rel_data['obj_canonical']})") # Too chatty
                        except Exception as e:
                            log_error(
                                f"Failed to merge relation ({rel_data['subj_canonical']})-[:{rel_data.get('relation_type_safe')}]->({rel_data['obj_canonical']}): {e}",
                                exc_info=True,
                            )

            log_info("KG update transaction completed.")
            return True
        except Exception as e:
            log_error(f"KG update process failed: {e}", exc_info=True)
            return False

    def retrieve_kg_context_for_prompt(
        self, entity_canonical_name: Optional[str] = None
    ) -> str:
        """
        Retrieves relevant KG facts based on a focus entity and formats as <kg_context> XML.
        If no entity_canonical_name is provided, returns an empty <kg_context>.
        Retrieves the focus entity and its 1-hop 'Canon' neighbors/relations.
        """
        log_info(f"Retrieving KG context for entity: {entity_canonical_name}")

        if not entity_canonical_name:
            log_info("No focus entity provided. Returning empty KG context.")
            return "<kg_context></kg_context>"

        if not self.neo4j_driver or not self.connect_neo4j():
            log_error("Cannot retrieve KG context: Not connected to Neo4j.")
            return "<kg_context>Error: Not connected to KG.</kg_context>"

        # Cypher query to get the focus entity and its 1-hop 'Canon' neighbors/relations
        # Assuming :Entity and relationships have 'status' property
        query = """
        MATCH (n:Entity {name: $canonicalName})
        WHERE n.status = 'Canon'
        OPTIONAL MATCH (n)-[r]-(m:Entity)
        WHERE m.status = 'Canon' AND r.status = 'Canon'
        RETURN n, r, m
        """
        parameters = {"canonicalName": entity_canonical_name}

        try:
            results = self.run_cypher_query(query, parameters)
            if not results:
                log_info(
                    f"No 'Canon' entity found for '{entity_canonical_name}' or no 'Canon' neighbors."
                )
                return "<kg_context></kg_context>"

            # Process results to build XML
            kg_context_root = etree.Element("kg_context")

            # Collect unique entities and relations from results
            entities_dict: Dict[str, Dict[str, Any]] = {}  # Use name as key
            relations_list: List[Dict[str, Any]] = []

            for record in results:
                # Add focus entity 'n'
                n_node = record.get("n")
                if n_node:
                    name = n_node.get("name")
                    if name and name not in entities_dict:
                        entities_dict[name] = n_node  # Store the node data

                # Add neighbor entity 'm' if exists
                m_node = record.get("m")
                if m_node:
                    name = m_node.get("name")
                    if name and name not in entities_dict:
                        entities_dict[name] = m_node  # Store the node data

                # Add relation 'r' if exists
                r_rel = record.get("r")
                if r_rel:
                    # Need source and target canonical names for relation XML
                    start_node_name = r_rel.start_node.get("name")
                    end_node_name = r_rel.end_node.get("name")
                    if start_node_name and end_node_name:
                        # Represent relation as a dict, including start/end names
                        rel_data = dict(r_rel)  # Get relation properties
                        rel_data["_type"] = r_rel.type  # Add type
                        rel_data["_start_node_name"] = start_node_name
                        rel_data["_end_node_name"] = end_node_name
                        relations_list.append(rel_data)

            # Add entities to KG context XML
            for name, props in entities_dict.items():
                entity_elem = etree.SubElement(
                    kg_context_root,
                    "entity",
                    id=name,  # Use canonical name as ID in context
                    canonical=name,
                    type=props.get("entityType"),
                    status=props.get("status"),
                )
                # Add other relevant properties as attributes
                for key, value in props.items():
                    if key not in [
                        "name",
                        "entityType",
                        "status",
                        "xml_ref",
                        "provenance",
                        "text_spans",
                    ]:  # Exclude internal/history properties
                        if isinstance(value, (str, int, float, bool)):
                            entity_elem.set(key, str(value))

            # Add relations to KG context XML
            # Need to deduplicate relations if multiple paths returned the same one
            processed_rels = (
                set()
            )  # Use set of tuples (start_name, type, end_name) to track added relations
            for rel_data in relations_list:
                start_name = rel_data["_start_node_name"]
                end_name = rel_data["_end_node_name"]
                rel_type = rel_data["_type"]

                if (start_name, rel_type, end_name) not in processed_rels:
                    relation_elem = etree.SubElement(
                        kg_context_root,
                        "relation",
                        type=rel_type,
                        subj=start_name,  # Use canonical name as subj/obj ID
                        obj=end_name,
                        status=rel_data.get("status"),
                    )
                    # Add other relevant properties as attributes
                    for key, value in rel_data.items():
                        if not key.startswith("_") and key not in [
                            "status",
                            "xml_ref",
                            "provenance",
                            "text_spans",
                        ]:
                            if isinstance(value, (str, int, float, bool)):
                                relation_elem.set(key, str(value))
                    processed_rels.add((start_name, rel_type, end_name))

            # Serialize the XML element to a string
            kg_context_xml_string = etree.tostring(
                kg_context_root, encoding="unicode", pretty_print=True
            )

            log_info(
                f"Generated KG context XML:\n---\n{kg_context_xml_string}\n---"
            )
            return kg_context_xml_string

        except Exception as e:
            log_error(
                f"Error retrieving or formatting KG context for '{entity_canonical_name}': {e}",
                exc_info=True,
            )
            return "<kg_context>Error retrieving KG context.</kg_context>"

    def process_user_instruction(
        self,
        user_instruction: str,
        focus_entity_canonical_name: Optional[str] = None,
    ) -> Optional[LLMResponseParsed]:
        """Handles the full workflow: build prompt, call LLM, parse response, update history."""
        log_info("Processing user instruction...")

        # 1. Retrieve relevant KG context
        kg_context_xml = self.retrieve_kg_context_for_prompt(
            focus_entity_canonical_name
        )

        # 2. Get conversation history XML for prompt
        # Pass potential token limit from config if available
        max_prompt_tokens = self.config.get("api", {}).get(
            "max_tokens"
        )  # Rough estimate
        conversation_history_xml = (
            self.get_conversation_history_xml_for_prompt(
                max_tokens=max_prompt_tokens
            )
        )

        # 3. Get Glossary XML content
        glossary_xml_content = (
            self._glossary_xml_content
        )  # Use the raw loaded content

        # 4. Build full prompt messages for LLM
        messages = self.build_llm_prompt(
            user_instruction,
            conversation_history_xml,
            kg_context_xml,
            glossary_xml_content,
        )
        if not messages:
            log_error("Failed to build LLM prompt.")
            return None

        # 5. Call LLM API
        llm_response_json = self.call_llm_api(messages)
        if not llm_response_json:
            log_error("LLM API call failed.")
            # UI handles displaying error message
            return None

        # 6. Parse XML response
        parsed_data = self.parse_llm_response_xml(llm_response_json)
        if not parsed_data:
            log_error("Failed to parse LLM response XML.")
            # UI handles displaying parsing error and raw XML
            return None

        # 7. Add turn to conversation history
        # Store the original user instruction text and the raw XML response
        self.add_turn(user_instruction, parsed_data.raw_xml)
        log_info("Added turn to conversation history.")

        # 8. Update KG from parsed data (entities, relations from LLM response)
        # Queries are NOT updated here, they are just parsed and displayed for user execution
        # Only update KG from entities/relations
        entities_to_update = parsed_data.entities
        relations_to_update = parsed_data.relations
        llm_parsed_subset_for_kg = LLMResponseParsed(
            raw_xml=parsed_data.raw_xml,  # Keep raw XML reference
            narrative_xml_element=parsed_data.narrative_xml_element,  # Keep narrative element
            entities=entities_to_update,
            relations=relations_to_update,
            queries=[],  # Exclude queries for KG update step
        )

        kg_updated = self.update_knowledge_graph(llm_parsed_subset_for_kg)
        if kg_updated:
            log_info("Knowledge Graph updated from LLM response.")
        else:
            log_warning(
                "Knowledge Graph update from LLM response failed or was partial."
            )

        # Return the full parsed data to the UI for display
        return parsed_data

    def apply_curation_updates(
        self, curated_item_data: Dict[str, Any], item_type: str
    ) -> bool:
        """
        Applies user-curated updates (e.g., status, canonical name, type) to the KG
        for a single entity or relation.
        This method takes data from the UI's Quick Curation Panel.
        """
        log_info(
            f"Applying user curation updates for item type '{item_type}': {curated_item_data}"
        )

        if not self.neo4j_driver or not self.connect_neo4j():
            log_error("Cannot apply curation updates: Not connected to Neo4j.")
            return False

        # Ensure provenance includes 'User_Curated'
        provenance = curated_item_data.get("provenance", [])
        if "User_Curated" not in provenance:
            provenance.append("User_Curated")
        curated_item_data["provenance"] = provenance

        try:
            with self.neo4j_driver.session() as session:
                if item_type == "entity":
                    # Identify the entity to update by its canonical name (primary key in KG)
                    # Or maybe by an internal KG node ID if the UI passes that?
                    # Let's assume the UI passes the current canonical name and the desired new data.
                    # Need a way to uniquely identify the node to update.
                    # If canonical name is being changed, use the OLD canonical name to match.
                    # The UI should pass the original canonical name or an internal node ID.
                    # Let's assume the UI passes the *original* canonical name for matching.
                    original_canonical_name = curated_item_data.get(
                        "_original_canonical"
                    )  # Assume UI adds this

                    if not original_canonical_name:
                        log_error(
                            "Cannot apply entity curation: Original canonical name not provided by UI."
                        )
                        return False

                    # Ensure status is valid
                    new_status = curated_item_data.get("status", "Pending")
                    if new_status not in ["Pending", "Canon", "Ignored"]:
                        log_warning(
                            f"Invalid status '{new_status}' provided for entity curation. Defaulting to 'Pending'."
                        )
                        new_status = "Pending"

                    # Sanitize new canonical name
                    new_canonical_name = curated_item_data.get(
                        "canonical", ""
                    ).strip()
                    if not new_canonical_name:
                        new_canonical_name = (
                            original_canonical_name  # Keep old if new is empty
                        )
                        log_warning(
                            f"New canonical name is empty for entity '{original_canonical_name}'. Keeping original."
                        )

                    # Sanitize entity type
                    new_entity_type = curated_item_data.get("entity_type")
                    if new_entity_type:
                        new_entity_type_safe = "".join(
                            c if c.isalnum() or c == "_" else "_"
                            for c in new_entity_type or "UnknownType"
                        )
                    else:
                        # If type is not provided/changed, don't set it or get existing?
                        # Let's require it if editing, or get it from KG if not provided
                        # For now, only set if provided
                        new_entity_type_safe = (
                            None  # Signal to query not to set if None
                        )

                    # Construct update query for entity
                    entity_update_query = """
                    MATCH (e:Entity {name: $original_canonical_name})
                    SET
                        e.canonical = $new_canonical_name, // Update canonical name
                        e.status = $new_status,            // Explicitly set status (overrides Pending)
                        e.provenance = CASE
                                         WHEN NOT $new_provenance[0] IN e.provenance THEN e.provenance + $new_provenance
                                         ELSE e.provenance
                                       END,
                        e.attributes = apoc.map.merge(e.attributes, $new_attributes) // Merge user-edited attributes
                    """
                    # Conditionally set entityType if provided
                    if new_entity_type_safe is not None:
                        entity_update_query += (
                            ", e.entityType = $new_entity_type"
                        )

                    summary = session.execute_write(
                        lambda tx: tx.run(
                            entity_update_query,
                            {
                                "original_canonical_name": original_canonical_name,
                                "new_canonical_name": new_canonical_name,
                                "new_status": new_status,
                                "new_provenance": [
                                    "User_Curated"
                                ],  # Always add User_Curated provenance on manual apply
                                "new_attributes": curated_item_data.get(
                                    "attributes", {}
                                ),
                                "new_entity_type": new_entity_type_safe,
                            },
                        ).consume()
                    )
                    log_info(
                        f"Applied curation for entity '{original_canonical_name}'. Summary: {summary.counters}"
                    )
                    return True

                elif item_type == "relation":
                    # Identify the relation to update. This is harder.
                    # Needs original subject canonical name, object canonical name, and relation type.
                    # The UI should pass these identifiers.
                    original_subj_canonical = curated_item_data.get(
                        "_original_subj_canonical"
                    )
                    original_obj_canonical = curated_item_data.get(
                        "_original_obj_canonical"
                    )
                    original_relation_type = curated_item_data.get(
                        "_original_relation_type"
                    )

                    if (
                        not original_subj_canonical
                        or not original_obj_canonical
                        or not original_relation_type
                    ):
                        log_error(
                            "Cannot apply relation curation: Original subject, object, or type not provided by UI."
                        )
                        return False

                    # Ensure status is valid
                    new_status = curated_item_data.get("status", "Pending")
                    if new_status not in ["Pending", "Canon", "Ignored"]:
                        log_warning(
                            f"Invalid status '{new_status}' provided for relation curation. Defaulting to 'Pending'."
                        )
                        new_status = "Pending"

                    # Sanitize new relation type if provided
                    new_relation_type_str = curated_item_data.get(
                        "relation_type"
                    )
                    # If changing type, it's complex - might need to delete and re-create rel
                    # For simplicity, let's ONLY allow updating status and attributes for now.
                    # Changing relation type or endpoints requires more complex logic (delete old, create new).
                    # Let's restrict curation to status/attributes for relations for now.
                    # If the user wants to change the type, they should probably edit the raw XML in history.

                    relation_update_query = f"""
                    MATCH (a:Entity {{name: $original_subj_canonical}})-[r:{original_relation_type}]->(b:Entity {{name: $original_obj_canonical}})
                    SET
                        r.status = $new_status, // Explicitly set status
                        r.provenance = CASE
                                         WHEN NOT $new_provenance[0] IN r.provenance THEN r.provenance + $new_provenance
                                         ELSE r.provenance
                                       END,
                        r.attributes = apoc.map.merge(r.attributes, $new_attributes) // Merge user-edited attributes
                    """
                    # Note: This query only updates properties on the *first* matching relationship found.
                    # If multiple relationships of the same type exist between the same two entities,
                    # this might not update the specific one intended by the user's click.
                    # Clicking the narrative highlights a specific occurrence, but Cypher matches a pattern.
                    # This is a limitation of the current approach. Passing XML_ref or an internal KG ID
                    # from the UI for relation updates would be more robust, but complex.

                    summary = session.execute_write(
                        lambda tx: tx.run(
                            relation_update_query,
                            {
                                "original_subj_canonical": original_subj_canonical,
                                "original_obj_canonical": original_obj_canonical,
                                "new_status": new_status,
                                "new_provenance": [
                                    "User_Curated"
                                ],  # Always add User_Curated provenance
                                "new_attributes": curated_item_data.get(
                                    "attributes", {}
                                ),
                            },
                        ).consume()
                    )
                    log_info(
                        f"Applied curation for relation ({original_subj_canonical})-[:{original_relation_type}]->({original_obj_canonical}). Summary: {summary.counters}"
                    )
                    return True

                else:
                    log_error(
                        f"Unsupported item type for curation: {item_type}"
                    )
                    return False

        except Exception as e:
            log_error(
                f"Error applying user curation updates for {item_type}: {e}",
                exc_info=True,
            )
            # TODO: Signal to UI for message box
            return False

    def execute_suggested_query(
        self, query_string: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Executes a user-selected query suggested by LLM."""
        log_info(f"Executing suggested query: {query_string}")
        # Execute read query for safety, unless query pattern explicitly indicates write
        # A robust version might parse the query string to check if it's read/write
        # For now, let's assume LLM suggested queries are intended for exploration (read)
        # If a suggested query is a write query, it will fail when using run_cypher_query
        # If it contains parameters, it will fail unless the function signature is changed
        # Let's run it as a simple read query without parameters for now.
        # WARNING: Executing arbitrary Cypher from LLM/User can be a security risk.
        # For a production app, strict validation or a dedicated query execution panel
        # allowing parameter input would be needed.

        # Use the read query method
        results = self.run_cypher_query(query_string)

        if results is not None:
            log_info(
                f"Query executed successfully. Returned {len(results)} records."
            )
            # log_info(f"Query results: {results}") # Too verbose
            return results
        else:
            log_error("Suggested query execution failed.")
            # run_cypher_query logs the specific error
            return None  # run_cypher_query already returns None on failure

    # --- Conversation History Management ---

    def new_session(self):
        """Clears the current conversation history and session file path."""
        log_info("Starting new session.")
        self._conversation_turns = []
        self._current_session_file = None
        self._emulator_response_index = (
            1  # Reset emulator index for new session
        )
        # TODO: Signal to UI to clear displays and confirm KG reset?
        log_info("Conversation history cleared.")
        # Note: This does NOT clear the KG. KG is only cleared during rebuild_kg_from_history.

    def load_session(self, file_path: Path) -> bool:
        """Loads conversation history from an XML file and triggers KG reconstruction."""
        log_info(f"Loading session from {file_path}")
        if not file_path.exists():
            log_error(f"Session file not found: {file_path}")
            return False

        try:
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(str(file_path), parser=parser)
            root = tree.getroot()

            if root is None or root.tag != "session":
                log_error(
                    f"Invalid session file format: Root tag is not <session> in {file_path}"
                )
                if parser.error_log:
                    log_warning(f"Parsing errors: {parser.error_log}")
                return False

            loaded_turns: List[ConversationTurn] = []
            for turn_elem in root.findall(".//turn"):
                turn = ConversationTurn.from_xml_element(turn_elem)
                if turn:
                    loaded_turns.append(turn)

            self._conversation_turns = loaded_turns
            self._current_session_file = file_path
            log_info(
                f"Successfully loaded {len(self._conversation_turns)} turns from {file_path}."
            )

            # --- Trigger KG Reconstruction ---
            log_info("Triggering KG reconstruction from loaded history...")
            # Get the raw glossary content to pass to reconstruction
            # We need a way to load the glossary content from its file path (potentially stored in config)
            # Or the UI needs to pass the current glossary content.
            # Let's assume the model can load it based on config or a default path for reconstruction.
            # A better approach might be to make load_session accept optional glossary content.
            # For now, let's add a method to get glossary content by path.

            glossary_content = self.load_glossary_content_by_path(
                self.config.get(
                    "glossary_file", "approach/glossary.xml"
                )  # Default path or from config
            )  # Assuming a default glossary path or config entry

            reconstruction_success = self.rebuild_kg_from_history(
                glossary_content
            )
            if reconstruction_success:
                log_info("KG reconstruction completed successfully.")
            else:
                log_error("KG reconstruction failed after loading session.")
                # TODO: Signal to UI for message box

            return True

        except Exception as e:
            log_error(
                f"Error loading session file {file_path}: {e}", exc_info=True
            )
            if parser.error_log:
                log_warning(f"Parsing errors: {parser.error_log}")
            return False

    def save_session(self, file_path: Optional[Path] = None) -> bool:
        """Saves current conversation history to an XML file."""
        save_path = (
            file_path if file_path is not None else self._current_session_file
        )

        if not save_path:
            log_error(
                "No session file path specified and no current session file is set."
            )
            return False

        log_info(f"Saving session to {save_path}")
        try:
            root = etree.Element("session")
            for turn in self._conversation_turns:
                root.append(turn.to_xml_element())

            # Create parent directories if they don't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Use etree.ElementTree to write with a header and pretty printing
            tree = etree.ElementTree(root)
            tree.write(
                str(save_path),
                pretty_print=True,
                encoding="utf-8",
                xml_declaration=True,
            )

            self._current_session_file = (
                save_path  # Update current path if saved with new name
            )
            log_info(f"Session saved successfully to {save_path}.")
            return True

        except Exception as e:
            log_error(
                f"Error saving session to {save_path}: {e}", exc_info=True
            )
            return False

    def add_turn(self, user_prompt_text: str, llm_response_raw_xml: str):
        """Adds a new turn to the in-memory conversation history."""
        new_turn = ConversationTurn(user_prompt_text, llm_response_raw_xml)
        self._conversation_turns.append(new_turn)
        # Note: Saving to file is handled explicitly by user action (Save Session)

    def edit_turn(
        self,
        turn_index: int,
        new_user_prompt_text: str,
        new_llm_response_raw_xml: str,
    ) -> bool:
        """Edits a specific turn and truncates history from that point. Triggers KG rebuild."""
        if not 0 <= turn_index < len(self._conversation_turns):
            log_error(f"Invalid turn index for editing: {turn_index}")
            return False

        log_info(
            f"Editing turn index {turn_index}. Truncating history from this point."
        )

        # Create the new turn data
        edited_turn_data = ConversationTurn(
            new_user_prompt_text,
            new_llm_response_raw_xml,
            timestamp=datetime.datetime.now(),  # Update timestamp for the edited turn
        )

        # Replace the turn and truncate the list
        self._conversation_turns[turn_index] = edited_turn_data
        self._conversation_turns = self._conversation_turns[: turn_index + 1]

        log_info(
            f"History truncated. New history length: {len(self._conversation_turns)} turns."
        )

        # Trigger KG Reconstruction from the modified history
        log_info("Triggering KG reconstruction after turn edit.")
        glossary_content = self.load_glossary_content_by_path(
            self.config.get("glossary_file", "approach/glossary.xml")
        )
        reconstruction_success = self.rebuild_kg_from_history(glossary_content)
        if reconstruction_success:
            log_info("KG reconstruction completed successfully.")
        else:
            log_error("KG reconstruction failed after turn edit.")
            # TODO: Signal to UI for message box

        return True

    def delete_turns_from(self, turn_index: int) -> bool:
        """Deletes a turn and all subsequent turns. Triggers KG rebuild."""
        if not 0 <= turn_index < len(self._conversation_turns):
            log_error(f"Invalid turn index for deletion: {turn_index}")
            return False

        log_info(f"Deleting turn index {turn_index} and all subsequent turns.")

        # Truncate the history list
        self._conversation_turns = self._conversation_turns[:turn_index]

        log_info(
            f"History truncated. New history length: {len(self._conversation_turns)} turns."
        )

        # Trigger KG Reconstruction from the modified history
        log_info("Triggering KG reconstruction after turn deletion.")
        glossary_content = self.load_glossary_content_by_path(
            self.config.get("glossary_file", "approach/glossary.xml")
        )
        reconstruction_success = self.rebuild_kg_from_history(glossary_content)
        if reconstruction_success:
            log_info("KG reconstruction completed successfully.")
        else:
            log_error("KG reconstruction failed after turn deletion.")
            # TODO: Signal to UI for message box

        return True

    def get_conversation_history_xml_for_prompt(
        self, max_tokens: Optional[int] = None
    ) -> str:
        """
        Generates the XML string representation of the conversation history
        formatted for inclusion in the LLM prompt.
        Handles potential token limits by truncating older turns.
        """
        if not self._conversation_turns:
            return "<conversation_history></conversation_history>"

        # Build history XML
        history_root = etree.Element("conversation_history")
        # Add turns in order
        for i, turn in enumerate(self._conversation_turns):
            # Format user prompt and raw XML response within the turn element for the prompt
            # This is the *representation* of past turns for the LLM, not the session save format
            turn_elem = etree.Element(
                "turn", index=str(i)
            )  # Add index for reference
            user_prompt_elem = etree.SubElement(
                turn_elem, "user_prompt_sent_to_llm"
            )
            # The user prompt sent to LLM might include history/KG context/glossary itself from previous turns
            # To avoid infinite nesting or complex reconstruction, store the original *text* user typed,
            # and format it along with the *previous* LLM's raw XML response for the prompt.
            # Let's simplify: just include the user's input text and the LLM's raw XML response for each past turn.
            # This assumes the LLM can process this linear history.

            # Note: Storing the *full prompt sent to LLM* including context in history would be complex.
            # Storing just user text + raw LLM response seems more manageable.
            # Redefining the history XML structure for the prompt:
            # <conversation_history>
            #   <turn>
            #     <user>...</user> // User's original input text
            #     <assistant>...</assistant> // LLM's raw XML response
            #   </turn>
            #   ...
            # </conversation_history>

            user_elem = etree.SubElement(turn_elem, "user")
            user_elem.text = (
                turn.user_prompt_text
            )  # Use the original user text

            assistant_elem = etree.SubElement(turn_elem, "assistant")
            assistant_elem.text = etree.CDATA(
                turn.llm_response_raw_xml
            )  # Use CDATA for raw XML

            history_root.append(turn_elem)

        history_xml_string = etree.tostring(
            history_root, encoding="unicode", pretty_print=True
        )

        # --- Token Limit Handling ---
        # This is a rough estimation. Actual token counting depends on the LLM and tokenization.
        # A proper implementation would use a specific tokenizer for the target model.
        # For now, use a simple word count or character count as a proxy.
        # TINS implies dropping *older* turns if max_tokens is exceeded.

        if max_tokens is not None and max_tokens > 0:
            estimated_tokens = (
                len(history_xml_string) / 4
            )  # Rough estimate: 1 token ~ 4 chars
            log_info(
                f"Estimated history tokens: {estimated_tokens} (max {max_tokens})"
            )

            if estimated_tokens > max_tokens:
                log_warning(
                    f"Conversation history exceeds estimated token limit ({estimated_tokens} > {max_tokens}). Truncating oldest turns for prompt."
                )
                # Need to truncate the XML string by removing elements from the beginning
                # This requires re-building the XML, selecting elements from the end.
                truncated_history_root = etree.Element("conversation_history")
                turns_to_keep = []
                current_estimated_tokens = 0
                # Iterate from the end to keep the most recent turns
                for turn in reversed(self._conversation_turns):
                    temp_turn_elem = etree.Element(
                        "turn"
                    )  # Build a temporary element
                    user_elem = etree.SubElement(temp_turn_elem, "user")
                    user_elem.text = turn.user_prompt_text
                    assistant_elem = etree.SubElement(
                        temp_turn_elem, "assistant"
                    )
                    assistant_elem.text = etree.CDATA(
                        turn.llm_response_raw_xml
                    )
                    turn_xml_string = etree.tostring(
                        temp_turn_elem, encoding="unicode"
                    )

                    turn_estimated_tokens = (
                        len(turn_xml_string) / 4
                    )  # Estimate for this turn

                    if (
                        current_estimated_tokens + turn_estimated_tokens
                        > max_tokens
                    ):
                        log_info(
                            f"Stopping history truncation: Adding turn {len(self._conversation_turns) - len(turns_to_keep) - 1} would exceed token limit."
                        )
                        break  # Stop if adding this turn exceeds the limit

                    turns_to_keep.insert(
                        0, turn
                    )  # Add to the beginning of the list
                    current_estimated_tokens += turn_estimated_tokens

                # Rebuild the XML with the turns to keep
                for turn in turns_to_keep:
                    turn_elem = etree.Element("turn")
                    user_elem = etree.SubElement(turn_elem, "user")
                    user_elem.text = turn.user_prompt_text
                    assistant_elem = etree.SubElement(turn_elem, "assistant")
                    assistant_elem.text = etree.CDATA(
                        turn.llm_response_raw_xml
                    )
                    truncated_history_root.append(turn_elem)

                history_xml_string = etree.tostring(
                    truncated_history_root,
                    encoding="unicode",
                    pretty_print=True,
                )
                log_info(
                    f"Truncated history XML estimated tokens: {len(history_xml_string) / 4}"
                )

        return history_xml_string

    def rebuild_kg_from_history(
        self, glossary_xml_content: Optional[str] = None
    ) -> bool:
        """Clears the KG and rebuilds it by processing glossary and all history turns."""
        log_info("Starting KG reconstruction from history and glossary...")

        if not self.neo4j_driver or not self.connect_neo4j():
            log_error("Cannot rebuild KG: Not connected to Neo4j.")
            # TODO: Signal to UI for message box
            return False

        try:
            with self.neo4j_driver.session() as session:
                # 1. Clear the existing KG
                log_warning("Clearing existing Knowledge Graph...")
                clear_query = "MATCH (n) DETACH DELETE n"
                try:
                    summary = session.execute_write(
                        lambda tx: tx.run(clear_query).consume()
                    )
                    log_info(
                        f"KG cleared. Nodes deleted: {summary.counters.nodes_deleted}, Relationships deleted: {summary.counters.relationships_deleted}"
                    )
                except Exception as e:
                    log_error(f"Failed to clear KG: {e}", exc_info=True)
                    # Decide if we should stop here or try processing anyway. Stopping is safer.
                    # TODO: Signal to UI
                    return False

                # 2. Process Glossary XML
                if glossary_xml_content:
                    log_info(
                        "Processing Glossary content for KG reconstruction."
                    )
                    # Need to parse the glossary XML into ParsedEntity/Relation objects
                    # Reuse the parsing logic, but it expects a JSON structure wrapping the XML.
                    # Create a mock JSON structure for parse_llm_response_xml
                    mock_llm_response = {
                        "raw_xml": glossary_xml_content,
                        "choices": [
                            {"message": {"content": glossary_xml_content}}
                        ],  # Mock OpenAI/Groq structure
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": glossary_xml_content}]
                                }
                            }
                        ],  # Mock Google structure
                        # Add other required keys parse_llm_response_xml might expect if needed
                    }
                    parsed_glossary_data = self.parse_llm_response_xml(
                        mock_llm_response
                    )

                    if parsed_glossary_data:
                        # Update KG using the standard update logic
                        # Create a subset containing only entities and relations for KG update
                        glossary_subset_for_kg = LLMResponseParsed(
                            raw_xml=glossary_xml_content,  # Keep raw XML reference
                            narrative_xml_element=etree.Element(
                                "narrative"
                            ),  # Dummy narrative element
                            entities=parsed_glossary_data.entities,
                            relations=parsed_glossary_data.relations,
                            queries=[],  # Exclude queries
                        )
                        log_info(
                            f"Applying {len(parsed_glossary_data.entities)} entities and {len(parsed_glossary_data.relations)} relations from Glossary."
                        )
                        update_success = self.update_knowledge_graph(
                            glossary_subset_for_kg
                        )  # Use the standard update logic
                        if update_success:
                            log_info("Successfully applied Glossary to KG.")
                        else:
                            log_error(
                                "Failed to apply Glossary to KG during reconstruction."
                            )
                            # Decide if we stop here... continue with history processing but log error.
                    else:
                        log_error(
                            "Failed to parse Glossary XML during reconstruction."
                        )
                        # TODO: Signal to UI

                # 3. Process History Turns Sequentially
                log_info(
                    f"Processing {len(self._conversation_turns)} history turns for KG reconstruction."
                )
                for i, turn in enumerate(self._conversation_turns):
                    log_info(f"Processing history turn {i}...")
                    # Parse the raw XML response for this turn
                    # Reuse the parsing logic, requires mock JSON
                    mock_llm_response = {
                        "raw_xml": turn.llm_response_raw_xml,
                        "choices": [
                            {"message": {"content": turn.llm_response_raw_xml}}
                        ],  # Mock OpenAI/Groq structure
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {"text": turn.llm_response_raw_xml}
                                    ]
                                }
                            }
                        ],  # Mock Google structure
                        # Add other required keys parse_llm_response_xml might expect if needed
                    }
                    parsed_turn_data = self.parse_llm_response_xml(
                        mock_llm_response
                    )

                    if parsed_turn_data:
                        # Update KG using the standard update logic
                        # Create a subset containing only entities and relations for KG update
                        turn_subset_for_kg = LLMResponseParsed(
                            raw_xml=turn.llm_response_raw_xml,  # Keep raw XML reference
                            narrative_xml_element=etree.Element(
                                "narrative"
                            ),  # Dummy narrative element
                            entities=parsed_turn_data.entities,
                            relations=parsed_turn_data.relations,
                            queries=[],  # Exclude queries
                        )
                        log_info(
                            f"Applying {len(parsed_turn_data.entities)} entities and {len(parsed_turn_data.relations)} relations from turn {i}."
                        )
                        update_success = self.update_knowledge_graph(
                            turn_subset_for_kg
                        )  # Use the standard update logic
                        if not update_success:
                            log_error(
                                f"Failed to apply data from turn {i} to KG during reconstruction."
                            )
                            # Decide if we stop or continue. Continue, but log error.
                    else:
                        log_error(
                            f"Failed to parse XML from history turn {i} during reconstruction."
                        )
                        # TODO: Signal to UI

            log_info("KG reconstruction process completed.")
            # TODO: Signal to UI to update KG State display
            return True

        except Exception as e:
            log_error(
                f"An unexpected error occurred during KG reconstruction: {e}",
                exc_info=True,
            )
            # TODO: Signal to UI for message box
            return False

    # --- Glossary Management ---

    def load_glossary_content_by_path(
        self, file_path: Optional[str] = None
    ) -> str:
        """Loads raw XML content of the glossary file from the given path or default."""
        glossary_path = (
            Path(file_path)
            if file_path
            else Path(
                self.config.get("glossary_file", "approach/glossary.xml")
            )
        )  # Default path

        if not glossary_path.exists():
            log_warning(
                f"Glossary file not found at {glossary_path}. Returning empty content."
            )
            self._glossary_xml_content = ""  # Clear stored content
            return ""

        log_info(f"Loading glossary content from {glossary_path}")
        try:
            with open(glossary_path, "r", encoding="utf-8") as f:
                content = f.read()
                self._glossary_xml_content = content  # Store raw content
                log_info("Glossary content loaded successfully.")
                return content
        except Exception as e:
            log_error(
                f"Error loading glossary file {glossary_path}: {e}",
                exc_info=True,
            )
            self._glossary_xml_content = ""  # Clear on error
            return ""

    def save_glossary_content(self, file_path: Path, content: str) -> bool:
        """Saves raw XML content to the glossary file."""
        log_info(f"Saving glossary content to {file_path}")
        try:
            # Optional: Basic XML validation before saving
            # parser = etree.XMLParser(recover=True)
            # etree.fromstring(content.encode('utf-8'), parser=parser)
            # if parser.error_log:
            #     log_warning(f"Glossary content has XML parsing errors, saving anyway: {parser.error_log}")

            file_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure directory exists
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            self._glossary_xml_content = (
                content  # Update stored content on successful save
            )
            log_info(f"Glossary content saved successfully to {file_path}.")
            return True
        except Exception as e:
            log_error(
                f"Error saving glossary content to {file_path}: {e}",
                exc_info=True,
            )
            return False

    def process_glossary_xml(self, raw_glossary_xml_content: str) -> bool:
        """Parses glossary XML and applies entities/relations to the KG."""
        log_info("Processing raw glossary XML content...")

        # Need to parse the glossary XML into ParsedEntity/Relation objects
        # Reuse the parsing logic, but it expects a JSON structure wrapping the XML.
        # Create a mock JSON structure for parse_llm_response_xml
        mock_llm_response = {
            "raw_xml": raw_glossary_xml_content,
            "choices": [
                {"message": {"content": raw_glossary_xml_content}}
            ],  # Mock OpenAI/Groq structure
            "candidates": [
                {"content": {"parts": [{"text": raw_glossary_xml_content}]}}
            ],  # Mock Google structure
            # Add other required keys parse_llm_response_xml might expect if needed
        }
        parsed_glossary_data = self.parse_llm_response_xml(mock_llm_response)

        if parsed_glossary_data:
            # Update KG using the standard update logic
            # Create a subset containing only entities and relations for KG update
            glossary_subset_for_kg = LLMResponseParsed(
                raw_xml=raw_glossary_xml_content,  # Keep raw XML reference
                narrative_xml_element=etree.Element(
                    "narrative"
                ),  # Dummy narrative element
                entities=parsed_glossary_data.entities,
                relations=parsed_glossary_data.relations,
                queries=[],  # Exclude queries
            )
            log_info(
                f"Applying {len(parsed_glossary_data.entities)} entities and {len(parsed_glossary_data.relations)} relations from Glossary content."
            )
            update_success = self.update_knowledge_graph(
                glossary_subset_for_kg
            )  # Use the standard update logic
            if update_success:
                log_info("Successfully applied Glossary content to KG.")
                self._glossary_xml_content = raw_glossary_xml_content  # Update internal content if applied successfully
                return True
            else:
                log_error("Failed to apply Glossary content to KG.")
                # TODO: Signal to UI
                return False
        else:
            log_error("Failed to parse Glossary XML content.")
            # TODO: Signal to UI
            return False

    # --- Placeholder for Alias/Canonical Name Management (Glosario) ---
    # These methods would interact with the KG to find existing canonical names
    # or create new alias relationships based on user curation actions.
    # This functionality is likely integrated into the apply_curation_updates workflow
    # or happens during KG reconstruction/updates if the LLM/Glossary provides alias tags.

    # def suggest_canonical(self, alias_name: str) -> Optional[str]:
    #     """Queries KG for existing canonical names similar to alias_name."""
    #     log_info(f"Suggesting canonical for alias: {alias_name}")
    #     # Implement Cypher query to find similar entity names/aliases in KG
    #     # Return a suggested canonical name if found
    #     return None # Placeholder

    # def create_alias_mapping(self, alias_name: str, canonical_name: str):
    #     """Creates a new alias mapping in the KG (e.g., :Alias node -> :Canonical node)."""
    #     log_info(f"Creating alias mapping: {alias_name} -> {canonical_name}")
    #     # Implement Cypher query to create the necessary nodes/relationships
    #     pass # Placeholder


# Note: Direct calls to CurationModel methods should be done from the main PySide application
# loop or event handlers, not directly in this file's top level or a __main__ block,
# as this file is intended to be imported.