<!-- Zero Source Specification v1.0 -->
<!-- ZS:PLATFORM:Desktop -->
<!-- ZS:LANGUAGE:Python -->
<!-- ZS:UI:PySide6 -->
<!-- ZS:DATA:Neo4j -->
<!-- ZS:DOMAIN:Knowledge Graph, LLM Interaction, NLP -->

# AI-Assisted Knowledge Graph Curation Tool

## Description

This application is a desktop graphical user interface (GUI) tool designed to assist users in curating knowledge graphs by leveraging the power of Large Language Models (LLMs). The tool allows users to provide natural language instructions to an LLM, receive structured data (entities, relations, queries) embedded within a narrative in XML format, visualize the parsed data, view the raw LLM output, monitor the state of a connected Neo4j knowledge graph, and manage application configuration and prompt templates. It aims to provide a user-friendly interface for interacting with LLM outputs intended for knowledge graph population and refinement.

## Functionality

### Core Features

- **LLM Interaction:** Send user instructions along with conversation history and relevant KG context (placeholder) to a configured LLM API.
- **Response Processing:** Receive and parse LLM responses, extracting structured data (entities, relations, Cypher queries) from the embedded XML.
- **Narrative Display:** Present the generated narrative text with visual highlighting for identified entities, relations, and queries.
- **Parsed Data Display:** Show the extracted entities, relations, and queries in a structured, readable format (e.g., JSON or formatted text).
- **Raw Output Display:** Provide access to the raw XML response received directly from the LLM for debugging and verification.
- **Knowledge Graph Monitoring:** Display a summary of the connected Neo4j database state, including entity and relation counts (distinguishing between 'Pending' and 'Canon' status).
- **Neo4j Browser Access:** Provide a button to easily open the Neo4j Browser web interface for detailed graph inspection.
- **Configuration Management:** Load, view, edit, and save application settings via a JSON editor in the GUI. This includes API configurations, XML format details, and prompt file paths.
- **Prompt Template Management:** Load, view, edit, and save LLM prompt templates (`.txt` files) via a text editor in the GUI.
- **Theming and Styling:** Allow users to select different GUI styles (based on PySide6/Qt available styles) and color schemes (light/dark).
- **Font Size Adjustment:** Allow users to dynamically adjust the font size for text display and input areas.
- **Persistent Settings:** Save window size, position, style, and font size between sessions.

### User Interface

The main window should utilize horizontal and vertical splitters to arrange content into distinct, resizable panes. Tab widgets should be used within these panes to organize related information and input types.

```
+-------------------------------------------------------------+
| Menubar (File, View)                                        |
+-------------------------------------------------------------+
|                                                             |
|  +-----------------------------+ +-------------------------+  |
|  | Narrative: (Left Pane)      | | Monitor Tabs: (Right Pane)|  |
|  | +-------------------------+ | | +---------------------+ |  |
|  | |                         | | | | Parsed Data Tab     | |  |
|  | | QTextEdit (Read-only) | | | | +-------------------+ |  |
|  | | (Highlighted narrative) | | | | QTextEdit (Parsed)| |  |
|  | |                         | | | | data JSON/text)   | |  |
|  | +-------------------------+ | | +-------------------+ |  |
|  |                             | | +---------------------+ |  |
|  +-----------------------------+ | | Raw XML Tab         | |  |
|                                 | | +-------------------+ |  |
|  +-----------------------------+ | | QTextEdit (Raw XML) | |  |
|  | Input Tabs: (Bottom Pane) | | | +-------------------+ |  |
|  | +-------------------------+ | | +---------------------+ |  |
|  | | User Instruction Tab    | | | | KG State Tab        | |  |
|  | | +-------------------+ | | | | +-------------------+ |  |
|  | | | QTextEdit (Input) | | | | | | QTextEdit (KG Sum.) | |  |
|  | | |                   | | | | | +-------------------+ |  |
|  | | +-------------------+ | | | | | Open Neo4j Browser| |  |
|  | +-------------------------+ | | | Button            | |  |
|  | | Prompt Templates Tab    | | | +-------------------+ |  |
|  | | +-------------------+ | | | +---------------------+ |  |
|  | | | ComboBox (Selector) | | | | History Tab         | |  |
|  | | | QTextEdit (Editor)  | | | | +-------------------+ |  |
|  | | +-------------------+ | | | | | QTextEdit (Logs)  | |  |
|  | +-------------------------+ | | | +-------------------+ |  |
|  | | Config Editor Tab       | | | +---------------------+ |  |
|  | | +-------------------+ | | | +-------------------------+  |
|  | | | QTextEdit (JSON Ed) | | |                             |  |
|  | | +-------------------+ | | +-------------------------+  |
|  | +-------------------------+                             |
|  | +-------------------------+                             |
|  | | Send to LLM Button      |                             |
|  | +-------------------------+                             |
|  +-----------------------------+-------------------------+  |
|                                                             |
+-------------------------------------------------------------+
| Toolbar (Style, Theme, Font Size) | Status Bar            |
+-------------------------------------------------------------+
```

- **Narrative Display (`QTextEdit`):** Read-only, displays text with inline highlighting/formatting based on parsed XML tags (e.g., entities with one background color, relations with another, queries formatted as code blocks). Should support rich text. Event filtering should be considered for clicking on highlighted elements to trigger actions (e.g., quick curation - *placeholder function in model*).
- **Monitor Tabs (`QTabWidget`):** Contains:
    - **Parsed Data Tab (`QTextEdit`):** Read-only, displays the structured data extracted from the XML response, ideally as formatted JSON.
    - **Raw XML Tab (`QTextEdit`):** Read-only, displays the complete text response from the LLM, including the XML structure.
    - **KG State Tab (`QTextEdit`):** Read-only, displays a summary of key metrics from the connected Neo4j database (e.g., node/relationship counts by status). Includes a button to open the Neo4j browser.
    - **History Tab (`QTextEdit`):** Read-only, displays application logs and conversation history.
- **Input Tabs (`QTabWidget`):** Contains:
    - **User Instruction Tab (`QTextEdit`):** Editable text area for the user to type their prompt/instruction for the LLM.
    - **Prompt Templates Tab (`QTextEdit`):** Allows selecting a prompt file from a dropdown and editing its content. Includes a save button.
    - **Config Editor Tab (`QTextEdit`):** Displays the current application configuration in a JSON editor. Includes a save button.
- **Toolbar:** Should include controls for selecting GUI style, toggling color scheme (Dark/Light), and adjusting font size.
- **Status Bar:** Display application status messages (e.g., "Sending to LLM...", "Ready", "Error: ...").
- **Buttons:** "Send to LLM" triggers the main workflow. "Open Neo4j Browser" opens the configured Neo4j URI in the default web browser. "Save Prompt Template" and "Save Config" save the content from the respective editors to files.
- **Menus:** "File" menu with actions for loading/saving configuration files and exiting. "View" menu with options for selecting color schemes and styles.

### User Flows

1.  **Initial Startup:**
    - Application loads configuration from `approach/default.json` or creates a default.
    - Application loads prompt templates from the `prompts/` directory.
    - GUI initializes with saved window state, style, and font size.
    - Attempts to connect to Neo4j using environment variables.
    - KG State summary is updated.
    - Status bar shows "Ready.".
2.  **Sending Instruction to LLM:**
    - User types instruction in the "User Instruction" tab.
    - User clicks "Send to LLM".
    - Application retrieves system and query prompt templates.
    - Application builds the full prompt, incorporating the user instruction, conversation history (stored previous XML responses), and KG context (placeholder, currently empty `<kg></kg>`).
    - Status bar shows "Sending to LLM...". "Send to LLM" button is disabled.
    - Application calls the configured LLM API with the built prompt.
    - Upon receiving a response (expected JSON with text containing XML), the application status bar updates.
    - Application parses the XML content from the response body.
    - Status bar shows "Processing LLM response...".
    - The parsed structured data (entities, relations, queries) is displayed in the "Parsed Data" tab.
    - The raw XML response is displayed in the "Raw XML" tab.
    - The narrative text within the XML is rendered in the "Narrative" display with highlighting applied to entities, relations, and queries based on their tags.
    - The parsed entities and relations are used to attempt an update/merge operation on the connected Neo4j graph.
    - The KG State summary is refreshed after the graph update attempt.
    - Status bar shows "LLM response received and processed." or an error message. "Send to LLM" button is re-enabled.
    - The raw XML response is appended to the internal conversation history.
3.  **Managing Configuration:**
    - User switches to the "Config Editor" tab.
    - The current configuration (loaded at startup or via "Load Config") is displayed as formatted JSON in the editor.
    - User can edit the JSON directly.
    - User clicks "Save Config" (or uses "Save Config..." from the File menu to choose a new file).
    - Application validates JSON format, saves the file, updates the internal config, and reloads prompt templates based on potentially changed file paths in the config.
    - Status bar confirms save or shows an error.
4.  **Managing Prompt Templates:**
    - User switches to the "Prompt Templates" tab.
    - User selects a `.txt` file from the `prompts/` directory using the combo box.
    - The content of the selected file loads into the editor.
    - User can edit the text.
    - User clicks "Save Prompt Template".
    - Application saves the content back to the selected file.
    - Status bar confirms save or shows an error.
5.  **Changing Style/Theme:**
    - User selects a style from the Style combo box or View menu.
    - Application applies the selected PySide style.
    - User clicks the Dark/Light mode button or selects a scheme from the View menu.
    - Application applies the corresponding color scheme (Dark, Light, or Auto via Qt style hints).
    - Status bar confirms change.
6.  **Adjusting Font Size:**
    - User changes the value in the Font Size spin box.
    - Application updates the font size for all relevant text widgets.

### Edge Cases

- **LLM API Call Failure:** The application should catch exceptions or bad responses from the API, log the error, inform the user via the status bar or a message box, and keep the UI responsive.
- **XML Parsing Errors:** If the LLM returns malformed XML, the parser (`lxml` with `recover=True`) should attempt recovery. If parsing fails critically, the raw XML should still be displayed, and the parsed data/narrative displays should show an error message. Errors should be logged.
- **Neo4j Connection Failure:** The application should attempt to connect/reconnect. If unsuccessful, KG-related features (state summary, graph updates) should be disabled or show an error message, and errors logged.
- **Empty User Instruction:** The "Send to LLM" button should be disabled or show a warning if the input is empty.
- **Invalid JSON/Prompt File Content:** Saving config or prompt files should include validation. If invalid, warn the user and prevent saving.
- **Missing Config/Prompt Files:** Application should handle missing default files by creating them. If specified files in config/prompts directory are missing, log errors and update the UI accordingly (e.g., disable prompt editor if file not found).
- **Clicking on Narrative:** If the event filter detects a click, it should identify the clicked element (text span), ideally linking it back to the parsed entity/relation/query data for potential quick curation actions (currently logs info).

## Technical Implementation

### Architecture

The application follows a largely monolithic architecture with a clear separation between the GUI (`main.py`) and the core logic/data handling (`model.py`).

- **GUI Layer (`main.py`):** Handles user interface creation, event handling, displaying information, and forwarding user actions to the model. Uses PySide6. Manages application settings using `QSettings`.
- **Model Layer (`model.py`):** Encapsulates the logic for interacting with external services (LLM API, Neo4j), parsing data, managing state (conversation history, config), and performing core operations like KG updates. Uses `requests` for API calls (placeholder), `neo4j` for database interaction, and `lxml` for XML parsing. Reads configuration and prompts from files. Uses `python-dotenv` for environment variables (Neo4j credentials). Includes custom logging utilities.

Communication is primarily from the GUI to the Model (user actions trigger model methods) and from the Model back to the GUI (model provides data/status updates for display).

### Data Model

**Application Internal Data Models (`model.py` classes):**

-   `ParsedEntity`: Represents an entity extracted from the LLM XML.
    ```python
    {
      "xml_id": str,       # ID from XML tag (e.g., "ent1")
      "text_span": str,    # The exact text from the narrative tag content
      "canonical": str,    # Canonical name (defaults to text_span if not in XML)
      "entity_type": Optional[str], # Type specified in XML (e.g., "Person", "Location")
      "status": str,       # Curation status ("Pending", "Canon", "Ignored") - defaults to "Pending"
      "provenance": List[str], # Source of the data point ("LLM_XML_Generated", "User_Curated", etc.)
      "attributes": Dict[str, Any] # Any other attributes from the XML tag
    }
    ```
-   `ParsedRelation`: Represents a relation extracted from the LLM XML.
    ```python
    {
      "xml_id": str,       # ID from XML tag (optional, auto-generated if not present)
      "text_span": str,    # The exact text from the narrative tag content
      "relation_type": str,# Type specified in XML (e.g., "ACTED_IN")
      "subj_id": str,      # XML ID of the subject entity
      "obj_id": str,       # XML ID of the object entity
      "status": str,       # Curation status ("Pending", "Canon", "Ignored") - defaults to "Pending"
      "provenance": List[str], # Source of the data point
      "attributes": Dict[str, Any] # Any other attributes from the XML tag
    }
    ```
-   `ParsedQuery`: Represents a suggested Cypher query extracted from the LLM XML.
    ```python
    {
      "xml_id": str,       # ID from XML tag (optional, auto-generated if not present)
      "purpose": str,      # Purpose specified in XML tag attribute
      "query_string": str  # The Cypher query text
    }
    ```
-   `LLMResponseParsed`: Container for the entire parsed LLM response.
    ```python
    {
      "raw_xml": str, # The original XML string received
      "narrative_xml_element": etree.Element, # The lxml element for the <narrative> tag
      "entities": List[ParsedEntity],
      "relations": List[ParsedRelation],
      "queries": List[ParsedQuery],
      "raw_response_json": Dict[str, Any] # The raw JSON response from the LLM API
    }
    ```

**Knowledge Graph Data Model (Neo4j):**

-   **Nodes:** Primarily `(:Entity)` nodes are expected.
    -   Properties:
        -   `name`: Canonical name of the entity (indexed, acts as primary key for merging).
        -   `xml_ref`: Original XML ID from the LLM response.
        -   `entityType`: Type of the entity (e.g., "Person", "Location", "Organization").
        -   `status`: Curation status ("Pending", "Canon"). Determines if the entity is considered canonical.
        -   `provenance`: List of strings indicating the source(s) of this node data (e.g., ["LLM_XML_Generated", "User_Curated"]).
        -   `attributes`: Map containing any additional arbitrary attributes parsed from the XML tag.
        -   `text_spans`: List of strings, storing the different text spans from the narrative that map to this canonical entity.
-   **Relationships:** Relationships between `(:Entity)` nodes have dynamic types determined by the LLM's `<relation type="...">` tag (e.g., `:ACTED_IN`, `:LOCATED_IN`).
    -   Properties:
        -   `xml_ref`: Original XML ID from the LLM response (optional on relation tag).
        -   `status`: Curation status ("Pending", "Canon").
        -   `provenance`: List of strings indicating the source(s) of this relationship data.
        -   `attributes`: Map containing any additional arbitrary attributes parsed from the XML tag.
        -   `text_spans`: List of strings, storing the different text spans that map to this relationship.

### Algorithms

-   **Prompt Building:** Concatenate a static system prompt, a query template (formatted with `user_instruction`, `conversation_history`, and `kg_context`), history XML, and KG context XML (placeholder).
-   **LLM API Call:** Abstracted function (`call_llm_api`) expected to make an HTTP POST request to an LLM endpoint (e.g., OpenAI compatible API, Groq, Google Gemini) with prompt messages and configured parameters (`temperature`, `max_tokens`). The specific API endpoint and key are expected to be handled internally by the `model.py` or dependencies (e.g., via environment variables).
-   **XML Parsing:** Use `lxml.etree` to parse the XML string extracted from the LLM response. Search for the configured root tag (`<response>`) and then the `<narrative>` tag. Recursively traverse the XML tree, identifying `<entity>`, `<relation>`, and `<query>` tags within the narrative or elsewhere. Extract data from tag attributes (`id`, `canonical`, `type`, `subj`, `obj`, `status`, `purpose`, etc.) and tag text content. Handle potential XML syntax errors with recovery.
-   **Narrative Rendering:** Convert the `lxml` element tree representing the `<narrative>` into HTML suitable for a `QTextEdit` display. Replace specific XML tags (`<entity>`, `<relation>`, `<query>`) with HTML `<span>` or `<code>` tags with inline styles (background color, display: block) and data attributes (`data-type`, `data-id`, `data-canonical`, etc.) to preserve the extracted metadata and enable potential future interaction (like clicking).
-   **Knowledge Graph Update:** Use Cypher `MERGE` statements executed within a Neo4j session to add or update entities and relationships based on the parsed data.
    -   Entities are merged based on `name` (canonical name).
    -   Relations require matching subject and object entities by `name` and then merging the relationship using a dynamic type (requires `apoc.merge.relationship`).
    -   On `CREATE`, set initial properties (`xml_ref`, `status`, `provenance`, `attributes`, `text_spans`).
    -   On `MATCH`, update properties: merge `attributes`, add new `provenance` sources, add new `text_spans`. Status update logic should prioritize existing 'Canon' status over incoming 'Pending', only allowing incoming 'Canon' or updating 'Pending'.
-   **KG State Summary:** Execute predefined Cypher queries (`MATCH (n:Entity) RETURN count(*)`, `MATCH ()-[r]-() RETURN count(*)`, `MATCH (n:Entity {status: 'Pending'}) RETURN count(*)`, etc.) to retrieve counts and display them.

### Dependencies

-   `PySide6`: For the graphical user interface.
-   `requests`: For making HTTP requests to the LLM API.
-   `neo4j`: For connecting to and interacting with the Neo4j database.
-   `python-dotenv`: For loading environment variables (specifically Neo4j credentials).
-   `lxml`: For efficient and robust XML parsing.

These dependencies are listed in the `requirements.txt` file.

### Configuration

The application relies on a JSON configuration file (default: `approach/default.json`). This file structures settings into sections:

```json
{
  "api": {
    "api_name": "groq",      // Identifier for the LLM API (used internally by model, though call is abstracted)
    "model_name": "...",     // Specific model name to use
    "temperature": 0.7,      // LLM generation temperature
    "max_tokens": 1024       // Maximum tokens for LLM response
  },
  "format": {
    "separator": "###END_XML_RESPONSE###", // Marker to find XML in response (not strictly used in parser but good practice)
    "response_root_tag": "response" // Expected root tag of the XML output
  },
  "prompts": {
    "system_prompt_file": "prompts/default_system.txt", // Path to the system prompt file
    "query_prompt_file": "prompts/default_query.txt"   // Path to the query prompt template file
  }
}
```

Neo4j connection details are loaded from environment variables (`.env` file or system environment): `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`. LLM API keys are also expected to be handled via environment variables or similar secure mechanisms, not within the config file itself.

### Expected LLM XML Structure

The application's parsing logic (`parse_llm_response_xml`) expects the LLM to return text containing an XML document. The core data should be wrapped in a root tag specified in the config (default `<response>`). Narrative text should ideally be within a `<narrative>` tag inside the root, but entities/relations/queries can potentially appear elsewhere within the root tag according to the current parsing logic.

Example expected structure (within the text response from the LLM):

```xml
<response>
  <narrative>
    Here is a sentence about <entity id="ent1" canonical="Alice" type="Person" status="Pending">Alice</entity>.
    Alice knows <entity id="ent2" canonical="Bob" type="Person">Bob</entity>.
    They <relation type="KNOWS" subj="ent1" obj="ent2">know each other</relation>.
  </narrative>
  <query purpose="Find all people named Alice">
    MATCH (p:Entity {name: 'Alice', entityType: 'Person'}) RETURN p
  </query>
</response>
```

Key tags and attributes:
-   `<response>`: Root tag (configurable).
-   `<narrative>`: Contains the human-readable text interspersed with tags.
-   `<entity>`: Represents an entity.
    -   `id` (required for relations to reference): Unique ID within this response (e.g., "ent1").
    -   `canonical` (recommended): The preferred, standardized name.
    -   `type` (recommended): Entity type (e.g., "Person", "Location").
    -   `status` (optional): Curation status suggestion ("Pending", "Canon"). Defaults to "Pending".
    -   Additional attributes are collected into the `attributes` map.
-   `<relation>`: Represents a relationship between two entities.
    -   `id` (optional): Unique ID for the relation.
    -   `type` (required): The type of relationship (e.g., "KNOWS", "LIVES_IN"). Should ideally be uppercase and snake_case for Neo4j.
    -   `subj` (required): The `id` of the subject entity.
    -   `obj` (required): The `id` of the object entity.
    -   `status` (optional): Curation status suggestion ("Pending", "Canon"). Defaults to "Pending".
    -   Additional attributes are collected into the `attributes` map.
-   `<query>`: Represents a suggested Cypher query.
    -   `id` (optional): Unique ID for the query.
    -   `purpose` (optional): Description of the query's intent.

## Style Guide

The application uses the PySide6/Qt styling system.
- Supports various platform-native styles available (`windows11`, `fusion`, etc.).
- Explicit support for toggling between Light and Dark color schemes using `Qt.ColorScheme`.
- Font size is user-adjustable and persistent.
- The narrative display applies specific background colors to highlight entities and relations. Queries are styled as block code.
- Basic iconography is used for buttons and menus (if available from the system style).

## Testing Scenarios

1.  Launch the application. Verify default config and prompts are loaded or created. Verify window state restore.
2.  Check Neo4j connection status in the KG State tab. If not connected, verify error message.
3.  Enter a simple instruction in the "User Instruction" tab (e.g., "Tell me about Alice and Bob."). Click "Send to LLM".
4.  Observe status bar messages during the LLM call and processing.
5.  Verify the "Narrative" display shows generated text. Check for highlighting of potential entities/relations.
6.  Check the "Parsed Data" tab. Verify that entities and relations are extracted with correct attributes (`id`, `canonical`, `type`, `subj`, `obj`). Check for extracted queries.
7.  Check the "Raw XML" tab. Verify the full XML structure is present and matches the expected format.
8.  Check the "KG State" tab. Verify that entity and relation counts are updated after the LLM response is processed (requires Neo4j connection).
9.  Enter an instruction expected to produce malformed XML. Verify the application handles it gracefully (displays raw XML, shows parsing error in logs/status, does not crash).
10. Go to "Config Editor". Edit the JSON (e.g., change a parameter). Click "Save Config". Verify the file is saved and the internal config is updated. Reload config from file to verify.
11. Go to "Prompt Templates". Select a prompt. Edit the text. Click "Save Prompt Template". Verify the file is saved. Reload the prompt to verify.
12. Use the Style selector and Theme button/menu to change styles and color schemes. Verify the UI updates visually.
13. Use the Font Size spin box. Verify text size changes in all relevant editors/displays.
14. If Neo4j is connected, click "Open Neo4j Browser" and verify the browser opens to the correct URI.
15. Close the application. Verify window size/position, style, and theme are saved. Relaunch to verify they are restored.

## Accessibility Requirements

-   Supports system-wide font size and potentially style preferences via Qt integration.
-   Keyboard navigation should be functional for all standard widgets (buttons, text areas, tabs, selectors).
-   Use of standard PySide6 widgets implies basic compatibility with screen readers, but this would require testing and potential addition of ARIA attributes if the PySide6 layer allows fine-grained control.
-   High contrast mode support depends on the selected PySide style and underlying OS theme, as well as the application's handling of custom colors (highlighting colors should ideally be customizable or adapt to themes).

## Performance Goals

-   UI should remain responsive during LLM calls (which occur in the background relative to the main event loop, though the button is disabled).
-   Parsing and display updates should be reasonably fast for typical LLM response sizes.
-   Neo4j updates should complete quickly, especially for large numbers of entities/relations in a single response. The current batch `UNWIND` Cypher queries should be efficient.
-   Handling large amounts of history or raw XML text in the QTextEdits should not degrade performance severely.

## Extended Features (Optional)

-   Implement the `call_llm_api` method in `model.py` to integrate with specific LLM providers (e.g., via `requests` to OpenAI API, Groq API, Gemini API).
-   Implement click handling on highlighted narrative elements to trigger quick curation actions (e.g., mark entity as Canon, change type, merge with existing).
-   Develop the `retrieve_kg_context_for_prompt` function to query Neo4j for relevant context (e.g., neighboring nodes, properties of referenced entities) to include in the LLM prompt.
-   Implement the `apply_curation_updates` function to modify entity/relation status, type, or attributes in the KG based on user actions triggered from the UI (currently lacks UI elements to trigger this).
-   Implement `suggest_canonical` and `create_alias_mapping` functionalities.
-   Add more robust error handling and user feedback for failed KG updates or parsing errors.
-   Add validation for JSON config and prompt files when saving from the editor.
-   Implement conversation history management (e.g., clear history, load/save history).
-   Add ability to execute suggested Cypher queries directly from the UI (e.g., from the Parsed Data or Raw XML view).