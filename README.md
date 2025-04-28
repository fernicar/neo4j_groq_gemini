<!-- Zero Source Specification v1.0 -->
<!-- ZS:PLATFORM:Desktop -->
<!-- ZS:LANGUAGE:Python -->
<!-- ZS:UI:PySide6 -->
<!-- ZS:DATA:Neo4j -->
<!-- ZS:DOMAIN:Knowledge Graph, LLM Interaction, NLP, Narrative Curation -->

# AI-Assisted Knowledge Graph Curation Tool

## Description

This application is a desktop graphical user interface (GUI) tool designed to assist users in curating knowledge graphs by leveraging the power of Large Language Models (LLMs). The tool allows users to provide natural language instructions to an LLM, receive structured data (entities, relations, queries) embedded within a narrative in XML format, visualize the parsed data with highlighting, view the raw LLM output, monitor the state of a connected Neo4j knowledge graph, manage application configuration and prompt templates, and manage a persistent conversation history which serves as the primary source for KG reconstruction. It aims to provide a user-friendly interface for interacting with LLM outputs intended for knowledge graph population and refinement, focusing the user's role on validation and status changes (`Pending` -> `Canon`).

*(Default UI Assumption: Desktop application using PySide6 version >= 6.9.0, employing the 'Fusion' style with an 'Auto' color scheme unless explicitly overridden by user settings or system defaults.)*

## Functionality

### Core Features

-   **LLM Interaction:** Send user instructions along with conversation history and relevant KG context to a configured LLM API (Groq, Google) or a local emulator.
-   **Response Processing:** Receive and parse LLM responses, extracting structured data (entities, relations, Cypher queries) from the embedded XML.
-   **Narrative Display & Interaction:** Present the generated narrative text with visual highlighting for identified entities, relations, and queries. Allow users to click on highlighted elements to trigger inspection/curation workflows.
-   **Parsed Data Display:** Show the extracted entities, relations, and queries in a structured, readable format (e.g., JSON).
-   **Raw Output Display:** Provide access to the raw XML response received directly from the LLM for debugging and verification.
-   **Knowledge Graph Monitoring:** Display a summary of the connected Neo4j database state, including entity and relation counts (distinguishing between 'Pending' and 'Canon' status). Provide a button to open the Neo4j Browser.
-   **Configuration Management:** Load, view, edit, and save application settings via a JSON editor in the GUI.
-   **Prompt Template Management:** Load, view, edit, and save LLM prompt templates (`.txt` files) via a text editor in the GUI.
-   **Conversation History Management:** Automatically save user prompts and LLM raw XML responses as turns in a persistent session file. Allow loading, saving, editing, and deleting individual turns or ranges of turns. The history is the source of truth for KG reconstruction.
-   **Glossary Management:** Manage a separate XML file containing background lore, entity/relation definitions, and potentially "hidden" attributes, parsed and applied to the KG using the same logic as narrative XML. This content is included as context for the LLM.
-   **Curation Workflow:** Allow users to inspect parsed entities and relations (primarily from the narrative or glossary), change their curation status (`Pending`, `Canon`, `Ignored`), edit properties like canonical name or type, and apply these changes to the Knowledge Graph.
-   **Query Execution:** Display LLM-suggested Cypher queries and allow users to execute them, displaying the results.
-   **Theming and Styling:** Allow users to select different GUI styles (based on PySide6/Qt available styles) and color schemes (light/dark/auto).
-   **Font Size Adjustment:** Allow users to dynamically adjust the font size for text display and input areas.
-   **Persistent Settings:** Save window size, position, style, font size, and the current conversation history file path between sessions.

### User Interface

The main window utilizes horizontal and vertical splitters. Tab widgets organize content within panes.

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
|  | | +-------------------+ | | | | | QTextEdit (Logs/Hist)| |  |
|  | +-------------------------+ | | | +-------------------+ |  |
|  | | Config Editor Tab       | | | +---------------------+ |  |
|  | | +-------------------+ | | | +-------------------------+  |
|  | | | QTextEdit (JSON Ed) | | |                             |  |
|  | | +-------------------+ | | +-------------------------+  |
|  | +-------------------------+                             |
|  | | Glossary Editor Tab     |                             |
|  | | +-------------------+ |                             |
|  | | | QTextEdit (XML Ed)  |                             |
|  | | | Load/Save/Apply Btn |                             |
|  | | +-------------------+ |                             |
|  | +-------------------------+                             |
|  +-----------------------------+-------------------------+  |
|                                                             |
+-------------------------------------------------------------+
| Toolbar (Style, Theme, Font Size) | Status Bar            |
+-------------------------------------------------------------+
```

-   **Narrative Display (`QTextEdit`):** Read-only, displays narrative text. Highlight entities/relations/queries with distinct background colors and tooltips showing basic metadata. Clicking highlighted text triggers the Quick Curation or Query Execution panel.
-   **Monitor Tabs (`QTabWidget`):**
    -   **Parsed Data Tab (`QTextEdit`):** Read-only, displays `ParsedEntity`, `ParsedRelation`, `ParsedQuery` data as formatted JSON.
    -   **Raw XML Tab (`QTextEdit`):** Read-only, displays the full text response from LLM.
    -   **KG State Tab (`QTextEdit`):** Read-only, displays a summary of Neo4j counts (`:Entity` Total, `Pending`, `Canon`; Relationships Total). Button to open Neo4j Browser.
    -   **History Tab (`QTextEdit`):** Displays application logs (streaming from file) *and* a user-friendly representation of the Conversation History turns.
-   **Input Tabs (`QTabWidget`):**
    -   **User Instruction Tab (`QTextEdit`):** Editable input for the next prompt.
    -   **Prompt Templates Tab (`QTextEdit`):** Selector for `.txt` prompt files, editor for selected file, Save button.
    -   **Config Editor Tab (`QTextEdit`):** Editor for application configuration JSON, Save button.
    -   **Glossary Editor Tab (`QTextEdit`):** Editor for Glossary XML content. Buttons to Load, Save, and "Apply Glossary to KG".
-   **Quick Curation Panel:** A non-modal dialog or widget that appears when an entity or relation in the narrative is clicked. Displays details of the item, allows editing (Canonical name, type, attributes), and changing `status` (`Pending`, `Canon`, `Ignored`). Includes Apply/Save and Cancel buttons.
-   **Query Execution Panel:** A non-modal dialog or section that appears when a query in the narrative is clicked. Displays the query string and purpose. Includes an "Execute" button and displays query results.
-   **Toolbar:** Contains controls for Style, Theme (Light/Dark), and Font Size.
-   **Status Bar:** Displays current application status and brief messages.
-   **Menus:** "File" (New/Load/Save Session, Load/Save Config, Exit). "View" (Color Scheme, Style).

### User Flows

1.  **Initial Startup:** Load settings, load default config/prompts/session (or create defaults), attempt Neo4j connection, update KG state, show UI.
2.  **Sending Instruction to LLM:** User enters text, clicks "Send". Build prompt (system + query template + history XML + KG context XML + Glossary XML). Call LLM API. Parse response XML. Store response in `_last_parsed_response`. Update conversation history (add new turn). Trigger KG update from parsed data. Update UI panels (Narrative, Parsed, Raw XML, KG State). Update history display. Re-enable Send button.
3.  **Narrative Curation:** User clicks highlighted entity/relation in Narrative. Identify item from `_last_parsed_response`. Open Quick Curation Panel, populated with item data. User edits data (especially status). User clicks "Apply". Call `model.apply_curation_updates` with edited data. Update KG. Refresh KG State. Close panel.
4.  **Executing Query:** User clicks highlighted query in Narrative. Identify query from `_last_parsed_response`. Open Query Execution panel. User clicks "Execute". Call `model.execute_suggested_query`. Display results (e.g., in History/Log tab).
5.  **Managing Configuration:** User edits JSON in Config Editor. Clicks "Save Config". Validate JSON, save file, update internal config, reload prompts.
6.  **Managing Prompt Templates:** User selects file in selector, edits text. Clicks "Save Prompt Template". Save file.
7.  **Managing Glossary:** User uses Load/Save buttons to manage Glossary XML file. User edits XML in Glossary Editor. User clicks "Apply Glossary to KG". Parse Glossary XML, apply to KG using standard update logic. Update KG State.
8.  **Managing Conversation History:**
    -   User uses File menu to Load/Save sessions. Loading triggers KG Reconstruction.
    -   User interacts with History Display (UI TBD) to Edit or Delete turns. Editing/Deleting turns triggers KG Reconstruction from the modified history.
9.  **KG Reconstruction:** (Triggered by loading/editing history or applying glossary). Clear Neo4j DB (`DETACH DELETE`). Apply Glossary XML. Sequentially apply XML from each turn's LLM response. Update KG State.

### User Interaction with Narrative/Parsed Data and Curation Workflow

-   **Narrative Display Interaction:**
    -   `narrative_display` (`QTextEdit`) must capture mouse click events.
    -   The rendering process (`_render_narrative_with_highlighting`) must embed sufficient metadata (e.g., `data-xml-id`, `data-type`) as HTML attributes within `<span>` or `<code>` tags surrounding the highlighted text.
    -   The `eventFilter` for the narrative display must:
        -   Identify the position of the click (`event.pos()`).
        -   Map the click position to the underlying text and its associated HTML/XML metadata (this is the complex part, potentially requiring iterating through the `QTextDocument`'s elements/formats or using specialized QTextEdit subclasses).
        -   Extract the `data-type` (`entity`, `relation`, `query`) and `data-xml-id` (or other relevant identifier) from the clicked element's metadata.
        -   If `data-type` is `entity` or `relation`: Find the corresponding `ParsedEntity` or `ParsedRelation` object in the stored `_last_parsed_response.entities` or `_last_parsed_response.relations` lists using the extracted ID. Call a method (`_show_quick_curation_for_item`) passing the item's data.
        -   If `data-type` is `query`: Find the corresponding `ParsedQuery` object. Call a method (`_show_query_execution_panel`) passing the query data.
-   **Quick Curation Panel:**
    -   A separate UI element (dialog/widget) displays details and allows editing key fields (`canonical`, `entity_type`, `relation_type`) and `status` (`Pending`, `Canon`, `Ignored`).
    -   "Apply" button triggers `model.apply_curation_updates` with the modified data for the specific item.
-   **Query Execution Panel:**
    -   A separate UI element displays the query string and purpose.
    -   "Execute" button calls `model.execute_suggested_query`.
    -   Displays results returned by the model method.
-   **Applying Curation Updates:** `model.apply_curation_updates` receives edited item data, constructs and executes Cypher queries to update the KG (setting status, provenance, properties). User-set 'Canon' status must not be overwritten by subsequent 'Pending' suggestions from LLM output processing (`update_knowledge_graph`).

### Glossary Management

-   **Content:** Glossary file contains XML with `<entity>` and `<relation>` tags (and attributes, including "hidden" ones like `gender`, `agenda`), possibly within a custom root tag like `<glossary>`.
-   **UI:** Dedicated "Glossary Editor" tab with text editor, Load/Save file buttons, and an "Apply Glossary to KG" button.
-   **Processing:** "Apply Glossary to KG" triggers `model.process_glossary_xml`. This method parses the glossary XML using the *same* core parsing logic as LLM responses (`parse_llm_response_xml` must be adaptable or reusable) to extract entities/relations. It then applies these to the KG using the *same* standard update/merge logic (`update_knowledge_graph`).
-   **LLM Context:** The *raw XML content* of the currently loaded Glossary file is included in the LLM prompt within a `<glossary_context>[Raw Glossary XML Content]</glossary_context>` tag.

### Conversation History Management

-   **Persistence:** Store history in a single XML file (`.ses` or `.chf`). Structure: `<session><turn timestamp="...">...</turn></session>`. Turn content includes `<user_prompt>` and `<llm_response_raw_xml>`.
-   **UI:**
    -   File menu actions: "New Session", "Load Session...", "Save Session", "Save Session As...".
    -   History tab displays history (user-friendly view).
    -   UI controls (context menu/buttons) on history items to "Edit Turn" (dialog for raw XML/prompt editing) and "Delete Turn and Subsequent".
-   **Model Logic:**
    -   `self._conversation_turns: List[Dict]` stores history internally.
    -   Methods: `load_session`, `save_session`, `add_turn`, `edit_turn`, `delete_turns_from`.
    -   `get_conversation_history_xml`: Formats internal turns into `<conversation_history>...</conversation_history>` XML for prompt. Must include current user prompt and KG context within the last turn's prompt structure. Must handle `max_tokens` by removing oldest turns from the *history XML generated for the prompt* (internal history state remains full).
-   **KG Reconstruction:** `model.rebuild_kg_from_history(glossary_xml_content)` method: Clears KG, applies Glossary XML, then sequentially applies XML from each turn's `<llm_response_raw_xml>`.

### Error Handling and Feedback

-   **Logging:** Use `log_info`, `log_warning`, `log_error` to `history.log` and console.
-   **Status Bar:** Short, transient status messages. Brief error summaries.
-   **Message Boxes (`QMessageBox`):** For critical errors (connection, file not found, invalid format) and required user confirmations.
-   **History/Log Display:** `history_display` streams live content of `history.log`. Auto-scroll.
-   **Specific Error Handling:**
    -   **LLM API Errors:** Catch exceptions, log details, status bar summary. Message box for critical failures (API key, persistent connection issue). Explicitly handle Google quota/rate limit errors as identified in documentation: log, status bar, potential message box warning user.
    -   **XML Parsing Errors:** Log `XMLSyntaxError` and raw text. Status bar/message box warning. Narrative/Parsed Data show error message. Raw XML shows full response.
    -   **Neo4j Errors:** Log details. KG State tab shows connection status/errors. Message box for connection failure on startup or crucial operations. Log Cypher errors.
    -   **File I/O Errors:** Message boxes for user-triggered file operation errors. Log details.
    -   **Validation Errors:** Message boxes for invalid JSON/XML format when saving from editors. Log details.

### Technical Implementation

### Architecture

Monolithic, clear separation: `main.py` (GUI, event handling, display, input) and `model.py` (Core logic: LLM/Neo4j interaction, data parsing/management, history/config/prompt handling, core curation/update logic). Communication via method calls and signals (where asynchronous updates are needed, e.g., model signalling status/errors to UI).

### Data Structures

-   `ParsedEntity`: `{ xml_id: str, text_span: str, canonical: str, entity_type: Optional[str], status: str, provenance: List[str], attributes: Dict[str, Any] }` - `attributes` stores lore/hidden details from narrative/Glossary.
-   `ParsedRelation`: `{ xml_id: str, text_span: str, relation_type: str, subj_id: str, obj_id: str, status: str, provenance: List[str], attributes: Dict[str, Any] }` - `attributes` stores details.
-   `ParsedQuery`: `{ xml_id: str, purpose: str, query_string: str }`
-   `LLMResponseParsed`: `{ raw_xml: str, narrative_xml_element: lxml.etree.Element, entities: List[ParsedEntity], relations: List[ParsedRelation], queries: List[ParsedQuery], raw_response_json: Dict[str, Any] }`
-   Neo4j Data Model: Primarily `(:Entity)` nodes and dynamic relationships. Nodes/Relationships have properties `name` (canonical, indexed), `xml_ref`, `entityType`, `status` (`Pending`, `Canon`, `Ignored`), `provenance`, `attributes`, `text_spans`. `MERGE` logic handles status and provenance updates.

### Algorithms

-   **Prompt Building:** Concatenate system prompt, query template (formatted with `user_instruction`, `<conversation_history>`, `<kg_context>`, `<glossary_context>` XML).
-   **LLM API Call:** Abstracted via `call_llm_api`. Implementation logic detailed in "LLM API Integration".
-   **XML Parsing:** Use `lxml.etree.fromstring` (with recovery consideration) to parse raw LLM XML and Glossary XML. Traverse tree to extract data from `<entity>`, `<relation>`, `<query>` tags and their attributes/content. Handle missing/malformed elements gracefully with logging.
-   **Narrative Rendering:** Convert `<narrative>` XML element (lxml object) into HTML for `QTextEdit` using `lxml` traversal and formatting with HTML tags (`<span>`, `<code>`) and `data-*` attributes for metadata.
-   **Knowledge Graph Update (`update_knowledge_graph`):** Use Cypher `MERGE` statements in a transaction. Entities merged by `name` (canonical). Relations merged by subject, object (canonical names), and type. `ON CREATE` sets initial properties. `ON MATCH` updates provenance, merges attributes, and conditionally updates status (only if KG status is 'Pending').
-   **Applying Curation (`apply_curation_updates`):** Construct Cypher queries to update status, provenance, and properties of specific entities/relations identified by their canonical name/ID based on user edits. This explicitly sets the status, potentially overriding 'Pending' in the KG.
-   **KG Context Retrieval (`retrieve_kg_context_for_prompt`):** Query Neo4j (read-only Cypher) based on trigger (user selected entity). Retrieve selected entity and 1-hop 'Canon' neighbors/relations. Format into `<kg_context>` XML.
-   **KG Reconstruction (`rebuild_kg_from_history`):** Execute `DETACH DELETE` Cypher. Parse and apply Glossary XML. Iterate history turns, parse each LLM response XML, apply to KG.

### Dependencies

-   `PySide6`: For the graphical user interface.
-   `requests`: For making HTTP requests (primarily for API calls if not using SDKs directly, or other web interactions).
-   `neo4j`: For connecting to and interacting with the Neo4j database.
-   `python-dotenv`: For loading environment variables.
-   `lxml`: For efficient and robust XML parsing.
-   `groq`: Python SDK for interacting with the Groq API.
-   `google-genai`: Python SDK for interacting with the Google Gemini API.

These dependencies should be listed in `requirements.txt`.

### Configuration

Relies on a JSON config file (default: `approach/default.json`) and environment variables (`.env`).
-   JSON config: `api` (name, model, params), `format` (XML details), `prompts` (file paths).
-   Environment variables: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GROQ_API_KEY`, `GOOGLE_API_KEY`.

### Expected LLM XML Structure

Expected output contains an XML document within the text response. Root tag is configurable (default `<response>`). Narrative text is within a `<narrative>` tag. Entities (`<entity>`), relations (`<relation>`), and queries (`<query>`) are embedded within or alongside the narrative.
-   `<entity id="..." canonical="..." type="..." status="..." attribute1="..." >text_span</entity>`
-   `<relation type="..." subj="..." obj="..." status="..." attribute1="..." >text_span</relation>`
-   `<query purpose="..." >query_string</query>`

### LLM API Integration

`call_llm_api` handles calls to `groq`, `google`, or `emulator`.
-   **API Selection:** Based on `api_name` in config.
-   **API Keys:** Read from `GROQ_API_KEY`, `GOOGLE_API_KEY` environment variables.
-   **Model Listing:** `fetch_available_models` method: uses SDKs for `groq` (excluding 'whisper', 'playai'), `google`, or returns dummy names for `emulator`.
-   **Groq:** Use `groq` SDK, standard OpenAI-compatible endpoint/payload, `Authorization: Bearer`.
-   **Google:** Use `google-genai` SDK, appropriate endpoint/payload, API key in URL query param. Implement basic timestamp rate limit *before* calling. **Crucially, detect and handle quota/rate limit errors based on Google API error codes/messages from SDK exceptions/responses**, logging and informing the user to avoid excessive calls.
-   **Emulator:** Read full content from numbered XML files (e.g., `emulator_responses/response_N.xml`), return a dictionary mimicking a real API response structure containing the XML. Cycle or stop if files run out.
-   **Return:** Raw API/emulator response dictionary, or `None` on failure/rate limit.

### Knowledge Graph Context Retrieval

`retrieve_kg_context_for_prompt` generates `<kg_context>`.
-   **Trigger:** User-initiated selection of a 'Canon' entity in the UI (future implementation).
-   **Relevance:** Retrieve the selected 'Canon' entity and its direct (1-hop) 'Canon' neighbors and connecting 'Canon' relations from Neo4j.
-   **Querying:** Use read-only Cypher queries (see `neo4j_query_help.md`).
-   **XML Format:** `<kg_context><entity id="..." canonical="..." .../><relation .../>...</kg_context>`. Use canonical name as `id` for entities within this context XML.
-   **Integration:** Included in the LLM prompt.
-   **Initial State:** Returns empty `<kg_context></kg_context>` if no entity is selected (or on first turn).

## Style Guide

Uses PySide6/Qt styling. Supports various platform styles (`Fusion` default). Explicit Light/Dark color scheme toggle using `Qt.ColorScheme`. User-adjustable font size applied application-wide. Narrative highlighting uses distinct background colors.

## Testing Scenarios

-   Load/Save config/prompts/session files.
-   Send instruction to LLM (Groq, Google, Emulator). Verify response panels (Narrative, Parsed, Raw XML, KG State).
-   Enter bad instruction (empty input).
-   Enter instruction for malformed XML output (test parsing robustness).
-   Click highlighted entity/relation. Verify Quick Curation Panel opens with correct data. Edit status/properties, apply. Verify KG State updates and changes persist in Neo4j.
-   Click highlighted query. Verify Query Execution panel opens. Execute query. View results.
-   Manage Glossary: Load, edit, save. Apply Glossary to KG. Verify KG State updates.
-   Manage History: Save current session. Load saved session. Verify history display, UI panels, and KG State after reconstruction. Edit a turn, delete turns. Reconstruct KG from modified history.
-   Test theme/style/font size changes and persistence.
-   Test Neo4j connection (with/without Neo4j running), verify error handling and KG State display.
-   Test LLM API key missing/invalid scenario.
-   Test Google API rate limit scenario (internal and API-signaled).

## Accessibility Requirements

Basic keyboard navigation for standard widgets. Respects system font scaling via application font size setting. Standard PySide6 widgets offer basic screen reader compatibility, though specific testing and potential ARIA attributes might be needed for complex custom interactions (like narrative clicking). High contrast depends on selected style and theme.

## Performance Goals

Responsive UI during LLM calls (backgrounding). Efficient parsing and display updates. Fast Neo4j interaction, especially for KG updates/reconstruction (Cypher `MERGE` and `UNWIND` needed). Handle large text in editors.

## Extended Features

-   Implement full `google-genai` SDK integration beyond basic text generation and error handling (e.g., model details, safety settings).
-   Refine prompt building XML structure based on LLM testing for optimal context utilization.
-   Implement dynamic KG Context retrieval triggered by automated analysis of user input/LLM output (beyond simple 1-hop neighbor on click).
-   Refine Narrative HTML rendering for more robust click mapping (mapping pixel position to XML element).
-   Implement "Ignored" status in KG update/curation logic.
-   Implement advanced Quick Curation features (e.g., merging entities, creating aliases, detailed attribute editing).
-   Implement manual triplet creation via narrative text selection.
-   Add more sophisticated Query Execution results viewer.
-   Implement token counting for conversation history to inform the user when truncation might occur for the prompt.
-   Add option to store history/glossary directly in Neo4j instead of files (alternative persistence).
-   Implement user authentication for Neo4j and potentially LLM APIs if needed for deployment scenarios.