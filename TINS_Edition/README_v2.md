<!-- Zero Source Specification v1.0 -->
<!-- ZS:COMPLEXITY:HIGH -->
<!-- ZS:PRIORITY:HIGH -->
<!-- ZS:PLATFORM:DESKTOP (Python/PySide6) -->
<!-- ZS:LANGUAGE:PYTHON -->

# Zero Source AI-Assisted Narrative & KG Curation Tool

## Description

This project specifies a desktop application designed to assist users in writing narrative content and simultaneously curating a Knowledge Graph (KG) based on that narrative. Unlike traditional approaches that rely heavily on NLP to extract structured data from arbitrary text, this tool leverages Large Language Models (LLMs) to generate narrative content that is *already enriched* with structured metadata (in XML format) directly usable for KG updates. The application provides a user interface focused on fast, visual curation of this enriched narrative and its corresponding KG updates, minimizing friction in the user's workflow. This XML output serves as the definitive 'source code' for both the narrative and the associated Knowledge Graph state, facilitating version control and future portability beyond a specific Neo4j (mentioned later) database dump.

The core hypothesis is that by having the LLM embed the KG structure within the narrative itself, and by providing a fluid point-and-click interface for human review and approval, the process of building a curated KG alongside creative writing becomes significantly more efficient and less burdensome than purely manual methods or methods relying on complex, potentially unreliable, NLP inference from raw text.

## Functionality

### Core Workflow

The user workflow revolves around iterative generation and curation:
1.  The user initiates a new narrative section or continues an existing one, potentially providing context or desired facts/triplets in a prompt.
2.  The tool sends a structured prompt (including previous conversation history, KG context, and user instructions) to the configured LLM.
3.  The LLM responds with narrative content embedded within an XML structure, containing explicit tags for entities, relations, metadata (canonical forms, types, status, provenance), and potentially suggested KG queries.
4.  The application parses the XML response and renders the narrative visually in a dedicated UI panel, using highlighting and interactive elements based on the XML tags and their attributes.
5.  The user reads the rendered narrative. As they read, they can interact with the highlighted elements (entities, relations) via mouse hover (to see suggested metadata like canonical form, type, provenance, status) and clicks (to open a quick-curation panel).
6.  Via the quick-curation panel, the user approves, rejects, or edits the metadata suggested by the LLM for entities and relations. This includes confirming canonical forms, assigning status (e.g., 'Canon', 'NeedsReview'), and managing provenance. They can also add entities/relations manually via a structured input if the LLM missed something.
7.  Based on the user's approved/edited data, the application updates the central Neo4j Knowledge Graph, ensuring accurate tracking of provenance and status for each node and relationship.
8.  The application logs the entire interaction (prompts, raw XML response, user curation actions, KG updates) for historical tracking and potential future analysis.
9.  Optionally, if the LLM suggested a KG query in the XML, the user can choose to execute it. The result of this query can be stored and included in future prompts to the LLM, allowing the LLM to leverage existing KG information for better generation.

### User Interface (PySide6)

The application will feature a multi-panel desktop interface designed for efficiency and clarity, prioritizing the user's reading and curation flow. Key UI panels include:
-   **Narrative Panel:** Displays the LLM-generated narrative parsed from the XML, with rich visual highlighting and interactive elements (mouseover, click) based on the XML tags (`<entity>`, `<relation>`, etc.) and their attributes (e.g., color coding by status/provenance, tooltips for metadata). Users can read and interact directly with the text here.
-   **Prompt/Context Panel:** Allows the user to view and edit the prompt being sent to the LLM, including user instructions, potentially previous conversation turns (in XML), and relevant KG context (`<kg>...</kg>` tags). Prompts for system and query will likely be edited via referencing external `.txt` files for multi-line content.
-   **Quick Curation Panel:** A transient or sidebar panel that appears when the user clicks on an entity or relation highlight in the Narrative Panel. Provides input fields and buttons to easily view, approve, reject, map canonical forms, assign status, and edit properties for the selected KG element.
-   **KG State/Review Panel:** Displays lists or a simplified view of the entities and relations currently in the Neo4j KG, potentially filterable by status, provenance, or type. This is for reviewing the curated KG, not the primary interface for initial curation. (Full KG exploration uses Neo4j Browser).
-   **History/Log Panel:** Shows the chronological log of interactions, including prompts sent, raw LLM responses (full XML), and user curation decisions.
-   **Query Panel:** Displays KG queries suggested by the LLM (`<query>`). Allows the user to execute them against Neo4j and potentially view results. Provides an interface to select/manage which KG query results should be added to the context for the *next* LLM prompt (`<kg>...</kg>`).

### Configuration Management

Configuration (LLM API, model, parameters, XML structure markers/tags, prompt file paths) is managed via a JSON file (e.g., `approach/default.json`), editable both manually in a text editor and via a dedicated configuration section within the PySide6 UI. API keys are loaded from `.env` variables.

### Knowledge Graph (Neo4j) Data Model

The Neo4j database serves as the central, single source of truth for the curated knowledge and glosario. The data model incorporates provenance and status tracking:
-   **Nodes:** Primary label `:Entity`. Additional labels for specific types or roles (`:Canonical`, `:Alias`, `:Person`, `:Place`, etc.). Properties include `name` (the canonical string), `status` (e.g., 'Pending', 'Canon', 'Rejected', 'NeedsReview'), `provenance` (list of sources, e.g., ['LLM_XML_Generated', 'User_Approved_T5', 'Glosario']), `canonical_ref` (relationship or property linking `:Alias` nodes to their `:Canonical` counterpart), `xml_ref` (reference to the location/ID in the source XML narrative).
-   **Relationships:** Types reflect canonical relations in the domain (e.g., `:LIVES_IN`, `:MET`, `:IS_A`). Properties include `status`, `provenance`, `confidence` (e.g., from LLM's implied confidence or user's rating), `xml_ref`.
-   **Glosario:** Integrated directly into the KG. `:Canonical` entities/relations form the core. `:Alias` nodes linked via `:HAS_ALIAS` relationships represent variations (including potential ASR errors in the future) that normalize to a canonical term. Provenance tracks if an alias/mapping came from user input vs. auto-suggestion.

### LLM Conversation Format

The conversation history maintained by the application (and potentially passed to the LLM for context) is structured in a consistent XML format. Each turn includes the user's prompt (potentially with `<kg>` context) and the LLM's complete XML response (`<response>` containing `<narrative>` and optional `<query>`).

## Technical Implementation

### Architecture

A desktop application built with PySide6. It interacts with:
-   Local Neo4j database via the Python `neo4j` driver.
-   External LLM APIs (Groq, Google) via the `requests` library.
-   Local file system for configurations (JSON) and prompt content (TXT).
-   Uses standard Python libraries for XML parsing (`lxml` recommended for robustness) and data handling.

### Algorithms

-   **LLM Interaction:** Constructing complex XML prompts (including KG context), sending via HTTP.
-   **XML Parsing:** Robustly parsing the LLM's XML response to extract narrative segments, entities, relations, attributes, and queries. Handling potential malformed XML.
-   **UI Rendering:** Iterating over parsed XML elements to display narrative text in a rich text format with interactive highlighting and tooltips in the PySide6 UI.
-   **Event Handling:** Capturing user clicks/interactions on UI elements (especially on the highlighted narrative text) and linking them to the underlying parsed XML data.
-   **Quick Curation Logic:** Presenting extracted/suggested metadata in a user-friendly panel, capturing user edits/approvals.
-   **KG Update Logic:** Translating user-approved/edited metadata from the parsed XML structure into Cypher queries (MERGE, SET) that update nodes, relations, properties, labels (`status`, `provenance`, `canonical_ref`, `xml_ref`) in Neo4j, handling existing data appropriately.
-   **Query Execution (Optional):** Running suggested Cypher queries against Neo4j via the driver.
-   **Context Management:** Formatting relevant parts of the KG or query results into `<kg>...</kg>` XML tags for inclusion in subsequent prompts to the LLM.
-   **API call Optimization:** The core process for identifying entities and relations for KG updates relies exclusively on parsing the structured data embedded within the LLM's XML output, intentionally bypassing traditional NLP-based extraction techniques from raw text in favor of trusting the LLM's explicit structural output.

### External Integrations

-   **Neo4j Database:** Local instance, accessed via Bolt protocol using `neo4j` driver.
-   **Groq API / Google Generative AI API:** Accessed via HTTP requests using `requests`. API keys from `.env`.
-   **PySide6 (or PyQt):** Framework for building the desktop GUI.
-   **`lxml`:** Recommended library for robust XML parsing and manipulation in Python.
-   **`json`:** Standard library for loading/saving JSON configuration.
-   **`python-dotenv`:** For environment variable management.

### UI Design Goals

-   Prioritize user workflow for reading and quick curation over comprehensive KG exploration.
-   Implement visual style resembling desktop applications with support for dark mode (e.g., using Fusion style and palette handling in Qt).
-   Use color coding and visual cues (highlighting, underlines, icons) to clearly indicate the status and provenance of suggested KG elements in the narrative text.
-   Design the Quick Curation Panel for minimal clicks and rapid data entry/selection for common tasks (approving, mapping canonicals).
-   Ensure responsiveness and fluid interaction even with longer narratives.

### Testing Scenarios

Initial tests will focus on core technical integrations and UI elements:
1.  UI Launch and rendering basic window structure.
2.  Loading and parsing JSON configuration.
3.  Loading prompt content from `.txt` files.
4.  Calling LLM API and receiving a raw XML response.
5.  Robustly parsing the LLM's XML response (even with minor errors).
6.  Rendering the parsed XML narrative in the UI with highlighting/mouseover working.
7.  Basic Neo4j connection and update (adding a simple node/relation manually via code).
Later tests will cover the core workflow:
8.  Clicking highlighted narrative elements opens Quick Curation panel.
9.  Approving an entity/relation via the panel correctly updates its status/provenance in the internal representation and Neo4j.
10. Mapping an alias to a canonical entity works correctly in internal data and Neo4j.
11. Manual addition of entities/relations via UI works.
12. Suggested KG query from LLM is displayed and executable.
13. KG context is correctly formatted and included in the next prompt.
14. Handling LLM errors, parsing errors, and Neo4j errors gracefully in the UI.
15. Persistence of configuration and logging.

### Logging

Comprehensive logging to `history.log` is maintained, including:
-   Timestamped entries for each interaction.
-   Full prompt XML sent to LLM.
-   Full raw XML response received from LLM.
-   Details of user curation actions (what was approved, rejected, edited).
-   Details of all KG updates (Cypher queries executed, results).
-   Errors and warnings.

## Getting Started

1.  Ensure Python 3.12.9 and a virtual environment are set up.
2.  Clone the repository.
3.  Install dependencies (`pip install -r requirements.txt`, including PySide6, requests, python-dotenv, lxml, neo4j).
4.  Install and run Neo4j Desktop. Configure credentials in a local `.env` file.
5.  Create `approach/` directory and a config file (e.g., `default.json`). Create `prompts/` and sample `.txt` files.
6.  Obtain LLM API keys (Groq, Google) and add to `.env`.
7.  Run the main Python script.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contributing

Contributions are welcome, particularly focused on refining the XML schema for LLM interaction, improving the robustness of XML parsing, enhancing the PySide6 UI for intuitive curation, and perfecting the KG update logic with provenance/status tracking.

## Implementation Notes (For the LLM generating code)

- The main application logic will reside in a PySide6 application class, not a linear script like `coreYYYYMMDD.py`.
- The `main.py` can remain a simple launcher or be integrated into the PySide6 app structure.
- Focus development iteratively: first, a basic UI window; then, LLM call & XML parsing; then, rendering narrative with basic highlighting; then, interactive curation panels and KG updates.
- The robustness of the XML parsing is critical. `lxml` is recommended for better error handling of potentially imperfect LLM output.
- The design of the XML schema (element names, attributes) is flexible but should be detailed in the JSON config and strictly followed in prompts to train the LLM.