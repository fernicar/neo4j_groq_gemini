# main.py
import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# Import PySide6 Components from best_gui.txt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QTabWidget, QSplitter, QMenuBar, QToolBar, QFileDialog,
    QMessageBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QSizePolicy,
    QDialog, QDialogButtonBox, QFormLayout, QStyleFactory, QStatusBar, QGroupBox,
    QRadioButton, QCheckBox, QToolButton, QCommandLinkButton, QDateTimeEdit,
    QSlider, QScrollBar, QDial, QProgressBar, QGridLayout, QMenu, QInputDialog
)
from PySide6.QtGui import QAction, QKeySequence, QTextCursor, QShortcut, QColor, QTextCharFormat, QBrush, QPalette
from PySide6.QtCore import Qt, Slot, QSize, QSettings, QFile, QTextStream, QDateTime, QTimer, QRect, QEvent

# Import the Model logic
from model import CurationModel, LLMResponseParsed, ParsedEntity, ParsedRelation, ParsedQuery, log_error, log_info, log_warning
from lxml import etree # Need lxml for generating HTML from parsed XML elements

# --- Constants ---
APP_NAME = "AI-Assisted KG Curation"
APP_VERSION = "1.0.0" # Updated version
SETTINGS_ORG = "YourOrgName" # Replace with your organization name
SETTINGS_APP = "AICurationTool" # Replace with your app name
DEFAULT_WINDOW_SIZE = QSize(1200, 800)
DEFAULT_FONT_SIZE = 11
RESOURCES_DIR = Path("resources") # Not strictly used for QSS anymore, but keep for potential
CONFIG_DIR = Path("approach") # Directory for JSON configs
PROMPTS_DIR = Path("prompts") # Directory for TXT prompts
DEFAULT_CONFIG_NAME = "default.json"
DEFAULT_PROMPT_SYSTEM = "default_system.txt"
DEFAULT_PROMPT_QUERY = "default_query.txt"


# Styles available based on your best_gui.txt
STYLE_THEMES = ['windows11', 'windowsvista', 'Windows', 'Fusion']
STYLE_SELECTED_THEME = STYLE_THEMES[3]  # Fusion style recommended for consistency
COLOR_SCHEMES = ['Auto', 'Light', 'Dark']
# Qt.ColorScheme maps to these indices: Unknown (Auto)=0, Light=1, Dark=2

# --- Main Application Window (View/Controller) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        # Ensure config and prompts directories exist
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        RESOURCES_DIR.mkdir(parents=True, exist_ok=True) # Keep resources dir

        # Create default config and prompt files if they don't exist (basic placeholders)
        default_config_path = CONFIG_DIR / DEFAULT_CONFIG_NAME
        default_system_prompt_path = PROMPTS_DIR / DEFAULT_PROMPT_SYSTEM
        default_query_prompt_path = PROMPTS_DIR / DEFAULT_PROMPT_QUERY

        if not default_config_path.exists():
             # Minimalist default config JSON structure
             default_config_data = {
                 "api": {
                     "api_name": "groq",
                     "model_name": "Allam-2-7b",
                     "temperature": 0.7,
                     "max_tokens": 1024 # Or match your model's default
                 },
                 "format": {
                     "separator": "###END_XML_RESPONSE###", # Updated separator convention
                     "response_root_tag": "response" # Expecting <response> as root
                 },
                 "prompts": {
                     "system_prompt_file": str(default_system_prompt_path),
                     "query_prompt_file": str(default_query_prompt_path)
                 }
             }
             try:
                 with open(default_config_path, 'w', encoding='utf-8') as f:
                     json.dump(default_config_data, f, indent=2)
                 logging.info(f"Created default config file: {default_config_path}")
             except Exception as e:
                 logging.error(f"Failed to create default config file: {e}")


        if not default_system_prompt_path.exists():
             try:
                 with open(default_system_prompt_path, 'w', encoding='utf-8') as f:
                     f.write("You are a helpful AI assistant that generates narrative enriched with XML tags for knowledge graph curation. Respond only with XML.") # Basic system prompt
                 logging.info(f"Created default system prompt file: {default_system_prompt_path}")
             except Exception as e:
                 logging.error(f"Failed to create default system prompt file: {e}")

        if not default_query_prompt_path.exists():
            # Example query prompt structure using placeholders.
            # The actual structure will need careful design based on LLM capability.
            # Placeholders {user_instruction}, {conversation_history}, {kg_context}
            default_query_content = """
            Generate the next part of the narrative based on the following instructions and context.
            Embed entities, relations, and metadata within the narrative using XML tags like <entity>, <relation>, etc.
            The entire response should be a single XML document.

            User Instruction: {user_instruction}

            <conversation_history>
            {conversation_history}
            </conversation_history>

            <kg_context>
            {kg_context}
            </kg_context>

            Generate the narrative section within a <narrative> tag inside a <response> root tag.
            Include entity tags (<entity id="ent1" canonical="..." type="..." status="Pending">text</entity>)
            and relation tags (<relation type="..." subj="ent1" obj="ent2" status="Pending">text</relation>)
            directly in the narrative flow.
            Optionally include <query> tags for suggested Cypher queries.

            <response>
              <narrative>
                <!-- Your generated narrative here with XML tags -->
              </narrative>
              <!-- Optional: <query purpose="...">...</query> -->
            </response>
            """
            try:
                with open(default_query_prompt_path, 'w', encoding='utf-8') as f:
                    f.write(default_query_content)
                logging.info(f"Created default query prompt file: {default_query_prompt_path}")
            except Exception as e:
                 logging.error(f"Failed to create default query prompt file: {e}")


        # Initialize the Model layer
        self.model = CurationModel(config_path=str(default_config_path)) # Use default config path

        self._init_ui()
        self._load_settings()
        self._apply_current_theme()

        # Try connecting to Neo4j on startup
        self.model.connect_neo4j()

        # Placeholder for the last parsed response
        self._last_parsed_response: Optional[LLMResponseParsed] = None
        self._conversation_history_xml: str = "" # Placeholder for conversation history

    def _init_ui(self):
        """Creates the user interface elements."""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setGeometry(100, 100, DEFAULT_WINDOW_SIZE.width(), DEFAULT_WINDOW_SIZE.height())

        # --- Central Widget & Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Menu Bar ---
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")
        # Add Open/Save Config Actions
        load_config_action = QAction("&Load Config...", self)
        load_config_action.triggered.connect(self._load_config_dialog)
        file_menu.addAction(load_config_action)

        save_config_action = QAction("&Save Config...", self)
        save_config_action.triggered.connect(self._save_config_dialog)
        file_menu.addAction(save_config_action)
        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menu_bar.addMenu("&View")
        # Color Scheme Submenu (using methods from best_gui.txt)
        color_scheme_menu = view_menu.addMenu("&Color Scheme")
        self.color_scheme_actions = []
        for i, scheme_name in enumerate(COLOR_SCHEMES):
             action = QAction(scheme_name, self)
             action.setCheckable(True)
             action.setData(i) # Qt.ColorScheme enum value
             action.triggered.connect(lambda checked, idx=i: self._on_color_scheme_selected(checked, force_index=idx))
             color_scheme_menu.addAction(action)
             self.color_scheme_actions.append(action)

        # Style Submenu (similar to best_gui.txt toolbar)
        style_menu = view_menu.addMenu("&Style")
        self.style_actions = []
        for style_name in QStyleFactory.keys():
            action = QAction(style_name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, name=style_name: self._on_style_selected(checked, style_name=name))
            style_menu.addAction(action)
            self.style_actions.append(action)


        # Theme Submenu (simplified from best_gui.txt - focus on color schemes)
        # Removed custom QSS loading from menu to simplify, keep toggle button.
        # If custom themes are needed, add back _load_custom_qss etc.

        # --- Main Horizontal Splitter (Top/Bottom Panes) ---
        main_splitter_h = QSplitter(Qt.Vertical)
        main_layout.addWidget(main_splitter_h, 1)

        # --- Top Pane (Display Area - Narrative & Info) ---
        top_pane_widget = QWidget()
        top_pane_layout = QHBoxLayout(top_pane_widget)
        top_pane_layout.setContentsMargins(0, 0, 0, 0)
        display_splitter_v = QSplitter(Qt.Horizontal)
        top_pane_layout.addWidget(display_splitter_v)
        main_splitter_h.addWidget(top_pane_widget)

        # --- Left Display (Narrative) ---
        left_display_widget = QWidget()
        left_display_layout = QVBoxLayout(left_display_widget)
        narrative_label = QLabel("Narrative:")
        left_display_layout.addWidget(narrative_label)
        self.narrative_display = QTextEdit() # Use QTextEdit for rich text/HTML display
        self.narrative_display.setReadOnly(True)
        self.narrative_display.setPlaceholderText("Generated narrative will appear here with highlighting...")
        self.narrative_display.setAcceptRichText(True) # Ensure it accepts rich text/HTML
        self.narrative_display.textChanged.connect(self._on_narrative_text_changed) # Connect for click tracking
        self.narrative_display.viewport().installEventFilter(self) # Install event filter for clicks
        left_display_layout.addWidget(self.narrative_display)
        display_splitter_v.addWidget(left_display_widget)

        # --- Right Display (Monitor Tabs) ---
        right_display_widget = QWidget()
        right_display_layout = QVBoxLayout(right_display_widget)
        self.monitor_tabs = QTabWidget()
        right_display_layout.addWidget(self.monitor_tabs)
        display_splitter_v.addWidget(right_display_widget)

        # Parsed Data Tab (Entities, Relations, Queries)
        parsed_tab = QWidget()
        parsed_layout = QVBoxLayout(parsed_tab)
        self.parsed_display = QTextEdit() # Show structured parsed data here (JSON or text)
        self.parsed_display.setReadOnly(True)
        self.parsed_display.setPlaceholderText("Parsed entities, relations, and queries from LLM response...")
        parsed_layout.addWidget(self.parsed_display)
        self.monitor_tabs.addTab(parsed_tab, "Parsed Data")

        # Raw XML Tab
        raw_xml_tab = QWidget()
        raw_xml_layout = QVBoxLayout(raw_xml_tab)
        self.raw_xml_display = QTextEdit() # Show raw XML response here
        self.raw_xml_display.setReadOnly(True)
        self.raw_xml_display.setPlaceholderText("Raw XML response from LLM...")
        raw_xml_layout.addWidget(self.raw_xml_display)
        self.monitor_tabs.addTab(raw_xml_tab, "Raw XML")

        # KG State/Review Tab (Simplified - can use Neo4j Browser for full review)
        kg_state_tab = QWidget()
        kg_state_layout = QVBoxLayout(kg_state_tab)
        self.kg_state_display = QTextEdit() # Show simple summary of KG state
        self.kg_state_display.setReadOnly(True)
        self.kg_state_display.setPlaceholderText("Summary of Knowledge Graph state (e.g., entity counts)...")
        kg_state_layout.addWidget(self.kg_state_display)
        # Add a button to open Neo4j Browser (needs external logic/instructions)
        open_neo4j_browser_button = QPushButton("Open Neo4j Browser")
        open_neo4j_browser_button.clicked.connect(self._open_neo4j_browser) # TODO: Implement this slot
        kg_state_layout.addWidget(open_neo4j_browser_button, alignment=Qt.AlignRight)
        self.monitor_tabs.addTab(kg_state_tab, "KG State")

        # History/Log Tab (Use the logger file)
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        self.history_display = QTextEdit() # Display log file content
        self.history_display.setReadOnly(True)
        self.history_display.setPlaceholderText("Application history and logs...")
        history_layout.addWidget(self.history_display)
        self.monitor_tabs.addTab(history_tab, "History")

        # --- Bottom Pane (Input Area Tabs) ---
        bottom_pane_widget = QWidget()
        bottom_pane_layout = QVBoxLayout(bottom_pane_widget)
        bottom_pane_layout.setContentsMargins(0, 5, 0, 0)
        self.input_tabs = QTabWidget()
        bottom_pane_layout.addWidget(self.input_tabs)
        main_splitter_h.addWidget(bottom_pane_widget)

        # User Input Tab
        user_input_tab = QWidget()
        user_input_layout = QVBoxLayout(user_input_tab)
        self.user_instruction_input = QTextEdit()
        self.user_instruction_input.setPlaceholderText("Type your instructions/prompt for the LLM here...")
        user_input_layout.addWidget(self.user_instruction_input)
        self.input_tabs.addTab(user_input_tab, "User Instruction")

        # Prompt Templates Tab (View/Edit TXT files)
        prompt_templates_tab = QWidget()
        prompt_templates_layout = QVBoxLayout(prompt_templates_tab)
        self.prompt_template_selector = QComboBox() # Selector for prompt files
        self._populate_prompt_template_selector()
        self.prompt_template_selector.currentTextChanged.connect(self._load_selected_prompt_template)
        prompt_templates_layout.addWidget(QLabel("Select Prompt File:"))
        prompt_templates_layout.addWidget(self.prompt_template_selector)
        self.current_prompt_template_editor = QTextEdit() # Editor for the selected prompt file
        self.current_prompt_template_editor.setPlaceholderText("Load a prompt template to view/edit...")
        prompt_templates_layout.addWidget(self.current_prompt_template_editor)
        save_prompt_button = QPushButton("Save Prompt Template")
        save_prompt_button.clicked.connect(self._save_current_prompt_template)
        prompt_templates_layout.addWidget(save_prompt_button, alignment=Qt.AlignRight)
        self.input_tabs.addTab(prompt_templates_tab, "Prompt Templates")

        # Config Editor Tab (View/Edit JSON config)
        config_editor_tab = QWidget()
        config_editor_layout = QVBoxLayout(config_editor_tab)
        self.config_editor = QTextEdit() # Editor for the JSON config
        self.config_editor.setPlaceholderText("Load a config file to view/edit JSON...")
        config_editor_layout.addWidget(QLabel("Current Config (JSON):"))
        config_editor_layout.addWidget(self.config_editor)
        save_config_button_tab = QPushButton("Save Config")
        save_config_button_tab.clicked.connect(self._save_config_from_editor)
        config_editor_layout.addWidget(save_config_button_tab, alignment=Qt.AlignRight)
        self.input_tabs.addTab(config_editor_tab, "Config Editor")

        # Quick Curation Panel (Initially hidden, shown via click)
        # This will likely be implemented as a QDialog or a QDockWidget
        # For now, we just have a placeholder or a simple debug display
        self.quick_curation_debug_label = QLabel("Quick Curation Info:") # Placeholder
        self.quick_curation_debug_label.hide() # Hide initially
        main_layout.addWidget(self.quick_curation_debug_label)

        # --- Initial Splitter Sizes ---
        main_splitter_h.setSizes([int(self.height() * 0.65), int(self.height() * 0.35)])
        display_splitter_v.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])

        # --- Bottom Toolbar ---
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolbar")  # Set objectName for state saving
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(Qt.BottomToolBarArea, toolbar)

        # Style Selector (ComboBox)
        toolbar.addWidget(QLabel(" Style: "))
        self.style_selector = QComboBox()
        self.style_selector.addItems(QStyleFactory.keys()) # Use available styles
        self.style_selector.setCurrentText(STYLE_SELECTED_THEME) # Set default
        self.style_selector.setMinimumWidth(150)
        self.style_selector.currentTextChanged.connect(lambda name: self._on_style_selected(True, style_name=name)) # Connect
        toolbar.addWidget(self.style_selector)

        # Color Scheme Toggle Button (QPushButton with checkable property)
        self.theme_button = QPushButton("Dark Mode")
        self.theme_button.setCheckable(True)
        self.theme_button.toggled.connect(self._toggle_color_scheme)
        toolbar.addWidget(self.theme_button)

        # Temperature (DoubleSpinBox) - Reflects config, maybe allow override?
        # For simplicity, let's assume config is the source of truth for now.
        # Add display/edit fields for API params later, linked to config_editor tab.
        # Placeholder for now:
        # toolbar.addWidget(QLabel(" Temp: "))
        # self.temp_spinbox = QDoubleSpinBox()
        # self.temp_spinbox.setRange(0.0, 2.0)
        # self.temp_spinbox.setSingleStep(0.1)
        # toolbar.addWidget(self.temp_spinbox)

        # Max Tokens (SpinBox) - Placeholder
        # toolbar.addWidget(QLabel(" Max Tokens: "))
        # self.max_tokens_spinbox = QSpinBox()
        # self.max_tokens_spinbox.setRange(50, 8192)
        # self.max_tokens_spinbox.setSingleStep(10)
        # toolbar.addWidget(self.max_tokens_spinbox)


        toolbar.addSeparator()

        # Font Size (SpinBox)
        toolbar.addWidget(QLabel(" Font Size: "))
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(8, 24)
        self.font_size_spinbox.setValue(DEFAULT_FONT_SIZE)
        self.font_size_spinbox.valueChanged.connect(self._update_font_size)
        toolbar.addWidget(self.font_size_spinbox)

        toolbar.addSeparator()

        # Send Button (Main Action) (QPushButton)
        self.send_button = QPushButton("Send to LLM")
        self.send_button.setToolTip("Send user instruction and context to LLM")
        self.send_button.clicked.connect(self._send_to_llm) # Connect to the actual LLM call logic
        toolbar.addWidget(self.send_button)

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing...")


        # --- Initial Content and State ---
        self._load_config_into_editor() # Load initial config into editor tab
        self._update_font_size(self.font_size_spinbox.value()) # Apply initial font size
        self._populate_prompt_template_selector() # Populate prompt selector
        self._load_selected_prompt_template() # Load default prompt template

        self.status_bar.showMessage("Ready.")


    # --- Slots for UI Interactions (Controller Logic) ---

    @Slot()
    def _send_to_llm(self):
        """Handles sending the user instruction and context to the LLM."""
        user_instruction = self.user_instruction_input.toPlainText()
        if not user_instruction.strip():
            QMessageBox.warning(self, "Input Error", "Please enter some instructions for the LLM.")
            return

        # TODO: Retrieve relevant KG context (using model.retrieve_kg_context_for_prompt)
        # For now, pass an empty KG context string
        kg_context_xml = "" # Placeholder

        # TODO: Retrieve conversation history XML
        # For now, pass an empty history string
        conversation_history_xml = self._conversation_history_xml # Use stored history

        self.status_bar.showMessage("Sending to LLM...")
        self.send_button.setEnabled(False) # Disable button while processing

        # Call the model method (this should ideally be done in a separate thread
        # to keep the UI responsive for long-running LLM calls)
        # For simplicity in this first draft, we'll call it directly.
        try:
            parsed_response = self.model.process_user_instruction(
                user_instruction=user_instruction,
                conversation_history_xml=conversation_history_xml, # Pass history
                # kg_context_xml=kg_context_xml # Pass KG context if implemented
            )

            if parsed_response:
                self.status_bar.showMessage("Processing LLM response...")
                self._last_parsed_response = parsed_response # Store for potential curation

                # Update history with the new turn (prompt + response)
                # This needs the *full* prompt XML including history/context
                # and the full response XML. Complex history management is a TODO.
                # For now, just append the new raw XML response to a simple string history.
                # A better approach would build/parse conversation XML turns.
                self._conversation_history_xml += parsed_response.raw_xml # Simple append


                # Update UI display panels
                self._update_display_panels(parsed_response)

                self.status_bar.showMessage("LLM response received and processed.", 3000)
            else:
                self.status_bar.showMessage("LLM call or parsing failed. Check logs.", 5000)

        except Exception as e:
            log_error(f"An error occurred during LLM processing: {e}", exc_info=True)
            self.status_bar.showMessage(f"Error: {e}", 5000)
        finally:
            self.send_button.setEnabled(True) # Re-enable button

    def _update_display_panels(self, parsed_response: LLMResponseParsed):
        """Updates the UI panels with data from the parsed LLM response."""
        # Display Raw XML
        self.raw_xml_display.setPlainText(parsed_response.raw_xml)

        # Display Parsed Data (JSON or text representation)
        # You could format this nicely, e.g., as JSON string
        parsed_data_dict = parsed_response.to_dict()
        # Remove potentially non-serializable items if just displaying structure
        parsed_data_dict.pop("narrative_xml_element", None)
        parsed_data_dict.pop("raw_response_json", None) # Already logged, maybe display less raw info here

        self.parsed_display.setPlainText(json.dumps(parsed_data_dict, indent=2))

        # Display Narrative with Highlighting (This is the complex part)
        self._render_narrative_with_highlighting(parsed_response.narrative_xml_element)

        # TODO: Update KG State summary (e.g., run a simple count query)
        self._update_kg_state_summary()

        # TODO: Display suggested queries in the Query Panel
        # This would involve iterating parsed_response.queries and adding them to the Query Panel UI


    def _render_narrative_with_highlighting(self, narrative_xml_element: etree.Element):
        """Renders the narrative XML element in the QTextEdit with highlighting."""
        if narrative_xml_element is None:
            self.narrative_display.setPlainText("Narrative element not found in XML.")
            return

        # Use lxml to traverse the tree and build HTML
        # This is a simplified approach; proper handling of mixed content and complex tags is needed.
        # We need to generate HTML like:
        # <p>Alice <span data-xml-id="rel1" data-type="relation" data-relation-type="LIVES_IN" data-subj="ent1" data-obj="ent2" style="color: green;">lived in</span> <span data-xml-id="ent2" data-type="entity" data-canonical="Wonderland" style="color: blue;">Wonderland</span>.</p>
        # The data attributes are crucial for later identifying clicks.

        def build_html_from_xml(element):
            html_content = ""
            if element.text:
                html_content += element.text.replace('\n', '<br/>') # Basic new line handling

            for child in element:
                # Determine the HTML tag and style based on the XML tag
                html_tag = 'span' # Default to span for entities and relations
                style = ''
                data_attrs = [] # List of data attributes for JS/Event handling

                if child.tag == 'entity':
                    html_tag = 'span'
                    style = 'background-color: #CCCCFF; color: black;' # Light blue background for entities
                    data_attrs.append(f'data-type="entity"')
                    # Add metadata as data attributes
                    for key, value in child.attrib.items():
                         data_attrs.append(f'data-{key}="{value.replace("\"", """)}"') # Sanitize quotes

                elif child.tag == 'relation':
                    html_tag = 'span'
                    style = 'background-color: #CCFFCC; color: black;' # Light green background for relations
                    data_attrs.append(f'data-type="relation"')
                    for key, value in child.attrib.items():
                         data_attrs.append(f'data-{key}="{value.replace("\"", """)}"') # Sanitize quotes

                elif child.tag == 'query':
                    # Queries might be better in a separate panel, but if incrusted:
                    html_tag = 'code' # Or pre
                    style = 'display: block; background-color: #EEEEEE; margin: 5px; padding: 5px;'
                    data_attrs.append(f'data-type="query"')
                    for key, value in child.attrib.items():
                         data_attrs.append(f'data-{key}="{{value.replace("\"", """)}}"')

                elif child.tag == 'sentence':
                    html_tag = 'p' # Treat sentence as a paragraph

                # Recursive call for children
                inner_html = build_html_from_xml(child)

                # Assemble the HTML tag
                if html_tag == 'span' or html_tag == 'code': # Elements with content inside tags
                     data_attrs_str = " ".join(data_attrs)
                     html_content += f'<{html_tag} style="{style}" {data_attrs_str}>{inner_html}</{html_tag}>'
                elif html_tag == 'p': # Paragraphs
                     html_content += f'<{html_tag}>{inner_html}</{html_tag}>'
                # Add other tags if your XML uses them (e.g., <alias>)
                # Handle <alias> similar to entity, linking to canonical

                if child.tail:
                    html_content += child.tail.replace('\n', '<br/>') # Add text after the tag

            return html_content

        try:
            # Start the HTML rendering
            html_output = build_html_from_xml(narrative_xml_element)
            # Wrap in basic HTML body for QTextEdit
            full_html = f"<html><body>{html_output}</body></html>"
            self.narrative_display.setHtml(full_html)
            log_info("Narrative rendered with highlighting.")
        except Exception as e:
             log_error(f"Error rendering narrative XML to HTML: {e}", exc_info=True)
             # Fallback: display raw XML if rendering fails
             self.narrative_display.setPlainText(etree.tostring(narrative_xml_element, pretty_print=True, encoding='unicode'))


    def eventFilter(self, source, event):
        """Filters events for the narrative display to handle clicks."""
        # This is how you capture clicks on specific elements in QTextEdit
        # Requires QTextEdit to render HTML with data attributes.
        # You need to map the position of the click back to the underlying HTML/XML structure.
        # This is non-trivial and often involves custom QTextEdit subclasses or complex event handling.
        # For simplicity, let's just log a click event for now as a placeholder.

        if source == self.narrative_display.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            log_warning("DEBUG: Using position().toPoint() instead of deprecated pos()")
            # Use position().toPoint() instead of pos() which is deprecated
            position = event.position().toPoint()
            cursor = self.narrative_display.cursorForPosition(position)
            # Get the character format at the cursor position
            char_format = cursor.charFormat()

            # Check if the format has a special attribute indicating a highlight
            # This requires that the HTML rendering added custom attributes or link names
            # which is more complex than simple span styles.
            # Example using text links (if you render <a href="#entity_id">text</a>)
            # url = char_format.anchorHref()
            # if url:
            #     entity_id = url.replace("#", "")
            #     log_info(f"Clicked on linked item with ID: {entity_id}")
            #     # TODO: Find the ParsedEntity/Relation by this ID and show quick curation panel
            #     self._show_quick_curation_for_item(entity_id)
            #     return True # Consume event

            # A more general approach involves checking properties/formats at cursor
            # This is complex without a custom widget or detailed HTML structure mapping.
            # For now, just log the position and the text around it.
            text_around_click = cursor.selectedText() # Might be empty if just a click
            if not text_around_click:
                 cursor.select(QTextCursor.WordUnderCursor)
                 text_around_click = cursor.selectedText()

            log_info(f"Mouse clicked at position {position} in narrative display. Text around click: '{text_around_click}'")
            # TODO: Map click position to the underlying XML tag and show curation panel
            # This requires storing the mapping between text ranges/positions and XML elements during rendering.

            # If you implement a custom QTextEdit or use more advanced HTML linking/events,
            # you would handle click events here and return True to consume them.
            # Otherwise, let the base class handle it.
            # return True # Uncomment to consume event
            pass

        return super().eventFilter(source, event) # Pass event to default handler

    # TODO: Implement _show_quick_curation_for_item(xml_id)
    # This would find the item in _last_parsed_response and populate/show the curation panel.

    # --- Methods for Config and Prompt Management ---

    def _load_config_dialog(self):
        """Opens file dialog to load a JSON config file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration File",
            str(CONFIG_DIR),
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.model.config_path = file_path # Update model's config path
            self.model._load_config() # Reload config in model
            self._load_config_into_editor() # Load into editor
            self.status_bar.showMessage(f"Config loaded from {os.path.basename(file_path)}", 3000)
            # TODO: Reload prompt file paths based on new config

    def _save_config_dialog(self):
        """Opens file dialog to save the current config to a JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration File",
            str(self.model.config_path), # Suggest current path
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.model.config_path = file_path # Update model's config path
            self._save_config_from_editor() # Save from editor to this path
            self.status_bar.showMessage(f"Config saved to {os.path.basename(file_path)}", 3000)


    @Slot()
    def _load_config_into_editor(self):
        """Loads the current config JSON into the editor tab."""
        if self.model.config:
            try:
                # Display the loaded config as formatted JSON
                config_json_text = json.dumps(self.model.config, indent=2)
                self.config_editor.setPlainText(config_json_text)
                log_info("Config loaded into editor.")
            except Exception as e:
                log_error(f"Error formatting config to JSON for editor: {e}")
                self.config_editor.setPlainText("Error loading config into editor.")
        else:
            self.config_editor.setPlainText("No config loaded.")

    @Slot()
    def _save_config_from_editor(self):
        """Saves the JSON content from the editor tab back to the config file."""
        json_text = self.config_editor.toPlainText()
        try:
            # Attempt to parse the text as JSON
            new_config_data = json.loads(json_text)
            # Update the model's config
            self.model.config = new_config_data
            # Save the updated config back to the file
            with open(self.model.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.model.config, f, indent=2)
            log_info(f"Config saved from editor to {self.model.config_path}")
            self.status_bar.showMessage("Config saved.", 3000)

            # TODO: Reload prompt templates if file paths changed in config
            self.model._load_prompt_templates()
            self._populate_prompt_template_selector()

        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Save Config Error", f"Invalid JSON format in editor: {e}")
            log_error(f"Invalid JSON format when saving config: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Save Config Error", f"Error saving config: {e}")
            log_error(f"Error saving config: {e}")


    def _populate_prompt_template_selector(self):
        """Populates the prompt template selector with .txt files in PROMPTS_DIR."""
        self.prompt_template_selector.clear()
        prompt_files = list(PROMPTS_DIR.glob("*.txt"))
        if prompt_files:
            file_names = [f.name for f in prompt_files]
            self.prompt_template_selector.addItems(file_names)
            log_info(f"Populated prompt selector with {len(file_names)} files.")
        else:
            self.prompt_template_selector.addItem("No .txt files found in prompts/")
            self.prompt_template_selector.setEnabled(False)
            self.current_prompt_template_editor.setEnabled(False)
            log_warning("No .txt files found in prompts/.")

    @Slot()
    def _load_selected_prompt_template(self):
        """Loads the content of the selected prompt template file into the editor."""
        selected_file_name = self.prompt_template_selector.currentText()
        if not selected_file_name or selected_file_name == "No .txt files found in prompts/":
            self.current_prompt_template_editor.setPlainText("")
            self.current_prompt_template_editor.setEnabled(False)
            return

        file_path = PROMPTS_DIR / selected_file_name
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.current_prompt_template_editor.setPlainText(f.read())
            self.current_prompt_template_editor.setEnabled(True)
            log_info(f"Loaded prompt template: {selected_file_name}")
        except FileNotFoundError:
             log_error(f"Prompt template file not found: {file_path}")
             self.current_prompt_template_editor.setPlainText(f"Error: File not found at {file_path}")
             self.current_prompt_template_editor.setEnabled(False)
        except Exception as e:
             log_error(f"Error loading prompt template {selected_file_name}: {e}")
             self.current_prompt_template_editor.setPlainText(f"Error loading file: {e}")
             self.current_prompt_template_editor.setEnabled(False)

    @Slot()
    def _save_current_prompt_template(self):
        """Saves the content from the prompt editor back to the file."""
        selected_file_name = self.prompt_template_selector.currentText()
        if not selected_file_name or selected_file_name == "No .txt files found in prompts/":
            QMessageBox.warning(self, "Save Prompt Error", "No prompt file selected or available.")
            return

        file_path = PROMPTS_DIR / selected_file_name
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.current_prompt_template_editor.toPlainText())
            log_info(f"Prompt template saved: {selected_file_name}")
            self.status_bar.showMessage(f"Prompt template '{selected_file_name}' saved.", 3000)

            # After saving, ensure the model reloads templates if needed (optional, depends on workflow)
            # self.model._load_prompt_templates()

        except Exception as e:
            QMessageBox.warning(self, "Save Prompt Error", f"Error saving prompt template: {e}")
            log_error(f"Error saving prompt template {selected_file_name}: {e}")


    # --- Theme and Style Methods (Adapted from best_gui.txt) ---

    @Slot(bool)
    def _on_style_selected(self, checked: bool, style_name: str):
         """Handles style selection from the menu or combobox."""
         if not checked: # Only act when checked (from menu) or always (from combobox)
             if self.sender() != self.style_selector:
                  return # Ignore unchecked from menu if not combobox event

         try:
             app = QApplication.instance()
             app.setStyle(QStyleFactory.create(style_name))

             # Make sure the style selector (combobox) is updated if changed from menu
             self.style_selector.setCurrentText(style_name)

             # Reapply color scheme to work with the new style
             # Get current color scheme from settings/UI state
             current_scheme_index = 0 # Default to Auto/Unknown
             for action in self.color_scheme_actions:
                  if action.isChecked():
                       current_scheme_index = action.data()
                       break
             # If toggle button is checked, force Dark (assuming toggle controls Light/Dark state)
             if self.theme_button.isChecked():
                  current_scheme_index = 2 # Dark

             # Apply the color scheme using the enum value
             app.styleHints().setColorScheme(Qt.ColorScheme(current_scheme_index))

             # Clear any custom stylesheet
             app.setStyleSheet('')

             self.status_bar.showMessage(f"{style_name} style applied", 3000)

         except Exception as e:
             log_error(f"Error applying {style_name} style: {e}", exc_info=True)
             QMessageBox.warning(self, "Style Error", f"Error applying {style_name} style: {str(e)}")

    @Slot(bool)
    def _toggle_color_scheme(self, checked: bool):
        """Toggles between light and dark color schemes."""
        app = QApplication.instance()

        # Clear any custom theme setting (custom QSS)
        self.settings.setValue("customTheme", "")
        app.setStyleSheet('') # Clear stylesheet

        if checked: # Dark mode requested (toggle button checked)
            app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
            scheme_index = 2 # Dark enum

            # Update the menu action state
            for action in self.color_scheme_actions:
                 action.setChecked(action.data() == scheme_index)

            self.theme_button.setText("Light Mode")
            self.settings.setValue("colorScheme", scheme_index) # Save setting

        else: # Light mode requested (toggle button unchecked)
            app.styleHints().setColorScheme(Qt.ColorScheme.Light)
            scheme_index = 1 # Light enum (or 0 for Auto/Unknown based on preferred default)

            # Update the menu action state
            for action in self.color_scheme_actions:
                 action.setChecked(action.data() == scheme_index)

            self.theme_button.setText("Dark Mode")
            self.settings.setValue("colorScheme", scheme_index) # Save setting

        self.status_bar.showMessage(f"{COLOR_SCHEMES[scheme_index]} color scheme applied", 3000)


    # Overridden method to apply theme/settings after app style is set
    def _apply_current_theme(self):
        """Applies the current theme and style settings."""
        app = QApplication.instance()
        global STYLE_SELECTED_THEME

        # Apply saved style first
        saved_style_name = self.settings.value("style", STYLE_SELECTED_THEME, type=str)
        if saved_style_name in QStyleFactory.keys():
            try:
                app.setStyle(QStyleFactory.create(saved_style_name))
                STYLE_SELECTED_THEME = saved_style_name # Update the global if needed
                self.style_selector.setCurrentText(saved_style_name) # Update selector UI
            except Exception as e:
                 log_error(f"Failed to apply saved style '{saved_style_name}': {e}")
                 # Fallback to default Fusion
                 app.setStyle(QStyleFactory.create(STYLE_SELECTED_THEME))
                 self.style_selector.setCurrentText(STYLE_SELECTED_THEME)
        else:
            # Apply default Fusion if saved style is invalid
            app.setStyle(QStyleFactory.create(STYLE_SELECTED_THEME))
            self.style_selector.setCurrentText(STYLE_SELECTED_THEME)


        # Apply saved color scheme
        # Default to 0 (Auto/Unknown) if no setting saved
        saved_color_scheme_index = self.settings.value("colorScheme", 0, type=int)
        try:
             app.styleHints().setColorScheme(Qt.ColorScheme(saved_color_scheme_index))
             # Update menu action check state
             for action in self.color_scheme_actions:
                  action.setChecked(action.data() == saved_color_scheme_index)

             # Update toggle button state based on saved color scheme
             if saved_color_scheme_index == 2: # Dark
                  self.theme_button.setChecked(True)
                  self.theme_button.setText("Light Mode")
             else: # Light or Auto
                  self.theme_button.setChecked(False)
                  self.theme_button.setText("Dark Mode")

        except Exception as e:
             log_error(f"Failed to apply saved color scheme index {saved_color_scheme_index}: {e}")
             # Fallback to Auto
             app.styleHints().setColorScheme(Qt.ColorScheme.Unknown)
             for action in self.color_scheme_actions:
                  action.setChecked(action.data() == 0)
             self.theme_button.setChecked(False)
             self.theme_button.setText("Dark Mode")


        # Apply font size settings
        font_size = self.settings.value("fontSize", DEFAULT_FONT_SIZE, type=int)
        self.font_size_spinbox.setValue(font_size)
        self._update_font_size(font_size) # This also saves the setting

    @Slot(int)
    def _update_font_size(self, size: int):
        """Updates the font size for all text elements in the UI."""
        # Save the setting
        self.settings.setValue("fontSize", size)

        # Create a font with the specified size
        app = QApplication.instance()
        font = app.font()
        font.setPointSize(size)

        # Apply to application-wide font
        app.setFont(font)

        # Apply to specific text elements that might need explicit updates
        for text_widget in [
            self.narrative_display,
            self.parsed_display,
            self.raw_xml_display,
            self.kg_state_display,
            self.history_display,
            self.user_instruction_input,
            self.current_prompt_template_editor,
            self.config_editor
        ]:
            text_widget.setFont(font)

        log_info(f"Font size updated to {size}pt")

        # Custom theme handling removed for simplification based on feedback emphasis
        # Ensure no stylesheet is applied by default if using color schemes
        app.setStyleSheet('')


    def _load_settings(self):
        """Loads UI settings like theme and font size."""
        # This method is primarily for loading geometry and font size now.
        # Theme/Style loading is handled in _apply_current_theme after QApplication is set up.
        font_size = self.settings.value("fontSize", DEFAULT_FONT_SIZE, type=int)
        self.font_size_spinbox.setValue(font_size) # Update the spinbox UI
        # _apply_current_theme calls _update_font_size which saves the setting


    def closeEvent(self, event):
        """Handle window close event."""
        log_info("Closing application.")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState()) # Save window state (dock widgets etc)
        # Save current style and color scheme settings
        self.settings.setValue("style", self.style_selector.currentText())
        # Get the current color scheme index from the UI state (menu actions)
        current_scheme_index = 0
        for action in self.color_scheme_actions:
             if action.isChecked():
                  current_scheme_index = action.data()
                  break
        # Or rely on the toggle button state if it strictly controls Light/Dark
        if self.theme_button.isChecked():
             current_scheme_index = 2 # Dark
        else:
             # If button is not checked, it could be Light or Auto.
             # Check menu actions to be precise, or default to Auto if toggle is off.
             # Let's save the state reflected in the menu actions.
             for action in self.color_scheme_actions:
                  if action.isChecked():
                       current_scheme_index = action.data()
                       break # Save the checked menu item's data (0, 1, or 2)

        self.settings.setValue("colorScheme", current_scheme_index)


        self.model.close_neo4j() # Ensure Neo4j connection is closed
        event.accept()

    # --- Helper for KG State Summary ---
    def _update_kg_state_summary(self):
        """Fetches and displays a simple summary of the KG state."""
        if not self.model.connect_neo4j():
            self.kg_state_display.setPlainText("Not connected to Neo4j.")
            return

        summary_text = "KG State Summary:\n"
        try:
            # Example: Count total entities and relations
            entity_count_result = self.model.run_cypher_query("MATCH (n:Entity) RETURN count(n) AS count")
            if entity_count_result:
                 summary_text += f"- Total Entities: {entity_count_result[0]['count']}\n"
            else:
                 summary_text += f"- Total Entities: N/A (query failed)\n"

            relation_count_result = self.model.run_cypher_query("MATCH ()-[r]-() RETURN count(r) AS count")
            if relation_count_result:
                 summary_text += f"- Total Relations: {relation_count_result[0]['count']}\n"
            else:
                 summary_text += f"- Total Relations: N/A (query failed)\n"

            # TODO: Add counts for Pending/Canon entities/relations
            pending_entities = self.model.run_cypher_query("MATCH (n:Entity {status: 'Pending'}) RETURN count(n) AS count")
            if pending_entities:
                 summary_text += f"- Pending Entities: {pending_entities[0]['count']}\n"

            canon_entities = self.model.run_cypher_query("MATCH (n:Entity {status: 'Canon'}) RETURN count(n) AS count")
            if canon_entities:
                 summary_text += f"- Canon Entities: {canon_entities[0]['count']}\n"


        except Exception as e:
            summary_text += f"Error retrieving summary: {e}\n"
            log_error(f"Error updating KG state summary: {e}", exc_info=True)

        self.kg_state_display.setPlainText(summary_text)

    # TODO: Implement _open_neo4j_browser slot - requires launching external process
    def _open_neo4j_browser(self):
        """Opens the default web browser to the Neo4j Browser URL."""
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        # Convert bolt:// to http:// or https:// and default port 7474
        browser_url = neo4j_uri.replace("bolt://", "http://").replace("neo4j://", "http://")
        # Check if port is included, otherwise append default browser port
        if ":" not in browser_url.split('/')[-1]:
             browser_url = browser_url.rstrip('/') + ":7474"

        log_info(f"Attempting to open Neo4j Browser at {browser_url}")
        import webbrowser
        try:
            webbrowser.open(browser_url)
            self.status_bar.showMessage(f"Opened Neo4j Browser at {browser_url}", 5000)
        except Exception as e:
            log_error(f"Failed to open web browser: {e}", exc_info=True)
            self.status_bar.showMessage(f"Failed to open Neo4j Browser. Access {browser_url} manually.", 10000)


    # --- Placeholder for QTextEdit Click Handling ---
    # This is complex and requires mapping pixel positions to underlying text/HTML elements
    # and then back to your parsed XML structure (_last_parsed_response).
    # A basic approach is to use QTextCharFormat.property and check for custom properties
    # added during HTML generation, but it requires careful implementation.
    # The eventFilter above is the entry point.

    @Slot()
    def _on_narrative_text_changed(self):
        """Slot connected to narrative_display textChanged signal."""
        # This signal fires whenever the text changes, which might not be what you want
        # for click detection. It's more useful for other text change related logic.
        pass # Keep it connected in case you find a use later

    # TODO: Implement _show_quick_curation_for_item(xml_id)
    # This function would be called from eventFilter after identifying the clicked item's xml_id.
    # It needs to find the corresponding ParsedEntity or ParsedRelation in _last_parsed_response
    # and then populate the Quick Curation Panel UI elements with its data.


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the logger is configured before the app starts
    # Logging is already configured at the top of model.py
    pass

    app = QApplication(sys.argv)
    app.setApplicationName(SETTINGS_APP)
    app.setOrganizationName(SETTINGS_ORG)

    # Load and apply saved style and color scheme before creating the window
    # This ensures the window is created with the correct appearance from the start.
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

    # Apply saved style
    saved_style_name = settings.value("style", STYLE_SELECTED_THEME, type=str)
    if saved_style_name in QStyleFactory.keys():
        try:
            app.setStyle(QStyleFactory.create(saved_style_name))
        except Exception:
            app.setStyle(QStyleFactory.create(STYLE_SELECTED_THEME)) # Fallback


    # Apply saved color scheme
    saved_color_scheme_index = settings.value("colorScheme", 0, type=int) # Default Auto
    try:
        # Use styleHints().setColorScheme (requires Qt 6.0+)
        app.styleHints().setColorScheme(Qt.ColorScheme(saved_color_scheme_index))
    except Exception:
         # Fallback if styleHints().setColorScheme is not available or fails
         # Older Qt versions relied more on stylesheets for dark mode
         # For PySide6 6.9.0 this should work.
         pass # Let it fail silently or log


    # Create and show the main window
    window = MainWindow()

    # Restore window geometry and state
    geometry = settings.value("geometry")
    if geometry:
        window.restoreGeometry(geometry)
    else:
        window.resize(DEFAULT_WINDOW_SIZE) # Resize if no saved geometry

    window_state = settings.value("windowState")
    if window_state:
         window.restoreState(window_state)


    window.show()

    sys.exit(app.exec())