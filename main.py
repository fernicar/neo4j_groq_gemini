# main.py
import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import webbrowser  # For opening Neo4j browser
from datetime import datetime
from lxml import etree

# Import PySide6 Components
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QTabWidget, QSplitter, QMenuBar, QToolBar, QFileDialog,
    QMessageBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QSizePolicy,
    QDialog, QDialogButtonBox, QFormLayout, QStyleFactory, QStatusBar, QGroupBox,
    QRadioButton, QCheckBox, QToolButton, QCommandLinkButton, QDateTimeEdit,
    QSlider, QScrollBar, QDial, QProgressBar, QGridLayout, QMenu, QInputDialog,
    QPlainTextEdit # Using for log display for potentially better performance with large text
)
from PySide6.QtGui import QAction, QKeySequence, QTextCursor, QColor, QTextCharFormat, QBrush, QPalette, QDesktopServices
from PySide6.QtCore import Qt, Slot, QSize, QSettings, QFile, QTextStream, QDateTime, QTimer, QRect, QEvent, QUrl, Signal, QRegularExpression, QPoint, QObject
from PySide6.QtXml import QDomDocument # Might be useful for more robust XML display/interaction, but lxml is used in model

# Import the Model logic and data structures
from model import CurationModel, LLMResponseParsed, ParsedEntity, ParsedRelation, ParsedQuery, log_error, log_info, log_warning, LOG_FILE # Import LOG_FILE path

# --- Constants ---
APP_NAME = "AI-Assisted KG Curation"
APP_VERSION = "1.3.0"
SETTINGS_ORG = "YourOrgName"  # Replace with your organization name
SETTINGS_APP = "AICurationTool"  # Replace with your app name
DEFAULT_WINDOW_SIZE = QSize(1200, 800)
DEFAULT_FONT_SIZE = 11
CONFIG_DIR = Path("approach")  # Directory for JSON configs
PROMPTS_DIR = Path("prompts")  # Directory for TXT prompts
EMULATOR_DIR = Path("emulator_responses")  # Directory for emulator XML files
DEFAULT_CONFIG_NAME = "default.json"
DEFAULT_PROMPT_SYSTEM = "default_system.txt"
DEFAULT_PROMPT_QUERY = "default_query.txt"
DEFAULT_GLOSSARY_NAME = "glossary.xml"  # Default glossary file name
DEFAULT_SESSION_NAME = "session.ses"  # Default session file name extension


# Styles available (using built-in Qt styles)
# Default assumption from TINS: 'Fusion' style
STYLE_SELECTED_THEME = "Fusion"  # Default style from TINS
COLOR_SCHEMES = ["Auto", "Light", "Dark"]
# Qt.ColorScheme maps to these indices: Unknown (Auto)=0, Light=1, Dark=2


# --- Custom Log Handler for UI Display ---
class QTextEditLoggingHandler(logging.Handler, QObject):
    """Custom logging handler that emits a Qt Signal."""

    # Signal to send log records to the UI thread
    log_signal = Signal(str, int)  # message, level (e.g., logging.INFO)

    def __init__(self, parent):
        super().__init__()
        logging.Handler.__init__(self)
        QObject.__init__(self, parent)
        self.parent = parent  # Store parent reference if needed, but signal is static

    def emit(self, record):
        """Emits the log record message via the signal."""
        msg = self.format(record)
        # Emit message and level (e.g., logging.INFO, logging.ERROR)
        self.log_signal.emit(msg, record.levelno)


# Connect the signal to a slot in MainWindow
# Need to create the handler and connect it after the QApplication is created
# and before the MainWindow is fully initialized.


# --- Main Application Window (View/Controller) ---
class MainWindow(QMainWindow):
    # Signal to update KG state summary display
    update_kg_state_summary_signal = Signal()
    # Signal to show a message box from the model thread
    show_message_box_signal = Signal(
        str, str, str
    )  # title, message, icon_type ('info', 'warning', 'critical')

    def __init__(self):
        super().__init__()
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        # Ensure necessary directories exist
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        EMULATOR_DIR.mkdir(parents=True, exist_ok=True)

        # Create default config, prompts, and glossary if they don't exist
        self._create_default_files()

        # Initialize the Model layer
        default_config_path_str = str(CONFIG_DIR / DEFAULT_CONFIG_NAME)
        self.model = CurationModel(config_path=default_config_path_str)

        # Connect model signals for UI updates/messages (assuming model emits them)
        # Note: Model needs to be updated to emit these signals if it doesn't yet
        # Placeholder connections, adjust based on actual signals in model.py
        # Example: self.model.kg_state_updated.connect(self._update_kg_state_summary)
        # For logging, connect the custom handler's signal

        self.update_kg_state_summary_signal.connect(
            self._update_kg_state_summary
        )
        self.show_message_box_signal.connect(self._show_message_box)

        self._init_ui()
        self._load_settings()  # Load UI settings (geometry, state, style, font, session file)
        # Apply theme/style/font size is handled within _load_settings/init_ui

        # Try connecting to Neo4j on startup
        self.model.connect_neo4j()
        self._update_kg_state_summary()  # Initial update of KG state display

        # Placeholder for the last parsed response received from the model
        self._last_parsed_response: Optional[LLMResponseParsed] = None

        # Load the default glossary file content on startup
        self.load_glossary_content(CONFIG_DIR / DEFAULT_GLOSSARY_NAME)

        # Load the last session file if saved in settings
        last_session_path_str = self.settings.value(
            "lastSessionFile", "", type=str
        )
        if last_session_path_str:
            last_session_path = Path(last_session_path_str)
            if last_session_path.exists():
                log_info(
                    f"Attempting to load last session from {last_session_path}"
                )
                load_success = self.model.load_session(last_session_path)
                if load_success:
                    self._current_session_file = (
                        last_session_path  # Store current file path
                    )
                    self._update_history_display()  # Refresh UI with loaded history
                    # KG Reconstruction already triggered by model.load_session
                    # KG state updated by model.load_session calling rebuild_kg_from_history
                else:
                    log_warning(
                        f"Failed to load last session from {last_session_path}"
                    )
                    self._current_session_file = None  # Clear invalid path
                    self.settings.setValue(
                        "lastSessionFile", ""
                    )  # Clear setting
                    self.model.new_session()  # Start a new empty session
                    self._update_history_display()  # Show empty history
                    self._update_kg_state_summary()  # Update KG state (likely empty after new session)
            else:
                log_info(
                    f"Last session file not found at {last_session_path}. Starting new session."
                )
                self._current_session_file = None
                self.settings.setValue("lastSessionFile", "")
                self.model.new_session()
                self._update_history_display()
                self._update_kg_state_summary()
        else:
            log_info("No last session file saved. Starting new session.")
            self._current_session_file = None
            self.model.new_session()
            self._update_history_display()
            self._update_kg_state_summary()

        self.status_bar.showMessage("Ready.")

    def _create_default_files(self):
        """Creates default config, prompts, glossary, emulator dirs if they don't exist."""
        # Ensure directories exist
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        EMULATOR_DIR.mkdir(parents=True, exist_ok=True)

        # Default config
        default_config_path = CONFIG_DIR / DEFAULT_CONFIG_NAME
        if not default_config_path.exists():
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
                    "system_prompt_file": str(
                        PROMPTS_DIR / DEFAULT_PROMPT_SYSTEM
                    ),
                    "query_prompt_file": str(
                        PROMPTS_DIR / DEFAULT_PROMPT_QUERY
                    ),
                },
                "glossary_file": str(
                    CONFIG_DIR / DEFAULT_GLOSSARY_NAME
                ),  # Add default glossary file path
            }
            try:
                with open(default_config_path, "w", encoding="utf-8") as f:
                    json.dump(default_config_data, f, indent=2)
                log_info(f"Created default config file: {default_config_path}")
            except Exception as e:
                log_error(f"Failed to create default config file: {e}")

        # Default prompts
        default_system_prompt_path = PROMPTS_DIR / DEFAULT_PROMPT_SYSTEM
        if not default_system_prompt_path.exists():
            try:
                with open(
                    default_system_prompt_path, "w", encoding="utf-8"
                ) as f:
                    f.write(
                        "You are a helpful AI assistant that generates narrative enriched with XML tags for knowledge graph curation. Respond only with XML."
                    )
                log_info(
                    f"Created default system prompt file: {default_system_prompt_path}"
                )
            except Exception as e:
                log_error(f"Failed to create default system prompt file: {e}")

        default_query_prompt_path = PROMPTS_DIR / DEFAULT_PROMPT_QUERY
        if not default_query_prompt_path.exists():
            default_query_content = """
<prompt_structure>
  <instructions>
    Generate the next part of the narrative based on the user's instruction and the provided context.
    Embed entities, relations, and optional Cypher queries within the narrative using XML tags.
    The entire response must be a single XML document rooted with `<response>`.
    Entities: `<entity id="unique_id" canonical="StandardName" type="Person|Place|Object|Concept" status="Pending|Canon">Text in narrative</entity>`
    Relations: `<relation type="REL_TYPE" subj="entity_id_of_subject" obj="entity_id_of_object" status="Pending|Canon">Text in narrative describing relation</relation>`
    Queries: `<query purpose="Why this query is useful">Cypher query string</query>`
    Narrative: `<narrative>Your generated story text with embedded tags.</narrative>`
  </instructions>

  <context>
    {conversation_history}
    {kg_context}
    {glossary_context}
    <user_input_for_this_turn>{user_instruction}</user_input_for_this_turn>
  </context>

  <!-- Your response must be a single XML document starting with <response> -->
  <response>
    <narrative>
      <!-- Generate the narrative here with embedded XML tags -->
    </narrative>
    <!-- Optional: Include query tags here or elsewhere in the response -->
    <!-- <query purpose="suggested query reason">MATCH (n) RETURN n LIMIT 10</query> -->
  </response>
</prompt_structure>
              """  # More structured query prompt for LLM
            try:
                with open(
                    default_query_prompt_path, "w", encoding="utf-8"
                ) as f:
                    f.write(default_query_content)
                log_info(
                    f"Created default query prompt file: {default_query_prompt_path}"
                )
            except Exception as e:
                log_error(f"Failed to create default query prompt file: {e}")

        # Default glossary
        default_glossary_path = CONFIG_DIR / DEFAULT_GLOSSARY_NAME
        if not default_glossary_path.exists():
            default_glossary_content = """
<glossary>
  <!-- Initial Glossary Entries -->
  <entity id="char_alice" canonical="Alice" type="Person" status="Canon">Alice</entity>
  <entity id="loc_wonderland" canonical="Wonderland" type="Place" status="Canon">Wonderland</entity>
  <relation type="LIVES_IN" subj="char_alice" obj="loc_wonderland" status="Canon">lives in</relation>
  <!-- Add entity with hidden attribute -->
  <entity id="char_madhatter" canonical="Mad Hatter" type="Person" status="Canon" personality="Eccentric">Mad Hatter</entity>
</glossary>
              """
            try:
                with open(default_glossary_path, "w", encoding="utf-8") as f:
                    f.write(
                        default_glossary_content.strip()
                    )  # Strip leading/trailing whitespace
                log_info(
                    f"Created default glossary file: {default_glossary_path}"
                )
            except Exception as e:
                log_error(f"Failed to create default glossary file: {e}")

        # Example emulator response file
        default_emulator_response_path = EMULATOR_DIR / "response_1.xml"
        if not default_emulator_response_path.exists():
            default_emulator_content = """
<response>
  <narrative>
    <entity id="ent1" canonical="Alice" type="Person" status="Pending">Alice</entity> was walking through the garden when she saw a <entity id="ent2" canonical="White Rabbit" type="Animal">White Rabbit</entity>. The rabbit was wearing a waistcoat and muttering. Alice decided to <relation type="FOLLOWED" subj="ent1" obj="ent2">follow</relation> it down a hole.
  </narrative>
  <query purpose="Find Alice's current location">
    MATCH (a:Entity {name: 'Alice'})-[:LOCATED_IN]->(l) RETURN l.name
  </query>
</response>
              """
            try:
                with open(
                    default_emulator_response_path, "w", encoding="utf-8"
                ) as f:
                    f.write(default_emulator_content.strip())
                log_info(
                    f"Created default emulator response file: {default_emulator_response_path}"
                )
            except Exception as e:
                log_error(
                    f"Failed to create default emulator response file: {e}"
                )

    def _init_ui(self):
        """Creates the user interface elements."""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        # Initial geometry is handled by _load_settings before show()
        # self.setGeometry(100, 100, DEFAULT_WINDOW_SIZE.width(), DEFAULT_WINDOW_SIZE.height())

        # --- Central Widget & Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Menu Bar ---
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")
        new_session_action = QAction("&New Session", self)
        new_session_action.triggered.connect(self._new_session)
        file_menu.addAction(new_session_action)
        load_session_action = QAction("&Load Session...", self)
        load_session_action.triggered.connect(self._load_session_dialog)
        file_menu.addAction(load_session_action)
        self.save_session_action = QAction(
            "&Save Session", self
        )  # Keep reference to enable/disable
        self.save_session_action.triggered.connect(
            lambda: self._save_session_dialog(save_as=False)
        )
        file_menu.addAction(self.save_session_action)
        save_session_as_action = QAction("Save Session &As...", self)
        save_session_as_action.triggered.connect(
            lambda: self._save_session_dialog(save_as=True)
        )
        file_menu.addAction(save_session_as_action)

        file_menu.addSeparator()

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
        # Color Scheme Submenu
        color_scheme_menu = view_menu.addMenu("&Color Scheme")
        self.color_scheme_actions = []
        for i, scheme_name in enumerate(COLOR_SCHEMES):
            action = QAction(scheme_name, self)
            action.setCheckable(True)
            action.setData(i)  # Store Qt.ColorScheme enum value
            action.triggered.connect(
                lambda checked, idx=i: self._on_color_scheme_selected(
                    checked, force_index=idx
                )
            )
            color_scheme_menu.addAction(action)
            self.color_scheme_actions.append(action)

        # Style Submenu
        style_menu = view_menu.addMenu("&Style")
        self.style_actions = []
        for style_name in QStyleFactory.keys():
            action = QAction(style_name, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked, name=style_name: self._on_style_selected(
                    checked, style_name=name
                )
            )
            style_menu.addAction(action)
            self.style_actions.append(action)

        # --- Main Horizontal Splitter (Top/Bottom Panes) ---
        main_splitter_h = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(main_splitter_h, 1)

        # --- Top Pane (Display Area - Narrative & Info) ---
        top_pane_widget = QWidget()
        top_pane_layout = QHBoxLayout(top_pane_widget)
        top_pane_layout.setContentsMargins(0, 0, 0, 0)
        display_splitter_v = QSplitter(Qt.Orientation.Horizontal)
        top_pane_layout.addWidget(display_splitter_v)
        main_splitter_h.addWidget(top_pane_widget)

        # --- Left Display (Narrative) ---
        left_display_widget = QWidget()
        left_display_layout = QVBoxLayout(left_display_widget)
        narrative_label = QLabel("Narrative:")
        left_display_layout.addWidget(narrative_label)
        self.narrative_display = (
            QTextEdit()
        )  # Use QTextEdit for rich text/HTML display and interaction
        self.narrative_display.setReadOnly(True)
        self.narrative_display.setPlaceholderText(
            "Generated narrative will appear here with highlighting..."
        )
        self.narrative_display.setAcceptRichText(
            True
        )  # Ensure it accepts rich text/HTML
        # self.narrative_display.textChanged.connect(self._on_narrative_text_changed) # textChanged is not for click detection
        self.narrative_display.viewport().installEventFilter(
            self
        )  # Install event filter for clicks
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
        self.parsed_display = (
            QTextEdit()
        )  # Show structured parsed data here (JSON or text)
        self.parsed_display.setReadOnly(True)
        self.parsed_display.setPlaceholderText(
            "Parsed entities, relations, and queries from LLM response..."
        )
        parsed_layout.addWidget(self.parsed_display)
        self.monitor_tabs.addTab(parsed_tab, "Parsed Data")

        # Raw XML Tab
        raw_xml_tab = QWidget()
        raw_xml_layout = QVBoxLayout(raw_xml_tab)
        self.raw_xml_display = QTextEdit()  # Show raw XML response here
        self.raw_xml_display.setReadOnly(True)
        self.raw_xml_display.setPlaceholderText("Raw XML response from LLM...")
        raw_xml_layout.addWidget(self.raw_xml_display)
        self.monitor_tabs.addTab(raw_xml_tab, "Raw XML")

        # KG State/Review Tab
        kg_state_tab = QWidget()
        kg_state_layout = QVBoxLayout(kg_state_tab)
        self.kg_state_display = QTextEdit()  # Show simple summary of KG state
        self.kg_state_display.setReadOnly(True)
        self.kg_state_display.setPlaceholderText(
            "Summary of Knowledge Graph state (e.g., entity counts)..."
        )
        kg_state_layout.addWidget(self.kg_state_display)
        # Button to open Neo4j Browser
        open_neo4j_browser_button = QPushButton("Open Neo4j Browser")
        open_neo4j_browser_button.clicked.connect(
            self.model._open_neo4j_browser
        )  # Connect to model method
        kg_state_layout.addWidget(
            open_neo4j_browser_button, alignment=Qt.AlignmentFlag.AlignRight
        )
        self.monitor_tabs.addTab(kg_state_tab, "KG State")

        # History/Log Tab (Display log file content and Conversation History)
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        # Split the history tab into two sections: Log and Conversation History
        history_splitter = QSplitter(Qt.Orientation.Vertical)
        history_layout.addWidget(history_splitter)

        # Log Display
        log_group = QGroupBox("Application Logs")
        log_layout = QVBoxLayout(log_group)
        self.log_display = (
            QPlainTextEdit()
        )  # Use PlainTextEdit for potentially better performance with large logs
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText(
            "Application logs will appear here..."
        )
        log_layout.addWidget(self.log_display)
        history_splitter.addWidget(log_group)
        self._setup_log_streaming()  # Setup log streaming to this widget

        # Conversation History Display (User-friendly view)
        conv_history_group = QGroupBox("Conversation History")
        conv_history_layout = QVBoxLayout(conv_history_group)
        self.conv_history_display = (
            QTextEdit()
        )  # Use QTextEdit for rich text formatting if needed
        self.conv_history_display.setReadOnly(True)
        self.conv_history_display.setPlaceholderText(
            "Conversation history turns will appear here..."
        )
        conv_history_layout.addWidget(self.conv_history_display)
        history_splitter.addWidget(conv_history_group)

        # Set initial sizes for history splitter
        history_splitter.setSizes([self.height() * 0.5, self.height() * 0.5])

        self.monitor_tabs.addTab(history_tab, "History")

        # Query Execution Panel (Integrated into Monitor Tabs for now, or as a separate dialog)
        # Let's add a simple Query Output tab for results
        query_output_tab = QWidget()
        query_output_layout = QVBoxLayout(query_output_tab)
        self.query_output_display = QTextEdit()
        self.query_output_display.setReadOnly(True)
        self.query_output_display.setPlaceholderText(
            "Results from executed queries will appear here..."
        )
        query_output_layout.addWidget(self.query_output_display)
        self.monitor_tabs.addTab(query_output_tab, "Query Output")
        # The Query Execution Panel itself (with Execute button) will pop up via clicking narrative

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
        self.user_instruction_input.setPlaceholderText(
            "Type your instructions/prompt for the LLM here..."
        )
        user_input_layout.addWidget(self.user_instruction_input)
        self.input_tabs.addTab(user_input_tab, "User Instruction")

        # Prompt Templates Tab (View/Edit TXT files)
        prompt_templates_tab = QWidget()
        prompt_templates_layout = QVBoxLayout(prompt_templates_tab)

        # Create a horizontal layout for the selector and save button
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select Prompt File:"))
        self.prompt_template_selector = QComboBox()  # Selector for prompt files
        self.prompt_template_selector.currentTextChanged.connect(
            self._load_selected_prompt_template
        )
        selector_layout.addWidget(self.prompt_template_selector)

        # Add stretch to push the save button to the right
        selector_layout.addStretch()

        # Add save button
        save_prompt_button = QPushButton("Save Prompt Template")
        save_prompt_button.clicked.connect(self._save_current_prompt_template)
        selector_layout.addWidget(save_prompt_button)

        # Add the horizontal layout to the main layout
        prompt_templates_layout.addLayout(selector_layout)

        # Add the editor below
        self.current_prompt_template_editor = QTextEdit()  # Editor for the selected prompt file
        self.current_prompt_template_editor.setPlaceholderText(
            "Load a prompt template to view/edit..."
        )
        prompt_templates_layout.addWidget(self.current_prompt_template_editor)
        self.input_tabs.addTab(prompt_templates_tab, "Prompt Templates")
        self._populate_prompt_template_selector()  # Populate prompt selector on init

        # Config Editor Tab (View/Edit JSON config)
        config_editor_tab = QWidget()
        config_editor_layout = QVBoxLayout(config_editor_tab)

        # Create a horizontal layout for the label and save button
        config_header_layout = QHBoxLayout()
        config_header_layout.addWidget(QLabel("Current Config (JSON):"))

        # Add stretch to push the save button to the right
        config_header_layout.addStretch()

        # Add save button
        save_config_button_tab = QPushButton("Save Config")
        save_config_button_tab.clicked.connect(self._save_config_from_editor)
        config_header_layout.addWidget(save_config_button_tab)

        # Add the horizontal layout to the main layout
        config_editor_layout.addLayout(config_header_layout)

        # Add the editor below
        self.config_editor = QTextEdit()  # Editor for the JSON config
        self.config_editor.setPlaceholderText(
            "Load a config file to view/edit JSON..."
        )
        config_editor_layout.addWidget(self.config_editor)
        self.input_tabs.addTab(config_editor_tab, "Config Editor")
        self._load_config_into_editor()  # Load initial config into editor tab

        # Glossary Editor Tab (View/Edit XML file)
        glossary_editor_tab = QWidget()
        glossary_editor_layout = QVBoxLayout(glossary_editor_tab)

        # Create a horizontal layout for the label and buttons
        glossary_header_layout = QHBoxLayout()
        glossary_header_layout.addWidget(QLabel("Glossary (XML):"))

        # Add stretch to push the buttons to the right
        glossary_header_layout.addStretch()

        # Add buttons to the header layout
        load_glossary_button = QPushButton("Load Glossary...")
        load_glossary_button.clicked.connect(self._load_glossary_dialog)
        glossary_header_layout.addWidget(load_glossary_button)

        save_glossary_button = QPushButton("Save Glossary...")
        save_glossary_button.clicked.connect(self._save_glossary_dialog)
        glossary_header_layout.addWidget(save_glossary_button)

        apply_glossary_button = QPushButton("Apply Glossary to KG")
        apply_glossary_button.clicked.connect(self._apply_glossary_to_kg)
        glossary_header_layout.addWidget(apply_glossary_button)

        # Add the horizontal layout to the main layout
        glossary_editor_layout.addLayout(glossary_header_layout)

        # Add the editor below
        self.glossary_editor = QTextEdit()  # Editor for the Glossary XML
        self.glossary_editor.setPlaceholderText(
            "Load a glossary XML file to view/edit..."
        )
        glossary_editor_layout.addWidget(self.glossary_editor)
        self.input_tabs.addTab(glossary_editor_tab, "Glossary Editor")
        # Content loaded via load_glossary_content in __init__

        # Quick Curation Panel (Implemented as a QDialog)
        # Created when needed, not part of the main window layout
        self.quick_curation_dialog = None  # Keep a reference

        # --- Initial Splitter Sizes ---
        main_splitter_h.setSizes(
            [int(self.height() * 0.65), int(self.height() * 0.35)]
        )
        display_splitter_v.setSizes(
            [int(self.width() * 0.6), int(self.width() * 0.4)]
        )

        # --- Bottom Toolbar ---
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolbar")  # Set objectName for state saving
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, toolbar)

        # Style Selector (ComboBox)
        toolbar.addWidget(QLabel(" Style: "))
        self.style_selector = QComboBox()
        self.style_selector.addItems(
            QStyleFactory.keys()
        )  # Use available styles
        self.style_selector.currentTextChanged.connect(
            lambda name: self._on_style_selected(True, style_name=name)
        )  # Connect
        toolbar.addWidget(self.style_selector)

        # Color Scheme Toggle Button
        self.theme_button = QPushButton(
            "Dark Mode"
        )  # Initial text reflects default light mode assumption
        self.theme_button.setCheckable(True)
        self.theme_button.toggled.connect(self._toggle_color_scheme)
        toolbar.addWidget(self.theme_button)

        toolbar.addSeparator()

        # Font Size (SpinBox)
        toolbar.addWidget(QLabel(" Font Size: "))
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(8, 24)
        self.font_size_spinbox.valueChanged.connect(self._update_font_size)
        toolbar.addWidget(self.font_size_spinbox)

        toolbar.addSeparator()

        # Send Button (Main Action)
        self.send_button = QPushButton("Send to LLM")
        self.send_button.setToolTip("Send user instruction and context to LLM")
        self.send_button.clicked.connect(
            self._send_to_llm
        )  # Connect to the actual LLM call logic
        toolbar.addWidget(self.send_button)

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing...")

        # --- Apply initial UI settings loaded before __init__ ---
        # Apply saved style and color scheme indices from settings
        self._apply_saved_ui_settings()
        # Apply font size (also loads value into spinbox)
        font_size = self.settings.value(
            "fontSize", DEFAULT_FONT_SIZE, type=int
        )
        self.font_size_spinbox.setValue(font_size)  # Update the spinbox UI
        self._update_font_size(font_size)  # This also saves the setting

        # --- Initial Content and State ---
        self._update_history_display()  # Show empty history or loaded history

    # --- Slots for UI Interactions (Controller Logic) ---

    @Slot()
    def _send_to_llm(self):
        """Handles sending the user instruction and context to the LLM."""
        user_instruction = self.user_instruction_input.toPlainText().strip()
        if not user_instruction:
            QMessageBox.warning(
                self,
                "Input Error",
                "Please enter some instructions for the LLM.",
            )
            return

        # TODO: Implement UI mechanism to select focus entity for KG context
        # For now, no focus entity is selected, so retrieve_kg_context_for_prompt will return empty.
        focus_entity_canonical_name = None  # Placeholder

        self.status_bar.showMessage(
            "Sending to LLM...", 0
        )  # 0 makes message persist
        self.send_button.setEnabled(False)  # Disable button while processing

        # Call the model method. This should ideally be done in a separate thread
        # to keep the UI responsive for long-running LLM calls.
        # For simplicity in this implementation, we'll call it directly for now.
        # For a responsive app, use QThread or QFuture.
        try:
            log_info("Calling model.process_user_instruction...")
            # Pass the current user instruction and the focus entity for KG context
            # Model will handle history and glossary context internally
            parsed_response = self.model.process_user_instruction(
                user_instruction=user_instruction,
                focus_entity_canonical_name=focus_entity_canonical_name,
            )

            if parsed_response:
                self.status_bar.showMessage("Processing LLM response...", 0)
                self._last_parsed_response = (
                    parsed_response  # Store for potential curation
                )

                # Update UI display panels
                self._update_display_panels(parsed_response)

                # Update history display with the new turn
                self._update_history_display()

                self.status_bar.showMessage(
                    "LLM response received and processed.", 5000
                )
            else:
                # Model logged the specific error (API failure, parsing error etc.)
                self.status_bar.showMessage(
                    "LLM call or parsing failed. Check History tab for logs.",
                    10000,
                )

        except Exception as e:
            # Catch unexpected errors not handled by model's internal logging
            log_error(
                f"An unexpected error occurred during LLM processing: {e}",
                exc_info=True,
            )
            self.status_bar.showMessage(f"Error: {e}", 10000)
            self._show_message_box(
                "Application Error",
                f"An unexpected error occurred: {e}",
                "critical",
            )
        finally:
            self.send_button.setEnabled(True)  # Re-enable button
            self.user_instruction_input.clear()  # Clear input field after sending

    def _update_display_panels(self, parsed_response: LLMResponseParsed):
        """Updates the UI panels with data from the parsed LLM response."""
        log_info("Updating UI display panels.")
        # Display Raw XML
        self.raw_xml_display.setPlainText(parsed_response.raw_xml)

        # Display Parsed Data (JSON representation)
        parsed_data_dict = parsed_response.to_dict()
        self.parsed_display.setPlainText(
            json.dumps(parsed_data_dict, indent=2)
        )

        # Display Narrative with Highlighting
        self._render_narrative_with_highlighting(
            parsed_response.narrative_xml_element
        )

        # KG State summary is updated after KG update by model.process_user_instruction
        # _update_kg_state_summary signal should be emitted by model

        # TODO: Display suggested queries in the Query Output tab or a dedicated panel
        # For now, just log them and add a placeholder in the Query Output tab
        query_output_text = "Suggested Queries:\n"
        if parsed_response.queries:
            for query in parsed_response.queries:
                query_output_text += (
                    f"\n--- Query (Purpose: {query.purpose}) ---\n"
                )
                query_output_text += f"{query.query_string}\n"
                # TODO: Add a way to click/select these queries to execute them
            self.query_output_display.setPlainText(query_output_text)
            self.monitor_tabs.setCurrentWidget(
                self.monitor_tabs.widget(
                    self.monitor_tabs.indexOf(
                        self.query_output_display.parentWidget()
                    )
                )
            )  # Switch to Query Output tab
        else:
            self.query_output_display.setPlainText(
                "No queries suggested in the last response."
            )

    def _render_narrative_with_highlighting(
        self, narrative_xml_element: etree.Element
    ):
        """Renders the narrative XML element in the QTextEdit with highlighting and data attributes."""
        if narrative_xml_element is None:
            self.narrative_display.setPlainText(
                "Narrative element not found in LLM response XML."
            )
            return

        # Use lxml to traverse the tree and build HTML with data attributes
        # This is crucial for later identifying clicks in eventFilter.

        def escape_html_attribute(value):
            """Helper to escape values for HTML attributes."""
            if value is None:
                return ""
            return str(value).replace('"', "&quot;").replace("'", "&#39;")

        def build_html_from_xml(element):
            html_content = ""
            # Add text before the element
            if element.text:
                html_content += escape_html_attribute(element.text).replace(
                    "\n", "<br/>"
                )  # Basic new line handling

            for child in element:
                # Determine the HTML tag and style based on the XML tag
                html_tag = "span"  # Default for inline elements
                style = ""
                data_attrs = []  # List of data attributes

                if child.tag == "entity":
                    html_tag = "span"
                    style = "background-color: #e0e0ff; color: black;"  # Light blue-ish background
                    data_attrs.append(f'data-type="entity"')
                    # Add metadata as data attributes
                    for key, value in child.attrib.items():
                        # Use safe attribute name and escaped value
                        safe_key = key.replace(
                            "-", "_"
                        )  # HTML data attributes can't have dashes directly if accessed via dataset
                        data_attrs.append(
                            f'data-{safe_key}="{escape_html_attribute(value)}"'
                        )

                elif child.tag == "relation":
                    html_tag = "span"
                    style = "background-color: #e0ffe0; color: black;"  # Light green-ish background
                    data_attrs.append(f'data-type="relation"')
                    for key, value in child.attrib.items():
                        safe_key = key.replace("-", "_")
                        data_attrs.append(
                            f'data-{safe_key}="{escape_html_attribute(value)}"'
                        )

                elif child.tag == "query":
                    # Wrap queries in a block element like <pre> or <code> for display consistency
                    html_tag = "pre"  # Use pre for block code display
                    style = "display: block; background-color: #f0f0f0; margin: 5px; padding: 5px; white-space: pre-wrap; font-family: monospace;"
                    data_attrs.append(f'data-type="query"')
                    for key, value in child.attrib.items():
                        safe_key = key.replace("-", "_")
                        data_attrs.append(
                            f'data-{safe_key}="{escape_html_attribute(value)}"'
                        )

                # Handle unknown tags - render their content but perhaps add a warning style
                elif isinstance(
                    child.tag, str
                ):  # Ensure it's a valid tag name
                    html_tag = "span"  # Default to span
                    style = "background-color: #ffffe0; color: black;"  # Yellow-ish warning
                    data_attrs.append(f'data-type="unknown_xml_tag"')
                    data_attrs.append(f'data-tag-name="{child.tag}"')
                    log_warning(
                        f"Unknown XML tag encountered in narrative: <{child.tag}>"
                    )
                else:  # Handle comments, processing instructions etc.
                    html_tag = None  # Don't render these as explicit elements

                # Recursive call for children
                inner_html = build_html_from_xml(child)

                # Assemble the HTML tag if we decided to render it
                if html_tag:
                    data_attrs_str = " ".join(data_attrs)
                    html_content += f'<{html_tag} style="{style}" {data_attrs_str}>{inner_html}</{html_tag}>'
                else:
                    # If not rendering a tag, just include its children's content
                    html_content += inner_html

                # Add text after the element
                if child.tail:
                    html_content += escape_html_attribute(child.tail).replace(
                        "\n", "<br/>"
                    )

            return html_content

        try:
            # Start the HTML rendering from the narrative root element
            html_output = build_html_from_xml(narrative_xml_element)
            # Wrap in basic HTML body for QTextEdit
            full_html = f"<html><body>{html_output}</body></html>"
            self.narrative_display.setHtml(full_html)
            log_info("Narrative rendered with highlighting.")
        except Exception as e:
            log_error(
                f"Error rendering narrative XML to HTML: {e}", exc_info=True
            )
            # Fallback: display raw XML string if rendering fails
            try:
                raw_xml_string = etree.tostring(
                    narrative_xml_element,
                    pretty_print=True,
                    encoding="unicode",
                )
                self.narrative_display.setPlainText(raw_xml_string)
                log_info(
                    "Narrative rendering failed, displayed raw XML element text."
                )
            except Exception as inner_e:
                log_error(
                    f"Failed to convert narrative element to string for fallback: {inner_e}"
                )
                self.narrative_display.setPlainText(
                    "Error rendering narrative."
                )

    def eventFilter(self, source, event):
        """Filters events for the narrative display to handle clicks on highlighted elements."""
        # This method requires mapping the click position (pixel) to the text cursor position,
        # and then inspecting the formatting or HTML attributes at that cursor position.
        # This is complex with standard QTextEdit and often simplified.
        # A robust implementation might involve:
        # 1. Subclassing QTextEdit to have more control over painting and hit testing, OR
        # 2. Using QTextCharFormat to embed custom properties/formats during HTML rendering, OR
        # 3. Parsing the HTML/XML structure and creating a mapping of character positions to XML elements.
        # Method 3 is likely the most feasible given the current structure.

        if (
            source == self.narrative_display.viewport()
            and event.type() == QEvent.Type.MouseButtonPress
        ):
            log_info(
                f"Mouse click detected in narrative display at position {event.pos()}."
            )
            # Get the cursor position from the mouse position
            cursor = self.narrative_display.cursorForPosition(event.pos())

            # --- Simplified Click Detection Logic (Placeholder) ---
            # This requires that the HTML rendering process added a way to identify the clicked element.
            # The _render_narrative_with_highlighting added HTML with data attributes.
            # QTextEdit doesn't easily expose HTML element data attributes via cursor.
            # A common technique is to embed `<a>` tags with `href` attributes linking to IDs,
            # as QTextEdit signals `anchorClicked(QUrl)`. Let's modify rendering to use <a> tags.

            # Reroute to the more reliable anchorClicked signal if rendering uses <a>
            # The eventFilter is still useful for other interactions (e.g., hover for tooltips)
            # For now, let's just log and return False to let the default handler proceed.

            # If using anchorClicked approach, the signal would be connected like:
            # self.narrative_display.anchorClicked.connect(self._on_narrative_anchor_clicked)
            # And eventFilter would primarily be for hover effects or other non-click interactions.

            # For now, just logging the click and letting default handler proceed.
            # TODO: Implement proper click-to-XML mapping and call _show_quick_curation_for_item or _show_query_execution_panel
            # based on the identified element type and ID.

            return False  # Pass the event on

        return super().eventFilter(
            source, event
        )  # Pass event to default handler

    @Slot(QUrl)
    def _on_narrative_anchor_clicked(self, url: QUrl):
        """Handles clicks on anchor links in the narrative display."""
        log_info(f"Narrative anchor clicked: {url.toString()}")
        # Assume anchor links are formatted like #type_xmlId (e.g., #entity_ent1, #relation_rel_abc, #query_q1)
        fragment = url.fragment()  # Get the part after '#'
        if not fragment:
            log_warning("Clicked anchor has no fragment (ID).")
            return

        parts = fragment.split("_", 1)  # Split into type and ID
        if len(parts) != 2:
            log_warning(
                f"Clicked anchor fragment '{fragment}' is not in expected format type_xmlId."
            )
            return

        item_type = parts[0]
        xml_id = parts[1]

        log_info(
            f"Identified clicked item: Type='{item_type}', XML ID='{xml_id}'"
        )

        if not self._last_parsed_response:
            log_warning("No parsed response available to find clicked item.")
            return

        # Find the corresponding item in the last parsed response
        item_data = None
        if item_type == "entity":
            for entity in self._last_parsed_response.entities:
                if entity.xml_id == xml_id:
                    item_data = entity
                    break
        elif item_type == "relation":
            for relation in self._last_parsed_response.relations:
                if relation.xml_id == xml_id:
                    item_data = relation
                    break
        elif item_type == "query":
            for query in self._last_parsed_response.queries:
                if query.xml_id == xml_id:
                    item_data = query
                    break

        if item_data:
            log_info(
                f"Found corresponding item data for {item_type} with XML ID {xml_id}."
            )
            if item_type == "entity" or item_type == "relation":
                self._show_quick_curation_for_item(item_data, item_type)
            elif item_type == "query":
                self._show_query_execution_panel(item_data)
        else:
            log_warning(
                f"Could not find corresponding item data for {item_type} with XML ID {xml_id} in the last parsed response."
            )

    def _show_quick_curation_for_item(self, item_data: Any, item_type: str):
        """Creates and shows the Quick Curation Dialog for an Entity or Relation."""
        log_info(
            f"Showing Quick Curation Dialog for {item_type} with XML ID: {getattr(item_data, 'xml_id', 'N/A')}"
        )

        if (
            self.quick_curation_dialog is not None
            and self.quick_curation_dialog.isVisible()
        ):
            self.quick_curation_dialog.close()  # Close previous dialog if open

        self.quick_curation_dialog = QDialog(self)
        self.quick_curation_dialog.setWindowTitle(
            f"Curate {item_type.capitalize()}"
        )
        layout = QFormLayout()

        # Store original data for model update call
        original_data = {}

        # Common fields
        layout.addRow("Text Span:", QLabel(item_data.text_span))
        original_data["text_span"] = (
            item_data.text_span
        )  # Keep original text span

        # Status selector
        status_options = ["Pending", "Canon", "Ignored"]
        status_selector = QComboBox()
        status_selector.addItems(status_options)
        status_selector.setCurrentText(item_data.status)
        layout.addRow("Status:", status_selector)

        # Attributes editor (Simplified - use a simple text edit for JSON/dict display for now)
        # A proper attribute editor would require dynamic form generation or a table view.
        attributes_editor = QTextEdit()
        try:
            attributes_editor.setPlainText(
                json.dumps(item_data.attributes, indent=2)
            )
        except Exception:
            attributes_editor.setPlainText(str(item_data.attributes))
        attributes_editor.setPlaceholderText("Edit attributes (JSON format)")
        attributes_editor.setFixedHeight(100)  # Limit height
        layout.addRow("Attributes (JSON):", attributes_editor)

        if item_type == "entity":
            # Entity specific fields
            canonical_editor = QLineEdit(item_data.canonical)
            layout.addRow("Canonical Name:", canonical_editor)
            entity_type_editor = QLineEdit(item_data.entity_type)
            layout.addRow("Entity Type:", entity_type_editor)

            original_data["_original_canonical"] = (
                item_data.canonical
            )  # Store original for model update

            # Store references to UI controls
            self.quick_curation_dialog.find_control = {
                "canonical_editor": canonical_editor,
                "entity_type_editor": entity_type_editor,
                "status_selector": status_selector,
                "attributes_editor": attributes_editor,
                "_original_data": original_data,  # Store original data including identifiers
                "_item_type": item_type,  # Store item type
            }

        elif item_type == "relation":
            # Relation specific fields
            layout.addRow(
                "Relation Type:", QLabel(item_data.relation_type)
            )  # For now, relation type not editable via panel
            layout.addRow("Subject XML ID:", QLabel(item_data.subj_id))
            layout.addRow("Object XML ID:", QLabel(item_data.obj_id))

            # TODO: Resolve XML IDs to canonical names for display here

            original_data["_original_subj_id"] = (
                item_data.subj_id
            )  # Store originals for model update
            original_data["_original_obj_id"] = item_data.obj_id
            original_data["_original_relation_type"] = (
                item_data.relation_type
            )  # Store original type

            # Store references to UI controls
            self.quick_curation_dialog.find_control = {
                "status_selector": status_selector,
                "attributes_editor": attributes_editor,
                "_original_data": original_data,
                "_item_type": item_type,
            }

        # Dialog buttons (Apply and Cancel)
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.button(QDialogButtonBox.Ok).setText(
            "Apply"
        )  # Rename Ok button to Apply
        button_box.accepted.connect(
            self._apply_curation_updates
        )  # Connect to apply slot
        button_box.rejected.connect(
            self.quick_curation_dialog.reject
        )  # Connect to reject slot
        layout.addRow(button_box)

        self.quick_curation_dialog.setLayout(layout)
        self.quick_curation_dialog.exec()  # Show as modal dialog

    @Slot()
    def _apply_curation_updates(self):
        """Collects data from Quick Curation Dialog and calls model to apply."""
        if (
            not self.quick_curation_dialog
            or not self.quick_curation_dialog.find_control
        ):
            log_error(
                "Apply curation called but dialog or controls not found."
            )
            return

        controls = self.quick_curation_dialog.find_control
        item_type = controls["_item_type"]
        original_data = controls[
            "_original_data"
        ]  # Retrieve original identifiers

        curated_item_data: Dict[str, Any] = {}
        curated_item_data.update(
            original_data
        )  # Start with original identifiers

        # Get updated status
        curated_item_data["status"] = controls["status_selector"].currentText()

        # Get updated attributes
        try:
            attributes_text = controls["attributes_editor"].toPlainText()
            curated_item_data["attributes"] = json.loads(attributes_text)
        except json.JSONDecodeError as e:
            QMessageBox.warning(
                self, "Invalid JSON", f"Attributes JSON is invalid: {e}"
            )
            return  # Stop if attributes JSON is bad
        except Exception as e:
            log_warning(
                f"Could not parse attributes JSON, sending empty dict: {e}"
            )
            curated_item_data["attributes"] = (
                {}
            )  # Send empty attributes on error

        if item_type == "entity":
            # Get updated entity-specific fields
            curated_item_data["canonical"] = controls[
                "canonical_editor"
            ].text()
            curated_item_data["entity_type"] = controls[
                "entity_type_editor"
            ].text()
            # UI doesn't handle provenance or text_span updates here, model will add 'User_Curated' provenance

            # Call model method
            success = self.model.apply_curation_updates(
                curated_item_data, "entity"
            )
            if success:
                log_info("Entity curation updates applied successfully.")
                self.status_bar.showMessage("Entity updates applied.", 3000)
                # Refresh KG state display
                self._update_kg_state_summary()
                self.quick_curation_dialog.accept()  # Close dialog
            else:
                log_error("Failed to apply entity curation updates.")
                # Model logs the error, maybe show a message box here?
                # self._show_message_box("Curation Failed", "Failed to apply entity updates to KG. See logs.", "warning")
                # Don't close dialog on model failure so user can retry or cancel

        elif item_type == "relation":
            # Only status and attributes are editable for relations via this panel
            # The original identifiers (_original_...) are already in curated_item_data

            # Call model method
            success = self.model.apply_curation_updates(
                curated_item_data, "relation"
            )
            if success:
                log_info("Relation curation updates applied successfully.")
                self.status_bar.showMessage("Relation updates applied.", 3000)
                # Refresh KG state display
                self._update_kg_state_summary()
                self.quick_curation_dialog.accept()  # Close dialog
            else:
                log_error("Failed to apply relation curation updates.")
                # self._show_message_box("Curation Failed", "Failed to apply relation updates to KG. See logs.", "warning")

    def _show_query_execution_panel(self, query_data: ParsedQuery):
        """Creates and shows the Query Execution Dialog."""
        log_info(
            f"Showing Query Execution Panel for query ID: {query_data.xml_id}"
        )

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Execute Query: {query_data.purpose}")
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Purpose:"))
        purpose_label = QLabel(query_data.purpose)
        purpose_label.setWordWrap(True)
        layout.addWidget(purpose_label)

        layout.addWidget(QLabel("Cypher Query:"))
        query_editor = QTextEdit(query_data.query_string)
        query_editor.setReadOnly(
            True
        )  # Make it read-only? Or allow editing before execution?
        query_editor.setFixedHeight(150)
        layout.addWidget(query_editor)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        execute_button = button_box.button(QDialogButtonBox.Ok)
        execute_button.setText("Execute Query")
        cancel_button = button_box.button(QDialogButtonBox.Cancel)

        # Connect signals
        execute_button.clicked.connect(
            lambda: self._execute_suggested_query(query_editor.toPlainText())
        )
        button_box.rejected.connect(dialog.reject)

        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec()  # Show as modal dialog

    @Slot(str)
    def _execute_suggested_query(self, query_string: str):
        """Calls model to execute the query and displays results."""
        log_info(f"User requested execution of query: {query_string}")
        self.status_bar.showMessage("Executing query...", 0)

        # Call model method (should ideally be in a thread)
        # For simplicity, direct call for now
        try:
            results = self.model.execute_suggested_query(query_string)

            # Display results
            self.monitor_tabs.setCurrentWidget(
                self.monitor_tabs.widget(
                    self.monitor_tabs.indexOf(
                        self.query_output_display.parentWidget()
                    )
                )
            )  # Switch to Query Output tab
            if results is not None:
                log_info("Query executed successfully.")
                # Format results nicely (simple JSON dump for now)
                results_text = f"Query Results:\n---\n{json.dumps(results, indent=2)}\n---\n"
                self.query_output_display.setPlainText(results_text)
                self.status_bar.showMessage(
                    f"Query executed successfully. {len(results)} records returned.",
                    5000,
                )
            else:
                log_error("Query execution failed.")
                self.query_output_display.setPlainText(
                    "Query execution failed. See logs."
                )
                self.status_bar.showMessage(
                    "Query execution failed. Check logs.", 10000
                )

        except Exception as e:
            log_error(
                f"An unexpected error occurred during query execution: {e}",
                exc_info=True,
            )
            self.query_output_display.setPlainText(
                f"An unexpected error occurred during query execution:\n{e}\nSee logs."
            )
            self.status_bar.showMessage(
                f"Error during query execution: {e}", 10000
            )

    # --- Methods for Config and Prompt Management ---

    def _load_config_dialog(self):
        """Opens file dialog to load a JSON config file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration File",
            str(CONFIG_DIR),
            "JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            log_info(f"User selected config file: {file_path}")
            self.model.config_path = Path(
                file_path
            )  # Update model's config path
            self.model._load_config()  # Reload config in model
            self._load_config_into_editor()  # Load into editor
            self.status_bar.showMessage(
                f"Config loaded from {os.path.basename(file_path)}", 3000
            )
            # Reload prompt file paths based on new config
            self.model._load_prompt_templates()
            self._populate_prompt_template_selector()
            # TODO: Reload glossary path and content if changed in config

    def _save_config_dialog(self):
        """Opens file dialog to save the current config to a JSON file."""
        # Suggest current path from model config
        suggested_path = (
            self.model.config_path
            if self.model.config_path
            else (CONFIG_DIR / DEFAULT_CONFIG_NAME)
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration File",
            str(suggested_path),
            "JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            log_info(f"User selected save config path: {file_path}")
            self.model.config_path = Path(
                file_path
            )  # Update model's config path
            self._save_config_from_editor()  # Save from editor to this path
            self.status_bar.showMessage(
                f"Config saved to {os.path.basename(file_path)}", 3000
            )
            # Reload prompt templates if file paths changed in config
            self.model._load_prompt_templates()
            self._populate_prompt_template_selector()
            # TODO: Reload glossary path and content if changed in config

    @Slot()
    def _load_config_into_editor(self):
        """Loads the current config JSON into the editor tab."""
        if self.model.config:
            try:
                config_json_text = json.dumps(self.model.config, indent=2)
                self.config_editor.setPlainText(config_json_text)
                log_info("Config loaded into editor.")
            except Exception as e:
                log_error(f"Error formatting config to JSON for editor: {e}")
                self.config_editor.setPlainText(
                    "Error loading config into editor."
                )
                self._show_message_box(
                    "Config Error",
                    "Error loading config into editor. See logs.",
                    "warning",
                )

        else:
            self.config_editor.setPlainText("No config loaded.")

    @Slot()
    def _save_config_from_editor(self):
        """Saves the JSON content from the editor tab back to the config file."""
        json_text = self.config_editor.toPlainText()
        if not self.model.config_path:
            log_error(
                "Cannot save config from editor: No config file path set."
            )
            self._show_message_box(
                "Save Config Error",
                "No config file path set. Use 'Save Config As...' first.",
                "warning",
            )
            return

        try:
            # Attempt to parse the text as JSON
            new_config_data = json.loads(json_text)
            # Update the model's config
            self.model.config = new_config_data
            # Save the updated config back to the file
            self.model.config_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure directory exists
            with open(self.model.config_path, "w", encoding="utf-8") as f:
                json.dump(self.model.config, f, indent=2)
            log_info(f"Config saved from editor to {self.model.config_path}")
            self.status_bar.showMessage("Config saved.", 3000)

            # Reload prompt templates based on potentially changed paths in config
            self.model._load_prompt_templates()
            self._populate_prompt_template_selector()
            # TODO: Reload glossary path and content if changed in config

        except json.JSONDecodeError as e:
            log_error(
                f"Invalid JSON format when saving config: {e}", exc_info=True
            )
            self._show_message_box(
                "Save Config Error",
                f"Invalid JSON format in editor: {e}",
                "warning",
            )
        except Exception as e:
            log_error(f"Error saving config from editor: {e}", exc_info=True)
            self._show_message_box(
                "Save Config Error", f"Error saving config: {e}", "warning"
            )

    def _populate_prompt_template_selector(self):
        """Populates the prompt template selector with .txt files in PROMPTS_DIR."""
        self.prompt_template_selector.clear()
        PROMPTS_DIR.mkdir(
            parents=True, exist_ok=True
        )  # Ensure directory exists
        prompt_files = list(PROMPTS_DIR.glob("*.txt"))
        if prompt_files:
            file_names = [f.name for f in prompt_files]
            self.prompt_template_selector.addItems(file_names)
            log_info(
                f"Populated prompt selector with {len(file_names)} files."
            )
            self.prompt_template_selector.setEnabled(True)
            # Load the default prompt file specified in the current config if possible
            if self.model.config and "prompts" in self.model.config:
                default_query_file = Path(
                    self.model.config["prompts"].get("query_prompt_file", "")
                ).name
                if default_query_file in file_names:
                    self.prompt_template_selector.setCurrentText(
                        default_query_file
                    )
                else:
                    self.prompt_template_selector.setCurrentIndex(
                        0
                    )  # Select first if default not found
            else:
                self.prompt_template_selector.setCurrentIndex(0)
            self._load_selected_prompt_template()  # Load the initially selected item
        else:
            self.prompt_template_selector.addItem(
                "No .txt files found in prompts/"
            )
            self.prompt_template_selector.setEnabled(False)
            self.current_prompt_template_editor.setPlainText("")
            self.current_prompt_template_editor.setEnabled(False)
            log_warning("No .txt files found in prompts/.")

    @Slot()
    def _load_selected_prompt_template(self):
        """Loads the content of the selected prompt template file into the editor."""
        selected_file_name = self.prompt_template_selector.currentText()
        if (
            not selected_file_name
            or selected_file_name == "No .txt files found in prompts/"
        ):
            self.current_prompt_template_editor.setPlainText("")
            self.current_prompt_template_editor.setEnabled(False)
            return

        file_path = PROMPTS_DIR / selected_file_name
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.current_prompt_template_editor.setPlainText(f.read())
            self.current_prompt_template_editor.setEnabled(True)
            log_info(f"Loaded prompt template: {selected_file_name}")
        except FileNotFoundError:
            log_error(f"Prompt template file not found: {file_path}")
            self.current_prompt_template_editor.setPlainText(
                f"Error: File not found at {file_path}"
            )
            self.current_prompt_template_editor.setEnabled(False)
            self._show_message_box(
                "Load Prompt Error",
                f"Prompt template file not found: {file_path}",
                "warning",
            )
        except Exception as e:
            log_error(
                f"Error loading prompt template {selected_file_name}: {e}",
                exc_info=True,
            )
            self.current_prompt_template_editor.setPlainText(
                f"Error loading file: {e}"
            )
            self.current_prompt_template_editor.setEnabled(False)
            self._show_message_box(
                "Load Prompt Error",
                f"Error loading prompt template {selected_file_name}: {e}",
                "warning",
            )

    @Slot()
    def _save_current_prompt_template(self):
        """Saves the content from the prompt editor back to the file."""
        selected_file_name = self.prompt_template_selector.currentText()
        if (
            not selected_file_name
            or selected_file_name == "No .txt files found in prompts/"
        ):
            self._show_message_box(
                "Save Prompt Error",
                "No prompt file selected or available.",
                "warning",
            )
            return

        file_path = PROMPTS_DIR / selected_file_name
        try:
            file_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure directory exists
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.current_prompt_template_editor.toPlainText())
            log_info(f"Prompt template saved: {selected_file_name}")
            self.status_bar.showMessage(
                f"Prompt template '{selected_file_name}' saved.", 3000
            )

            # After saving, ensure the model reloads templates
            self.model._load_prompt_templates()

        except Exception as e:
            log_error(
                f"Error saving prompt template {selected_file_name}: {e}",
                exc_info=True,
            )
            self._show_message_box(
                "Save Prompt Error",
                f"Error saving prompt template: {e}",
                "warning",
            )

    # --- Glossary Management ---

    def load_glossary_content(self, file_path: Optional[Path] = None):
        """Loads glossary content from a file and displays it in the editor."""
        glossary_path = (
            file_path
            if file_path is not None
            else (CONFIG_DIR / DEFAULT_GLOSSARY_NAME)
        )

        log_info(f"Loading glossary content from {glossary_path}")
        content = self.model.load_glossary_content_by_path(str(glossary_path))
        self.glossary_editor.setPlainText(content)
        if (
            not content and glossary_path.exists()
        ):  # File existed but was empty or had read error
            self._show_message_box(
                "Load Glossary Error",
                f"Error loading glossary file: {glossary_path}. See logs.",
                "warning",
            )

    def _load_glossary_dialog(self):
        """Opens file dialog to load a glossary XML file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Glossary File",
            str(CONFIG_DIR),  # Suggest config directory
            "XML Files (*.xml);;All Files (*)",
        )
        if file_path:
            self.load_glossary_content(Path(file_path))
            self.status_bar.showMessage(
                f"Glossary loaded from {os.path.basename(file_path)}", 3000
            )

    def _save_glossary_dialog(self):
        """Opens file dialog to save the current glossary content."""
        # Suggest current path if known, otherwise default
        suggested_path = (
            CONFIG_DIR / DEFAULT_GLOSSARY_NAME
        )  # Default save location
        # If glossary path is stored in config and config is loaded, use that
        if self.model.config and "glossary_file" in self.model.config:
            cfg_glossary_path = Path(self.model.config["glossary_file"])
            if (
                cfg_glossary_path.parent.exists()
            ):  # Check if config path is valid
                suggested_path = cfg_glossary_path

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Glossary File",
            str(suggested_path),
            "XML Files (*.xml);;All Files (*)",
        )
        if file_path:
            content = self.glossary_editor.toPlainText()
            save_success = self.model.save_glossary_content(
                Path(file_path), content
            )
            if save_success:
                self.status_bar.showMessage(
                    f"Glossary saved to {os.path.basename(file_path)}", 3000
                )
                # Optional: Update config if saved to a different path
                # if self.model.config and self.model.config.get('glossary_file') != file_path:
                #      self.model.config['glossary_file'] = file_path
                #      # Save config? Or wait for user to save config manually?
                #      # Best to wait for user to save config manually to avoid unexpected saves.
            else:
                self._show_message_box(
                    "Save Glossary Error",
                    "Failed to save glossary file. See logs.",
                    "warning",
                )

    @Slot()
    def _apply_glossary_to_kg(self):
        """Triggers model to process glossary XML and apply to KG."""
        content = self.glossary_editor.toPlainText()
        if not content.strip():
            QMessageBox.information(
                self, "Apply Glossary", "Glossary editor is empty."
            )
            return

        # Optional: Ask for confirmation as this might overwrite existing KG data
        reply = QMessageBox.question(
            self,
            "Apply Glossary to KG",
            "Applying the glossary content will parse its XML and merge entities/relations into the Knowledge Graph. This may update or overwrite existing data based on merge logic. Proceed?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

        self.status_bar.showMessage("Applying Glossary to KG...", 0)
        # Call model method (should ideally be in a thread)
        # For simplicity, direct call for now
        try:
            success = self.model.process_glossary_xml(content)
            if success:
                self.status_bar.showMessage("Glossary applied to KG.", 3000)
                self._update_kg_state_summary()  # Refresh KG state display
            else:
                # Model logs error
                self.status_bar.showMessage(
                    "Failed to apply Glossary to KG. Check logs.", 10000
                )
                # self._show_message_box("Apply Glossary Failed", "Failed to apply glossary to KG. See logs.", "warning")
        except Exception as e:
            log_error(
                f"An unexpected error occurred applying glossary: {e}",
                exc_info=True,
            )
            self.status_bar.showMessage(f"Error applying glossary: {e}", 10000)
            self._show_message_box(
                "Application Error",
                f"An unexpected error occurred applying glossary: {e}",
                "critical",
            )

    # --- Conversation History Management ---

    @Slot()
    def _new_session(self):
        """Starts a new conversation session."""
        reply = QMessageBox.question(
            self,
            "New Session",
            "Starting a new session will clear the current conversation history and Knowledge Graph.\n\nDo you want to save the current session before starting a new one?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )

        if reply == QMessageBox.Save:
            # Attempt to save first
            save_success = self._save_session_dialog(
                save_as=False
            )  # Try saving to current file
            if not save_success and self._current_session_file is None:
                # If no current file and save failed, prompt Save As
                save_success = self._save_session_dialog(save_as=True)

            if not save_success:
                log_warning(
                    "New Session cancelled by user during save prompt."
                )
                return  # Don't proceed if saving failed or was cancelled

        elif reply == QMessageBox.Cancel:
            log_info("New Session cancelled by user.")
            return

        # If we reached here, the user chose Discard or Save (and save succeeded)
        self.model.new_session()  # Clear history in model
        self._current_session_file = None  # Clear current file path
        self.settings.setValue("lastSessionFile", "")  # Clear setting

        # Clear UI displays related to the last turn
        self.narrative_display.clear()
        self.parsed_display.clear()
        self.raw_xml_display.clear()
        self.query_output_display.clear()
        self.user_instruction_input.clear()

        # Trigger KG Reconstruction (will result in empty KG since history is empty)
        # Need to load glossary content again for the rebuild
        glossary_content = self.model.load_glossary_content_by_path(
            self.model.config.get(
                "glossary_file", "approach/glossary.xml"
            )  # Use current config path
        )
        reconstruction_success = self.model.rebuild_kg_from_history(
            glossary_content
        )
        if reconstruction_success:
            log_info("New session started and KG rebuilt (cleared).")
            self.status_bar.showMessage("New session started.", 3000)
            self._update_kg_state_summary()  # Refresh KG state display
            self._update_history_display()  # Refresh history display (should be empty)
        else:
            log_error("Failed to rebuild KG after starting new session.")
            self.status_bar.showMessage(
                "New session started, but KG rebuild failed. See logs.", 10000
            )
            # Still clear history and start new session even if KG rebuild fails

    @Slot()
    def _load_session_dialog(self):
        """Opens file dialog to load a conversation session file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Conversation Session",
            str(
                Path(".")
            ),  # Suggest current directory or default sessions directory
            f"Session Files (*{DEFAULT_SESSION_NAME.split('.')[-1]});;XML Files (*.xml);;All Files (*)",
        )
        if file_path:
            log_info(f"User selected session file to load: {file_path}")
            # Ask for confirmation as loading clears current state
            reply = QMessageBox.question(
                self,
                "Load Session",
                "Loading a session will clear the current conversation history and Knowledge Graph, replacing them with the loaded content. Proceed?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                log_info("Load Session cancelled by user confirmation.")
                return

            load_success = self.model.load_session(Path(file_path))
            if load_success:
                self._current_session_file = Path(
                    file_path
                )  # Store current file path
                self.settings.setValue(
                    "lastSessionFile", str(self._current_session_file)
                )  # Save to settings
                log_info(f"Session {file_path} loaded.")
                self.status_bar.showMessage(
                    f"Session '{os.path.basename(file_path)}' loaded.", 3000
                )
                self._update_history_display()  # Refresh UI with loaded history
                self._update_kg_state_summary()  # KG state updated by load_session
                # Clear last parsed response and narrative display as they might not match history state point?
                # Or only display the last turn's narrative/parsed data? Let's display the last turn's data.
                if self.model._conversation_turns:
                    last_turn = self.model._conversation_turns[-1]
                    # Need to re-parse the last turn's XML to get the Parsed objects and narrative element
                    mock_llm_response = {
                        "raw_xml": last_turn.llm_response_raw_xml,
                        "choices": [
                            {
                                "message": {
                                    "content": last_turn.llm_response_raw_xml
                                }
                            }
                        ],
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": last_turn.llm_response_raw_xml
                                        }
                                    ]
                                }
                            }
                        ],
                    }
                    last_parsed_response = self.model.parse_llm_response_xml(
                        mock_llm_response
                    )
                    if last_parsed_response:
                        self._last_parsed_response = last_parsed_response
                        self._update_display_panels(
                            last_parsed_response
                        )  # Update display panels
                    else:
                        log_error(
                            "Failed to re-parse last turn XML after loading session."
                        )
                        self.narrative_display.setPlainText(
                            "Error displaying last turn narrative after load."
                        )
                        self.parsed_display.setPlainText(
                            "Error displaying last turn parsed data after load."
                        )
                        self.raw_xml_display.setPlainText(
                            last_turn.llm_response_raw_xml
                        )  # Show raw xml anyway
                        self.query_output_display.clear()  # Clear queries
                else:
                    self._last_parsed_response = None
                    self.narrative_display.clear()
                    self.parsed_display.clear()
                    self.raw_xml_display.clear()
                    self.query_output_display.clear()
                    self.kg_state_display.setPlainText(
                        "KG State Summary:\n- KG cleared during session load."
                    )  # Clear KG state display temporarily

            else:
                log_error(f"Failed to load session from {file_path}")
                self.status_bar.showMessage(
                    f"Failed to load session '{os.path.basename(file_path)}'. Check logs.",
                    10000,
                )
                self._show_message_box(
                    "Load Session Error",
                    f"Failed to load session from {file_path}. See logs.",
                    "warning",
                )

    @Slot(bool)
    def _save_session_dialog(self, save_as: bool = False) -> bool:
        """Opens file dialog to save the current conversation session or saves to current file."""
        save_path = None
        if save_as or not self._current_session_file:
            # Prompt for file name if Save As or no current file
            suggested_name = (
                self._current_session_file
                if self._current_session_file
                else Path(
                    f"session_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.ses"
                )
            )
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Conversation Session",
                str(suggested_name),
                f"Session Files (*{DEFAULT_SESSION_NAME.split('.')[-1]});;XML Files (*.xml);;All Files (*)",
            )
            if file_path:
                save_path = Path(file_path)
            else:
                log_info("Save Session dialog cancelled.")
                return False  # User cancelled

        else:
            # Save to the current file
            save_path = self._current_session_file

        log_info(f"Saving session to {save_path}")
        save_success = self.model.save_session(save_path)

        if save_success:
            self._current_session_file = (
                save_path  # Update current file path if Save As
            )
            self.settings.setValue(
                "lastSessionFile", str(self._current_session_file)
            )  # Save to settings
            self.status_bar.showMessage(
                f"Session saved to {os.path.basename(save_path)}", 3000
            )
            log_info(f"Session {save_path} saved successfully.")
            return True
        else:
            log_error(f"Failed to save session to {save_path}")
            self.status_bar.showMessage(
                f"Failed to save session '{os.path.basename(save_path)}'. Check logs.",
                10000,
            )
            self._show_message_box(
                "Save Session Error",
                f"Failed to save session to {save_path}. See logs.",
                "warning",
            )
            return False

    def _update_history_display(self):
        """Refreshes the Conversation History display in the History tab."""
        self.conv_history_display.clear()
        if not self.model._conversation_turns:
            self.conv_history_display.setPlainText(
                "Conversation history is empty."
            )
            self.save_session_action.setEnabled(
                False
            )  # Disable Save button if history is empty
            return

        self.save_session_action.setEnabled(True)  # Enable Save button

        # Format history for display (user-friendly, possibly with rich text)
        history_text = ""
        for i, turn in enumerate(self.model._conversation_turns):
            history_text += f"--- Turn {i+1} ({turn.timestamp.strftime('%Y-%m-%d %H:%M:%S')}) ---\n"
            history_text += f"User:\n{turn.user_prompt_text}\n"
            # Optionally display a summary or link to the raw XML instead of the full raw XML
            # For now, show raw XML, but this might be too much.
            # history_text += f"LLM Response (Raw XML):\n{turn.llm_response_raw_xml[:500]}...\n\n" # Truncate for display
            # Or provide a link/indicator to show/edit the full XML/prompt

            # User-friendly display: Show parsed narrative if available
            # Requires re-parsing the turn's XML to get the narrative element - potentially slow
            # Alternative: Store parsed data with the turn object (increases memory usage)
            # Let's just indicate LLM response received
            history_text += "LLM Response Received (details in Parsed/Raw XML tabs for last turn).\n\n"

            # TODO: Implement context menu or buttons for Edit/Delete Turn

        self.conv_history_display.setPlainText(history_text)
        self.conv_history_display.verticalScrollBar().setValue(
            self.conv_history_display.verticalScrollBar().maximum()
        )  # Auto-scroll

        # TODO: Implement context menu on turns for Edit/Delete
        self.conv_history_display.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.conv_history_display.customContextMenuRequested.connect(
            self._show_history_context_menu
        )

    @Slot(QPoint)
    def _show_history_context_menu(self, pos: QPoint):
        """Shows context menu for history display."""
        cursor = self.conv_history_display.cursorForPosition(pos)
        block = cursor.block()
        block_number = block.blockNumber()

        # Find which turn this block belongs to.
        # This requires mapping line/block number back to turn index.
        # If each turn starts with "--- Turn X ---", find that line.
        text_at_cursor = block.text()
        match = QRegularExpression(r"--- Turn (\d+) \(").match(text_at_cursor)
        if match.hasMatch():
            turn_index_str = match.captured(1)
            try:
                turn_index = (
                    int(turn_index_str) - 1
                )  # Convert 1-based to 0-based index
                if 0 <= turn_index < len(self.model._conversation_turns):
                    # Valid turn found, show menu
                    menu = QMenu(self)
                    edit_action = menu.addAction("Edit Turn...")
                    delete_action = menu.addAction(
                        "Delete Turn and Subsequent"
                    )

                    # Connect actions with turn index
                    edit_action.triggered.connect(
                        lambda: self._edit_history_turn(turn_index)
                    )
                    delete_action.triggered.connect(
                        lambda: self._delete_history_turns_from(turn_index)
                    )

                    menu.exec(self.conv_history_display.mapToGlobal(pos))
                    log_info(
                        f"History context menu shown for turn {turn_index + 1}."
                    )

            except ValueError:
                log_warning(
                    f"Could not parse turn number from history display block: {text_at_cursor}"
                )
        else:
            log_info("Click in history display did not land on a turn header.")

    def _edit_history_turn(self, turn_index: int):
        """Opens a dialog to edit a specific history turn's raw data."""
        log_info(f"Editing history turn index: {turn_index}")
        if not 0 <= turn_index < len(self.model._conversation_turns):
            log_error(f"Invalid turn index for editing: {turn_index}")
            return

        turn = self.model._conversation_turns[turn_index]

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Turn {turn_index + 1}")
        layout = QVBoxLayout()

        layout.addWidget(QLabel("User Prompt (Text sent to LLM):"))
        user_prompt_editor = QTextEdit(turn.user_prompt_text)
        layout.addWidget(user_prompt_editor)

        layout.addWidget(QLabel("LLM Response (Raw XML):"))
        llm_response_editor = QTextEdit(turn.llm_response_raw_xml)
        layout.addWidget(llm_response_editor)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        if dialog.exec() == QDialog.Accepted:
            new_user_prompt = user_prompt_editor.toPlainText()
            new_llm_response_xml = llm_response_editor.toPlainText()

            # Confirm KG rebuild
            reply = QMessageBox.question(
                self,
                "Confirm Edit & Rebuild",
                "Editing this turn will truncate the history from this point and trigger a full KG rebuild from the modified history.\n\nProceed?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                log_info("Edit Turn cancelled by user confirmation.")
                return

            # Call model method to edit turn and trigger rebuild
            success = self.model.edit_turn(
                turn_index, new_user_prompt, new_llm_response_xml
            )
            if success:
                log_info(
                    f"Turn {turn_index + 1} edited and history truncated. KG reconstruction started."
                )
                self.status_bar.showMessage(
                    f"Turn {turn_index + 1} edited. Rebuilding KG...", 0
                )
                self._update_history_display()  # Refresh history display
                # KG state updated by edit_turn calling rebuild_kg_from_history
            else:
                log_error(f"Failed to edit turn {turn_index + 1}.")
                self.status_bar.showMessage(
                    f"Failed to edit turn {turn_index + 1}. See logs.", 10000
                )
                # self._show_message_box("Edit Turn Failed", f"Failed to edit turn {turn_index + 1}. See logs.", "warning")

    def _delete_history_turns_from(self, turn_index: int):
        """Deletes a turn and all subsequent turns after confirmation. Triggers KG rebuild."""
        log_info(f"Request to delete history turns from index: {turn_index}")
        if not 0 <= turn_index < len(self.model._conversation_turns):
            log_error(f"Invalid turn index for deletion: {turn_index}")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Deletion & Rebuild",
            f"This will delete turn {turn_index + 1} and all subsequent turns ({len(self.model._conversation_turns) - turn_index} turns total).\n\nThis action will also trigger a full KG rebuild from the remaining history.\n\nAre you sure you want to proceed?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.No:
            log_info("Delete Turns cancelled by user confirmation.")
            return

        # Call model method to delete turns and trigger rebuild
        success = self.model.delete_turns_from(turn_index)
        if success:
            log_info(
                f"Turns from index {turn_index + 1} onwards deleted. KG reconstruction started."
            )
            self.status_bar.showMessage(f"Turns deleted. Rebuilding KG...", 0)
            self._update_history_display()  # Refresh history display
            # KG state updated by delete_turns_from calling rebuild_kg_from_history
        else:
            log_error(f"Failed to delete turns from index {turn_index + 1}.")
            self.status_bar.showMessage(
                f"Failed to delete turns. See logs.", 10000
            )
            # self._show_message_box("Delete Turns Failed", "Failed to delete turns. See logs.", "warning")

    # --- Theme and Style Methods ---

    @Slot(bool)
    def _on_style_selected(self, checked: bool, style_name: str):
        """Handles style selection from the menu or combobox."""
        # This slot is connected to both menu actions (checkable) and combobox (currentTextChanged).
        # checked=True for menu action when selected, checked=False when deselected (ignore).
        # combobox sends no 'checked' state, so we check the sender.
        if self.sender() == self.style_selector:
            # Event came from combobox, apply the new text directly
            pass  # style_name is already the current text
        elif checked:
            # Event came from menu action and it's checked
            pass  # style_name is the action's text

        try:
            app = QApplication.instance()
            app.setStyle(QStyleFactory.create(style_name))

            # Make sure the style selector (combobox) is updated if changed from menu
            self.style_selector.setCurrentText(style_name)

            # Reapply color scheme to work with the new style
            # Get current color scheme from settings/UI state
            # Prioritize the state of the theme_button (Dark/Light) then the menu actions
            if self.theme_button.isChecked():
                current_scheme_index = 2  # Dark
            else:
                # Check which menu action is checked
                current_scheme_index = 0  # Default to Auto if none checked (shouldn't happen if init is correct)
                for action in self.color_scheme_actions:
                    if action.isChecked():
                        current_scheme_index = action.data()
                        break

            # Apply the color scheme using the enum value
            app.styleHints().setColorScheme(
                Qt.ColorScheme(current_scheme_index)
            )

            # Clear any custom stylesheet (QSS) if applying a standard style/scheme
            app.setStyleSheet("")

            log_info(f"{style_name} style applied")
            self.status_bar.showMessage(f"{style_name} style applied", 3000)

            # Save the setting
            self.settings.setValue("style", style_name)

        except Exception as e:
            log_error(f"Error applying {style_name} style: {e}", exc_info=True)
            self.status_bar.showMessage(f"Error applying style: {e}", 5000)
            self._show_message_box(
                "Style Error",
                f"Error applying {style_name} style: {str(e)}",
                "warning",
            )

    @Slot(bool)
    def _toggle_color_scheme(self, checked: bool):
        """Toggles between light and dark color schemes."""
        app = QApplication.instance()

        # Clear any custom theme setting (custom QSS)
        self.settings.setValue("customTheme", "")
        app.setStyleSheet("")  # Clear stylesheet

        scheme_index = 0  # Default to Auto
        if checked:  # Dark mode requested (toggle button checked)
            scheme_index = 2  # Dark enum
            self.theme_button.setText("Light Mode")
            app.styleHints().setColorScheme(Qt.ColorScheme.Dark)

        else:  # Light mode requested (toggle button unchecked)
            # Could be Light or Auto depending on default preference
            # Let's make the button strictly toggle between Dark and Light
            scheme_index = 1  # Light enum
            self.theme_button.setText("Dark Mode")
            app.styleHints().setColorScheme(Qt.ColorScheme.Light)

        # Update the menu action state to match the button state
        for action in self.color_scheme_actions:
            action.setChecked(action.data() == scheme_index)

        log_info(f"Color scheme toggled to {COLOR_SCHEMES[scheme_index]}")
        self.status_bar.showMessage(
            f"{COLOR_SCHEMES[scheme_index]} color scheme applied", 3000
        )

        # Save the setting
        self.settings.setValue("colorScheme", scheme_index)

    @Slot(bool)
    def _on_color_scheme_selected(
        self, checked: bool, force_index: Optional[int] = None
    ):
        """Handles color scheme selection from the menu."""
        if not checked and force_index is None:
            return  # Ignore unchecked from menu unless forced

        app = QApplication.instance()
        scheme_index = (
            force_index if force_index is not None else self.sender().data()
        )  # Get index from action data

        try:
            app.styleHints().setColorScheme(Qt.ColorScheme(scheme_index))

            # Update menu action check states (uncheck others, check this one)
            for action in self.color_scheme_actions:
                action.setChecked(action.data() == scheme_index)

            # Update toggle button state to match the selected scheme
            if scheme_index == 2:  # Dark
                self.theme_button.setChecked(True)
                self.theme_button.setText("Light Mode")
            else:  # Light or Auto
                self.theme_button.setChecked(False)
                self.theme_button.setText("Dark Mode")

            # Clear any custom stylesheet (QSS)
            app.setStyleSheet("")

            log_info(
                f"{COLOR_SCHEMES[scheme_index]} color scheme applied (from menu)"
            )
            self.status_bar.showMessage(
                f"{COLOR_SCHEMES[scheme_index]} color scheme applied", 3000
            )

            # Save the setting
            self.settings.setValue("colorScheme", scheme_index)

        except Exception as e:
            log_error(
                f"Error applying color scheme index {scheme_index}: {e}",
                exc_info=True,
            )
            self.status_bar.showMessage(
                f"Error applying color scheme: {e}", 5000
            )
            self._show_message_box(
                "Theme Error",
                f"Error applying color scheme: {str(e)}",
                "warning",
            )

    @Slot(int)
    def _update_font_size(self, size: int):
        """Updates the font size for all text elements in the UI."""
        # Save the setting
        self.settings.setValue("fontSize", size)

        # Create a font with the specified size
        app = QApplication.instance()
        font = app.font()
        font.setPointSize(size)

        # Apply to application-wide font - affects most standard widgets
        app.setFont(font)

        # Note: For some widgets (like QTextEdit rendering complex HTML)
        # or when using stylesheets, the application font might not fully apply.
        # Explicitly setting font on text widgets might be needed if styling overrides it.
        # For now, relying on app.setFont(). If needed, uncomment loop below.

        # for text_widget in [
        #     self.narrative_display,
        #     self.parsed_display,
        #     self.raw_xml_display,
        #     self.kg_state_display,
        #     self.log_display, # Use log_display (PlainTextEdit)
        #     self.conv_history_display, # Use conv_history_display (QTextEdit)
        #     self.user_instruction_input,
        #     self.current_prompt_template_editor,
        #     self.config_editor,
        #     self.glossary_editor
        # ]:
        #    try:
        #         text_widget.setFont(font)
        #    except Exception as e:
        #         log_warning(f"Could not set font on {type(text_widget).__name__}: {e}")

        log_info(f"Font size updated to {size}pt")

    def _apply_saved_ui_settings(self):
        """Applies style, color scheme settings saved in QSettings."""
        app = QApplication.instance()

        # Apply saved style first
        saved_style_name = self.settings.value(
            "style", STYLE_SELECTED_THEME, type=str
        )
        if saved_style_name in QStyleFactory.keys():
            try:
                app.setStyle(QStyleFactory.create(saved_style_name))
                self.style_selector.setCurrentText(
                    saved_style_name
                )  # Update selector UI
                log_info(f"Applied saved style: {saved_style_name}")
            except Exception as e:
                log_error(
                    f"Failed to apply saved style '{saved_style_name}': {e}"
                )
                # Fallback to default Fusion
                app.setStyle(QStyleFactory.create(STYLE_SELECTED_THEME))
                self.style_selector.setCurrentText(STYLE_SELECTED_THEME)
                log_warning(
                    f"Failed to apply saved style, falling back to {STYLE_SELECTED_THEME}"
                )
        else:
            # Apply default Fusion if saved style is invalid or not available
            app.setStyle(QStyleFactory.create(STYLE_SELECTED_THEME))
            self.style_selector.setCurrentText(STYLE_SELECTED_THEME)
            log_info(
                f"No saved style found or invalid, applied default: {STYLE_SELECTED_THEME}"
            )

        # Apply saved color scheme
        # Default to 0 (Auto/Unknown) if no setting saved
        saved_color_scheme_index = self.settings.value(
            "colorScheme", 0, type=int
        )
        try:
            # Use styleHints().setColorScheme (requires Qt 6.0+)
            app.styleHints().setColorScheme(
                Qt.ColorScheme(saved_color_scheme_index)
            )

            # Update menu action check state
            for action in self.color_scheme_actions:
                action.setChecked(action.data() == saved_color_scheme_index)

            # Update toggle button state based on saved color scheme
            if saved_color_scheme_index == 2:  # Dark
                self.theme_button.setChecked(True)
                self.theme_button.setText("Light Mode")
            else:  # Light or Auto (0 or 1)
                self.theme_button.setChecked(False)
                self.theme_button.setText("Dark Mode")

            log_info(
                f"Applied saved color scheme index: {saved_color_scheme_index} ({COLOR_SCHEMES[saved_color_scheme_index]})"
            )

        except Exception as e:
            log_error(
                f"Failed to apply saved color scheme index {saved_color_scheme_index}: {e}"
            )
            # Fallback to Auto
            app.styleHints().setColorScheme(Qt.ColorScheme.Unknown)
            for action in self.color_scheme_actions:
                action.setChecked(action.data() == 0)
            self.theme_button.setChecked(False)
            self.theme_button.setText("Dark Mode")
            log_warning(
                f"Failed to apply saved color scheme, falling back to Auto"
            )

        # Font size loading is handled in _update_font_size called from __init__

        # Clear any custom stylesheet if using standard styles/schemes
        app.setStyleSheet("")

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
        # Save geometry and state
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

        # Save current style and color scheme settings are saved in _on_style_selected and _toggle_color_scheme / _on_color_scheme_selected

        # Save the path of the current session file
        if self._current_session_file:
            self.settings.setValue(
                "lastSessionFile", str(self._current_session_file)
            )
        else:
            self.settings.setValue(
                "lastSessionFile", ""
            )  # Clear setting if no session file

        self.model.close_neo4j()  # Ensure Neo4j connection is closed
        event.accept()

    # --- Helper for KG State Summary ---
    @Slot()
    def _update_kg_state_summary(self):
        """Fetches and displays a simple summary of the KG state."""
        log_info("Updating KG state summary display.")
        if not self.model.connect_neo4j():
            self.kg_state_display.setPlainText("Not connected to Neo4j.")
            return

        summary_text = "KG State Summary:\n"
        try:
            # Example: Count total entities and relations, and counts by status
            # Use model's run_cypher_query
            total_entities_result = self.model.run_cypher_query(
                "MATCH (n:Entity) RETURN count(n) AS count"
            )
            summary_text += f"- Total Entities: {total_entities_result[0]['count'] if total_entities_result else 'N/A (query failed)'}\n"

            total_relations_result = self.model.run_cypher_query(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            )  # Count directed relations
            summary_text += f"- Total Relationships: {total_relations_result[0]['count'] if total_relations_result else 'N/A (query failed)'}\n"

            pending_entities = self.model.run_cypher_query(
                "MATCH (n:Entity {status: 'Pending'}) RETURN count(n) AS count"
            )
            summary_text += f"- Pending Entities: {pending_entities[0]['count'] if pending_entities else 'N/A'}\n"

            canon_entities = self.model.run_cypher_query(
                "MATCH (n:Entity {status: 'Canon'}) RETURN count(n) AS count"
            )
            summary_text += f"- Canon Entities: {canon_entities[0]['count'] if canon_entities else 'N/A'}\n"

            ignored_entities = self.model.run_cypher_query(
                "MATCH (n:Entity {status: 'Ignored'}) RETURN count(n) AS count"
            )
            summary_text += f"- Ignored Entities: {ignored_entities[0]['count'] if ignored_entities else 'N/A'}\n"

            # Add relationship status counts if desired (requires indexing or can be slow)
            # pending_relations = self.model.run_cypher_query("MATCH ()-[r {status: 'Pending'}]->() RETURN count(r) AS count")
            # canon_relations = self.model.run_cypher_query("MATCH ()-[r {status: 'Canon'}]->() RETURN count(r) AS count")
            # ignored_relations = self.model.run_cypher_query("MATCH ()-[r {status: 'Ignored'}]->() RETURN count(r) AS count")

        except Exception as e:
            summary_text += f"Error retrieving summary: {e}\n"
            log_error(f"Error updating KG state summary: {e}", exc_info=True)

        self.kg_state_display.setPlainText(summary_text)

    # --- Logging Streaming to UI ---
    def _setup_log_streaming(self):
        """Sets up the custom log handler to stream logs to the QPlainTextEdit."""
        # Create the logging handler instance
        self.log_handler = QTextEditLoggingHandler(self)
        # Set format for the handler
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.log_handler.setFormatter(formatter)
        # Add the handler to the root logger
        logging.getLogger().addHandler(self.log_handler)
        # Connect the signal to the slot
        self.log_handler.log_signal.connect(self.append_log_message)
        log_info("Log streaming setup completed.")

        # Load existing log file content on startup
        try:
            if Path(LOG_FILE).exists():
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    self.log_display.setPlainText(f.read())
                # Scroll to bottom
                self.log_display.verticalScrollBar().setValue(
                    self.log_display.verticalScrollBar().maximum()
                )
                log_info(f"Loaded existing log file: {LOG_FILE}")
            else:
                log_info("No existing log file found to load.")
        except Exception as e:
            log_error(f"Error loading existing log file {LOG_FILE}: {e}")
            self.log_display.setPlainText(
                f"Error loading existing log file: {e}\n"
            )

    @Slot(str, int)
    def append_log_message(self, message: str, level: int):
        """Appends a log message to the QPlainTextEdit display."""
        # Optional: Apply color based on level
        color = "black"
        if level >= logging.ERROR:
            color = "red"
        elif level >= logging.WARNING:
            color = "orange"
        elif level >= logging.INFO:
            color = "black"  # Default color

        # Append message with color
        self.log_display.appendHtml(
            f"<span style='color: {color};'>{message}</span>"
        )
        # Auto-scroll
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

    # --- Message Box Signaling ---
    @Slot(str, str, str)
    def _show_message_box(self, title: str, message: str, icon_type: str):
        """Slot to display a message box from potentially another thread."""
        # Ensure this slot is invoked on the GUI thread
        try:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)

            icon = QMessageBox.Information  # Default
            if icon_type == "warning":
                icon = QMessageBox.Warning
            elif icon_type == "critical":
                icon = QMessageBox.Critical
            elif icon_type == "question":
                icon = (
                    QMessageBox.Question
                )  # Not explicitly in TINS, but useful
            else:  # 'info'
                icon = QMessageBox.Information

            msg_box.setIcon(icon)
            msg_box.exec()  # Show the message box
            log_info(
                f"Displayed message box: Title='{title}', Message='{message}', Icon='{icon_type}'"
            )

        except Exception as e:
            log_error(
                f"Error displaying message box '{title}': {e}", exc_info=True
            )
            # Fallback: print to console if UI message box fails
            print(f"FATAL UI ERROR - Could not display message box:")
            print(f"Title: {title}")
            print(f"Message: {message}")
            print(f"Icon: {icon_type}")
            print(f"Error: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the logger is configured before the app starts
    # Logging is already configured at the top of model.py and setup in MainWindow
    pass  # Keep this pass

    app = QApplication(sys.argv)
    app.setApplicationName(SETTINGS_APP)
    app.setOrganizationName(SETTINGS_ORG)

    # Apply initial/saved style and color scheme settings *before* creating the main window
    # This ensures widgets are styled correctly from instantiation.
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

    # Get saved style and apply
    saved_style_name = settings.value("style", STYLE_SELECTED_THEME, type=str)
    if saved_style_name in QStyleFactory.keys():
        try:
            app.setStyle(QStyleFactory.create(saved_style_name))
            log_info(f"Applied initial application style: {saved_style_name}")
        except Exception as e:
            log_error(
                f"Failed to apply initial saved style '{saved_style_name}': {e}"
            )
            app.setStyle(
                QStyleFactory.create(STYLE_SELECTED_THEME)
            )  # Fallback
            log_warning(
                f"Falling back to default style: {STYLE_SELECTED_THEME}"
            )
    else:
        app.setStyle(
            QStyleFactory.create(STYLE_SELECTED_THEME)
        )  # Apply default
        log_info(
            f"No saved style found or invalid, applied default style: {STYLE_SELECTED_THEME}"
        )

    # Get saved color scheme index and apply
    saved_color_scheme_index = settings.value(
        "colorScheme", 0, type=int
    )  # Default Auto
    try:
        # Use styleHints().setColorScheme (requires Qt 6.0+). TINS specified >= 6.9.0.
        app.styleHints().setColorScheme(
            Qt.ColorScheme(saved_color_scheme_index)
        )
        log_info(
            f"Applied initial color scheme index: {saved_color_scheme_index} ({COLOR_SCHEMES[saved_color_scheme_index]})"
        )
    except Exception as e:
        log_error(
            f"Failed to apply initial saved color scheme index {saved_color_scheme_index}: {e}"
        )
        # Fallback to Auto
        app.styleHints().setColorScheme(Qt.ColorScheme.Unknown)
        log_warning("Falling back to Auto color scheme.")

    # Create and show the main window
    window = MainWindow()

    # Restore window geometry and state
    geometry = settings.value("geometry")
    if geometry:
        window.restoreGeometry(geometry)
        log_info("Restored window geometry.")
    else:
        window.resize(DEFAULT_WINDOW_SIZE)  # Resize if no saved geometry
        log_info(
            f"No saved geometry, using default size: {DEFAULT_WINDOW_SIZE}"
        )

    window_state = settings.value("windowState")
    if window_state:
        window.restoreState(window_state)
        log_info("Restored window state.")
    else:
        log_info("No saved window state.")

    window.show()

    sys.exit(app.exec())
