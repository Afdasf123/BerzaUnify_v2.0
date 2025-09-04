markdown
# BerzaUnify v2.0 - Mixture of Experts Orchestration System

Welcome to BerzaUnify v2.0, an advanced "Mixture of Experts" (MoE) orchestration system designed to tackle complex tasks by leveraging a team of specialized AI models. This version has been fully upgraded to use the latest `google-genai` Python SDK, ensuring future-proof compatibility and performance with Google's Gemini family of models.

The system is supported by an intelligent, trigger-based AutoHotkey Sentry that provides real-time external context to the AI team, enabling more accurate and relevant responses.

## Table of Contents
1.  [Key Features](#key-features)
2.  [System Architecture](#system-architecture)
3.  [Prerequisites](#prerequisites)
4.  [Installation Guide](#installation-guide)
5.  [How to Run](#how-to-run)
6.  [Usage Flow](#usage-flow)
7.  [File Descriptions](#file-descriptions)
8.  [Troubleshooting](#troubleshooting)

## Key Features

*   **Mixture of Experts (MoE) Core:** Decomposes complex user requests into smaller, manageable sub-tasks.
*   **Parallel AI Processing:** Sub-tasks are executed in parallel by a team of "Worker" AIs, significantly speeding up response time.
*   **Specialized AI Roles:**
    *   **Intake AI:** Analyzes the initial request and conversation history to create a structured execution plan.
    *   **Worker AIs:** Execute the individual sub-tasks from the plan.
    *   **Synthesizer AI:** Combines the results from all workers into a single, coherent, and polished final response.
*   **Modern Google Gemini Integration:** Utilizes the official and recommended `google-genai` library for robust and efficient communication with Gemini Pro and Gemini Flash models.
*   **Intelligent AHK Sentry:** An AutoHotkey script passively monitors the application and, upon task execution, automatically scrapes the content of an active external chat window to provide up-to-the-minute context.
*   **Real-time UI:** A simple web interface powered by Flask-SocketIO provides real-time logging of the entire orchestration process.

## System Architecture

The application follows a decoupled and event-driven architecture:

```
+-----------------+      +-----------------+      +------------------------+
|   User via UI   |----->|  Flask Server   |----->|   Orchestrator v2.0    |
|  (index.html)   |      |    (app.py)     |      | (orchestrator_v2.py)   |
+-----------------+      +-----------------+      +----------+-------------+
        ^                        |                           |
        |                        | Logs to                   | Uses google-genai SDK
        | (Real-time Logs)       v                           v
        |                  +-----------------+      +------------------------+
        +------------------| application.log |      |   Google Gemini APIs   |
                           +-----------------+      | (Pro & Flash)          |
                                     ^              +------------------------+
                                     |
                                     | Monitors
                           +-----------------+
                           |   AHK Sentry    |
                           |   (sentry.ahk)  |
                           +-----------------+
                                     | Scrapes & Writes
                                     v
                           +-------------------------+
                           | conversation_memory.txt |
                           +-------------------------+
```

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
2.  **AutoHotkey**: [Download AutoHotkey](https://www.autohotkey.com/) (Required for the Sentry context scraper on Windows).
3.  **Google AI API Keys**: You need API keys for both Gemini Pro and Gemini Flash.
    *   Go to [Google AI Studio](https://aistudio.google.com/app/apikey) to create your API keys.

## Installation Guide

Follow these steps to set up BerzaUnify v2.0 on your local machine.

### Step 1: Clone the Repository

Clone this repository to your local machine using Git:
```bash
git clone <repository-url>
cd berzaunify-v2
```

### Step 2: Set up a Python Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all the required Python packages using pip. The `requirements.txt` file is configured to install `google-genai`, the modern SDK for Gemini.

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

You need to provide your Google AI API keys in an environment file.

1.  Make a copy of the template file `.env.template` and name it `.env`.
2.  Open the new `.env` file in a text editor.
3.  Paste your API keys obtained from Google AI Studio.

Your `.env` file should look like this:
```env
# .env
GEMINI_PRO_API_KEY="your-gemini-pro-api-key-here"
GEMINI_FLASH_API_KEY="your-gemini-flash-api-key-here"
```

## How to Run

To run the application, you must start both the AHK Sentry and the Flask web server.

### Step 1: Start the Sentry

Navigate to the project directory and double-click the `sentry.ahk` file. You should see a new green "H" icon appear in your system tray, indicating that the Sentry is active and monitoring for the trigger signal.

### Step 2: Start the Web Application

With your virtual environment still active, run the main Python application from your terminal:

```powershell
python app.py
```

You should see output indicating that the Flask server is running on `http://127.0.0.1:5000`.

### Step 3: Access the User Interface

Open your web browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

You are now ready to use BerzaUnify v2.0.

## Usage Flow

Here is the step-by-step process of using the application:

1.  **Prepare Context:** Open an external application window that contains the conversation or context you want the AI to be aware of (e.g., a customer support chat, an email thread, or a document). Make sure this is the active window on your screen.
2.  **Define Goal:** In the BerzaUnify web UI, type your primary objective or question into the text area. For example: "Based on the user's last message, draft a polite response that addresses their concern about the delivery delay and offer a 10% discount."
3.  **Execute:** Click the **"Execute Task"** button.
4.  **Automatic Scrape:** The AHK Sentry instantly detects the `--- EXECUTION_TRIGGERED ---` signal in the log. It scrapes the content of your active external window and saves it to `conversation_memory.txt`.
5.  **Orchestration Begins:** The orchestrator reads your goal and the newly saved conversation memory.
6.  **AI Planning:** The Intake AI (Gemini Flash) analyzes the combined information and generates a JSON plan with parallelizable sub-tasks.
7.  **Parallel Execution:** Multiple Worker AIs (Gemini Flash) are spawned in separate threads to execute the sub-tasks simultaneously. You will see their progress in the real-time log.
8.  **Synthesis:** Once all workers are finished, the Synthesizer AI (Gemini Pro) reviews all the intermediate results and crafts a final, high-quality response.
9.  **View Result:** The final synthesized answer is displayed in the UI log.

## File Descriptions

*   `app.py`: The main Flask-SocketIO web server that handles communication with the UI and initiates the orchestration process.
*   `orchestrator_v2.py`: The core logic of the MoE system. It manages the AI team, handles parallel processing, and uses the `google-genai` library to communicate with the Gemini models.
*   `index.html`: The front-end user interface for interacting with the system.
*   `sentry.ahk`: The AutoHotkey script that acts as the intelligent Sentry, watching the application log and scraping external window content on demand.
*   `requirements.txt`: A list of all Python dependencies required for the project.
*   `.env`: Your local environment file containing your secret API keys. (This file is ignored by Git).
*   `.env.template`: A template file to show the required format for the `.env` file.
*   `conversation_memory.txt`: A temporary file used by the Sentry to pass scraped context to the orchestrator.
*   `application.log`: A log file where the application writes its status. This is the file monitored by `sentry.ahk`.

## Troubleshooting

*   **"API Key not found" or Authentication Errors:**
    *   Ensure you have correctly copied `.env.template` to `.env`.
    *   Verify that your API keys are pasted correctly inside the `.env` file and that there are no extra spaces or characters.
    *   Make sure you have enabled the necessary APIs in your Google Cloud project associated with the keys.

*   **Sentry is not scraping context:**
    *   Confirm that AutoHotkey is installed on your Windows machine.
    *   Check if the `sentry.ahk` script is running (look for the "H" icon in the system tray). If not, double-click it to start it.

*   **Python `ModuleNotFoundError`:**
    *   Make sure your Python virtual environment is activated.
    *   Run `pip install -r requirements.txt` again to ensure all dependencies are installed correctly.

*   **Web UI is not loading:**
    *   Check the terminal where you ran `python app.py` for any error messages.
    *   Ensure no other application is using port 5000.