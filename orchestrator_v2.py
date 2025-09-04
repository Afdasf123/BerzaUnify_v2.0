# orchestrator_v2.py
# The core 'Mixture of Experts' orchestration logic using the google-genai SDK.

import threading
import time
import random
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# NOTE: The google-genai package has changed names/entry-points across
# releases (eg. `google.genai`, `google.generativeai`, or `google_genai`).
# Importing it at module-import time can cause the whole app to fail if
# the runtime environment's package layout differs. We import it lazily
# inside the caller and try multiple fallback names so the app can start
# even when the AI SDK isn't available. This allows the UI to run and
# surface a clear error message to the user instead of crashing at import.


def _get_genai_module():
    """Try to import the google genai library under several known names.

    Returns the imported module object on success, or None if not found.
    """
    candidates = ["google.genai", "google.generativeai", "google_genai"]
    for name in candidates:
        try:
            module = __import__(name, fromlist=['*'])
            return module
        except Exception:
            continue
    return None

# --- 1. CONFIGURATION ---


# --- 1. CONFIGURATION AND INITIALIZATION (MULTI-KEY, THREAD-SAFE) ---

# Load environment variables from the .env file
load_dotenv()

# Load all 4 of your API keys from the environment file
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4")
]
# Filter out any keys that might be empty or not set
API_KEYS = [key for key in API_KEYS if key]

if not API_KEYS:
    raise ValueError("FATAL: No Gemini API keys found in the .env file. Please set GEMINI_API_KEY_1, etc.")

# A thread-safe way to cycle through your keys
_api_key_index = 0
_api_key_lock = threading.Lock()

def get_next_api_key():
    """Cycles through the available API keys in a thread-safe manner."""
    global _api_key_index
    with _api_key_lock:
        key = API_KEYS[_api_key_index]
        _api_key_index = (_api_key_index + 1) % len(API_KEYS)
        return key

# Define the models to be used for different stages
PLANNER_MODEL = "gemini-2.5-pro-latest"
SYNTHESIZER_MODEL = "gemini-2.5-pro-latest"
DEFAULT_WORKER_MODEL = "gemini-2.5-pro-latest"

# Backwards-compatible names used elsewhere in the file
INTAKE_MODEL = PLANNER_MODEL
WORKER_MODEL_FAST = DEFAULT_WORKER_MODEL
WORKER_MODEL_PRO = SYNTHESIZER_MODEL

# Lock to protect concurrent writes to the shared worker_results dict
results_lock = threading.Lock()

# Debug log lock and path for raw AI responses
ai_debug_lock = threading.Lock()
AI_DEBUG_LOG = 'ai_debug.log'

def log_ai_debug(entry: dict):
    """Append a JSON-like entry to the ai_debug.log file (threads safe).

    entry will be serialized using json.dumps with non-serializable values converted to str().
    """
    try:
        payload = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            **entry
        }
        with ai_debug_lock:
            with open(AI_DEBUG_LOG, 'a', encoding='utf-8') as f:
                try:
                    f.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")
                except Exception:
                    f.write(str(payload) + "\n")
    except Exception:
        # Never raise from logging
        pass


# --- 2. AI CALLER ---

def call_ai(prompt, model_name, is_json=False, socketio=None, emit_log=None):
    """
    A robust function to call a specified Gemini model with a given prompt.

    Args:
        prompt (str): The prompt to send to the AI model.
        model_name (str): The name of the model to use (e.g., 'gemini-2.5-pro').

    Returns:
        str: The text response from the model, or a descriptive error message.
    """
    # Lazy import of the genai module/api to avoid import-time crashes.
    genai = _get_genai_module()
    if genai is None:
        error_message = "AI SDK not installed or importable (tried google.genai, google.generativeai, google_genai)."
        print(error_message)
        return error_message

    # Implement retry with exponential backoff for transient errors (e.g., 5xx, 429, model overloaded)
    MAX_RETRIES = 4
    BACKOFF_FACTOR = 1.0  # base seconds
    MAX_JITTER = 0.5  # seconds
    TRANSIENT_ERROR_CODES = {429, 500, 502, 503, 504}

    # Normalize model names the planner might emit (e.g., 'gemini-2.5-pro-latest' -> 'gemini-2.5-pro')
    model_name = model_name.replace('-latest', '').strip()

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        api_key = get_next_api_key()

        # Lazy import of the genai module/api to avoid import-time crashes.
        genai = _get_genai_module()
        if genai is None:
            error_message = "AI SDK not installed or importable (tried google.genai, google.generativeai, google_genai)."
            if emit_log:
                emit_log(error_message)
            return None

        MAX_RETRIES = 4
        INITIAL_BACKOFF = 1  # Start with a 1-second wait

        for attempt in range(MAX_RETRIES):
            try:
                # --- The Multi-Key Magic ---
                api_key_to_use = get_next_api_key()

                if emit_log:
                    emit_log(f"-> Attempt {attempt + 1}/{MAX_RETRIES}: Calling '{model_name}' with Key #{_api_key_index}... ")
                if socketio:
                    try:
                        socketio.emit('log', {'data': f"-> Attempt {attempt + 1}/{MAX_RETRIES}: Calling '{model_name}' with Key #{_api_key_index}... "})
                    except Exception:
                        pass

                # Prefer GenerativeModel API when available
                if hasattr(genai, 'GenerativeModel'):
                    model = genai.GenerativeModel(model_name, api_key=api_key_to_use)
                    generation_config = None
                    if is_json and hasattr(genai, 'types'):
                        try:
                            generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
                        except Exception:
                            generation_config = None

                    # Try calling generate_content where available
                    try:
                        if generation_config is not None:
                            res = model.generate_content(prompt, generation_config=generation_config)
                        else:
                            # some SDKs expect just prompt
                            try:
                                res = model.generate_content(prompt)
                            except TypeError:
                                res = model.generate(prompt)
                    except Exception as e:
                        raise

                    # Try to extract text
                    raw_text = None
                    if hasattr(res, 'text'):
                        raw_text = getattr(res, 'text')
                    else:
                        # candidates -> candidate.content.parts -> join
                        cand = getattr(res, 'candidates', None)
                        if cand and len(cand) > 0:
                            first = cand[0]
                            # nested content.parts
                            content = getattr(first, 'content', None) or first.get('content') if isinstance(first, dict) else None
                            if content:
                                parts = getattr(content, 'parts', None) or content.get('parts') if isinstance(content, dict) else None
                                if parts:
                                    raw_text = ''.join(getattr(p, 'text', str(p)) for p in parts)

                    # Finalize
                    if raw_text:
                        log_ai_debug({'model': model_name, 'api_key_hint': api_key_to_use[:8] if api_key_to_use else None, 'response': raw_text})
                        if is_json:
                            s = raw_text.strip()
                            if s.startswith('```json'):
                                s = s[s.find('\n')+1: s.rfind('```')].strip()
                            try:
                                return json.loads(s)
                            except Exception:
                                return None
                        return raw_text

                # Fallback: Client-style API
                if hasattr(genai, 'Client'):
                    client = genai.Client(api_key=api_key_to_use)
                    models_api = getattr(client, 'models', None)
                    if models_api and hasattr(models_api, 'generate_content'):
                        try:
                            res = models_api.generate_content(model=model_name, contents=prompt)
                        except TypeError:
                            res = models_api.generate_content(model_name, prompt)

                        # Log raw response dict/object
                        log_ai_debug({'model': model_name, 'api_key_hint': api_key_to_use[:8] if api_key_to_use else None, 'raw_response': res})

                        # Handle explicit error dict
                        if isinstance(res, dict) and 'error' in res:
                            err = res['error']
                            err_msg = str(err)
                            if '503' in err_msg or 'overloaded' in err_msg or (isinstance(err, dict) and err.get('code') in (429, 500, 502, 503, 504)):
                                # transient; will be retried below
                                raise Exception(err_msg)
                            # non-retriable
                            if emit_log:
                                emit_log(f"ERROR: A non-retriable error occurred with model '{model_name}'. Reason: {err_msg}")
                            if socketio:
                                try:
                                    socketio.emit('log', {'data': f"ERROR: A non-retriable error occurred with model '{model_name}'. Reason: {err_msg}"})
                                except Exception:
                                    pass
                            return None

                        # Extract text from known shapes
                        if hasattr(res, 'text') and res.text:
                            log_ai_debug({'model': model_name, 'response': res.text})
                            if is_json:
                                try:
                                    return json.loads(res.text)
                                except Exception:
                                    return None
                            return res.text

                        # dict candidates
                        if isinstance(res, dict) and 'candidates' in res and res['candidates']:
                            first = res['candidates'][0]
                            if isinstance(first, dict):
                                for k in ('content', 'text', 'output', 'message'):
                                    if k in first:
                                        val = first[k]
                                        if isinstance(val, dict) and 'text' in val:
                                            return val['text']
                                        if isinstance(val, str):
                                            return val
                            return str(first)

                # If we reach here, treat as transient and retry
                raise Exception("Empty or unknown response shape; will retry")

            except Exception as e:
                error_str = str(e)
                # transient overloads
                if '503' in error_str or 'overloaded' in error_str or 'UNAVAILABLE' in error_str or '429' in error_str:
                    wait_time = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                    if emit_log:
                        emit_log(f"--- WARNING: Model is overloaded (503). Retrying in {wait_time:.2f} seconds... ---")
                    if socketio:
                        try:
                            socketio.emit('log', {'data': f"--- WARNING: Model is overloaded (503). Retrying in {wait_time:.2f} seconds... ---"})
                        except Exception:
                            pass
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retriable error
                    error_message = f"ERROR: A non-retriable error occurred with model '{model_name}'. Reason: {error_str}"
                    if emit_log:
                        emit_log(error_message)
                    if socketio:
                        try:
                            socketio.emit('log', {'data': error_message})
                        except Exception:
                            pass
                    return None

        # All retries exhausted
        final_error = f"FATAL: AI call failed after {MAX_RETRIES} attempts. The service may be down or your keys may have issues."
        if emit_log:
            emit_log(final_error)
        if socketio:
            try:
                socketio.emit('log', {'data': final_error})
            except Exception:
                pass
        return None
    
    # Protect shared write with a lock to avoid race conditions
    with results_lock:
        results_dict[task_id] = result
    emit_log(f"✅ Finished worker task: '{task_id}'.")


def handle_synthesis_task(synthesis_task, worker_results, original_prompt, conversation_context, emit_log):
    """
    Executes the final synthesis step. It combines worker results with the original context
    and prompt to generate a comprehensive final answer.
    
    Args:
        synthesis_task (dict): The plan for the synthesis task.
        worker_results (dict): The collected results from all worker AIs.
        original_prompt (str): The initial prompt from the user.
        conversation_context (str): The context from the conversation memory file.
        emit_log (function): The function to send log messages back to the frontend.
    """
    emit_log("🔄 Synthesizing final answer from worker results...")
    
    # This is the default prompt for the synthesizer if one isn't provided in the plan.
    synthesis_prompt_template = synthesis_task.get('prompt', """
    You are a master synthesizer AI. Your job is to take an original user request, relevant conversation context, and the results from several specialist AIs to craft a single, comprehensive, and coherent final answer.

    **Original User Request:**
    {original_prompt}

    **Conversation Context:**
    {conversation_context}

    **Results from Specialist AI Workers:**
    {worker_results_formatted}

    **Your Task:**
    Based on all the information above, provide a final, well-structured response that directly addresses the user's original request. Do not just list the worker results; integrate them into a complete answer. The final output should be ready to be presented to the user.
    """)

    # Format the worker results nicely for inclusion in the prompt.
    worker_results_formatted = "\n\n".join(
        [f"--- Result from Task '{task_id}' ---\n{result}" for task_id, result in worker_results.items()]
    )

    final_prompt = synthesis_prompt_template.format(
        original_prompt=original_prompt,
        conversation_context=conversation_context,
        worker_results_formatted=worker_results_formatted
    )
    
    synthesizer_model = synthesis_task.get('model', SYNTHESIZER_MODEL)
    
    final_answer = call_ai(final_prompt, synthesizer_model)

    # If synthesizer exhausted retries and failed, do one last fallback attempt with a smaller model
    if isinstance(final_answer, str) and final_answer.startswith('AI call failed after'):
        fallback_model = 'gemini-2.5-pro-latest'
        emit_log(f"⚠️ Synthesizer failed on {synthesizer_model}. Attempting fallback model: {fallback_model}...")
        log_ai_debug({'event': 'synthesizer_fallback_attempt', 'original_model': synthesizer_model, 'fallback_model': fallback_model, 'reason': final_answer})
        final_answer = call_ai(final_prompt, fallback_model)
        emit_log(f"Fallback model ({fallback_model}) result:")
        emit_log(final_answer)

    emit_log("--- FINAL ANSWER ---")
    emit_log(final_answer)
    emit_log("--- EXECUTION_COMPLETE ---")

def run_ai_task_in_thread(task, worker_results, emit_log):
    """
    A wrapper function designed to be executed in a separate thread.
    It runs a single AI task, logs its progress, and stores the result in a shared dictionary.
    
    Args:
        task (dict): A dictionary containing task details ('task_id', 'prompt', 'model').
        results_dict (dict): A shared dictionary to store the output of the task.
        emit_log (function): The function to send log messages back to the frontend.
    """
    task_id = task.get('task_id', 'unknown_task')
    prompt = task.get('prompt', '')
    model = task.get('model', DEFAULT_WORKER_MODEL) # Use the default if not specified
    
    emit_log(f"▶️ Starting worker task: '{task_id}' (Model: {model})...")
    
    result = call_ai(prompt, model, socketio=None, emit_log=emit_log) # Pass emit_log for logging
    
    # This should use the thread-safe lock we designed
    with results_lock:
        worker_results[task_id] = result
        
    emit_log(f"✅ Finished worker task: '{task_id}'.")

# --- 4. MAIN ORCHESTRATOR ---

def main_orchestrator(user_prompt, emit_log):
    """
    The main controller for the Mixture of Experts system.

    1. Reads conversation context from a file.
    2. Uses an Intake AI to generate a JSON execution plan.
    3. Executes the plan's worker tasks in parallel threads.
    4. Waits for all workers to complete.
    5. Executes a final synthesis task to combine the results into a single answer.
    
    Args:
        user_prompt (str): The user's input from the web interface.
        emit_log (function): The function to send log messages back to the frontend.
    """
    emit_log("Orchestrator V2 Initialized. Reading context...")

    # Step 1: Read Conversation Memory
    try:
        with open('conversation_memory.txt', 'r', encoding='utf-8') as f:
            conversation_context = f.read()
        emit_log("Conversation memory loaded." if conversation_context else "Conversation memory is empty.")
    except FileNotFoundError:
        conversation_context = "No previous conversation context found."
        emit_log("No conversation memory file found. Starting fresh.")
    except Exception as e:
        conversation_context = f"Error reading conversation memory: {e}"
        emit_log(f"⚠️ Error reading conversation_memory.txt: {e}")

    # Step 2: Intake AI - Plan Generation
    emit_log(f"🧠 Generating execution plan using model: {INTAKE_MODEL}...")

    # A detailed prompt that instructs the Intake AI on how to create the JSON plan.
    planner_prompt = f"""
You are an expert AI orchestrator. Your task is to analyze a user's prompt and conversation context, then break it down into a series of parallel tasks for a team of AI workers. You must then define a final synthesis task to combine their results.

**Conversation Context:**
---
{conversation_context}
---

**User's Current Prompt:**
---
{user_prompt}
---

**Instructions:**
1.  Analyze the user's prompt in the context of the conversation.
2.  Create a set of distinct, parallelizable "worker_tasks". Each task should be a specific prompt for an AI model to execute. Think about different angles or sub-problems (e.g., one task for research, one for code generation, one for creative writing).
3.  Define a final "synthesis_task". This task's prompt will be given the results of all worker tasks to produce the final, cohesive answer for the user.
4.  Your output MUST be a valid JSON object with the specified structure. Do not add any text or explanations outside of the JSON object.

**Model Assignment Strategy:**
- For simple, boilerplate files like `requirements.txt`, `.env.template`, or a basic `index.html`, you MUST assign the `gemini-2.5-flash` model.
- For complex, core logic files like `app.py` or `orchestrator_v2.py`, you MUST assign the `gemini-2.5-pro-latest` model.
- If my Master Prompt explicitly requests a parallel review, you should assign BOTH models to the same critical task.
- For the final 'synthesis' task, you MUST always assign the `gemini-2.5-pro-latest` model.

**JSON Structure:**
{{
    "worker_tasks": [
        {{
            "task_id": "a_unique_and_descriptive_id",
            "prompt": "The detailed prompt for this specific worker.",
            "model": "model_name_for_worker"
        }}
    ],
    "synthesis_task": {{
        "task_id": "synthesis",
        "prompt": "The detailed prompt for the synthesizer AI, which will be given the worker results later.",
        "model": "model_name_for_synthesis"
    }}
}}

Now, generate the JSON plan for the user's prompt provided above.
"""
    
    plan_json_str = call_ai(planner_prompt, INTAKE_MODEL)

    # Try to extract the first JSON object from model output
    json_start = plan_json_str.find('{')
    json_end = plan_json_str.rfind('}')
    if json_start != -1 and json_end != -1 and json_end > json_start:
        plan_json_str = plan_json_str[json_start:json_end+1]

    # Quick validation: require double-quoted JSON. If the model returned an error dict
    # (often represented with single quotes) or plain-text error, stop and emit useful logs.
    if '"' not in plan_json_str:
        emit_log("🔴 FATAL ERROR: Intake AI did not return valid JSON (no double quotes detected).")
        emit_log(f"Raw response was:\n{plan_json_str}")
        emit_log("--- EXECUTION_HALTED ---")
        return

    try:
        plan = json.loads(plan_json_str)
        emit_log("✅ Plan generated successfully.")
        emit_log("--- EXECUTION_PLAN ---")
        emit_log(json.dumps(plan, indent=2))
        emit_log("----------------------")
    except json.JSONDecodeError as e:
        emit_log(f"🔴 FATAL ERROR: Failed to decode JSON plan from Intake AI. Error: {e}")
        emit_log(f"Raw response was:\n{plan_json_str}")
        emit_log("--- EXECUTION_HALTED ---")
        return

    # Step 3: Execute Worker Tasks in Parallel
    worker_tasks = plan.get('worker_tasks', [])
    if not worker_tasks:
        emit_log("⚠️ Warning: No worker tasks found in the plan. Proceeding directly to synthesis.")
        worker_results = {}
    else:
        threads = []
        worker_results = {} # Shared dictionary for results
        emit_log(f"🚀 Launching {len(worker_tasks)} worker task(s) in parallel...")
        
        for task in worker_tasks:
            # Create and start a thread for each worker task
            thread = threading.Thread(target=run_ai_task_in_thread, args=(task, worker_results, emit_log))
            threads.append(thread)
            thread.start()

        # Step 4: Wait for all worker threads to complete
        for thread in threads:
            thread.join()
        
        emit_log("🏁 All worker tasks have completed.")

    # Step 5: Execute the Synthesis Task
    synthesis_task = plan.get('synthesis_task')
    if synthesis_task:
        handle_synthesis_task(
            synthesis_task,
            worker_results,
            user_prompt,
            conversation_context,
            emit_log
        )
    else:
        # This is a fallback in case the plan is malformed
        emit_log("🔴 FATAL ERROR: No synthesis task found in the plan.")
        emit_log("--- DUMPING RAW WORKER RESULTS ---")
        for task_id, result in worker_results.items():
            emit_log(f"--- Result from Task '{task_id}' ---")
            emit_log(result)
        emit_log("--- EXECUTION_HALTED ---")