# app.py
# Main Flask-SocketIO server application for BerzaUnify v2.0
# Serves the user interface and handles task execution events.

import eventlet
# Monkey patch is crucial for socketio background tasks to work correctly with standard libraries
eventlet.monkey_patch()

from flask import Flask, render_template
from flask_socketio import SocketIO
import orchestrator_v2
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask application
app = Flask(__name__)
# A secret key is required for Flask sessions and SocketIO
app.config['SECRET_KEY'] = 'b3rz@uN1fy_v2.0_s3cr3t_k3y!'

# Initialize SocketIO with eventlet as the async mode, as specified in requirements.txt
# Eventlet is a high-performance, non-blocking I/O library suitable for long-running background tasks.
socketio = SocketIO(app, async_mode='eventlet')

@app.route('/')
def index():
    """
    Serves the main user interface file, index.html.
    This is the entry point for users interacting with the system.
    """
    logging.info("Serving index.html to a client.")
    return render_template('index.html')

def log_to_client(message):
    """
    A callback function to emit log messages to the connected client.
    This function will be passed to the orchestrator to allow it to communicate
    its progress back to the user interface in real-time.
    It also prints to the server console for debugging purposes.
    """
    logging.info(f"Emitting log to client: {message}")
    socketio.emit('log_message', {'message': str(message)})

@socketio.on('connect')
def handle_connect():
    """
    Handles a new client connection. Logs the connection on the server.
    """
    logging.info("Client connected.")
    log_to_client("System Ready: Welcome to BerzaUnify v2.0.")

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handles a client disconnection. Logs the event on the server.
    """
    logging.info("Client disconnected.")

@socketio.on('handle_execute_task')
def handle_execute_task(json_data):
    """
    The core event handler that triggers the "Mixture of Experts" orchestration.
    
    This function is triggered when the user clicks the 'Execute' button in the UI.
    Its primary responsibilities are:
    1. Immediately emitting a specific trigger message for the AutoHotkey Sentry.
    2. Starting the main orchestrator logic in a background thread to keep the UI responsive.
    """
    # Accept both 'prompt' and 'task' for compatibility
    user_prompt = (json_data.get('prompt') or json_data.get('task') or '').strip()
    if not user_prompt:
        log_to_client("Error: No prompt provided.")
        return
    try:
        # Start orchestrator in background with positional args
        socketio.start_background_task(orchestrator_v2.main_orchestrator, user_prompt, log_to_client)
        log_to_client("--- EXECUTION_TRIGGERED ---")
        logging.info(f"Started background task for orchestrator with prompt: {user_prompt}")
    except Exception as e:
        log_to_client(f"Error starting orchestrator: {e}")


if __name__ == '__main__':
    """
    Main entry point for the application.
    Runs the Flask-SocketIO server.
    Host '0.0.0.0' makes the server accessible on the local network.
    """
    logging.info("Starting BerzaUnify v2.0 server...")
    # The use of eventlet is explicitly managed by the SocketIO.run command.
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)