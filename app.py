# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, url_for, request, redirect, flash, session, get_flashed_messages, make_response, jsonify # ADDED get_flashed_messages
import os
import sys
import requests
from dotenv import load_dotenv
import paramiko # For SSHing into Lambda instance
from fasthtml.common import *
from typing import Any
# Import fast_html components

# Load environment variables from .env file
load_dotenv()

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'a_very_secret_key_for_flash_messages')
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# --- Configuration for Lambda Cloud ---
LAMBDA_CLOUD_API_KEY = os.getenv('LAMBDA_CLOUD_API_KEY')
LAMBDA_CLOUD_SSH_KEY_NAME = os.getenv('LAMBDA_CLOUD_SSH_KEY_NAME')
# Manually get this from your provisioned instance
LAMBDA_INSTANCE_IP = os.getenv('LAMBDA_INSTANCE_IP', "YOUR_LAMBDA_INSTANCE_PUBLIC_IP")
LAMBDA_INSTANCE_USER = os.getenv('LAMBDA_INSTANCE_USER', "ubuntu")
SSH_PRIVATE_KEY_PATH = os.getenv('SSH_PRIVATE_KEY_PATH')
DOCKER_IMAGE_NAME = os.getenv('DOCKER_IMAGE_NAME', "your-dockerhub-username/controlnet-generator:latest")
REGION = "us-west-1"
# the associated function.
# --- SSH Utilities (Same as before) ---
def run_ssh_command(ip, username, private_key_path, command):
    try:
        key = paramiko.RSAKey.from_private_key_file(private_key_path)
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=ip, username=username, pkey=key)

        print(f"Executing SSH command: {command}")
        stdin, stdout, stderr = client.exec_command(command)
        output = stdout.read().decode('utf-8').strip()
        errors = stderr.read().decode('utf-8').strip()

        if errors:
            print(f"SSH Command Error: {errors}")
        print(f"SSH Command Output: {output}")

        client.close()
        return output, errors
    except Exception as e:
        print(f"SSH connection or command execution failed: {e}")
        return None, str(e)

# --- Helper function for flashing messages ---
def get_flashed_html_messages():
    messages_html = []
    for category, message in get_flashed_messages(with_categories=True):
        messages_html.append(
            Div(message, class_=f"alert alert-{category}")
        )
    if messages_html:
        return Div(*messages_html, class_="messages")
    return ""

def base_layout(title: str, content: Any, scripts: Any = None, navigation : Any = None):
    if navigation is None:
        navigation = Nav(class_="container")  # An empty Nav object
    else:
        navigation = Nav(navigation, class_="container")

    return Div(
    Head(
            Meta(charset="UTF-8"),
                Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
                Title(f"Interactive Flask App - {title}"),
                Link(rel="stylesheet", href=url_for('static', filename='css/style.css'))
        ),
        Body(
            Header(
                    navigation
                ),
                Main(
                    Div(content, class_="container")
                ),
                Footer(
                    P("&copy; 2025 Interactive Flask App")
                ),
                Script(src=url_for('static', filename='js/script.js')),
                scripts or "" # Add page-specific scripts if provided
        )
    )

@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def index():
    is_connected = session.get('lambda_connected', False)
    action_button_section = Div()
    if not is_connected:
        # If NOT connected (or connection failed), show the "Connect" button.
        action_button_section =Div(
            P("Please initialize your Lambda Cloud connection and ensure the Docker image is ready."),
                P(
                    A("Connect to Lambda & Pull Docker Image", href=url_for('connect_lambda'), class_="button-link")
                )
            )

    else:
        # If connected (session['lambda_connected'] is True), show the "Go to Generate" button.
        action_button_section = Div(
            P("Connection established and Docker image pulled. You can now proceed to generate images."),
                P(
                    A("Go to the Image Generation Page", href=url_for('inference'), class_="button-link")
                ),
                P(
                    A("Go to the fine tuning Page", href=url_for('training'), class_="button-link")
                ),
                P(
                   A("Go to the results Page", href=url_for('results'), class_="button-link")
                )
        )
    content = Div(
        H1("Welcome to the Lambda ControlNet App!"),
            P("This application allows you to generate images using a Flux ControlNet model running on Lambda Cloud."),
            P("Before proceeding, ensure you have:"),
            Ul(
                Li("A Lambda Cloud account and API Key."),
                    Li("An SSH key pair uploaded to Lambda Cloud and the private key accessible locally."),
                    Li(
                        f"A running Lambda Cloud instance with Docker installed and SSH accessible at {LAMBDA_INSTANCE_IP}."),
                    Li(f"Your Docker image {DOCKER_IMAGE_NAME} pushed to Docker Hub.")
            ),
            action_button_section,
            get_flashed_html_messages()  # Display messages here too if any
    )

    # In the index() function:
    # Get the final HTML string from fasthtml

    # Return it as a tuple: (body, status_code, headers)
    html_obj = base_layout("Connect to lambda cloud", content)
    print("DEBUG TYPE OF html_obj:", type(html_obj))
    return str(html_obj), 200, {'Content-Type': 'text/html'}

@app.route('/connect_lambda')
def connect_lambda():
    if not LAMBDA_INSTANCE_IP or not SSH_PRIVATE_KEY_PATH:
        flash('Lambda Instance IP or SSH Private Key Path not configured in .env!', 'error')
        return redirect(url_for('index'))
    session['lambda_connected'] = False
    print(f"Attempting to connect to Lambda IP: {LAMBDA_INSTANCE_IP} and pull Docker image.")
    # Command to pull the Docker image
    docker_pull_command = f"docker pull {DOCKER_IMAGE_NAME}"

    stdout, stderr = run_ssh_command(LAMBDA_INSTANCE_IP, LAMBDA_INSTANCE_USER, SSH_PRIVATE_KEY_PATH, docker_pull_command)

    if stderr and "no space left on device" in stderr.lower():
        flash(f'Error pulling Docker image: No space left on device. Please clean up your Lambda instance. Error: {stderr}', 'error')
    elif stderr:
        flash(f'Failed to connect to Lambda or pull Docker image. Check SSH config and Lambda instance. Error: {stderr}', 'error')
    else:
        flash(f'Successfully connected to Lambda and pulled Docker image: {DOCKER_IMAGE_NAME}', 'success')
        session['lambda_connected'] = True

    response = make_response((redirect(url_for('index'))))
    response.mimetype = 'text/html'
    response.status_code = 200  # Optional
    return response

@app.route('/inference')
# ‘/’ URL is bound with hello_world() function.
def inference():
    return

@app.route('/inference/controlnet')
# ‘/’ URL is bound with hello_world() function.
def inference_controlnet():
    return
@app.route('/inference/controlnetReduced')
# ‘/’ URL is bound with hello_world() function.
def inference_controlnet_reduced():
    return

@app.route('/inference/controlnetHed')
# ‘/’ URL is bound with hello_world() function.
def inference_controlnet_hed():
    return
@app.route('/training')
# ‘/’ URL is bound with hello_world() function.
def training():
    return

@app.route('/training/controlnet')
# ‘/’ URL is bound with hello_world() function.
def training_controlnet():
    return
@app.route('/training/controlnetReduced')
# ‘/’ URL is bound with hello_world() function.
def training_controlnet_reduced():
    return

@app.route('/training/controlnetHed')
# ‘/’ URL is bound with hello_world() function.
def training_controlnet_hed():
    return
@app.route('/results')
# ‘/’ URL is bound with hello_world() function.
def results():
    return

@app.route('/results/controlnet')
# ‘/’ URL is bound with hello_world() function.
def results_controlnet():
    return
@app.route('/results/controlnetReduced')
# ‘/’ URL is bound with hello_world() function.
def results_controlnet_reduced():
    return

@app.route('/results/controlnetHed')
# ‘/’ URL is bound with hello_world() function.
def results_controlnet_hed():
    return

# A simple API endpoint for JavaScript to interact with
@app.route('/api/greet', methods=['POST'])
def greet_api():
    data = request.json
    name = data.get('name', 'Guest')
    message = f"Hello, {name}! This message came from Flask via JavaScript."
    return

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    # It's good practice to set a FLASK_SECRET_KEY in your .env
    if not app.secret_key:
        print("WARNING: FLASK_SECRET_KEY is not set in .env. Using a default for development.")
        print("Set FLASK_SECRET_KEY=your_random_string_here for production.")
    if not LAMBDA_CLOUD_API_KEY:
        print("WARNING: LAMBDA_CLOUD_API_KEY is not set in .env.")
    if not SSH_PRIVATE_KEY_PATH:
        print("WARNING: SSH_PRIVATE_KEY_PATH is not set in .env.")

    app.run(debug=True, host='0.0.0.0', port=5000)
