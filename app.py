# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
import cloudinary
from flask import Flask, url_for, request, redirect, flash, session, get_flashed_messages, make_response, jsonify # ADDED get_flashed_messages
import os
import sys
import requests
from dotenv import load_dotenv
import paramiko # For SSHing into Lambda instance
from fasthtml.common import *
from typing import Any
# Import fast_html components
import time #for cache busting
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# Load environment variables from .env file
load_dotenv()
import threading
#serializing request to lambda
inference_lock = threading.Lock()
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
app.secret_key = os.urandom(12)
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# --- Configuration for Lambda Cloud ---

LAMBDA_INSTANCE_ID = os.getenv('LAMBDA_INSTANCE_ID')
LAMBDA_CLOUD_API_BASE = "https://cloud.lambdalabs.com/api/v1/instances"
LAMBDA_CLOUD_API_KEY = os.getenv('LAMBDA_CLOUD_API_KEY')
# Manually get this from your provisioned instance
LAMBDA_INSTANCE_IP = os.getenv('LAMBDA_INSTANCE_IP', "YOUR_LAMBDA_INSTANCE_PUBLIC_IP")
LAMBDA_INSTANCE_USER = os.getenv('LAMBDA_INSTANCE_USER', "ubuntu")
SSH_PRIVATE_KEY_PATH = os.getenv('SSH_PRIVATE_KEY_PATH')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
DOCKER_IMAGE_NAME = os.getenv('DOCKER_IMAGE_NAME', "your-dockerhub-username/controlnet-generator:latest")
# for when the docker is ready
REGION = "us-west-3"
CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)
# the associated function.
# SSH Utilities
def get_lambda_instance_info(instance_id):
    headers = {
        "Authorization": f"Bearer {LAMBDA_CLOUD_API_KEY}"
    }
    url = f"{LAMBDA_CLOUD_API_BASE}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        try:
            data = response.json()
            instances = data.get("data", [])
            print("DEBUG: Instance list from Lambda API:")
            print(instances)

            for inst in instances:
                if inst.get("id") == instance_id:
                    print("DEBUG: Matched instance:", inst)
                    return inst

            print("Instance ID not found.")
            return None
        except Exception as e:
            print(f"Failed to parse Lambda instance JSON: {e}")
            return None
    else:
        print(f"Failed to get instance list: {response.status_code} {response.text}")
        return None

def run_ssh_command(ip, username, private_key_path, command):
    try:
        key = paramiko.RSAKey.from_private_key_file(private_key_path)
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=ip, username=username, pkey=key)

        print(f"Executing SSH command: {command}")
        stdin, stdout, stderr = client.exec_command(command)
        output = stdout.read().decode('utf-8', errors='ignore').strip()
        errors = stderr.read().decode('utf-8', errors='ignore').strip()

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
    # Default navigation if none is provided
    cache_buster = int(time.time())  # Current Unix timestamp
    if navigation is None:
        navigation = Nav(
            A("Connect to Lambda & Pull Docker Image", href=url_for('connect_lambda'), class_= "nav-link"),
            A("Connect to Lambda & Pull Docker Image", href=url_for('connect_lambda'), class_="nav-link")
        ,class_="main-nav")
    else:
        # If navigation is provided, ensure it's wrapped in a Nav and has the class
        navigation = Nav(navigation, class_="main-nav")

    return Div(
        Head(
            Meta(charset="UTF-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
            Title(f"Interactive Flask App - {title}"),
            Link(rel="stylesheet", href=url_for('static', filename='css/style.css', v=cache_buster))
        ),
        Body(
            Header(
                H1("ControlNet App", class_="site-title"),
                navigation,
            ),
            Main(
                Div(content, class_="container")
            ),
            Footer(
                P("2025 Lambda ControlNet App. All rights reserved.")
            ),
            Script(src=url_for('static', filename='js/script.js')),
            scripts or "" # Add page-specific scripts if provided
        )
    )

@app.route('/')
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
                    #Li(f"Your Docker image {DOCKER_IMAGE_NAME} pushed to Docker Hub.")
            ),
            action_button_section,
            get_flashed_html_messages()  # Display messages here too if any
    )
    html_obj = base_layout("Connect to lambda cloud", content)
    print("DEBUG TYPE OF html_obj:", type(html_obj))
    return str(html_obj), 200, {'Content-Type': 'text/html'}

@app.route('/connect_lambda')
def connect_lambda():
    if not LAMBDA_INSTANCE_ID or not LAMBDA_CLOUD_API_KEY:
        flash('Lambda instance ID or API key is not configured!', 'error')
        return redirect(url_for('index'))

    instance_data = get_lambda_instance_info(LAMBDA_INSTANCE_ID)

    if not instance_data:
        flash("Lambda instance ID not found or failed to retrieve data.", "error")
        return redirect(url_for('index'))

    status = instance_data.get("status")
    public_ip = instance_data.get("ip")

    if status != "active" or not public_ip:
        flash(f"Lambda instance is not active or missing IP (status: {status})", "error")
        return redirect(url_for('index'))

    global LAMBDA_INSTANCE_IP
    LAMBDA_INSTANCE_IP = public_ip
    command = "source venv_flux/bin/activate && cd tesiControlNetFlux/Src"
    stdout, stderr = run_ssh_command(public_ip, LAMBDA_INSTANCE_USER, SSH_PRIVATE_KEY_PATH, command)#command for docker

    if stderr and "no space left on device" in stderr.lower():
        flash(f'Error during SSH: No space left on device. {stderr}', 'error')
    elif stderr:
        flash(f'SSH failed: {stderr}', 'error')
    else:
        flash('Successfully verified Lambda instance is up and SSH is working.', 'success')
        session['lambda_connected'] = True

    return redirect(url_for('index'))

model_map = {
    "hed": "tommycik/controlFluxAlcolHed",
    "reduced": "tommycik/controlFluxAlcolReduced",
    "standard": "tommycik/controlFluxAlcol",
}

@app.route('/inference', methods=["GET", "POST"])
def inference():
    if request.method == "POST":
        prompt = request.form['prompt']
        scale = request.form.get('scale', 0.2)
        steps = request.form.get('steps', 50)
        guidance = request.form.get('guidance', 6.0)
        model = request.form.get('model', 'standard')
        if not inference_lock.acquire(blocking=False):
            return jsonify({"error": "Inference is already running. Please wait."}), 429

        try:
            model_type = "canny"
            N4 = False
            if model == "hed":
                model_type = "hed"
            elif model == "reduced":
                N4 = True

            # Call Lambda via SSH
            command = (
                f"export CLOUDINARY_CLOUD_NAME={CLOUDINARY_CLOUD_NAME} &&"
                f"export HUGGINGFACE_TOKEN={HUGGINGFACE_TOKEN} &&"
                f"export CLOUDINARY_API_KEY={CLOUDINARY_API_KEY} &&"
                f"export CLOUDINARY_API_SECRET={CLOUDINARY_API_SECRET} &&"
                f"source venv_flux/bin/activate && cd tesiControlNetFlux/Src && python3 scripts/controlnet_infer_api.py "
                f"--prompt \"{prompt}\" --scale {scale} --steps {steps} --guidance {guidance} --controlnet_model {model_map[model]} --N4 {N4} --controlnet_type {model_type}"
            )
            output, errors = run_ssh_command(LAMBDA_INSTANCE_IP, LAMBDA_INSTANCE_USER, SSH_PRIVATE_KEY_PATH, command)

            result_url = None
            if output and "https" in output:
                result_url = output.strip()

            content = Div(
                H2("Inference Result"),
                P(f"Prompt: {prompt}"),
                P(f"Result:"),
                Img(src=result_url, style="max-width: 500px;") if result_url else P("Error during inference."),
                P(A("Back to Form", href=url_for('inference')))
            )
        finally:
            inference_lock.release()
        return str(base_layout("Result", content, navigation=A("Back to Home", href=url_for('index')))), 200

    # GET method: render form
    form = Form(
        Select(
            Option("ControlNet HED", value="hed"),
            Option("ControlNet Canny reduced", value="reduced"),
            Option("ControlNet Canny", value="standard"),
            id="model"
        ),
        Label("Prompt:"),
        Input(type="text", name="prompt", required=True),
        Label("Conditioning Scale (0.2):"),
        Input(type="number", name="scale", step="0.1", value="0.2"),
        Label("Steps (50):"),
        Input(type="number", name="steps", value="50"),
        Label("Guidance Scale (6.0):"),
        Input(type="number", name="guidance", step="0.5", value="6.0"),
        Button("Submit", type="submit")
    , method="post")

    return str(base_layout("ControlNet HED Inference", form, navigation=A("Back to Inference Menu", href=url_for('inference')))), 200
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