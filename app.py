import queue
import shlex
import tempfile
import threading
import time
import uuid
from io import BytesIO
from typing import Any
from scp import SCPClient

import os
import cloudinary
import cv2
import numpy as np
import paramiko
import requests
import yaml
from PIL import Image
from cloudinary.api import resources
from dotenv import load_dotenv
from fasthtml.common import *
from flask import Flask, url_for, request, redirect, flash, session, get_flashed_messages, jsonify
from huggingface_hub import HfApi, hf_hub_download
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.urandom(12)
LAMBDA_CLOUD_API_BASE = "https://cloud.lambdalabs.com/api/v1/instances"
LAMBDA_CLOUD_API_KEY = os.getenv('LAMBDA_CLOUD_API_KEY')
LAMBDA_INSTANCE_IP = os.getenv('LAMBDA_INSTANCE_IP', "YOUR_LAMBDA_INSTANCE_PUBLIC_IP")
LAMBDA_INSTANCE_USER = os.getenv('LAMBDA_INSTANCE_USER', "ubuntu")
SSH_PRIVATE_KEY_PATH = os.getenv('SSH_PRIVATE_KEY_PATH')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
DOCKER_IMAGE_NAME = os.getenv('DOCKER_IMAGE_NAME',
                              "your-dockerhub-username/controlnet-generator:latest")  # when docker is ready
REGION = "us-west-3"
CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)
HF_NAMESPACE = "tommycik"
api = HfApi()
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
# serializing request to lambda
work_queue = queue.Queue()
# Dictionary to store results, keyed by a unique job ID
results_db = {}
# A lock to serialize the SSH command
worker_lock = threading.Lock()


def convert_to_canny(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    out_path = input_path.replace(".jpg", "_canny.jpg")
    cv2.imwrite(out_path, edges)
    return out_path


def convert_to_hed(input_path):
    from hed_infer import hed_from_path
    return hed_from_path(input_path)


def models_list(namespace=HF_NAMESPACE):
    models = api.list_models(author=namespace)
    result = []
    for model in models:
        try:
            info = api.model_info(model.modelId)
            result.append({
                "id": model.modelId,
                "card_data": info.card_data or {}
            })
        except Exception as e:
            print(f"Skipping {model.modelId}: {e}")
    return result


def model_info(model_id):
    try:
        yaml_path = hf_hub_download(repo_id=model_id,
                                    filename="training_config.yaml",
                                    repo_type="model")
        with open(yaml_path, "r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e:
        print(f"No config for {model_id}: {e}")
        return {}


def scp_to_lambda(local_path, remote_path):
    key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY_PATH)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Attempting to SCP from {local_path} to {remote_path}")
    ssh.connect(LAMBDA_INSTANCE_IP, username=LAMBDA_INSTANCE_USER, pkey=key)

    # Ensure the remote directory exists
    remote_dir = os.path.dirname(remote_path)
    stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {remote_dir}')
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        raise RuntimeError(f"Failed to create remote directory {remote_dir}: {stderr.read().decode()}")

    with SCPClient(ssh.get_transport()) as scp:
        scp.put(local_path, remote_path)


class SSHManager:
    def __init__(self, ip, username, key_path):
        self.ip = ip
        self.username = username
        self.key_path = key_path
        self.client = None
        self.lock = threading.Lock()
        self.connect()

    def connect(self):
        try:
            key = paramiko.RSAKey.from_private_key_file(self.key_path)
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(hostname=self.ip, username=self.username, pkey=key)
            print("SSH connected.")
        except Exception as e:
            print(f"SSH connection failed: {e}")
            self.client = None

    def is_connected(self):
        if not self.client:
            return False
        try:
            transport = self.client.get_transport()
            return transport and transport.is_active()
        except:
            return False

    def reconnect_if_needed(self):
        if not self.is_connected():
            print("SSH not connected, reconnecting...")
            self.connect()

    def run_command(self, command):
        with self.lock:
            self.reconnect_if_needed()
            if not self.client:
                return None, "SSH client not connected"
            stdin, stdout, stderr = self.client.exec_command(command)
            output = stdout.read().decode('utf-8', errors='replace').strip()
            errors = stderr.read().decode('utf-8', errors='replace').strip()
            return output, errors


def worker():
    print("Worker thread has started and is ready to process jobs.")
    while True:
        job = work_queue.get()
        if job is None:
            break

        job_id = job['job_id']
        command = job['command']

        try:
            print(f"[{time.ctime()}] Worker: Starting job {job_id}")

            output, errors = ssh_manager.run_command(command)

            print(f"[{time.ctime()}] Worker: Job {job_id} completed.")
            print("SSH OUTPUT:", output)
            print("SSH ERRORS:", errors)

            result_url = None
            url_match = re.search(r'(https?://\S+)', output)
            if url_match:
                result_url = url_match.group(0)

            if result_url:
                # inference complete
                results_db[job_id] = {"status": "done", "output": result_url}

            elif "Step" in output:
                # training log: parse progress
                match = re.search(r"Step (\d+)/(\d+)", output)
                if match:
                    current, total = map(int, match.groups())
                    progress = int((current / total) * 100)
                    results_db[job_id] = {"status": "running", "progress": progress}
                else:
                    results_db[job_id] = {"status": "running", "message": output[-500:]}

            elif errors:
                results_db[job_id] = {"status": "error", "message": errors}

            else:
                results_db[job_id] = {"status": "unknown", "message": output[-500:]}

        except Exception as e:
            print(f"[{time.ctime()}] Worker: Exception for job {job_id}: {e}")
            results_db[job_id] = {"status": "error", "message": str(e)}
        finally:
            work_queue.task_done()


worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()


# SSH Utilities
def get_lambda_info():
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

            for instance in instances:
                if instance.get("ip") == LAMBDA_INSTANCE_IP:
                    print("DEBUG: Matched instance:", instance)
                    return instance

            print("Instance ID not found.")
            return None
        except Exception as e:
            print(f"Failed to parse Lambda instance JSON: {e}")
            return None
    else:
        print(f"Failed to get instance list: {response.status_code} {response.text}")
        return None


ssh_manager = SSHManager(LAMBDA_INSTANCE_IP, LAMBDA_INSTANCE_USER, SSH_PRIVATE_KEY_PATH)


# Helper function for messages
def flash_html_messages():
    messages_html = []
    for category, message in get_flashed_messages(with_categories=True):
        messages_html.append(
            Div(message, class_=f"alert alert-{category}")
        )
    if messages_html:
        return Div(*messages_html, class_="messages")
    return ""


def base_layout(title: str, content: Any, extra_scripts: list[str] = None):
    cache_buster = int(time.time())
    is_connected = session.get('lambda_connected', False)
    if is_connected:

        navigation = Nav(
            A("Inference", href=url_for('inference'), class_="nav-link"),
            A("Training", href=url_for('training'), class_="nav-link"),
            A("Results", href=url_for('results'), class_="nav-link"),
            class_="main-nav"
        )
    else:
        navigation = Nav(
            A("Connect to Lambda", href=url_for('connect_lambda'), class_="nav-link"),
        )

    scripts = [Script(src=url_for('static', filename='js/script.js', v=cache_buster))]
    if extra_scripts:
        scripts.extend([Script(src=url_for('static', filename=path, v=cache_buster)) for path in extra_scripts])

    return Div(
        Head(
            Meta(charset="UTF-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
            Title(f"ControlNet App - {title}"),
            Link(rel="stylesheet", href=url_for('static', filename='css/style.css', v=cache_buster))
        ),
        Body(
            Header(H1("ControlNet App", class_="site-title"), navigation),
            Main(Div(content, id="main_div", class_="container")),
            Footer(P("© 2025 Lambda ControlNet App")),
            *scripts
        )
    )


@app.route('/')
def index():
    is_connected = session.get('lambda_connected', False)

    if not is_connected:
        action_section = Div(
            P("Before using this app, connect to your Lambda Cloud instance and ensure the Docker image is ready."),
            A("Connect to Lambda & Pull Docker Image", href=url_for('connect_lambda'), cls="button primary"),
        )
    else:
        action_section = Div(
            H2("Lambda instance is connected and Docker image is ready."),
            Div(
                A("Go to Inference Page", href=url_for('inference'), cls="link"),
                A("Go to Training Page", href=url_for('training'), cls="link"),
                A("Go to Results Page", href=url_for('results'), cls="link"),
                class_="points"),
        )

    content = Div(
        H1("Lambda ControlNet App", cls="hero-title"),
        P("Generate and fine-tune images with ControlNet models running on Lambda Cloud.",
          cls="hero-subtitle"),
        Hr(),
        H2("Getting Started"),
        Div(
            P("Create a Lambda Cloud account & API Key."),
            P("Upload an SSH key pair to Lambda Cloud."),
            P(f"Start a Lambda instance with Docker installed (IP: {LAMBDA_INSTANCE_IP})."),
            class_="points"
        ),
        Hr(),
        action_section,
        flash_html_messages()
        , id="content", class_="container")

    return str(base_layout("Home", content)), 200


@app.route('/connect_lambda')
def connect_lambda():
    if not LAMBDA_CLOUD_API_KEY:
        flash('Lambda API key is not configured!', 'error')
        return redirect(url_for('index'))

    instance_data = get_lambda_info()

    if not instance_data:
        flash("Lambda instance IP not found or failed to retrieve data.", "error")
        return redirect(url_for('index'))

    status = instance_data.get("status")

    #todo caricare docker
    if status != "active":
        flash(f"Lambda instance is not active (status: {status})", "error")
        return redirect(url_for('index'))

    command = "source venv_flux/bin/activate && cd tesiControlNetFlux/Src"
    stdout, stderr = ssh_manager.run_command(command)  # command for docker

    if stderr and "no space left on device" in stderr.lower():
        flash(f'Error during SSH: No space left on device. {stderr}', 'error')
    elif stderr:
        flash(f'SSH failed: {stderr}', 'error')
    else:
        flash('Successfully verified Lambda instance is up and SSH is working.', 'success')
        session['lambda_connected'] = True

    return redirect(url_for('index'))


@app.route('/job_result/<job_id>', methods=["GET"])
def get_result(job_id):
    result = results_db.get(job_id)
    if result is None:
        return jsonify({"status": "pending"})
    return jsonify(result)


@app.route('/inference', methods=["GET", "POST"])
def inference():
    if request.method == "POST":
        prompt = request.form['prompt']
        scale = request.form.get('scale', 0.2)
        steps = request.form.get('steps', 50)
        guidance = request.form.get('guidance', 6.0)
        model_id = request.form["model"]
        params = model_info(model_id)
        model_type = params.get("controlnet_type", "canny")
        n4 = params.get("N4", False)
        precision = params.get("mixed_precision", "fp16")  # todo aggiungere questo controllo
        control_image_path = None
        remote_control_path = None

        image_url = request.form.get('control_image_data')
        if image_url and "," in image_url:
            header, encoded = image_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(binary_data)).convert("RGB")

            # Check if image is completely white (or blank)
            np_image = np.array(image)
            if not np.all(np_image == [0, 0, 0]):
                import tempfile
                control_image_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
                image.save(control_image_path)
                remote_control_path = f"/home/ubuntu/tesiControlNetFlux/remote_inputs/{uuid.uuid4()}.jpg"
                scp_to_lambda(control_image_path, remote_control_path)
                if control_image_path and os.path.exists(control_image_path):
                    os.remove(control_image_path)
        # Call Lambda via SSH
        command = (
            f"export CLOUDINARY_CLOUD_NAME={CLOUDINARY_CLOUD_NAME} &&"
            f"export HUGGINGFACE_TOKEN={HUGGINGFACE_TOKEN} &&"
            f"export CLOUDINARY_API_KEY={CLOUDINARY_API_KEY} &&"
            f"export CLOUDINARY_API_SECRET={CLOUDINARY_API_SECRET} &&"
            f"source venv_flux/bin/activate && cd tesiControlNetFlux/Src && python3 scripts/controlnet_infer_api.py "
            f"--prompt \"{prompt}\" --scale {scale} --steps {steps} --guidance {guidance} --controlnet_model {model_id} --N4 {n4} --controlnet_type {model_type}"
        )
        if remote_control_path:
            command += f" --control_image {remote_control_path} "
        job_id = str(uuid.uuid4())
        work_queue.put({"job_id": job_id, "command": command})

        content = Div(
            H2("Inference Job Submitted"),
            P(f"Prompt: {prompt}"),
            P(f"Job ID: {job_id}"),
            Div("Waiting for result...", id="result-section"),
            id="content")
        return str(base_layout("Waiting for Inference", content,
                               extra_scripts=["js/polling.js"])), 200
    models = models_list()
    options = [Option(m["id"].split("/")[-1], value=m["id"]) for m in models]

    form = Form(
        Select(*options, id="model", name="model"),
        Label("Conditioning Scale (0.2):"),
        Input(type="number", name="scale", step="0.1", value="0.2"),
        Label("Steps (50):"),
        Input(type="number", name="steps", value="50"),
        Label("Guidance Scale (6.0):"),
        Input(type="number", name="guidance", step="0.5", value="6.0"),
        Label("Prompt:"), Input(type="text", name="prompt", required=True, cls="input"),
        Label("Upload Images:"), Input(type="file", id="uploadInput", accept="image/*", multiple=True),
        Canvas(id="drawCanvas", width="512", height="512", style="border:1px solid #ccc;"),
        Button("Clear Canvas", type="button", id="clearCanvasBtn", cls="button"),
        Input(type="hidden", name="control_image_data", id="controlImageData"),
        Button("Submit", type="submit", cls="button"),
        method="post", id="content", enctype="multipart/form-data")

    return str(base_layout("Inference", form, extra_scripts=["js/inference.js"])), 200


@app.route("/preprocess_image", methods=["POST"])
def preprocess_image():
    try:
        files = request.files.getlist("images")
        model_id = request.form["model"]  #todo testare, potrebbe dare errore causa usa nome repo e non id
        print(model_id)
        params = model_info(model_id)
        controlnet_type = params.get("controlnet_type", "canny")
        merged_image = None

        for file in files:
            image = Image.open(file.stream).convert("RGB")
            temp_path = f"/tmp/{uuid.uuid4()}.jpg"
            image.save(temp_path)

            if controlnet_type == "canny":
                result_path = convert_to_canny(temp_path)
            elif controlnet_type == "hed":
                result_path = convert_to_hed(temp_path)
            else:
                return jsonify({"status": "error", "error": "Invalid model_type"})

            img_arr = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
            if merged_image is None:
                merged_image = img_arr
            else:
                merged_image = cv2.add(merged_image, img_arr)

        # Convert merged_image back to RGB for canvas
        merged_rgb = cv2.cvtColor(merged_image, cv2.COLOR_GRAY2RGB)
        merged_path = f"/tmp/{uuid.uuid4()}_merged.jpg"
        cv2.imwrite(merged_path, merged_rgb)

        with open(merged_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            data_url = f"data:image/png;base64,{encoded}"
            return jsonify({"status": "ok", "converted_data_url": data_url})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


@app.route('/training', methods=["GET", "POST"])
def training():
    if request.method == "POST":
        mode = request.form["mode"]

        if mode == "existing":
            model_id = request.form["existing_model"]
            hub_model_id = model_id
            params = model_info(hub_model_id)
            reuse = request.form.get("reuse_as_controlnet", "yes")
            if reuse == "yes":
                controlnet_model = hub_model_id
                controlnet_type = params.get("controlnet_type", "canny")
            else:
                controlnet_type = request.form["controlnet_type"]
                if controlnet_type == "canny":
                    controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
                elif controlnet_type == "hed":
                    controlnet_model = "Xlabs-AI/flux-controlnet-hed-diffusers"
                else:
                    controlnet_model = hub_model_id

        else:  # new model
            new_name = request.form["new_model_name"]
            hub_model_id = f"{HF_NAMESPACE}/{new_name}"

            # check existence
            existing = [m["id"] for m in models_list()]
            if hub_model_id in existing:
                flash("Model with this name already exists on HuggingFace!", "error")
                return redirect(url_for("training"))

            controlnet_type = request.form["controlnet_type"]

            # User chooses controlnet
            controlnet_source = request.form.get("controlnet_source", "default")
            if controlnet_source == "default":
                #Pretrained models
                if controlnet_type == "canny":
                    controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
                elif controlnet_type == "hed":
                    controlnet_model = "Xlabs-AI/flux-controlnet-hed-diffusers"
                else:
                    controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"  # fallback
            elif controlnet_source == "existing":
                # User picked an existing model from HF repo
                controlnet_model = request.form["existing_controlnet_model"]
            else:
                controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"

        learning_rate = request.form.get("learning_rate", "2e-6")
        steps = request.form["steps"]
        train_batch_size = request.form["train_batch_size"]
        n4 = request.form["N4"]#todo mixed precison e n4 se modello presistente gia quantizzato non può scegliere
        gradient_accumulation_steps = None
        if "gradient_accumulation_steps" in request.form:
            gradient_accumulation_steps = request.form["gradient_accumulation_steps"]
        resolution = None
        if "resolution" in request.form:
            resolution = request.form["resolution"]
            if resolution and int(resolution) > 512:
                resolution = 512
        checkpointing_steps = None
        if "checkpointing_steps" in request.form:
            checkpointing_steps = request.form["checkpointing_steps"]
        validation_steps = None
        if "validation_steps" in request.form:
            validation_steps = request.form["validation_steps"]
        mixed_precision = None
        if "mixed_precision" in request.form:
            mixed_precision = request.form["mixed_precision"]
        prompt = None
        remote_validation_path = None
        if 'validation_image' in request.files:
            val_img = request.files['validation_image']
            if val_img and val_img.filename:
                filename = secure_filename(val_img.filename)
                validation_image_path = os.path.join(tempfile.gettempdir(), filename)
                val_img.save(validation_image_path)
                remote_validation_path = f"/home/ubuntu/tesiControlNetFlux/remote_inputs/{uuid.uuid4()}_{filename}"
                scp_to_lambda(validation_image_path, remote_validation_path)
                prompt = request.form.get("prompt")  # only then get prompt
                if validation_image_path and os.path.exists(validation_image_path):
                    os.remove(validation_image_path)

        cmd = [
            f"export HUGGINGFACE_TOKEN={shlex.quote(str(HUGGINGFACE_TOKEN or ''))}", "&&",
            "source venv_flux/bin/activate && cd tesiControlNetFlux/Src &&",
            "python3 scripts/controlnet_training_api.py",
            f"--learning_rate {shlex.quote(str(learning_rate))}",
            f"--steps {shlex.quote(str(steps))}",
            f"--hub_model_id {shlex.quote(hub_model_id)}",
            f"--controlnet_model {shlex.quote(controlnet_model)}",
            f"--controlnet_type {shlex.quote(controlnet_type)}",
            f"--N4 {str(bool(n4)).lower()}",
            f"--train_batch_size {shlex.quote(str(train_batch_size))}",
        ]

        if resolution:
            cmd.append(f"--resolution {shlex.quote(str(resolution))}")
        if validation_steps:
            cmd.append(f"--validation_steps {shlex.quote(str(validation_steps))}")
        if mixed_precision:
            cmd.append(f"--mixed_precision {shlex.quote(str(mixed_precision))}")
        if checkpointing_steps:
            cmd.append(f"--checkpointing_steps {shlex.quote(str(checkpointing_steps))}")
        if gradient_accumulation_steps:
            cmd.append(f"--gradient_accumulation_steps {shlex.quote(str(gradient_accumulation_steps))}")
        if remote_validation_path:
            cmd.append(f"--validation_image {shlex.quote(remote_validation_path)}")
        if prompt:
            cmd.append(f"--prompt {shlex.quote(prompt)}")

        command = " ".join(cmd)
        job_id = str(uuid.uuid4())
        work_queue.put({"job_id": job_id, "command": command})
        train_config = {
            "controlnet_type": controlnet_type,
            "controlnet_model": controlnet_model,
            "N4": n4,
            "mixed_precision": mixed_precision,
            "steps": steps,
            "train_batch_size": train_batch_size,
            "learning_rate": learning_rate,
            "resolution": resolution,
            "checkpointing_steps": checkpointing_steps,
            "validation_steps": validation_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "validation_image": remote_validation_path or "default",
            "hub_model_id": hub_model_id,
        }

        yaml_path = os.path.join(tempfile.gettempdir(), "training_config.yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump(train_config, f)

        api = HfApi()
        api.upload_file(
            path_or_fileobj=yaml_path,
            path_in_repo="training_config.yaml",
            repo_id=hub_model_id,
            repo_type="model"
        )
        content = Div(
            H2("Training Job Submitted"),
            P(f"Model: {hub_model_id}"),
            P(f"Job ID: {job_id}"),
            Div("Waiting for training...", id="result-section"),
            id="content")
        return str(base_layout("Waiting for Training", content,
                               extra_scripts=["js/polling.js"])), 200

    models = models_list()
    options = []
    for m in models:
        model_id = m["id"]
        params = model_info(model_id)
        options.append(
            Option(
                model_id.split("/")[-1],
                value=model_id,
                **{
                    "data_controlnet_type": params.get("controlnet_type", "canny"),
                    "data_N4": params.get("N4", False),
                    "data_steps": params.get("steps", 1000),
                    "data_train_batch_size": params.get("train_batch_size", 2),
                    "data_learning_rate": params.get("learning_rate", "2e-6"),
                    "data_mixed_precision": params.get("mixed_precision", "fp16"),
                    "data_gradient_accumulation_steps": params.get("gradient_accumulation_steps", 1),
                    "data_resolution": params.get("resolution", 512),
                    "data_checkpointing_steps": params.get("checkpointing_steps", 250),
                    "data_validation_steps": params.get("validation_steps", 125),
                }
            )
        )
    form = Form(
        Label("Mode:"),
        Select(
            Option("Fine-tune existing model", value="existing"),
            Option("Create new model", value="new"),
            id="mode", name="mode"
        ),

        Div(
            Label("Existing Model:"),
            Select(*options,
                   id="existingModel", name="existing_model"),
            Label("Use existing model as ControlNet model too?"),
            Select(Option("no", value="no"), Option("yes", value="yes"),
                   name="reuse_as_controlnet", id="reuse_as_controlnet"),

            id="existingModelWrapper"

        ),

        Div(
            Label("ControlNet Source:"),
            Select(Option("Use default (based on Canny/HED)", value="default"), Option("Use existing model as ControlNet", value="existing"),
                   name="controlnet_source", id="controlnetSource"),
            Div(
                Label("Select Existing ControlNet Model:"),
                Select(
                    *[
                        Option(m["id"].split("/")[-1], value=m["id"])
                        for m in models
                    ],
                    name="existing_controlnet_model"
                ),
                id="existingControlnetWrapper",
                style="display:none;"
            ),
            Label("New Model Name (Hub ID suffix):"),
            Input(name="new_model_name", id="newModelName", cls="input"),
            id="newModelWrapper", style="display:none;"
        ),

        Label("ControlNet Type:", for_="controlnet_type"),
        Input(id="controlnet_type", name="controlnet_type", required=True, cls="input"),

        Label("N4:", for_="N4"),
        Select(Option("No", value="false"), Option("Yes", value="true"), id="N4", name="N4", cls="input"),

        Label("Learning Rate:"),
        Input(id="learning_rate", name="learning_rate", value="2e-6", cls="input"),

        Label("Training Steps:", for_="steps"),
        Input(id="steps", name="steps", type="number", required=True, cls="input"),

        Label("Train Batch Size:", for_="train_batch_size"),
        Input(id="train_batch_size", name="train_batch_size", type="number", required=True, cls="input"),

        Label("Gradient Accumulation Steps:"),
        Input(id="gradient_accumulation_steps", name="gradient_accumulation_steps", type="number", value="1"),

        Label("Resolution:"),
        Input(id="resolution", name="resolution", type="number", value="512"),

        Label("Checkpointing Steps:"),
        Input(id="checkpointing_steps", name="checkpointing_steps", type="number", value="250"),

        Label("Validation Steps:"),
        Input(id="validation_steps", name="validation_steps", type="number", value="125"),

        Label("Validation Image (JPG):"),
        Input(id="validationImage", name="validation_image", type="file", accept=".jpg,.jpeg", cls="input"),

        Label("Mixed Precision:"),
        Select(Option("fp16", value="fp16"), Option("bf16", value="bf16"),
               id="mixed_precision", name="mixed_precision"),
        Div(
            Label("Prompt:", for_="prompt"),
            Input(id="prompt", name="prompt", cls="input"),
            id="promptWrapper",
            style="display:none;"
        ),

        Button("Start Training", type="submit", cls="button"),

        method="post", enctype="multipart/form-data", id="trainingForm", cls="form"
    ),
    return str(base_layout("Training", form, extra_scripts=["js/training.js"])), 200


@app.route('/results', methods=["GET"])
def results():
    #todo finire
    models = models_list()
    selected_model = request.args.get("model", "all")
    page = int(request.args.get("page", 1))
    per_page = 20

    image_urls = []
    has_more = False

    try:
        if selected_model == "all":
            for m in models:
                model_name = m["id"].split("/")[-1]
                res = resources(type="upload", prefix=f"{HF_NAMESPACE}/{model_name}_results/", max_results=per_page)
                image_urls.extend([item["secure_url"] for item in res["resources"]])
        else:
            model_name = selected_model.split("/")[-1]
            res = resources(type="upload", prefix=f"{HF_NAMESPACE}/{model_name}_results/", max_results=per_page,
                            context=True)
            image_urls = [item["secure_url"] for item in res["resources"]]
            has_more = "next_cursor" in res

            next_cursor = res.get("next_cursor")
    except Exception as e:
        flash(f"Error fetching results: {e}", "error")

    # Model selector
    options = [Option("All Models", value="all", selected=(selected_model == "all"))]
    for m in models:
        options.append(
            Option(
                m["id"].split("/")[-1],
                value=m["id"],
                selected=(selected_model == m["id"])
            )
        )

    model_selector = Form(
        Label("Select Model:", for_="model"),
        Select(*options, name="model", id="model"),
        Button("Show Results", type="submit", cls="button"),
        method="get", cls="form"
    )

    # Image gallery
    cards = [Div(Img(src=url, cls="card-img"), cls="card") for url in image_urls]

    # Pagination controls
    pagination = Div(cls="pagination")
    if page > 1:
        pagination.append(
            A("⬅ Previous", href=url_for("results", model=selected_model, page=page - 1), cls="button secondary")
        )
    if has_more:
        pagination.append(
            A("Next ➡", href=url_for("results", model=selected_model, page=page + 1), cls="button secondary")
        )

    content = Div(
        H1("Model Results", cls="hero-title"),
        model_selector,
        Div(*cards, cls="card-grid"),
        pagination
    )

    return str(base_layout("Results", content)), 200


# main driver function
if __name__ == '__main__':

    if not app.secret_key:
        print("WARNING: FLASK_SECRET_KEY is not set in .env. Using a default for development.")
        print("Set FLASK_SECRET_KEY=your_random_string_here for production.")
    if not LAMBDA_CLOUD_API_KEY:
        print("WARNING: LAMBDA_CLOUD_API_KEY is not set in .env.")
    if not SSH_PRIVATE_KEY_PATH:
        print("WARNING: SSH_PRIVATE_KEY_PATH is not set in .env.")
    app.run(debug=False, host='0.0.0.0', port=5000)
