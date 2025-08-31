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
import sys
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
REGION = os.getenv('REGION')
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

def validate_model_or_fallback(model_id: str, default_model: str):
    try:
        files = api.list_repo_files(model_id)
        if "config.json" in files:
            return model_id
        else:
            print(f"[WARNING] Repo {model_id} non valido, uso fallback {default_model}")
            return default_model
    except Exception as e:
        print(f"[ERROR] Impossibile accedere al repo {model_id}: {e}")
        return default_model

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
            # Imposta il keepalive ogni 30 secondi
            transport = self.client.get_transport()
            if transport:
                transport.set_keepalive(30)
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
            out_chunks = []
            err_chunks = []
            for line in iter(stdout.readline, ""):
                out_chunks.append(line)
            for line in iter(stderr.readline, ""):
                err_chunks.append(line)
            return "".join(out_chunks), "".join(err_chunks)


# def setup_lambda_instance():
#     if not LAMBDA_CLOUD_API_KEY:
#         app.logger.error("Lambda API key is not configured — skipping setup.")
#         return
#
#     instance_data = get_lambda_info()
#     if not instance_data or instance_data.get("status") != "active":
#         app.logger.error("Lambda instance is not active — skipping setup.")
#         return
#
#     # Check if image exists remotely
#     check_image_cmd = f"sudo docker images -q {DOCKER_IMAGE_NAME}"
#     image_id, _ = ssh_manager.run_command(check_image_cmd)
#
#     if not image_id.strip():
#         app.logger.info(f"Pulling Docker image {DOCKER_IMAGE_NAME} on Lambda...")
#         ssh_manager.run_command(f"sudo docker pull {DOCKER_IMAGE_NAME}")
#     else:
#         app.logger.info(f"Image {DOCKER_IMAGE_NAME} already present on Lambda — skipping pull.")
#
#     # Stop & remove any existing container
#     ssh_manager.run_command("sudo docker stop controlnet || true")
#     ssh_manager.run_command("sudo docker rm controlnet || true")
#
#     # Start the container (ensure Python is available in the image!)
#     run_cmd = f"""
#         sudo docker run -d --gpus all \
#             --name controlnet \
#             {DOCKER_IMAGE_NAME} sleep infinity
#     """
#     stdout, stderr = ssh_manager.run_command(run_cmd)
#     if stderr:
#         app.logger.error(f"Error starting container: {stderr}")
#
#     # Update repo inside the container
#     update_repo_cmd = """
#         sudo docker exec controlnet bash -c '
#             cd /workspace/tesiControlNetFlux &&
#             git reset --hard &&
#             git pull origin main
#         '
#     """
#     ssh_manager.run_command(update_repo_cmd)
#     app.logger.info("Lambda container is ready.")
#
#
# def run_setup_in_context():
#     with app.app_context():
#         setup_lambda_instance()
#
# # Start a background thread so app boot isn't blocked
# threading.Thread(target=run_setup_in_context, daemon=True).start()

@app.route('/connect_lambda')
def connect_lambda():
    if not LAMBDA_CLOUD_API_KEY:
        return str(base_layout("Error", P("Lambda API key is not configured!"))), 400

    instance_data = get_lambda_info()
    if not instance_data:
        return str(base_layout("Error", P("Lambda instance not found."))), 400
    if instance_data.get("status") != "active":
        return str(base_layout("Error", P(f"Instance not active (status={instance_data['status']})"))), 400

    log_lines = []
    def log(msg):
        log_lines.append(msg)
        print(msg)

    # Step 1: fix DNS, docker config
    log("Configuring Docker daemon...")
    ssh_manager.run_command("echo 'nameserver 8.8.8.8' | sudo tee /etc/resolv.conf")
    ssh_manager.run_command("echo '{ \"ipv6\": false }' | sudo tee /etc/docker/daemon.json")
    ssh_manager.run_command("sudo systemctl restart docker")

    # Step 2: pull image
    log(f"Pulling Docker image {DOCKER_IMAGE_NAME}...")
    out, err = ssh_manager.run_command(f"sudo docker pull {DOCKER_IMAGE_NAME}")
    log(out or err)

    # Step 3: ensure container fresh
    log("Removing any old container...")
    ssh_manager.run_command("sudo docker rm -f controlnet || true")

    # Step 4: run container
    log("Starting new container...")
    out, err = ssh_manager.run_command(
        f"sudo docker run -d --gpus all --name controlnet "
        f"--entrypoint python3 {DOCKER_IMAGE_NAME} -c 'import time; time.sleep(1e9)'"
    )
    log(out or err)

    log("Installing extra Python packages (pyyaml, huggingface_hub)...")
    ssh_manager.run_command(
        "sudo docker exec controlnet pip install --upgrade pip && "
        "sudo docker exec controlnet pip install pyyaml huggingface_hub"
    )
    # Step 5: check running
    status, _ = ssh_manager.run_command("sudo docker inspect -f '{{.State.Running}}' controlnet || echo false")
    if "true" not in status.lower():
        exists, _ = ssh_manager.run_command("sudo docker ps -a --format '{{.Names}}' | grep -w controlnet || true")
        if not exists.strip():
            log("Container was not created at all.")
        else:
            logs, _ = ssh_manager.run_command("sudo docker logs controlnet")
            log("Container failed to start:\n" + logs)
        return str(base_layout("Connect Lambda Failed", Pre("\n".join(log_lines)))), 500

    # Step 6: update repo
    log("Updating repo inside container...")
    ssh_manager.run_command(
        "sudo docker exec controlnet bash -c 'cd /workspace/tesiControlNetFlux &&  git pull '"
    )

    session['lambda_connected'] = True
    log("Lambda is ready.")

    content = Div(
        H1("Connect Lambda Logs"),
        Pre("\n".join(log_lines), style="text-align:left; background:#222; color:#0f0; padding:1rem; border-radius:8px; max-height:60vh; overflow:auto;"),
        A("Continue to Home", href=url_for('index'), cls="button primary")
    )
    return str(base_layout("Connect Lambda", content)), 200


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
            status_cmd = "sudo docker inspect -f '{{.State.Running}}' controlnet 2>/dev/null || echo false"
            out, _ = ssh_manager.run_command(status_cmd)
            if "true" not in out.lower():
                ssh_manager.run_command(
                    f"sudo docker rm -f controlnet || true && "
                    f"sudo docker start controlnet || "
                    f"sudo docker run -d --gpus all --name controlnet "
                    f"--entrypoint python3 {DOCKER_IMAGE_NAME} -c \"import time; time.sleep(1e9)\""
                )
            output, errors = ssh_manager.run_command(command)

            print(f"[{time.ctime()}] Worker: Job {job_id} completed.")
            print("SSH OUTPUT:", output)
            print("SSH ERRORS:", errors)

            result_url = None
            url_match = re.search(r'(https?://\S+)', output)
            if url_match:
                result_url = url_match.group(0)

            if result_url:
                print("done")
                results_db[job_id] = {"status": "done", "output": result_url}
            elif "Step" in output and not re.search(r"_complete\b", output, re.IGNORECASE):
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
                results_db[job_id] = {"status": "done", "output": "done"}

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
            A("Inference", href=url_for('inference'), cls="nav-link"),
            A("Training", href=url_for('training'), cls="nav-link"),
            A("Results", href=url_for('results'), cls="nav-link"),
            cls="nav",
        )
    else:
        navigation = Nav(
            A("Connect to Lambda", href=url_for('connect_lambda'), cls="nav-link"),
            cls="nav",
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
            Header(H1("ControlNet App", cls="site-title"), navigation),
            Main(Div(content, id="main_div", cls="container")),
            Footer(P("© 2025 Lambda ControlNet App")),
            *scripts,
            #todo navtoggle per menu mobile
            Script(f"""
                document.addEventListener("DOMContentLoaded", () => {{
                  const nav = document.querySelector(".nav");
                  const toggle = document.getElementById("navToggle");
                  if (toggle && nav){{
                    toggle.addEventListener("click", () => nav.classList.toggle("open"));
                  }}
            
                  document.querySelectorAll(".messages .alert").forEach(el => {{
                    setTimeout(() => {{ el.style.transition = "opacity .4s"; el.style.opacity = "0"; }}, 4000);
                  }});
            
            
                  if (location.hash){{
                    const t = document.querySelector(location.hash);
                    if (t) t.scrollIntoView({{behavior:"smooth", block:"start"}});
                  }}
                }});
            """)
        )
    )


@app.route('/')
def index():
    is_connected = session.get('lambda_connected', False)

    if not is_connected:
        action_section = Div(
            P("Before using this app, connect to your Lambda Cloud instance and ensure the Docker image is ready."),
            A("Connect to Lambda & Pull Docker Image",id="connectBtn", href=url_for('connect_lambda'), cls="button "
                                                                                                           "primary"),
            Script(f"""
                document.getElementById("connectBtn").addEventListener("click", function(){{
                  this.innerText = "Connecting...";
                  this.classList.add("loading");
                  }});
                """),
            cls="center-box"
        )
    else:
        action_section = Div(
            H2("Lambda instance is connected and Docker image is ready."),
            Div(
                A("Go to Inference Page", cls="button primary", href=url_for('inference')),
                A("Go to Training Page", cls="button primary", href=url_for('training')),
                A("Go to Results Page", cls="button primary", href=url_for('results')),

                cls="points_links"
            ),
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
            cls="points"
        ),
        Hr(),
        action_section,
        flash_html_messages()
        , id="content", class_="container")

    return str(base_layout("Home", content)), 200


@app.route('/job_result/<job_id>', methods=["GET"])
def get_result(job_id):
    result = results_db.get(job_id)
    if result is None:
        return jsonify({"status": "waiting"}), 200
    return jsonify(result), 200


@app.route('/inference', methods=["GET", "POST"])
def inference():
    is_connected = session.get('lambda_connected', False)
    if not is_connected:
        return redirect(url_for('index'))
    if request.method == "POST":
        prompt = request.form['prompt']
        scale = request.form.get('scale', 0.2)
        steps = request.form.get('steps', 50)
        guidance = request.form.get('guidance', 6.0)
        model_chosen = request.form["model"]
        default_canny = "InstantX/FLUX.1-dev-Controlnet-Canny"
        model_id = validate_model_or_fallback(model_chosen, default_canny)
        if model_id != model_chosen:
            flash(f"Repo {model_chosen} non valido, uso modello di default {default_canny}", "error")
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
            if not np.all(np_image == 0):
                import tempfile
                control_image_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
                image.save(control_image_path)
                remote_control_path = f"/home/ubuntu/tesiControlNetFlux/remote_inputs/{uuid.uuid4()}.jpg"
                scp_to_lambda(control_image_path, remote_control_path)
                ssh_manager.run_command(
                    f"sudo docker cp /home/ubuntu/tesiControlNetFlux/remote_inputs/. controlnet:/workspace/tesiControlNetFlux/remote_inputs/")
                remote_control_path = f"/workspace/tesiControlNetFlux/remote_inputs/{uuid.uuid4()}.jpg"
                if control_image_path and os.path.exists(control_image_path):
                    os.remove(control_image_path)
        # Call Lambda via SSH
        command = (
            f"sudo docker exec -e CLOUDINARY_CLOUD_NAME={CLOUDINARY_CLOUD_NAME} "
            f"-e HUGGINGFACE_TOKEN={HUGGINGFACE_TOKEN} "
            f"-e CLOUDINARY_API_KEY={CLOUDINARY_API_KEY} "
            f"-e CLOUDINARY_API_SECRET={CLOUDINARY_API_SECRET} "
            f"controlnet python3 /workspace/tesiControlNetFlux/Src/scripts/controlnet_infer_api.py "
            f"--prompt \"{prompt}\" --scale {scale} --steps {steps} --guidance {guidance} "
            f"--controlnet_model {model_id} --N4 {n4} --controlnet_type {model_type}"
        )
        if remote_control_path:
            command += f" --control_image {remote_control_path} "
        job_id = str(uuid.uuid4())
        work_queue.put({"job_id": job_id, "command": command})

        content = Div(
            H2("Inference Job Submitted", id="job-status"),
            P(f"Prompt: {prompt}"),
            P(f"Job ID: {job_id}"),
            Div("Waiting for result...", id="result-section"),
            Script(f"""
                    async function pollResult() {{
                        const res = await fetch("{url_for('get_result', job_id=job_id)}");
                        const data = await res.json();
                        let resultDiv = document.getElementById("result-section");
                        if (data.status === "done") {{
                            document.getElementById("job-status").innerText = "Inference Job Terminated";
                            if (data.output && data.output.startsWith("http")) {{
                                resultDiv.innerHTML = `<img src="${{data.output}}" style='max-width: 500px;'/>`;
                            }} else {{
                                resultDiv.innerHTML = "<p>" + data.output + "</p>";
                            }}
                        }} else if (data.status === "running") {{
                            resultDiv.innerHTML = `<p>Progress: ${{data.progress || 0}}%</p>`;
                            setTimeout(pollResult, 2000);
                        }} else {{
                            resultDiv.innerHTML = "<p>Waiting...</p>";
                            setTimeout(pollResult, 2000);
                        }}
                            }}
                    pollResult();
                    """),
            id="content")

        return str(base_layout("Waiting for Inference", content)), 200
    models = models_list()
    options = [Option(m["id"].split("/")[-1], value=m["id"]) for m in models]

    form = Form(
        Fieldset(
            Legend("Model Selection"),
            Select(*options, id="model", name="model"),
        ),
        Fieldset(
            Legend("Parameters"),
            Div(
                Label("Scale:"), Input(type="number", name="scale", step="0.1", value="0.2"),
                Label("Steps:"), Input(type="number", name="steps", value="50"),
                Label("Guidance:"), Input(type="number", name="guidance", step="0.5", value="6.0"),
                cls="form-row"
            )
        ),
        Fieldset(
            Legend("Prompt & Control Image"),
            Label("Prompt:"), Input(type="text", name="prompt", required=True, cls="input"),
            Label("Upload Image:"),
            Input(type="file", name="images", id="uploadInput", accept="image/*", multiple=True),
            Canvas(id="drawCanvas", width="512", height="512", style="border:1px solid #ccc;"),
            Button("Clear Canvas", type="button", id="clearCanvasBtn", cls="button secondary"),
            Input(type="hidden", name="control_image_data", id="controlImageData")
        ),
        Button("Run Inference", type="submit", cls="button primary"),
        Script(f"""
                    document.addEventListener("DOMContentLoaded", function () {{
                        const canvas = document.getElementById("drawCanvas");
                        const ctx = canvas.getContext("2d");
                    
                        ctx.fillStyle = "black";
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                        let drawing = false;
                    
                        canvas.addEventListener("mousedown", () => drawing = true);
                        canvas.addEventListener("mouseup", () => {{
                            drawing = false;
                            ctx.beginPath();
                        }});
                        canvas.addEventListener("mousemove", function(e) {{
                            if (!drawing) return;
                            ctx.lineWidth = 0.5;
                            ctx.lineCap = "round";
                            ctx.strokeStyle = "white";
                            ctx.lineTo(e.offsetX, e.offsetY);
                            ctx.stroke();
                            ctx.beginPath();
                            ctx.moveTo(e.offsetX, e.offsetY);
                        }});
                    
                        function clearCanvas() {{
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.fillStyle = "black";
                            ctx.fillRect(0, 0, canvas.width, canvas.height);
                            window.convertedImage = null;
                        }}
                    
                        document.getElementById("clearCanvasBtn").addEventListener("click", clearCanvas);
                    
                        document.getElementById("uploadInput").addEventListener("change", async function (e) {{
                            const files = e.target.files;
                            if (!files.length) return;
                    
                            const model = document.getElementById("model").value;
                            const formData = new FormData();
                    
                            for (let file of files) {{
                                formData.append("images", file);
                            }}
                            formData.append("model", model);
                    
                            const response = await fetch("/preprocess_image", {{ method: "POST", body: formData }});
                            const data = await response.json();
                    
                            if (data.status === "ok") {{
                                const img = new Image();
                                img.onload = function () {{
                                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                                    window.convertedImage = img;
                                }};
                                img.src = data.converted_data_url;
                            }} else {{
                                alert("Error: " + data.error);
                                }}
                        }});
                    
                        document.querySelector("form").addEventListener("submit", function () {{
                            const dataURL = canvas.toDataURL("image/png");
                            document.getElementById("controlImageData").value = dataURL;
                            }});
                        }});
        """),
        method="post",
        id="content",
        enctype="multipart/form-data",
        cls="form-card",)

    return str(base_layout("Inference", form, extra_scripts=["js/inference.js"])), 200


@app.route("/preprocess_image", methods=["POST"])
def preprocess_image():
    try:
        print("Received form data:", request.files)
        files = request.files.getlist("images")
        model_id = request.form["model"]
        print(model_id)
        params = model_info(model_id)
        controlnet_type = params.get("controlnet_type", "canny")
        merged_image = None

        temp_files_to_clean = []

        for file_storage in files:
            temp_path = os.path.join(tempfile.gettempdir(),
                                     secure_filename(file_storage.filename) + "_" + str(uuid.uuid4()))
            temp_files_to_clean.append(temp_path)
            file_storage.save(temp_path)

            result_path = None
            if controlnet_type == "canny":
                result_path = convert_to_canny(temp_path)
            elif controlnet_type == "hed":
                result_path = convert_to_hed(temp_path)
            else:
                return jsonify({"status": "error", "error": f"Invalid model_type: {controlnet_type}"})

            # CRITICAL: Verify the file exists before proceeding.
            if not os.path.exists(result_path):
                raise FileNotFoundError(
                    f"Processed image file was not created by the conversion function: {result_path}")

            # Add the processed file to the cleanup list
            if result_path and result_path != temp_path:
                temp_files_to_clean.append(result_path)

            img_arr = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
            if img_arr is None:
                raise ValueError(f"Failed to read processed image at {result_path}")

            desired_size = (512, 512)
            img_arr = cv2.resize(img_arr, desired_size, interpolation=cv2.INTER_AREA)
            if merged_image is None:
                merged_image = img_arr
            else:
                merged_image = cv2.add(merged_image, img_arr)

        if merged_image is None:
            return jsonify({"status": "error", "error": "No valid images were processed."}), 500

        # Convert merged_image back to RGB for canvas
        merged_rgb = cv2.cvtColor(merged_image, cv2.COLOR_GRAY2RGB)
        merged_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_merged.jpg")
        temp_files_to_clean.append(merged_path)
        cv2.imwrite(merged_path, merged_rgb)

        with open(merged_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            data_url = f"data:image/jpg;base64,{encoded}"

        return jsonify({"status": "ok", "converted_data_url": data_url})
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "error": str(e)})
    finally:
        # Cleanup temporary files in a robust way
        for fpath in temp_files_to_clean:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except OSError as e:
                print(f"Error removing temporary file {fpath}: {e}")


@app.route('/training', methods=["GET", "POST"])
def training():
    is_connected = session.get('lambda_connected', False)
    if not is_connected:
        return redirect(url_for('index'))
    if request.method == "POST":
        mode = request.form["mode"]

        controlnet_type = request.form["controlnet_type"]
        if (controlnet_type.lower() != "canny") & (controlnet_type.lower() != "hed"):
            flash("Unexpected type of controlnet, using Canny as default", "error")
            controlnet_type = "canny"
        if mode == "existing":
            model_id = request.form["existing_model"]
            default_canny = "InstantX/FLUX.1-dev-Controlnet-Canny"

            hub_model_id = validate_model_or_fallback(model_id, default_canny)
            if hub_model_id != model_id:
                flash(f"Repo {model_id} non valido, uso modello di default {default_canny}", "error")
            reuse = request.form.get("reuse_as_controlnet", "yes")
            if reuse == "yes":
                controlnet_model = hub_model_id
            else:
                controlnet_source = request.form.get("controlnet_source_existing", "canny")
                if controlnet_source == "canny":
                    controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
                elif controlnet_source == "hed":
                    controlnet_model = "Xlabs-AI/flux-controlnet-hed-diffusers"
                elif controlnet_source == "existing":
                    controlnet_model_tmp = request.form["existing_controlnet_model_existing"]
                    default_canny = "InstantX/FLUX.1-dev-Controlnet-Canny"

                    controlnet_model = validate_model_or_fallback(controlnet_model_tmp, default_canny)
                    if controlnet_model != controlnet_model_tmp:
                        flash(f"Repo {controlnet_model_tmp} non valido, uso modello di default {default_canny}",
                              "error")

                else:
                    controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"

        else:  # new model
            new_name = request.form["new_model_name"]
            hub_model_id = f"{HF_NAMESPACE}/{new_name}"

            # check existence
            existing = [m["id"] for m in models_list()]
            if hub_model_id in existing:
                flash("Model with this name already exists on HuggingFace!", "error")
                return redirect(url_for("training"))

            # User chooses controlnet
            controlnet_source = request.form.get("controlnet_source", "canny")
            if controlnet_source == "canny":
                controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
            elif controlnet_source == "hed":
                controlnet_model = "Xlabs-AI/flux-controlnet-hed-diffusers"
            elif controlnet_source == "existing":
                controlnet_model_tmp = request.form["existing_controlnet_model"]
                default_canny = "InstantX/FLUX.1-dev-Controlnet-Canny"

                controlnet_model = validate_model_or_fallback(controlnet_model_tmp, default_canny)
                if controlnet_model != controlnet_model_tmp:
                    flash(f"Repo {controlnet_model_tmp} non valido, uso modello di default {default_canny}", "error")

            else:
                controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"

        learning_rate = request.form.get("learning_rate", "2e-6")
        steps = request.form["steps"]
        train_batch_size = request.form["train_batch_size"]
        n4 = request.form["N4"]
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
        if str(n4).lower() in ["true", "yes", "1"]:
            # Force disable AMP if N4 is enabled
            mixed_precision = "no"
        else:
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
                ssh_manager.run_command(
                    f"sudo docker cp /home/ubuntu/tesiControlNetFlux/remote_inputs/. controlnet:/workspace/tesiControlNetFlux/remote_inputs/")
                remote_validation_path = f"/workspace/tesiControlNetFlux/remote_inputs/{uuid.uuid4()}_{filename}"
                prompt = request.form.get("prompt")  # only then get prompt
                if validation_image_path and os.path.exists(validation_image_path):
                    os.remove(validation_image_path)
        print(hub_model_id)
        print(controlnet_model)
        cmd = [
            f"sudo docker exec -e HUGGINGFACE_TOKEN={shlex.quote(str(HUGGINGFACE_TOKEN or ''))} "
            f"controlnet python3 /workspace/tesiControlNetFlux/Src/scripts/controlnet_training_api.py "
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
        print(command)
        job_id = str(uuid.uuid4())
        work_queue.put({"job_id": job_id, "command": command})

        content = Div(
            H2("Training Job Submitted", id = "job-status"),
            P(f"Model: {hub_model_id}"),
            P(f"Job ID: {job_id}"),
            Div("Waiting for training...", id="result-section"),
            Script(f"""
                    async function pollResult() {{
                        const res = await fetch("{url_for('get_result', job_id=job_id)}");
                        const data = await res.json();
                        let resultDiv = document.getElementById("result-section");
                        if (data.status === "done") {{
                            document.getElementById("job-status").innerText = "Training Job Terminated";
                            window.location.href = "{url_for('inference')}";
                        }} else if (data.status === "running") {{
                            resultDiv.innerHTML = `<p>Progress: ${{data.progress || 0}}%</p>`;
                            setTimeout(pollResult, 2000);
                        }} else {{
                            resultDiv.innerHTML = "<p>Waiting...</p>";
                            setTimeout(pollResult, 2000);
                        }}
                            }}
                    pollResult();
                """),
            id="content")
        return str(base_layout("Waiting for Training", content)), 200

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
                    "data_N4": params.get("N4", "false"),
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
        Fieldset(
            Legend("Mode"),
            Select(
                Option("Fine-tune existing model", value="existing"),
                Option("Create new model", value="new"),
                id="mode", name="mode"
            ),
            Div(  # Existing model block
                Label("Existing Model:"),
                Select(*options, id="existingModel", name="existing_model"),
                Label("ControlNet Type:"), Input(id="controlnet_type_existing", name="controlnet_type"),
                Label("Reuse as ControlNet?"),
                Select(Option("No", value="no"), Option("Yes", value="yes"), name="reuse_as_controlnet", id="reuse_as_controlnet"),
                Div(
                    Label("ControlNet Source:"),
                    Select(
                        Option("Default - Canny", value="canny"),
                        Option("Default - HED", value="hed"),
                        Option("Use Existing", value="existing"),
                        id="controlnet_source_existing", name="controlnet_source_existing"
                    ),
                    Div(
                        Label("Choose Existing ControlNet:"),
                        Select(*options, id="existingControlnetModel_existing", name="existing_controlnet_model_existing"),
                        id="existingControlnetWrapper_existing", style="display:none;"
                    ),
                    cls="form-row",
                    id="existingControlnetSourceWrapper", style="display:none;"
                ),

                id="existingModelWrapper"
            ),
            Div(  # New model block
                Label("New Model Name:"), Input(name="new_model_name"),
                Label("ControlNet Type:"), Input(id="controlnet_type", name="controlnet_type"),
                Div(
                    Label("ControlNet Source:"),
                    Select(
                        Option("Default - Canny", value="canny"),
                        Option("Default - HED", value="hed"),
                        Option("Use Existing", value="existing"),
                        id="controlnet_source", name="controlnet_source"
                    ),
                    Div(
                        Label("Choose Existing ControlNet:"),
                        Select(*options, id="existingControlnetModel", name="existing_controlnet_model"),
                        id="existingControlnetWrapper", style="display:none;"
                    ),
                    cls="form-row"
                ),
                id="newModelWrapper", style="display:none;"
            )
        ),
        Fieldset(
            Legend("Quantization & Precision"),
            Div(
                Label("Use N4 Quantization"),
                Select(
                    Option("No", value="false"),
                    Option("Yes", value="true"),
                    id="N4", name="N4"
                ),
                cls="form-row"
            ),
            Div(
                Label("Mixed Precision"),
                Select(
                    Option("no", value="no"),
                    Option("fp16", value="fp16"),
                    Option("bf16", value="bf16", selected=True),
                    name="mixed_precision", id="mixed_precision"
                ),
                id="mixed_precision_group",
                cls="form-row"
            )
        ),
        Fieldset(
            Legend("Training Parameters"),
            Div(
                Label("Learning Rate:"), Input(id="learning_rate", name="learning_rate", value="2e-6"),
                Label("Steps:"), Input(id="steps", name="steps", type="number"),
                cls="form-row"
            ),
            Div(
                Label("Batch Size:"), Input(id="train_batch_size", name="train_batch_size", type="number"),
                Label("Gradient Accumulation:"),
                Input(id="gradient_accumulation_steps", name="gradient_accumulation_steps", type="number", value="1"),
                cls="form-row"
            ),
            Div(
                Label("Resolution:"), Input(id="resolution", name="resolution", type="number", value="512"),
                Label("Checkpoint Steps:"),
                Input(id="checkpointing_steps", name="checkpointing_steps", type="number", value="250"),
                cls="form-row"
            ),
            Div(
                Label("Validation Steps:"),
                Input(id="validation_steps", name="validation_steps", type="number", value="125"),
                cls="form-row"
            )
        ),
        Fieldset(
            Legend("Validation"),
            Label("Validation Image:"),
            Input(id="validationImage", name="validation_image", type="file", accept=".jpg,.jpeg"),
            Div(Label("Prompt:"), Input(id="prompt", name="prompt"), id="promptWrapper", style="display:none;")
        ),
        Button("Start Training", type="submit", cls="button primary"),
        Script(f"""
            document.getElementById("existingModel").addEventListener("change", function() {{
                const selected = this.options[this.selectedIndex];
                document.getElementById("controlnet_type").value = selected.dataset.controlnet_type;
                document.getElementById("N4").value = selected.dataset.n4.toString().toLowerCase();;
                document.getElementById("steps").value = selected.dataset.steps;
                document.getElementById("train_batch_size").value = selected.dataset.train_batch_size;
                document.getElementById("learning_rate").value = selected.dataset.learning_rate;
                document.getElementById("mixed_precision").value = selected.dataset.mixed_precision;
                document.getElementById("gradient_accumulation_steps").value = selected.dataset.gradient_accumulation_steps;
                document.getElementById("resolution").value = selected.dataset.resolution;
                document.getElementById("checkpointing_steps").value = selected.dataset.checkpointing_steps;
                document.getElementById("validation_steps").value = selected.dataset.validation_steps;
            }});
            document.getElementById("reuse_as_controlnet").addEventListener("change", function() {{
                const wrapper = document.getElementById("existingControlnetSourceWrapper");
                if (this.value === "no") {{
                    wrapper.style.display = "block";
                }} else {{
                    wrapper.style.display = "none";
                }}
            }});
            document.getElementById("controlnet_source").addEventListener("change", function() {{
                const existingWrapper = document.getElementById("existingControlnetWrapper");
                if (this.value === "existing") {{
                    existingWrapper.style.display = "block";
                }} else {{
                    existingWrapper.style.display = "none";
                }}
            }});
            document.getElementById("controlnet_source_existing").addEventListener("change", function() {{
                const existingWrapper = document.getElementById("existingControlnetWrapper_existing");
                if (this.value === "existing") {{
                    existingWrapper.style.display = "block";
                }} else {{
                    existingWrapper.style.display = "none";
                }}
            }});    
            // Show/hide prompt input if a validation image is uploaded
            document.getElementById("validationImage").addEventListener("change", function () {{
                const wrapper = document.getElementById("promptWrapper");
                
                if (this.files.length > 0) {{
                    wrapper.style.display = "block";
                    document.getElementById("prompt").setAttribute("required", "true");
                }} else {{
                    wrapper.style.display = "none";
                    document.getElementById("prompt").removeAttribute("required");
                }}
            }});
    
            document.getElementById("mode").addEventListener("change", function() {{
                const newWrapper = document.getElementById("newModelWrapper");
                const existingWrapper = document.getElementById("existingModelWrapper");
                if (this.value === "new") {{
                    newWrapper.style.display = "block";
                    existingWrapper.style.display = "none";
                }} else {{
                    newWrapper.style.display = "none";
                    existingWrapper.style.display = "block";
                }}
            }});
    
            document.getElementById("N4").addEventListener("change", function() {{
              const n4Select = document.getElementById("N4");
              const mpSelect = document.getElementById("mixed_precision");
              const mpGroup = document.getElementById("mixed_precision_group");
            
            if (n4Select.value.toLowerCase() === "true") {{
              mpGroup.style.display = "none";
            }} else {{
              mpGroup.style.display = "block";
            }}
    
            }});
            
            document.addEventListener("DOMContentLoaded", function() {{
              const n4Select = document.getElementById("N4");
              const mpSelect = document.getElementById("mixed_precision");
              const mpGroup = document.getElementById("mixed_precision_group");
            
            if (n4Select.value.toLowerCase() === "true") {{
              mpGroup.style.display = "none";
            }} else {{
              mpGroup.style.display = "block";
            }}
    
            }});
            """),
        method="post",
        id="trainingForm",
        cls="form-card",
        action=url_for("training"), enctype="multipart/form-data"
    )
    return str(base_layout("Training", form)), 200


@app.route('/results', methods=["GET"])
def results():
    is_connected = session.get('lambda_connected', False)
    if not is_connected:
        return redirect(url_for('index'))

    models = models_list()
    selected_model = request.args.get("model", "all")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 4))

    # struttura session cursors: session['results_cursors'] = { model_id: { page_number: next_cursor, ... }, ... }
    if 'results_cursors' not in session:
        session['results_cursors'] = {}

    def fetch_page(prefix, model_key, page_num, per_page, start_cursor=None, filter_repo_image=False):
        """
        Ritorna tuple: (items_list, next_cursor)
        items_list: lista di resources (dizionari) limitata a per_page
        Questo metodo gestisce la navigazione in avanti: può iterare usando next_cursor fino a raggiungere la page desiderata.
        """
        collected = []
        cursor = start_cursor
        current_page = 1

        # se start_cursor fornito, assumo che corrisponda a pagina 1 (o alla posizione richiesta)
        while True:
            try:
                # richiedo max_results=per_page (ogni chiamata prende al più per_page risultati)
                res = resources(type="upload", prefix=prefix, max_results=per_page, next_cursor=cursor)
            except Exception as e:
                # Fall-back: ritorna vuoto
                return [], None

            # filtra se richiesto (utile per prefix generico HF_NAMESPACE/)
            batch = res.get("resources", [])
            if filter_repo_image:
                batch = [r for r in batch if "/repo_image/" in r.get("secure_url", "") or "/repo_image/" in r.get("public_id", "")]
            # append in ordine finché non raggiungiamo per_page
            for r in batch:
                collected.append(r)
                if len(collected) >= per_page:
                    break

            next_cursor = res.get("next_cursor")
            # se abbiamo raccolto per_page interrompiamo e restituiamo next_cursor così il client può andare avanti
            if len(collected) >= per_page:
                return collected[:per_page], next_cursor

            # se non ci sono più risultati
            if not next_cursor:
                # abbiamo tutto quello che c'era (potrebbe essere meno di per_page)
                return collected, None

            # altrimenti continuiamo la richiesta con next_cursor
            cursor = next_cursor
            # loop - ma raramente continua più di poche iterazioni

    # == Ottieni i risultati solo per la pagina corrente ==
    image_resources = []
    next_cursor = None
    model_key = selected_model

    try:
        if selected_model == "all":
            # prefisso generico e poi filtro per repo_image
            prefix = f"{HF_NAMESPACE}/"
            # vediamo se abbiamo in session un cursor già per questa pagina
            cursors_for_model = session['results_cursors'].get(prefix, {})
            start_cursor = None

            # se page==1 e non abbiamo cursor, lasciamo start_cursor None
            if page > 1:
                # se abbiamo il cursor per page-1 usalo come start
                prev_cursor = cursors_for_model.get(str(page-1))
                if prev_cursor:
                    start_cursor = prev_cursor
                else:
                    # fallback: iteriamo dalla pagina 1 arrivando a page-1 (sono poche chiamate)
                    start_cursor = None

            items, next_cursor = fetch_page(prefix, prefix, page, per_page, start_cursor=start_cursor, filter_repo_image=True)
            image_resources = items

            # salva cursor per questa page (se non nullo)
            if prefix not in session['results_cursors']:
                session['results_cursors'][prefix] = {}
            session['results_cursors'][prefix][str(page)] = next_cursor

        else:
            # specific model
            model_name = selected_model.split("/")[-1]
            prefix = f"{HF_NAMESPACE}/{model_name}_results/repo_image/"
            cursors_for_model = session['results_cursors'].get(prefix, {})
            start_cursor = None
            if page > 1:
                prev_cursor = cursors_for_model.get(str(page-1))
                if prev_cursor:
                    start_cursor = prev_cursor
                else:
                    start_cursor = None

            items, next_cursor = fetch_page(prefix, prefix, page, per_page, start_cursor=start_cursor, filter_repo_image=False)
            image_resources = items

            if prefix not in session['results_cursors']:
                session['results_cursors'][prefix] = {}
            session['results_cursors'][prefix][str(page)] = next_cursor

        # costruisci le card dalla singola page ONLY
        grids = []
        for r in image_resources:
            url = r.get("secure_url")
            # costruiamo control e text URL basandoci sulla stessa struttura
            control_url = url.replace("/repo_image/", "/repo_control/").rsplit(".", 1)[0] + "_control.jpg"
            # per i raw text Cloudinary spesso mette raw/upload; proviamo a costruire l'url raw sostituendo /image/upload/ con /raw/upload/
            text_url = url.replace("/image/upload/", "/raw/upload/").replace("/repo_image/", "/repo_text/").rsplit(".", 1)[0] + "_text"

            params_text, prompt_text, control_img_tag = "", "", None

            # fetch only the text for the current items
            try:
                resp = requests.get(text_url, timeout=5)
                if resp.status_code == 200:
                    params_text = resp.text
                    for line in params_text.splitlines():
                        if line.lower().startswith("prompt:"):
                            prompt_text = line
            except Exception:
                params_text = ""

            # test control image existence HEAD
            try:
                h = requests.head(control_url, timeout=4)
                if h.status_code == 200:
                    control_img_tag = Img(src=control_url, cls="card-img")
            except Exception:
                control_img_tag = None

            resized_url = f"{url}?w=512&h=512&c=fit"

            if not (control_img_tag or params_text or prompt_text):
                grid = Div(Div(Img(src=resized_url, cls="card-img"), cls="grid-item full"), cls="result-grid")
            else:
                grid = Div(
                    Div(Img(src=resized_url, cls="card-img"), cls="grid-item"),
                    Div(control_img_tag or "", cls="grid-item"),
                    Div(Pre(params_text), cls="grid-item") if params_text else Div("", cls="grid-item"),
                    Div(prompt_text or "", cls="grid-item") if prompt_text else Div("", cls="grid-item"),
                    cls="result-grid"
                )

            grids.append(grid)

    except Exception as e:
        flash(f"Error fetching results: {e}", "error")
        grids = []

    # Paginazione: Previous e Next
    pagination_children = []
    prev_link = None
    next_link = None

    # costruisci previous utilizzando i cursors salvati (se esistono)
    prefix_key = (f"{HF_NAMESPACE}/{selected_model.split('/')[-1]}_results/repo_image/") if selected_model != "all" else f"{HF_NAMESPACE}/"
    cursors_for_model = session['results_cursors'].get(prefix_key, {})

    if page > 1:
        # se abbiamo il cursor per page-1 allora il link Previous può includerlo (in query string fetch via page-1)
        prev_link = url_for("results", model=selected_model, page=page-1, per_page=per_page)
        pagination_children.append(A("⬅ Previous", href=prev_link, cls="button secondary"))

    # Next: se next_cursor esiste per questa page
    if next_cursor:
        next_link = url_for("results", model=selected_model, page=page+1, per_page=per_page, next_cursor=next_cursor)
        pagination_children.append(A("Next ➡", href=next_link, cls="button secondary"))

    pagination = Div(*pagination_children, cls="pagination")

    # Selettore modello
    model_options = [Option("All Models", value="all", selected=(selected_model == "all"))]
    for m in models:
        model_options.append(
            Option(
                m["id"].split("/")[-1],
                value=m["id"],
                selected=(selected_model == m["id"])
            )
        )
    selector = Form(
        Select(*model_options, name="model", onchange="this.form.submit()", value=selected_model, style="width:100%"),
        method="get",
        style="width:100%; margin-bottom:1rem;"
    )

    content = Div(
        H1("Model Results", cls="hero-title"),
        selector,
        Div(*grids, cls="card-grid"),
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
