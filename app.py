import queue
import shlex
import tempfile
import threading
import time
import uuid
from io import BytesIO
from scp import SCPClient
import json
from collections import defaultdict
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
from flask import Flask, url_for, request, redirect, flash, session, get_flashed_messages, jsonify, Response
from huggingface_hub import HfApi, hf_hub_download
from werkzeug.utils import secure_filename
from flask_babel import Babel, _

app = Flask(__name__)
#Automatic translation
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

def get_locale():
    # Use session setting if available
    if 'lang' in session:
        return session['lang']
    # Otherwise use browser preference
    return request.accept_languages.best_match(['en', 'it'])

babel = Babel(app, locale_selector=get_locale)

@app.route('/set_lang/<lang>')
def set_language(lang):
    if lang in ['en', 'it']:
        session['lang'] = lang
    return redirect(request.referrer or url_for('index'))

app.secret_key = os.urandom(12)
LAMBDA_CLOUD_API_BASE = "https://cloud.lambdalabs.com/api/v1/instances"
LAMBDA_CLOUD_API_KEY = os.getenv('LAMBDA_CLOUD_API_KEY')
LAMBDA_INSTANCE_IP = os.getenv('LAMBDA_INSTANCE_IP', "YOUR_LAMBDA_INSTANCE_PUBLIC_IP")
LAMBDA_INSTANCE_USER = os.getenv('LAMBDA_INSTANCE_USER', "ubuntu")
SSH_PRIVATE_KEY_PATH = os.getenv('SSH_PRIVATE_KEY_PATH')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
WANDB_TOKEN = os.getenv('WANDB_TOKEN')
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
job_queues = defaultdict(queue.Queue)   # in-memory per-job queues to push events to browser
results_lock = threading.Lock()
# A lock to serialize the SSH command
worker_lock = threading.Lock()



def sanitize_number(val, default, force_int=False):
    try:
        value = float(val)
        if math.isnan(value):
            return default
        if value < 0:
            value = 1
        return int(round(value)) if force_int else value
    except Exception:
        return default

def sanitize_text(val, default):
    try:
        if val is None:
            return default
        value = str(val).strip()
        if not value or value.lower() in ["nan", "undefined", "none", "null"]:
            return default
        return value
    except Exception:
        return default

def as_str(v, default=""):
    if isinstance(v, bool): return "True" if v else "False"
    return str(v if v is not None else default)

def publish(job_id: str, payload: dict):
    #Pushing status for SSE consumers
    with results_lock:
        results_db[job_id] = payload
    try:
        job_queues[job_id].put_nowait(payload)
    except Exception:
        # if queue full or consumer gone, ignore
        pass

def load_image_safe(filepath):
    pil_img = Image.open(filepath)
    if pil_img.mode == "RGBA":
        background = Image.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    else:
        pil_img = pil_img.convert("RGB")
    return pil_img

def convert_to_canny(input_path):
    pil_img = load_image_safe(input_path)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    root, _ = os.path.splitext(input_path)
    out_path = f"{root}_canny.png"  # always save as PNG
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
            print(f"[WARNING] Repository {model_id} not valid, using fallback {default_model}")
            return default_model
    except Exception as e:
        print(f"[ERROR] Repository {model_id} not accessible : {e}, using default")
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
            # Keep alive
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

    def run_command_streaming(self, command, job_id=None, timeout=None):
        with self.lock:
            self.reconnect_if_needed()
            if not self.client:
                return "", _("SSH client not connected")

            transport = self.client.get_transport()
            chan = transport.open_session()
            chan.get_pty()  # helps remote Python flush
            chan.exec_command(command)
            chan.settimeout(0.2)

            output_chunks, error_chunks = [], []
            buf_out, buf_err = "", ""

            def emit_lines_from(buffer_text):
                # normalize CR-only progress updates; keep the trailing partial
                normalized = buffer_text.replace("\r\n", "\n").replace("\r", "\n")
                parts = normalized.split("\n")
                return parts[:-1], parts[-1]

            while True:
                if chan.recv_ready():
                    chunk = chan.recv(4096).decode("utf-8", errors="ignore")
                    output_chunks.append(chunk)
                    buf_out += chunk
                    lines, buf_out = emit_lines_from(buf_out)
                    for line in lines:
                        if not line:
                            continue
                        if job_id:
                            print(line)

                            progress = None
                            msg = None

                            # Inference style (plain tqdm without prefix)
                            m = re.search(r'^\s*(\d{1,3})%\|.*?(\d+)/(\d+)', line)
                            if m:
                                progress = int(m.group(1))
                                msg = _("Progress: %(progress)d%%", progress=progress)

                            # Training steps
                            elif line.strip().startswith("Steps:"):
                                m = re.search(r'(\d{1,3})%\|.*?(\d+)/(\d+)', line)
                                if m:
                                    progress = int(m.group(1))
                                    msg = _("Progress: %(progress)d%%", progress=progress)

                            # Map evaluation
                            elif line.strip().startswith("Map:"):
                                m = re.search(r'(\d{1,3})%\|.*?(\d+)/(\d+)', line)
                                if m:
                                    progress = int(m.group(1))
                                    msg = _("Map: %(progress)d%%", progress=progress)

                            # Fetching files
                            elif line.strip().startswith("Fetching"):
                                m = re.search(r'(\d{1,3})%\|.*?(\d+)/(\d+)', line)
                                if m:
                                    progress = int(m.group(1))
                                    msg = _("Fetching files: %(progress)d%%", progress=progress)

                            # Loading checkpoint shards
                            elif line.strip().startswith("Loading checkpoint shards"):
                                m = re.search(r'(\d{1,3})%\|.*?(\d+)/(\d+)', line)
                                if m:
                                    progress = int(m.group(1))
                                    msg = _("Loading checkpoint: %(progress)d%%", progress=progress)

                            # Uploading files
                            elif line.strip().startswith("Processing Files"):
                                m = re.search(r'(\d{1,3})%\|', line)
                                if m:
                                    progress = int(m.group(1))
                                    msg = _("Processing Files: %(progress)d%%", progress=progress)

                            if msg is not None:
                                publish(job_id, {
                                    "status": "running",
                                    "progress": progress if progress is not None else 0,
                                    "message": msg
                                })

                if chan.recv_stderr_ready():
                    chunk = chan.recv_stderr(4096).decode("utf-8", errors="ignore")
                    error_chunks.append(chunk)
                    buf_err += chunk  # stderr buffered

                if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                    break

                time.sleep(0.05)

            # publish any trailing line
            if buf_out.strip() and job_id:
                publish(job_id, {"status": "running", "message": buf_out.strip()})

            exit_code = chan.recv_exit_status()
            output = "".join(output_chunks)
            errors = "".join(error_chunks) if exit_code != 0 else ""
            return output, errors

    #Wrapper to run a command using Screen
    def run_in_screen(self, job_id, command):
        # Start screen session with logging to /tmp/{job_id}.log
        log_cmd = (
            f"screen -L -Logfile /tmp/{job_id}.log -dmS job_{job_id} bash -c {shlex.quote(command)}"
        )
        self.run_command(log_cmd)

        # Get the PID of the screen process
        out, _ = self.run_command(f"screen -ls | grep job_{job_id} | awk '{{print $1}}'")
        if out:
            return out.strip().split('.')[0]
        return None

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
            # ensure container is running
            status_cmd = "sudo docker inspect -f '{{.State.Running}}' controlnet 2>/dev/null || echo false"
            out, _ = ssh_manager.run_command(status_cmd)
            if "true" not in out.lower():
                # Restarting docker if not running
                ssh_manager.run_command(
                    f"sudo docker rm -f controlnet || true && "
                    f"sudo docker start controlnet || "
                    f"sudo docker run -d --gpus all --name controlnet "
                    f"--entrypoint python3 {DOCKER_IMAGE_NAME} -c \"import time; time.sleep(1e9)\""
                )

            start_time = time.time()
            # First status
            publish(job_id, {"status": "running", "message": "Work started", "started": start_time})

            pid = ssh_manager.run_in_screen(job_id, command)
            if not pid:
                raise RuntimeError("Failed to start screen job")

            log_file = f"/tmp/{job_id}.log"

            # Resilient loop
            output, errors = "", ""
            while True:
                try:
                    cmd = f"tail --pid={pid} -f {log_file}"
                    chunk_out, chunk_err = ssh_manager.run_command_streaming(cmd, job_id)
                    output += chunk_out
                    errors += chunk_err
                    # Finished normally
                    break
                except Exception as e:
                    print(f"[{time.ctime()}] Worker: Lost connection for job {job_id}: {e}")
                    publish(job_id, {"status": "running", "message": "⚠ Connection lost, retrying..."})
                    time.sleep(3)
                    ssh_manager.reconnect_if_needed()

                    # Check if job already finished
                    still_running, _ = ssh_manager.run_command(f"ps -p {pid} || true")
                    if "defunct" in still_running or not still_running.strip():
                        # Fetch remaining logs
                        final_out, _ = ssh_manager.run_command(f"cat {log_file} || true")
                        output += final_out
                        # Exit loop
                        break

            elapsed = int(time.time() - start_time)

            # Detect final outputs
            url_match = re.search(r'(https?://\S+)', output)
            finished_match = re.search(r"_complete\b", output, re.IGNORECASE)
            if url_match:
                publish(job_id, {"status": "done", "output": url_match.group(0), "elapsed": elapsed})
            elif finished_match:
                publish(job_id, {"status": "done", "message": "done", "elapsed": elapsed})
            elif errors:
                publish(job_id, {"status": "error", "message": errors, "elapsed": elapsed})
            else:
                publish(job_id, {"status": "done", "output": output or "done", "elapsed": elapsed})

        except Exception as e:
            print(f"[{time.ctime()}] Worker: Exception for job {job_id}: {e}")
            publish(job_id, {"status": "error", "message": str(e)})
        finally:
            work_queue.task_done()

@app.route("/events/<job_id>")
def sse_events(job_id):
    def stream():
        try:
            out, _ = ssh_manager.run_command(f"cat /tmp/{job_id}.log || true")
            if out:
                for line in out.splitlines():
                    yield f"data: {json.dumps({'status': 'running', 'message': line})}\n\n"
        except Exception as e:
            print(f"Replay failed for {job_id}: {e}")

            # Stream live updates
        q = job_queues[job_id]
        while True:
            payload = q.get()
            try:
                yield f"data: {json.dumps(payload)}\n\n"
            except GeneratorExit:
                break

    # Help avoid buffering
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream(), mimetype="text/event-stream", headers=headers)

@app.route('/connect_lambda')
def connect_lambda():
    if not LAMBDA_CLOUD_API_KEY:
        return str(base_layout((_("Error")), P(_("Lambda API key is not configured!")))), 400

    instance_data = get_lambda_info()
    if not instance_data:
        return str(base_layout(_("Error"), P(_("Lambda instance not found.")))), 400
    if instance_data.get("status") != "active":
        return str(base_layout(
            _("Error"),
            P(_("Instance not active (status=%(status)s)") % {"status": instance_data['status']})
        )), 400

    log_lines = []

    def log(msg):
        log_lines.append(msg)
        print(msg)

    # Step 1: fix DNS, docker config
    log(_("Configuring Docker daemon..."))
    ssh_manager.run_command("echo 'nameserver 8.8.8.8' | sudo tee /etc/resolv.conf")
    ssh_manager.run_command("echo '{ \"ipv6\": false }' | sudo tee /etc/docker/daemon.json")
    ssh_manager.run_command("sudo systemctl restart docker")

    # Step 2: pull image
    log(_("Pulling Docker image %(image)s...") % {"image": DOCKER_IMAGE_NAME})
    out, err = ssh_manager.run_command(f"sudo docker pull {DOCKER_IMAGE_NAME}")
    log(out or err)

    # Step 3: ensure container fresh
    log(_("Removing any old container..."))
    ssh_manager.run_command("sudo docker rm -f controlnet || true")

    # Step 4: run container
    log(_("Starting new container..."))
    out, err = ssh_manager.run_command(
        f"sudo docker run -d --gpus all --name controlnet "
        f"--entrypoint python3 {DOCKER_IMAGE_NAME} -c 'import time; time.sleep(1e9)'"
    )
    log(out or err)

    log(_("Installing extra Python packages"))
    ssh_manager.run_command(
        "sudo docker exec controlnet pip install --upgrade pip && "
        "sudo docker exec controlnet pip install pyyaml huggingface_hub tqdm screen"
    )
    # Step 5: check running
    status, tmp = ssh_manager.run_command("sudo docker inspect -f '{{.State.Running}}' controlnet || echo false")
    if "true" not in status.lower():
        exists, tmp = ssh_manager.run_command("sudo docker ps -a --format '{{.Names}}' | grep -w controlnet || true")
        if not exists.strip():
            log(_("Container was not created at all."))
        else:
            logs, tmp = ssh_manager.run_command("sudo docker logs controlnet")
            log(_("Container failed to start:") + "\n" + logs)

        content = Div(
            H1(_("Connect Lambda Failed")),
            Pre("\n".join(log_lines),
                style="text-align:left; background:#222; color:#f33; padding:1rem; border-radius:8px; max-height:60vh; overflow:auto;"),
            A(_("Continue to Home"), href=url_for('index'), cls="button primary")
        )
        return str(base_layout(_("Connect Lambda"), content)), 500

    # Step 6: update repo
    log(_("Updating repo inside container..."))
    ssh_manager.run_command(
        "sudo docker exec controlnet bash -c 'cd /workspace/tesiControlNetFlux &&  git pull '"
    )

    session['lambda_connected'] = True
    log(_("Lambda is ready."))

    content = Div(
        H1(_("Connect Lambda Logs")),
        Pre("\n".join(log_lines), style="text-align:left; background:#222; color:#0f0; padding:1rem; border-radius:8px; max-height:60vh; overflow:auto;"),
        A(_("Continue to Home"), href=url_for('index'), cls="button primary")
    )
    return str(base_layout(_("Connect Lambda"), content)), 200

def base_layout(title: str, content: Any, extra_scripts: list[str] = None):
    cache_buster = int(time.time())
    is_connected = session.get('lambda_connected', False)
    nav_links = []
    # Always show Home if not on index
    if request.endpoint != 'index':
        nav_links.append(A(_("Home"), href=url_for('index'), cls="nav-link"))

    if is_connected:
        nav_links.extend([
            A(_("Inference"), href=url_for('inference'), cls="nav-link"),
            A(_("Training"), href=url_for('training'), cls="nav-link"),
            A(_("Results"), href=url_for('results'), cls="nav-link"),
        ])
    else:
        nav_links.append(A(_("Connect to Lambda"), href=url_for('connect_lambda'), cls="nav-link"))

    navigation = Nav(*nav_links, cls="nav")
    current_lang = session.get('lang', 'en')
    lang_buttons = Div(
        A("EN", href=url_for("set_language", lang="en"), cls="button secondary small {'active' if current_lang == 'en' else ''}"),
        A("IT", href=url_for("set_language", lang="it"), cls="button secondary small {'active' if current_lang == 'it' else ''}"),
        cls="lang-toggle"
    )
    scripts = [Script(src=url_for('static', filename='js/script.js', v=cache_buster))]
    if extra_scripts:
        scripts.extend([Script(src=url_for('static', filename=path, v=cache_buster)) for path in extra_scripts])

    return Div(
        Head(
            Meta(charset="UTF-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
            Title(f"Flux Designer - {title}"),
            Link(rel="stylesheet", href=url_for('static', filename='css/style.css', v=cache_buster))
        ),
        Body(
            Header(H1(_("Flux Designer"), cls="site-title"), navigation, lang_buttons),
            Main(Div(flash_html_messages(), content, id="main_div", cls="container")),
            Footer(P("© 2025 Lambda ControlNet App")),
            *scripts,
            Script(f"""
                document.addEventListener("DOMContentLoaded", () => {{
                // Possibility to add a toggle for nav
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


@app.route("/")
def index():
    is_connected = session.get('lambda_connected', False)

    if not is_connected:
        action_section = Div(
            P(_("Before using this app, ensure the lambda remote machine is ready.")),
            A(_("Connect to Lambda & Pull Docker Image"),
              id="connectBtn",
              href=url_for('connect_lambda'),
              cls="button primary"),
            Script(f"""
                document.getElementById("connectBtn").addEventListener("click", function(){{
                  this.innerText = "{_('Connecting...')}";
                  this.classList.add("loading");
                  }});
                """),
            cls="center-box"
        )
    else:
        action_section = Div(
            H2(_("Lambda instance is connected and Docker image is ready.")),
            Div(
                A(_("Go to Inference Page"), cls="button primary", href=url_for('inference')),
                A(_("Go to Training Page"), cls="button primary", href=url_for('training')),
                A(_("Go to Results Page"), cls="button primary", href=url_for('results')),
                cls="points_links"
            ),
        )

    content = Div(
        H1(_("Flux Designer"), cls="hero-title"),
        P(_("Generate images and fine-tune models with ControlNet models running on Lambda Cloud."),
          cls="hero-subtitle"),
        Hr(),
        H2(_("Getting Started")),
        Div(
            P(_("Create a Lambda Cloud account & API key.")),
            P(_("Start the Lambda machine and wait until it's started")),
            P(_("Make sure the Lambda machine IP, Lambda API key and region are set correctly (IP: %(ip)s, Region: %(region)s).",
                ip=LAMBDA_INSTANCE_IP, region=REGION)),
            cls="points"
        ),
        Hr(),
        action_section,
        id="content", class_="container")

    return str(base_layout(_("Home"), content)), 200

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
                return jsonify({"status": "error", "error": _("Invalid model_type: %(ctype)s", ctype=controlnet_type)})

            # Verify the file exists before proceeding.
            if not os.path.exists(result_path):
                raise FileNotFoundError(
                    _("Processed image file was not created by the conversion function: %(path)s", path=result_path))

            if result_path and result_path != temp_path:
                temp_files_to_clean.append(result_path)

            img_arr = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
            if img_arr is None:
                raise ValueError(_("Failed to read processed image at %(path)s", path=result_path))

            desired_size = (512, 512)
            img_arr = cv2.resize(img_arr, desired_size, interpolation=cv2.INTER_AREA)
            if merged_image is None:
                merged_image = img_arr
            else:
                merged_image = cv2.add(merged_image, img_arr)

        if merged_image is None:
            return jsonify({"status": "error", "error": _("No valid images were processed.")}), 500

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
        for fpath in temp_files_to_clean:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except OSError as e:
                print(f"Error removing temporary file {fpath}: {e}")


@app.route('/inference', methods=["GET", "POST"])
def inference():
    is_connected = session.get('lambda_connected', False)
    if not is_connected:
        return redirect(url_for('index'))

    if request.method == "POST":
        prompt = sanitize_text(request.form.get("prompt"), "tall glass on white background")
        scale = sanitize_number(request.form.get('scale', 0.2), 0.2, False)
        steps = sanitize_number(request.form.get('steps', 50), 50, True)
        guidance = sanitize_number(request.form.get('guidance', 6.0), 6.0, False)
        model_chosen = request.form["model"]
        default_canny = "InstantX/FLUX.1-dev-Controlnet-Canny"
        model_id = validate_model_or_fallback(model_chosen, default_canny)
        if model_id != model_chosen:
            flash(_("Repo %(repo)s not valid, using default model %(default)s",
                    repo=model_chosen, default=default_canny), "error")

        params = model_info(model_id)
        model_type = sanitize_text(params.get("controlnet_type", "canny"), "canny")
        n4 = params.get("N4", False)
        # future possible control
        precision = sanitize_text(params.get("mixed_precision", "bf16"), "bf16")
        control_image_path = None
        remote_control_path = None

        image_url = request.form.get('control_image_data')
        if image_url and "," in image_url:
            header, encoded = image_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(binary_data)).convert("RGB")
            np_image = np.array(image)
            if not np.all(np_image == 0):
                import tempfile
                unique_id = str(uuid.uuid4())
                control_image_path = os.path.join(tempfile.gettempdir(), f"{unique_id}.jpg")
                image.save(control_image_path)
                remote_control_host_path = f"/home/ubuntu/tesiControlNetFlux/remote_inputs/{unique_id}.jpg"
                scp_to_lambda(control_image_path, remote_control_host_path)
                ssh_manager.run_command(
                    "sudo docker exec controlnet mkdir -p /workspace/tesiControlNetFlux/Src/remote_inputs"
                )
                ssh_manager.run_command(
                    f"sudo docker cp /home/ubuntu/tesiControlNetFlux/remote_inputs/{unique_id}.jpg "
                    f"controlnet:/workspace/tesiControlNetFlux/Src/remote_inputs/{unique_id}.jpg"
                )
                remote_control_path = f"remote_inputs/{unique_id}.jpg"
                if control_image_path and os.path.exists(control_image_path):
                    os.remove(control_image_path)

        # Command to run inference (unchanged except for variables)
        command = (
            f"sudo docker exec -e PYTHONUNBUFFERED=1 "
            f"-e TQDM_MININTERVAL=0 "
            f"-e TQDM_MAXINTERVAL=0 "
            f"-e CLOUDINARY_CLOUD_NAME={CLOUDINARY_CLOUD_NAME} "
            f"-e HUGGINGFACE_TOKEN={HUGGINGFACE_TOKEN} "
            f"-e CLOUDINARY_API_KEY={CLOUDINARY_API_KEY} "
            f"-e CLOUDINARY_API_SECRET={CLOUDINARY_API_SECRET} "
            f"controlnet python3 -u /workspace/tesiControlNetFlux/Src/scripts/controlnet_infer_api.py "
            f"--prompt \"{prompt}\" --scale {scale} --steps {steps} --guidance {guidance} "
            f"--controlnet_model {model_id} --controlnet_type {model_type}"
        )
        if remote_control_path:
            command += f" --control_image {remote_control_path} "
        if n4:
            command += f" --N4"
        print(command)

        job_id = str(uuid.uuid4())
        work_queue.put({"job_id": job_id, "command": command})
        sse_url = url_for("sse_events", job_id=job_id)

        # JS translations
        js_job_done = _("Inference Job Terminated")
        js_open_full = _("Open full resolution")
        js_elapsed = _("Elapsed time: %(seconds)s seconds")
        js_error = _("Error")
        js_lost = _("Connection lost, retrying in 3s...")
        js_retry = _("Connection lost, retrying...")

        content = Div(
            H2(_("Inference Job Submitted"), id="job-status"),
            P(_("Model: %(model)s", model=model_id)),
            P(_("Prompt: %(prompt)s", prompt=prompt)),
            P(_("Job ID: %(jobid)s", jobid=job_id)),
            P(_("Elapsed time"), style="display:none;", id="time"),
            Div(
                A(_("Generate another image"),
                  href=url_for('inference'),
                  id="inference_link",
                  cls="button primary",
                  style="display: none; width: fit-content;"),
                style="background=none;border:none;box-shadow:none;display:flex;justify-content:center;",
            ),
            Div(_("Waiting for your turn..."), id="result-section"),
            Script(f"""
                const status = document.getElementById("job-status");
                const result_div = document.getElementById("result-section");
                const time_el = document.getElementById("time");
                function connectSSE() {{
                  const es = new EventSource("{sse_url}");

                  es.onmessage = (e) => {{
                    const data = JSON.parse(e.data);
                    if (data.status === "done") {{
                        status.innerText = "{js_job_done}";
                        if (data.output && data.output.startsWith("http")) {{
                            result_div.innerHTML = `<img class="generated preview" src="${{data.output}}" 
                            style='max-width:100%;height:auto;'/><p><a class="button secondary" href="${{
                            data.output}}" target="_blank">{js_open_full}</a></p>`;
                            time_el.innerHTML = data.elapsed ? `{js_elapsed}`.replace("%(seconds)s", data.elapsed) : "";                       
                            time_el.style.display = "block"; 
                            document.getElementById("inference_link").style.display = "block";                    
                        }} else {{
                            result_div.innerHTML = "<pre>" + (data.output || data.message || "") + "</pre>";
                            time_el.innerHTML = data.elapsed ? `{js_elapsed}`.replace("%(seconds)s", data.elapsed) : "";                       
                            time_el.style.display = "block";    
                        }}
                        es.close();
                    }} else if (data.status === "running") {{
                        result_div.innerHTML = `<pre class="progress-log">${{(data.message||'').slice(-800)}}</pre>`;
                    }} else if (data.status === "error") {{
                        result_div.innerHTML = `<p style="color:#f66">{js_error}: ${{data.message || 'unknown'}}</p>`;
                        es.close();
                    }}
                }};

                  es.onerror = () => {{
                     console.warn("{js_lost}");
                     result_div.innerHTML = `<p style="color:orange">{js_retry}</p>`;
                     es.close();
                     setTimeout(connectSSE, 3000);
                  }};
                }}
                connectSSE();
            """),
            id="content"
        )
        return str(base_layout(_("Waiting for Inference"), content)), 200

    # GET method
    models = models_list()
    options = [Option(m["id"].split("/")[-1], value=m["id"]) for m in models]

    form = Form(
        Fieldset(
            Legend(_("Model Selection")),
            Select(*options, id="model", name="model"),
        ),
        Fieldset(
            Legend(_("Parameters")),
            Div(
                Label(_("Scale:")), Input(type="number", name="scale", step="0.1", value="0.2"),
                Label(_("Steps:")), Input(type="number", name="steps", value="50"),
                Label(_("Guidance:")), Input(type="number", name="guidance", step="0.5", value="6.0"),
                cls="form-row"
            )
        ),
        Fieldset(
            Legend(_("Prompt & Control Image")),
            Label(_("Prompt:")), Input(type="text", name="prompt", required=True, cls="input"),
            Label(_("Upload Image:")),
            Input(type="file", name="images", id="uploadInput", accept=".jpg,.jpeg,.png", multiple=True),
            Canvas(id="drawCanvas", width="512", height="512", style="border:1px solid #ccc;"),
            Div(
                Button(_("Pencil"), type="button", id="pencilBtn", cls="button secondary"),
                Button(_("Eraser"), type="button", id="eraserBtn", cls="button secondary"),
                Button(_("Undo"), type="button", id="undoBtn", cls="button secondary"),
                Button(_("Redo"), type="button", id="redoBtn", cls="button secondary"),
                Button(_("Clear"), type="button", id="clearCanvasBtn", cls="button secondary"),
                cls="button-group"
            ),
            Input(type="hidden", name="control_image_data", id="controlImageData")
        ),
        Button(_("Run Inference"), type="submit", cls="button primary"),
        Script(f"""
                    (function(){{
                          function san(el, d, asInt){{
                            let n = Number(el.value);
                            if (!Number.isFinite(n)) {{ el.value = d; return; }}
                            if (n < 0) n = 1;
                            el.value = asInt ? Math.round(n) : n;
                          }}
                          const scale = document.querySelector('input[name="scale"]');
                          const steps = document.querySelector('input[name="steps"]');
                          const guidance = document.querySelector('input[name="guidance"]');
                          if (scale)   scale.addEventListener('change',   ()=>san(scale,   0.2, false));
                          if (steps)   steps.addEventListener('change',   ()=>san(steps,     50, true));
                          if (guidance)guidance.addEventListener('change',()=>san(guidance, 6.0, false));

                          // text guards (prompt)
                          function cleanText(el, d){{
                            const v = (el.value||"").trim().toLowerCase();
                            if (!v || ["nan","undefined","none","null"].includes(v)) {{
                              el.value = d;
                            }}
                          }}
                          const textIds = ["prompt"];
                          textIds.forEach(id=>{{
                            const el = document.getElementById(id);
                            if (el) el.addEventListener("change", ()=>cleanText(el, el.defaultValue||""));
                          }});
                        }})();
                    document.addEventListener("DOMContentLoaded", function () {{
                        const canvas = document.getElementById("drawCanvas");
                        const ctx = canvas.getContext("2d");
                    
                        ctx.fillStyle = "black";
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                        let drawing = false;
                        let tool = "pencil"; // default tool
                    
                        let undoStack = [];
                        let redoStack = [];
                    
                        function saveState() {{
                            undoStack.push(canvas.toDataURL());
                            redoStack = [];
                        }}
                    
                        function restoreState(dataURL) {{
                            let img = new Image();
                            img.onload = () => {{
                                ctx.clearRect(0, 0, canvas.width, canvas.height);
                                ctx.drawImage(img, 0, 0);
                            }};
                            img.src = dataURL;
                        }}
                    
                        canvas.addEventListener("mousedown", () => {{
                            drawing = true;
                            saveState();
                        }});
                    
                        canvas.addEventListener("mouseup", () => {{
                            drawing = false;
                            ctx.beginPath();
                        }});
                    
                        canvas.addEventListener("mousemove", function(e) {{
                            if (!drawing) return;
                            ctx.lineWidth = 0.5;
                            ctx.lineCap = "round";
                            ctx.strokeStyle = (tool === "pencil") ? "white" : "black";
                            ctx.lineTo(e.offsetX, e.offsetY);
                            ctx.stroke();
                            ctx.beginPath();
                            ctx.moveTo(e.offsetX, e.offsetY);
                        }});
                    
                        // Buttons
                        document.getElementById("clearCanvasBtn").addEventListener("click", () => {{
                            saveState();
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.fillStyle = "black";
                            ctx.fillRect(0, 0, canvas.width, canvas.height);
                        }});
                    
                        document.getElementById("pencilBtn").addEventListener("click", () => tool = "pencil");
                        document.getElementById("eraserBtn").addEventListener("click", () => tool = "eraser");
                    
                        document.getElementById("undoBtn").addEventListener("click", () => {{
                            if (undoStack.length > 0) {{
                                redoStack.push(canvas.toDataURL());
                                restoreState(undoStack.pop());
                            }}
                        }});
                    
                        document.getElementById("redoBtn").addEventListener("click", () => {{
                            if (redoStack.length > 0) {{
                                undoStack.push(canvas.toDataURL());
                                restoreState(redoStack.pop());
                            }}
                        }});
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
                        // Save on submit
                        document.querySelector("form").addEventListener("submit", function () {{
                            const dataURL = canvas.toDataURL("image/png");
                            document.getElementById("controlImageData").value = dataURL;
                        }});
                    }});
        """),
        method="post",
        id="content",
        enctype="multipart/form-data",
        cls="form-card",
    )

    return str(base_layout(_("Inference"), form)), 200

@app.route('/training', methods=["GET", "POST"])
def training():
    is_connected = session.get('lambda_connected', False)
    if not is_connected:
        return redirect(url_for('index'))

    if request.method == "POST":
        mode = request.form["mode"]
        controlnet_type = sanitize_text(request.form.get("controlnet_type"), "canny")
        if (controlnet_type.lower() != "canny") and (controlnet_type.lower() != "hed"):
            flash(_("Unexpected type of controlnet, using Canny as default"), "error")
            controlnet_type = "canny"

        if mode == "existing":
            model_id = request.form["existing_model"]
            default_canny = "InstantX/FLUX.1-dev-Controlnet-Canny"
            hub_model_id = validate_model_or_fallback(model_id, default_canny)
            if hub_model_id != model_id:
                flash(_("Repo %(repo)s not valid, using default model %(default)s",
                        repo=model_id, default=default_canny), "error")

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
                        flash(_("Repo %(repo)s not valid, using default model %(default)s",
                                repo=controlnet_model_tmp, default=default_canny), "error")
                else:
                    controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
        else:  # new
            new_name = sanitize_text(request.form.get("new_model_name"), "my-default-model")
            hub_model_id = f"{HF_NAMESPACE}/{new_name}"
            existing = [m["id"] for m in models_list()]
            if hub_model_id in existing:
                flash(_("Model with this name already exists on HuggingFace!"), "error")
                return redirect(url_for("training"))

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
                    flash(_("Repo %(repo)s not valid, using default model %(default)s",
                            repo=controlnet_model_tmp, default=default_canny), "error")
            else:
                controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"

        learning_rate = sanitize_number(request.form.get("learning_rate", "2e-6"), 2e-6, False)
        steps = sanitize_number(request.form.get("steps", "500"), 500, True)
        train_batch_size = sanitize_number(request.form.get("train_batch_size", "2"), 2, True)
        n4 = request.form["N4"]
        gradient_accumulation_steps = None
        if "gradient_accumulation_steps" in request.form:
            gradient_accumulation_steps = sanitize_number(request.form.get("gradient_accumulation_steps", "1"), 1, True)
        resolution = None
        if "resolution" in request.form:
            resolution = sanitize_number(request.form.get("resolution", "512"), 512,  True)
            if resolution and int(resolution) > 512:
                resolution = 512
        checkpointing_steps = None
        if "checkpointing_steps" in request.form:
            checkpointing_steps = sanitize_number(request.form.get("checkpointing_steps", "250"), 250, True)
        validation_steps = None
        if "validation_steps" in request.form:
            validation_steps = sanitize_number(request.form.get("validation_steps", "125"),   125, True)

        mixed_precision = None
        if n4.lower() in ["true", "yes", "1"]:
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

                # local save
                local_val_path = os.path.join(tempfile.gettempdir(), filename)
                val_img.save(local_val_path)

                # unique name
                unique_id = str(uuid.uuid4())
                remote_val_host_path = f"/home/ubuntu/tesiControlNetFlux/remote_inputs/{unique_id}_{filename}"

                # send to host
                scp_to_lambda(local_val_path, remote_val_host_path)

                ssh_manager.run_command(
                    "sudo docker exec controlnet mkdir -p /workspace/tesiControlNetFlux/Src/remote_inputs"
                )

                # copy the file in the docker
                ssh_manager.run_command(
                    f"sudo docker cp {remote_val_host_path} "
                    f"controlnet:/workspace/tesiControlNetFlux/Src/remote_inputs/{unique_id}_{filename}"
                )

                remote_validation_path = f"/workspace/tesiControlNetFlux/Src/remote_inputs/{unique_id}_{filename}"

                prompt = sanitize_text(request.form.get("prompt"), "")
                if local_val_path and os.path.exists(local_val_path):
                    os.remove(local_val_path)
        print(hub_model_id)
        print(controlnet_model)
        cmd = [
            f"sudo docker exec -e PYTHONUNBUFFERED=1 "
            f"-e HUGGINGFACE_TOKEN={shlex.quote(str(HUGGINGFACE_TOKEN or ''))} "
            f"-e WANDB_TOKEN={shlex.quote(str(WANDB_TOKEN or ''))} "
            f"controlnet python3 -u /workspace/tesiControlNetFlux/Src/scripts/controlnet_training_api.py "
            f"--learning_rate {shlex.quote(str(learning_rate))}",
            f"--steps {shlex.quote(str(steps))}",
            f"--hub_model_id {shlex.quote(hub_model_id)}",
            f"--controlnet_model {shlex.quote(controlnet_model)}",
            f"--controlnet_type {shlex.quote(controlnet_type)}",
            f"--train_batch_size {shlex.quote(str(train_batch_size))}",
        ]
        if n4.lower() == "true":
            cmd.append(f"--N4")
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
        sse_url = url_for("sse_events", job_id=job_id)

        content = Div(
            H2(_("Training Job Submitted"), id="job-status"),
            P(_("Model: %(model)s", model=hub_model_id)),
            P(_("Job ID: %(job_id)s", job_id=job_id)),
            P(_("Elapsed time"), style="display:none;", id="time"),
            Div(_("Waiting for your turn..."), id="result-section"),
            Div(
                A(_("Go to Inference Page"),
                  href=url_for('inference'),
                  id="inference_link",
                  cls="button primary",
                  style="display: none; width: fit-content;"),
                style="background=none;border:none;box-shadow:none;",
                cls="center-box"
            ),
            Script(f"""
            const status = document.getElementById("job-status");
            const result_div = document.getElementById("result-section");
            const time_el = document.getElementById("time");
            function connectSSE() {{
              const es = new EventSource("{sse_url}");
              es.onmessage = (e) => {{
                const data = JSON.parse(e.data);
                if (data.status === "done") {{
                    status.innerText = "{_('Training Job Terminated')}";
                    if (data.output) {{
                        result_div.innerHTML = "<pre>" + (data.output || data.message || "") + "</pre>";
                        time_el.innerHTML = data.elapsed ? "{_('Elapsed time:')} " + data.elapsed + " " + "{_('seconds')}" : "";
                        time_el.style.display = "block";
                        document.getElementById("inference_link").style.display = "block";
                    }}
                    es.close();
                }} else if (data.status === "running") {{
                    result_div.innerHTML = `<pre class="progress-log">${{(data.message||'').slice(-800)}}</pre>`;
                }} else if (data.status === "error") {{
                    result_div.innerHTML = `<p style="color:#f66">{_('Error')}: ${{data.message || 'unknown'}}</p>`;
                    es.close();
                }}
              }};
              es.onerror = () => {{
                 console.warn("SSE connection lost, retrying in 3s...");
                 result_div.innerHTML = `<p style="color:orange">{_('Connection lost, retrying...')}</p>`;
                 es.close();
                 setTimeout(connectSSE, 3000);
              }};
            }}
            connectSSE();
            const inference_link = document.getElementById("inference_link");
            """),
            id="content", cls="result"
        )
        return str(base_layout(_("Waiting for Training"), content)), 200

    # GET: form
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
                    "data_N4": as_str(params.get("N4", False), "False"),
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
            Legend(_("Mode")),
            Select(
                Option(_("Fine-tune existing model"), value="existing"),
                Option(_("Create new model"), value="new"),
                id="mode", name="mode"
            ),
            Div(
                Label(_("Existing Model:")),
                Select(*options, id="existingModel", name="existing_model"),
                Label(_("ControlNet Type:")), Input(id="controlnet_type_existing", name="controlnet_type"),
                Label(_("Reuse as ControlNet?")),
                Select(Option(_("No"), value="no"), Option(_("Yes"), value="yes"),
                       name="reuse_as_controlnet", id="reuse_as_controlnet"),
                Div(
                    Label(_("ControlNet Source:")),
                    Select(
                        Option(_("Default - Canny"), value="canny"),
                        Option(_("Default - HED"), value="hed"),
                        Option(_("Use Existing"), value="existing"),
                        id="controlnet_source_existing", name="controlnet_source_existing"
                    ),
                    Div(
                        Label(_("Choose Existing ControlNet:")),
                        Select(*options, id="existingControlnetModel_existing", name="existing_controlnet_model_existing"),
                        id="existingControlnetWrapper_existing", style="display:none;"
                    ),
                    cls="form-row",
                    id="existingControlnetSourceWrapper", style="display:none;"
                ),
                id="existingModelWrapper"
            ),
            Div(
                Label(_("New Model Name:")), Input(name="new_model_name"),
                Label(_("ControlNet Type:")), Input(id="controlnet_type", name="controlnet_type"),
                Div(
                    Label(_("ControlNet Source:")),
                    Select(
                        Option(_("Default - Canny"), value="canny"),
                        Option(_("Default - HED"), value="hed"),
                        Option(_("Use Existing"), value="existing"),
                        id="controlnet_source", name="controlnet_source"
                    ),
                    Div(
                        Label(_("Choose Existing ControlNet:")),
                        Select(*options, id="existingControlnetModel", name="existing_controlnet_model"),
                        id="existingControlnetWrapper", style="display:none;"
                    ),
                    cls="form-row"
                ),
                id="newModelWrapper", style="display:none;"
            )
        ),
        Fieldset(
            Legend(_("Quantization & Precision")),
            Div(
                Label(_("Use N4 Quantization")),
                Select(
                    Option(_("No"), value="false", selected=True),
                    Option(_("Yes"), value="true"),
                    id="N4", name="N4"
                ),
                cls="form-row"
            ),
            Div(
                Label(_("Mixed Precision")),
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
            Legend(_("Training Parameters")),
            Div(
                Label(_("Learning Rate:")), Input(id="learning_rate", name="learning_rate", value="2e-6"),
                Label(_("Steps:")), Input(id="steps", name="steps", type="number", value="500"),
                cls="form-row"
            ),
            Div(
                Label(_("Batch Size:")), Input(id="train_batch_size", name="train_batch_size", type="number"),
                Label(_("Gradient Accumulation:")),
                Input(id="gradient_accumulation_steps", name="gradient_accumulation_steps", type="number", value="1"),
                cls="form-row"
            ),
            Div(
                Label(_("Resolution:")), Input(id="resolution", name="resolution", type="number", value="512"),
                Label(_("Checkpoint Steps:")),
                Input(id="checkpointing_steps", name="checkpointing_steps", type="number", value="250"),
                cls="form-row"
            ),
            Div(
                Label(_("Validation Steps:")),
                Input(id="validation_steps", name="validation_steps", type="number", value="125"),
                cls="form-row"
            )
        ),
        Fieldset(
            Legend(_("Validation")),
            Label(_("Validation Image:")),
            Input(id="validationImage", name="validation_image", type="file", accept=".jpg,.jpeg,.png"),
            Div(Label(_("Prompt:")), Input(id="prompt", name="prompt"), id="promptWrapper", style="display:none;")
        ),
        Button("Start Training", type="submit", cls="button primary"),
        Script(f"""
            (function(){{
                  function san(el, d, asInt){{
                    let n = Number(el.value);
                    if (!Number.isFinite(n)) {{ el.value = d; return; }}
                    if (n < 0) n = 1;
                    el.value = asInt ? Math.round(n) : n;
                  }}
                  const ids = [
                    ["learning_rate", 2e-6, false],
                    ["steps", 500, true],
                    ["train_batch_size", 2, true],
                    ["gradient_accumulation_steps", 1, true],
                    ["resolution", 512, true],
                    ["checkpointing_steps", 250, true],
                    ["validation_steps", 125, true],
                  ];
                  ids.forEach(([id, d, asInt])=>{{
                    const el = document.getElementById(id);
                    if (el) el.addEventListener('change', ()=>san(el, d, asInt));
                  }});
                
                  
                  function cleanText(el, d){{
                    const v = (el.value||"").trim().toLowerCase();
                    if (!v || ["nan","undefined","none","null"].includes(v)) {{
                      el.value = d;
                    }}
                  }}
                  const textIds = ["controlnet_type"];
                  textIds.forEach(id=>{{
                    const el = document.getElementById(id);
                    if (el) el.addEventListener("change", ()=>cleanText(el, el.defaultValue||"canny"));
                  }});
                
              
                  const defaults = {{
                    canny: {{
                      controlnet_type: "canny",
                      N4: "false",
                      learning_rate: "2e-6",
                      steps: 500,
                      train_batch_size: 2,
                      mixed_precision: "bf16",
                      gradient_accumulation_steps: 1,
                      resolution: 512,
                      checkpointing_steps: 250,
                      validation_steps: 125
                    }},
                    hed: {{
                      controlnet_type: "hed",
                      N4: "false",
                      learning_rate: "2e-6",
                      steps: 500,
                      train_batch_size: 2,
                      mixed_precision: "bf16",
                      gradient_accumulation_steps: 1,
                      resolution: 512,
                      checkpointing_steps: 250,
                      validation_steps: 125
                    }}
                  }};
                
                  function setVal(id, v){{ const el=document.getElementById(id); if(el) el.value=v; }}
                  function applyDefaultsFor(kind){{
                    const d = defaults[kind] || defaults.canny;
                    setVal("controlnet_type", d.controlnet_type);
                    setVal("N4", d.N4);
                    setVal("learning_rate", d.learning_rate);
                    setVal("steps", d.steps);
                    setVal("train_batch_size", d.train_batch_size);
                    setVal("mixed_precision", d.mixed_precision);
                    setVal("gradient_accumulation_steps", d.gradient_accumulation_steps);
                    setVal("resolution", d.resolution);
                    setVal("checkpointing_steps", d.checkpointing_steps);
                    setVal("validation_steps", d.validation_steps);
                
                    const mpGroup = document.getElementById("mixed_precision_group");
                    if (mpGroup) mpGroup.style.display = (String(d.N4).toLowerCase()==="true") ? "none" : "block";
                  }}
                
                  function maybeApplyDefaults(){{
                    const mode = document.getElementById("mode").value;
                    const source = document.getElementById("controlnet_source").value;
                    if (mode === "new" && (source === "canny" || source === "hed")){{
                      applyDefaultsFor(source);
                    }}
                  }}
                
                  const sourceSel = document.getElementById("controlnet_source");
                  const modeSel   = document.getElementById("mode");
                  if (sourceSel) sourceSel.addEventListener("change", maybeApplyDefaults);
                  if (modeSel)   modeSel.addEventListener("change", maybeApplyDefaults);
                  document.addEventListener("DOMContentLoaded", maybeApplyDefaults);
                }})();
              function fillFromSelected(selected){{
                  if (!selected) return;
                  const d = selected.dataset;
                
                  // read attributes with camelCase (dataset auto-converts)
                  const n4 = (d.n4 || "false").toString().toLowerCase();
                
                  document.querySelectorAll("#controlnet_type, #controlnet_type_existing").forEach(el=>{{
                    if (el) el.value = d.controlnetType || "canny";
                  }});
                
                  const setVal = (id, v) => {{ const el=document.getElementById(id); if(el && v !== undefined) el.value=v; }};
                
                  setVal("N4", n4);
                  setVal("steps", d.steps);
                  setVal("train_batch_size", d.trainBatchSize);
                  setVal("learning_rate", d.learningRate);
                  setVal("mixed_precision", d.mixedPrecision);
                  setVal("gradient_accumulation_steps", d.gradientAccumulationSteps);
                  setVal("resolution", d.resolution);
                  setVal("checkpointing_steps", d.checkpointingSteps);
                  setVal("validation_steps", d.validationSteps);
                
                  // toggle MP when N4 true
                  const mpGroup = document.getElementById("mixed_precision_group");
                  if (mpGroup) mpGroup.style.display = (n4 === "true") ? "none" : "block";
              }}
            
              function toggleMode(){{
                const mode = document.getElementById("mode").value;
                const newWrapper = document.getElementById("newModelWrapper");
                const existingWrapper = document.getElementById("existingModelWrapper");
                if (mode === "new"){{ newWrapper.style.display="block"; existingWrapper.style.display="none"; }}
                else {{ newWrapper.style.display="none"; existingWrapper.style.display="block"; }}
              }}
            
              function toggleExistingControlnetSource(){{
                const v = document.getElementById("reuse_as_controlnet").value;
                const wrapper = document.getElementById("existingControlnetSourceWrapper");
                wrapper.style.display = (v === "no") ? "block" : "none";
              }}
                
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
              function toggleExistingPicker(){{
                const v = document.getElementById("controlnet_source_existing").value;
                const w = document.getElementById("existingControlnetWrapper_existing");
                w.style.display = (v === "existing") ? "block" : "none";
              }}
            
              function toggleNewPicker(){{
                const v = document.getElementById("controlnet_source").value;
                const w = document.getElementById("existingControlnetWrapper");
                w.style.display = (v === "existing") ? "block" : "none";
              }}
            
              function togglePromptOnValidation(){{
                const f = document.getElementById("validationImage");
                const wrapper = document.getElementById("promptWrapper");
                if (!f || !wrapper) return;
                wrapper.style.display = (f.files && f.files.length>0) ? "block" : "none";
                const inp = document.getElementById("prompt");
                if (inp){{
                  if (f.files && f.files.length>0) inp.setAttribute("required", "true");
                  else inp.removeAttribute("required");
                }}
              }}
            
              // EVENT HOOKS
              document.getElementById("existingModel").addEventListener("change", function(){{
                fillFromSelected(this.options[this.selectedIndex]);
              }});
              document.getElementById("existingControlnetModel").addEventListener("change", function(){{
                fillFromSelected(this.options[this.selectedIndex]);
              }});
              document.getElementById("mode").addEventListener("change", toggleMode);
              document.getElementById("reuse_as_controlnet").addEventListener("change", toggleExistingControlnetSource);
              document.getElementById("controlnet_source_existing").addEventListener("change", toggleExistingPicker);
              document.getElementById("controlnet_source").addEventListener("change", toggleNewPicker);
              document.getElementById("validationImage").addEventListener("change", togglePromptOnValidation);
            
              // INITIALIZE ON LOAD
              document.addEventListener("DOMContentLoaded", function(){{
                toggleMode();
                toggleExistingControlnetSource();
                toggleExistingPicker();
                toggleNewPicker();
                togglePromptOnValidation();
                const sel = document.getElementById("existingModel");
                if (sel) fillFromSelected(sel.options[sel.selectedIndex]);
              }});
            """),
        method="post",
        id="trainingForm",
        cls="form-card",
        action=url_for("training"), enctype="multipart/form-data"
    )
    return str(base_layout(_("Training"), form)), 200


@app.route('/results', methods=["GET"])
def results():
    is_connected = session.get('lambda_connected', False)
    if not is_connected:
        return redirect(url_for('index'))

    models = models_list()
    selected_model = request.args.get("model", "all")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 4))

    if 'results_cursors' not in session:
        session['results_cursors'] = {}

    # Page num not used as cloudinary don't use it
    def fetch_page(prefix, model_key, page_num, per_page, start_cursor = None, filter_repo_image=False):
        collected = []
        cursor = start_cursor

        while True:
            try:
                res = resources(type="upload", prefix=prefix, max_results=per_page, next_cursor=cursor)
            except Exception:
                return [], None

            batch = res.get("resources", [])
            if filter_repo_image:
                batch = [r for r in batch if
                         "/repo_image/" in r.get("secure_url", "") or "/repo_image/" in r.get("public_id", "")]

            for r in batch:
                if len(collected) < per_page:
                    collected.append(r)

            next_cursor = res.get("next_cursor")

            # If we've filled the page, decide whether a "Next" truly exists by peeking ahead.
            if len(collected) >= per_page:
                if not next_cursor:
                    return collected[:per_page], None

                # Look ahead to see if at least one more filtered item exists.
                look_cursor = next_cursor
                while look_cursor:
                    try:
                        look_res = resources(type="upload", prefix=prefix, max_results=per_page,
                                             next_cursor=look_cursor)
                    except Exception:
                        break
                    look_batch = look_res.get("resources", [])
                    if filter_repo_image:
                        look_batch = [r for r in look_batch if
                                      "/repo_image/" in r.get("secure_url", "") or "/repo_image/" in r.get("public_id",
                                                                                                           "")]
                    if look_batch:
                        return collected[:per_page], next_cursor
                    look_cursor = look_res.get("next_cursor")
                # No more items
                return collected[:per_page], None

            # If this batch didn't fill the page and there's no cursor, we're out of items
            if not next_cursor:
                return collected, None

            # Otherwise continue fetching until either page is full or no more items
            cursor = next_cursor

    next_cursor = None
    try:
        if selected_model == "all":
            prefix = f"{HF_NAMESPACE}/"
            cursors_for_model = session['results_cursors'].get(prefix, {})
            start_cursor = cursors_for_model.get(str(page-1)) if page > 1 else None
            items, next_cursor = fetch_page(prefix, prefix, page, per_page, start_cursor=start_cursor, filter_repo_image=True)
            image_resources = items
            session['results_cursors'].setdefault(prefix, {})[str(page)] = next_cursor
        else:
            model_name = selected_model.split("/")[-1]
            prefix = f"{HF_NAMESPACE}/{model_name}_results/repo_image/"
            cursors_for_model = session['results_cursors'].get(prefix, {})
            start_cursor = cursors_for_model.get(str(page-1)) if page > 1 else None
            items, next_cursor = fetch_page(prefix, prefix, page, per_page, start_cursor=start_cursor, filter_repo_image=False)
            image_resources = items
            session['results_cursors'].setdefault(prefix, {})[str(page)] = next_cursor

        grids = []
        for r in image_resources:
            url = r.get("secure_url")
            control_url = url.replace("/repo_image/", "/repo_control/").rsplit(".", 1)[0] + "_control.jpg"
            text_url = url.replace("/image/upload/", "/raw/upload/").replace("/repo_image/", "/repo_text/").rsplit(".", 1)[0] + "_text"

            params_text, prompt_text, control_img_tag = "", "", None
            try:
                resp = requests.get(text_url, timeout=5)
                if resp.status_code == 200:
                    params_text = resp.text
                    for line in params_text.splitlines():
                        if line.lower().startswith("prompt:"):
                            prompt_text = line
                    if params_text:
                        params_text = "\n".join(
                            l for l in params_text.splitlines()
                            if not l.lower().startswith("prompt:")
                        )
            except Exception:
                params_text = ""

            try:
                h = requests.head(control_url, timeout=4)
                if h.status_code == 200:
                    control_img_tag = Img(src=control_url, cls="card-img")
            except Exception:
                control_img_tag = None

            resized_url = url
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
        flash(_("Error fetching results: %(err)s", err=e), "error")
        grids = []

    pagination_children = []
    if page > 1:
        pagination_children.append(
            A("⬅ " + _("Previous"),
              href=url_for("results", model=selected_model, page=page - 1, per_page=per_page),
              cls="button secondary")
        )
    if next_cursor:
        pagination_children.append(
            A(_("Next") + " ➡",
              href=url_for("results", model=selected_model, page=page + 1, per_page=per_page),
              cls="button secondary")
        )
    pagination = Div(*pagination_children, cls="pagination") if pagination_children else ""

    model_options = [Option(_("All Models"), value="all", selected=(selected_model == "all"))]
    for m in models:
        model_options.append(
            Option(m["id"].split("/")[-1], value=m["id"], selected=(selected_model == m["id"]))
        )
    selector = Form(
        Select(*model_options, name="model", onchange="this.form.submit()", value=selected_model, style="width:100%"),
        method="get",
        style="width:100%; margin-bottom:1rem;"
    )

    content = Div(
        H1(_("Model Results"), cls="hero-title"),
        selector,
        Div(*grids, cls="card-grid"),
        pagination
    )
    return str(base_layout(_("Results"), content)), 200

ssh_manager = SSHManager(LAMBDA_INSTANCE_IP, LAMBDA_INSTANCE_USER, SSH_PRIVATE_KEY_PATH)
#Worker start
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

# main driver function
if __name__ == '__main__':

    if not app.secret_key:
        print("WARNING: FLASK_SECRET_KEY is not set in .env. Using a default for development.")
        print("Set FLASK_SECRET_KEY=your_random_string_here for production.")
    if not LAMBDA_CLOUD_API_KEY:
        print("WARNING: LAMBDA_CLOUD_API_KEY is not set in .env.")
    if not SSH_PRIVATE_KEY_PATH:
        print("WARNING: SSH_PRIVATE_KEY_PATH is not set in .env.")
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
