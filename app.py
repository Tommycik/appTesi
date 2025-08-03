import uuid
from io import BytesIO

import cloudinary
from flask import Flask, url_for, request, redirect, flash, session, get_flashed_messages, make_response, jsonify
import requests
from dotenv import load_dotenv
import paramiko
from fasthtml.common import *
from typing import Any
import time  #for cache busting
import threading
import queue
from PIL import Image
import cv2
import numpy as np
from werkzeug.utils import secure_filename


def scp_to_lambda(local_path, remote_path):
    import paramiko
    from scp import SCPClient

    key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY_PATH)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(LAMBDA_INSTANCE_IP, username=LAMBDA_INSTANCE_USER, pkey=key)

    with SCPClient(ssh.get_transport()) as scp:
        scp.put(local_path, remote_path)

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

#serializing request to lambda
inference_queue = queue.Queue()
# Dictionary to store results, keyed by a unique job ID
results_db = {}
# A lock to serialize the SSH command
worker_lock = threading.Lock()

def worker():
    while True:
        job = inference_queue.get()
        if job is None:
            break
        try:
            output, errors = ssh_manager.run_command(job["command"])
            result_url = None
            if output and "https" in output:
                result_url = output.strip()
            results_db[job["job_id"]] = result_url or errors or "Unknown error"
        except Exception as e:
            results_db[job["job_id"]] = f"Exception during inference: {e}"
        inference_queue.task_done()

app = Flask(__name__)
app.secret_key = os.urandom(12)

LAMBDA_INSTANCE_ID = os.getenv('LAMBDA_INSTANCE_ID')
LAMBDA_CLOUD_API_BASE = "https://cloud.lambdalabs.com/api/v1/instances"
LAMBDA_CLOUD_API_KEY = os.getenv('LAMBDA_CLOUD_API_KEY')
LAMBDA_INSTANCE_IP = os.getenv('LAMBDA_INSTANCE_IP', "YOUR_LAMBDA_INSTANCE_PUBLIC_IP")
LAMBDA_INSTANCE_USER = os.getenv('LAMBDA_INSTANCE_USER', "ubuntu")
SSH_PRIVATE_KEY_PATH = os.getenv('SSH_PRIVATE_KEY_PATH')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
DOCKER_IMAGE_NAME = os.getenv('DOCKER_IMAGE_NAME',
                              "your-dockerhub-username/controlnet-generator:latest")  #when docker is ready
REGION = "us-west-3"

CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

model_map = {
    "hed": "tommycik/controlFluxAlcolHed",
    "reduced": "tommycik/controlFluxAlcolReduced",
    "standard": "tommycik/controlFluxAlcol",
}
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

            for instance in instances:
                if instance.get("id") == instance_id:
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
def get_flashed_html_messages():
    messages_html = []
    for category, message in get_flashed_messages(with_categories=True):
        messages_html.append(
            Div(message, class_=f"alert alert-{category}")
        )
    if messages_html:
        return Div(*messages_html, class_="messages")
    return ""


def base_layout(title: str, content: Any, scripts: Any = None, navigation: Any = None):
    cache_buster = int(time.time())  # Current Unix timestamp
    if navigation is None:

        navigation = Nav(
            A("Connect to Lambda & Pull Docker Image", href=url_for('connect_lambda'), class_="nav-link"),
            A("Connect to Lambda & Pull Docker Image", href=url_for('connect_lambda'), class_="nav-link")
            , class_="main-nav")
    else:
        navigation = Nav(navigation, class_="main-nav")

    return Div(

        Head(
            Meta(charset="UTF-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
            Title(f"Interactive Flask App - {title}"),
            Link(rel="stylesheet", href=url_for('static', filename='css/style.css', v=cache_buster))
        ),
        Body(

            Header(H1("ControlNet App", class_="site-title"), navigation),

            Main(Div(content, class_="container")),

            Footer(P("2025 Lambda ControlNet App. All rights reserved.")),

            Script(src=url_for('static', filename='js/script.js')),
            scripts or ""  # Page-specific scripts
        )
    )

def convert_to_canny(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    out_path = input_path.replace(".png", "_canny.png")
    cv2.imwrite(out_path, edges)
    return out_path

def convert_to_hed(input_path):
    # Placeholder: use OpenCV's structured forest edge detection or deep HED model if available
    # For now just use canny as placeholder
    return convert_to_canny(input_path)

@app.route('/')
def index():
    is_connected = session.get('lambda_connected', False)
    if not is_connected:
        action_button_section = Div(
            P("Please initialize your Lambda Cloud connection and ensure the Docker image is ready."),
            P(
                A("Connect to Lambda & Pull Docker Image", href=url_for('connect_lambda'), class_="button-link")
            )
        )

    else:
        action_button_section = Div(
            P("Lambda initialize and Docker image pulled. You can now proceed to use the models."),
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
        get_flashed_html_messages()
    )
    html_obj = base_layout("Connect to lambda cloud", content)
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
    stdout, stderr = ssh_manager.run_command(command)  #command for docker

    if stderr and "no space left on device" in stderr.lower():
        flash(f'Error during SSH: No space left on device. {stderr}', 'error')
    elif stderr:
        flash(f'SSH failed: {stderr}', 'error')
    else:
        flash('Successfully verified Lambda instance is up and SSH is working.', 'success')
        session['lambda_connected'] = True

    return redirect(url_for('index'))

@app.route('/inference_result/<job_id>', methods=["GET"])
def get_inference_result(job_id):
    result = results_db.get(job_id)
    if result is None:
        return jsonify({"status": "pending"})
    return jsonify({"status": "done", "output": result})

@app.route('/inference', methods=["GET", "POST"])
def inference():
    if request.method == "POST":
        prompt = request.form['prompt']
        scale = request.form.get('scale', 0.2)
        steps = request.form.get('steps', 50)
        guidance = request.form.get('guidance', 6.0)
        model = request.form.get('model', 'standard')


        model_type = "canny"
        n4 = False
        if model == "hed":
            model_type = "hed"
        elif model == "reduced":
            n4 = True
        control_image_path = None
        remote_control_path = None

        data_url = request.form.get('control_image_data')
        if data_url and "," in data_url:
            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(binary_data)).convert("RGB")

            # Check if image is completely white (or blank)
            np_image = np.array(image)
            if np_image.max() != 255 or np_image.min() != 255:
                control_image_path = f"/tmp/{uuid.uuid4()}.png"
                image.save(control_image_path)
                remote_control_path = f"/home/ubuntu/tesiControlNetFlux/remote_inputs/{uuid.uuid4()}.png"
                scp_to_lambda(control_image_path, remote_control_path)
        # Call Lambda via SSH
        command = (
            f"export CLOUDINARY_CLOUD_NAME={CLOUDINARY_CLOUD_NAME} &&"
            f"export HUGGINGFACE_TOKEN={HUGGINGFACE_TOKEN} &&"
            f"export CLOUDINARY_API_KEY={CLOUDINARY_API_KEY} &&"
            f"export CLOUDINARY_API_SECRET={CLOUDINARY_API_SECRET} &&"
            f"source venv_flux/bin/activate && cd tesiControlNetFlux/Src && python3 scripts/controlnet_infer_api.py "
            f"--prompt \"{prompt}\" --scale {scale} --steps {steps} --guidance {guidance} --controlnet_model {model_map[model]} --N4 {n4} --controlnet_type {model_type}"
        )
        if remote_control_path:
            command += f"--control_image {remote_control_path} "
        job_id = str(uuid.uuid4())
        inference_queue.put({"job_id": job_id, "command": command})

        content = Div(
            H2("Inference Job Submitted"),
            P(f"Prompt: {prompt}"),
            P(f"Job ID: {job_id}"),
            Div("Waiting for result...", id="result-section"),
            Script(f"""
            async function pollResult() {{
                const res = await fetch("{url_for('get_inference_result', job_id=job_id)}");
                const data = await res.json();
                if (data.status === "done") {{
                    let resultDiv = document.getElementById("result-section");
                    if (data.output.startsWith("http")) {{
                        resultDiv.innerHTML = `<img src="${{data.output}}" style='max-width: 500px;'/>`;
                    }} else {{
                        resultDiv.innerHTML = "<p>" + data.output + "</p>";
                    }}
                }} else {{
                    setTimeout(pollResult, 2000);
                }}
            }}
            pollResult();
            """)
        )

        return str(
            base_layout("Waiting for Inference", content, navigation=A("Back to Home", href=url_for('index')))), 200

    # GET method: render form
    form = Form(
        Select(
            Option("ControlNet Canny", value="standard"),
            Option("ControlNet HED", value="hed"),
            Option("ControlNet Canny reduced", value="reduced"),
            id="model"
        ),
        Label("Prompt:"),
        Input(type="text", name="prompt", required=True),
        Div(
            Label("Upload Image:"),
            Input(type="file", id="uploadInput", accept="image/*"),

            Br(), Br(),

            Canvas(id="drawCanvas", width="512", height="512", style="border:1px solid black;"),

            Br(),

            Button("Clear Canvas", type="button", onclick="clearCanvas()"),

            Br(), Br(),

            Input(type="hidden", name="control_image_data", id="controlImageData"),

            Script("""
                const canvas = document.getElementById("drawCanvas");
                const ctx = canvas.getContext("2d");
                let drawing = false;
            
                canvas.addEventListener("mousedown", () => drawing = true);
                canvas.addEventListener("mouseup", () => {
                    drawing = false;
                    ctx.beginPath();
                });
                canvas.addEventListener("mousemove", draw);
            
                function draw(e) {
                    if (!drawing) return;
                    ctx.lineWidth = 3;
                    ctx.lineCap = "round";
                    ctx.strokeStyle = "black";
                    ctx.lineTo(e.offsetX, e.offsetY);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(e.offsetX, e.offsetY);
                }
            
                function clearCanvas() {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    if (window.convertedImage) {
                        ctx.drawImage(window.convertedImage, 0, 0, canvas.width, canvas.height);
                    }
                }
            
                document.getElementById("uploadInput").addEventListener("change", async function (e) {
                    const file = e.target.files[0];
                    if (!file) return;
            
                    const modelType = document.getElementById("model").value; // hed or canny
                    const formData = new FormData();
                    formData.append("image", file);
                    formData.append("model_type", modelType);
            
                    const response = await fetch("/preprocess_image", { method: "POST", body: formData });
                    const data = await response.json();
            
                    if (data.status === "ok") {
                        const img = new Image();
                        img.onload = function () {
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                            window.convertedImage = img;
                        };
                        img.src = data.converted_data_url;
                    } else {
                        alert("Error: " + data.error);
                    }
                });
            
                document.querySelector("form").addEventListener("submit", function (e) {
                    const dataURL = canvas.toDataURL("image/png");
                    document.getElementById("controlImageData").value = dataURL;
                });
                """
            )
        ),

        Label("Conditioning Scale (0.2):"),
        Input(type="number", name="scale", step="0.1", value="0.2"),
        Label("Steps (50):"),
        Input(type="number", name="steps", value="50"),
        Label("Guidance Scale (6.0):"),
        Input(type="number", name="guidance", step="0.5", value="6.0"),
        Button("Submit", type="submit")
        , method="post", enctype="multipart/form-data")

    return str(base_layout("ControlNet Inference", form,
                           navigation=A("Back to Inference Menu", href=url_for('inference')))), 200

@app.route("/preprocess_image", methods=["POST"])
def preprocess_image():
    try:
        file = request.files["image"]
        model_type = request.form["model_type"]
        image = Image.open(file.stream).convert("RGB")

        temp_path = f"/tmp/{uuid.uuid4()}.png"
        image.save(temp_path)

        if model_type == "canny":
            result_path = convert_to_canny(temp_path)
        elif model_type == "hed":
            result_path = convert_to_hed(temp_path)
        else:
            return jsonify({"status": "error", "error": "Invalid model_type"})

        with open(result_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            data_url = f"data:image/png;base64,{encoded}"
            return jsonify({"status": "ok", "converted_data_url": data_url})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})
@app.route('/training')
def training():
    return


@app.route('/training/controlnet')
def training_controlnet():
    return


@app.route('/training/controlnetReduced')
def training_controlnet_reduced():
    return


@app.route('/training/controlnetHed')
def training_controlnet_hed():
    return


@app.route('/results')
def results():
    return


@app.route('/results/controlnet')
def results_controlnet():
    return


@app.route('/results/controlnetReduced')
def results_controlnet_reduced():
    return


@app.route('/results/controlnetHed')
def results_controlnet_hed():
    return


# main driver function
if __name__ == '__main__':
    threading.Thread(target=worker, daemon=True).start()
    if not app.secret_key:
        print("WARNING: FLASK_SECRET_KEY is not set in .env. Using a default for development.")
        print("Set FLASK_SECRET_KEY=your_random_string_here for production.")
    if not LAMBDA_CLOUD_API_KEY:
        print("WARNING: LAMBDA_CLOUD_API_KEY is not set in .env.")
    if not SSH_PRIVATE_KEY_PATH:
        print("WARNING: SSH_PRIVATE_KEY_PATH is not set in .env.")

    app.run(debug=True, host='0.0.0.0', port=5000)
