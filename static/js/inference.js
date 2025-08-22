const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

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
    ctx.strokeStyle = "white";
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    window.convertedImage = null;
}

document.getElementById("uploadInput").addEventListener("change", async function (e) {
    const files = e.target.files;
    if (!files.length) return;

    const model = document.getElementById("model").value;
    const formData = new FormData();

    for (let file of files) {
        formData.append("images", file);
    }
    formData.append("model", model);

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

document.querySelector("form").addEventListener("submit", function () {
    const dataURL = canvas.toDataURL("image/png");
    document.getElementById("controlImageData").value = dataURL;
});