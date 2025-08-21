// Autofill training parameters when selecting an existing model
document.getElementById("existingModel").addEventListener("change", function() {
  const selected = this.options[this.selectedIndex];
  const ds = selected.dataset;
  if (!ds) return;

  document.getElementById("controlnet_type").value = ds.controlnetType || "canny";
  document.getElementById("N4").value = ds.n4 || "false";
  document.getElementById("steps").value = ds.steps || "1000";
  document.getElementById("train_batch_size").value = ds.trainBatchSize || "2";
  document.getElementById("learning_rate").value = ds.learningRate || "2e-6";
  document.getElementById("mixed_precision").value = ds.mixedPrecision || "fp16";
});

// Show/hide prompt input if a validation image is uploaded
document.getElementById("validationImage").addEventListener("change", function () {
  const wrapper = document.getElementById("promptWrapper");
  if (this.files.length > 0) {
    wrapper.style.display = "block";
    document.getElementById("prompt").setAttribute("required", "true");
  } else {
    wrapper.style.display = "none";
    document.getElementById("prompt").removeAttribute("required");
  }
});

document.getElementById("mode").addEventListener("change", function() {
    const wrapper = document.getElementById("newModelControlnetWrapper");
    if (this.value === "new") {
        wrapper.style.display = "block";
    } else {
        wrapper.style.display = "none";
    }
});

document.getElementById("controlnetSource").addEventListener("change", function() {
    const existingWrapper = document.getElementById("existingControlnetWrapper");
    if (this.value === "existing") {
        existingWrapper.style.display = "block";
    } else {
        existingWrapper.style.display = "none";
    }
});