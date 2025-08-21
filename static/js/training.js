// Autofill training parameters when selecting an existing model
document.getElementById("existingModel").addEventListener("change", function() {
    const selected = this.options[this.selectedIndex];
    document.getElementById("controlnetType").value = selected.dataset.controlnet_type;
    document.getElementById("N4").value = selected.dataset.n4;
    document.getElementById("steps").value = selected.dataset.steps;
    document.getElementById("trainBatchSize").value = selected.dataset.train_batch_size;
    document.getElementById("learningRate").value = selected.dataset.learning_rate;
    document.getElementById("mixedPrecision").value = selected.dataset.mixed_precision;
    document.getElementById("gradient_accumulation_steps").value = selected.dataset.gradient_accumulation_steps;
    document.getElementById("resolution").value = selected.dataset.resolution;
    document.getElementById("checkpointing_steps").value = selected.dataset.checkpointing_steps;
    document.getElementById("validation_steps").value = selected.dataset.validation_steps;
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