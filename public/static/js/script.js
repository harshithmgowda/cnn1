/* ===================================================
   CIFAR-10 — Swiss International Typographic Style
   Pure interaction logic. No decoration.
   =================================================== */

const fileElem = document.getElementById("fileElem");
const chooseBtn = document.getElementById("chooseBtn");
const dropArea = document.getElementById("drop-area");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");
const message = document.getElementById("message");
const previewSection = document.getElementById("previewSection");
const resultSection = document.getElementById("resultSection");
const imagePreview = document.getElementById("imagePreview");
const predictionText = document.getElementById("predictionText");
const confidenceValue = document.getElementById("confidenceValue");
const progressBar = document.getElementById("progressBar");
const top3List = document.getElementById("top3List");

let selectedFile = null;

/* --- Reset --- */
function resetUI() {
  selectedFile = null;
  fileElem.value = "";
  predictBtn.disabled = true;
  resetBtn.style.display = "none";
  previewSection.style.display = "none";
  resultSection.style.display = "none";
  imagePreview.innerHTML = "";
  predictionText.textContent = "\u2014";
  confidenceValue.textContent = "\u2014";
  progressBar.style.width = "0%";
  top3List.innerHTML = "";
  message.textContent = "";
}

/* --- File Select --- */
chooseBtn.addEventListener("click", () => {
  fileElem.click();
});

fileElem.addEventListener("change", (e) => {
  handleFile(e.target.files[0]);
});

/* --- Drag & Drop --- */
["dragenter", "dragover"].forEach((evt) => {
  dropArea.addEventListener(evt, (e) => {
    e.preventDefault();
    dropArea.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropArea.addEventListener(evt, (e) => {
    e.preventDefault();
    dropArea.classList.remove("dragover");
  });
});

dropArea.addEventListener("drop", (e) => {
  handleFile(e.dataTransfer.files[0]);
});

/* --- Handle File --- */
function handleFile(f) {
  if (!f) return;

  if (f.size > 4 * 1024 * 1024) {
    message.textContent = "FILE TOO LARGE \u2014 MAXIMUM 4 MB";
    return;
  }

  if (!f.type.startsWith("image/")) {
    message.textContent = "INVALID FILE TYPE \u2014 IMAGE REQUIRED";
    return;
  }

  selectedFile = f;
  predictBtn.disabled = false;
  resetBtn.style.display = "inline-block";
  message.textContent = "";
  showPreview(f);
}

/* --- Preview --- */
function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded image" />`;
    previewSection.style.display = "block";
  };
  reader.readAsDataURL(file);
}

/* --- Predict --- */
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    message.textContent = "NO IMAGE SELECTED";
    return;
  }

  predictBtn.disabled = true;
  message.textContent = "CLASSIFYING\u2026";
  predictionText.textContent = "\u2026";
  confidenceValue.textContent = "";
  progressBar.style.width = "0%";
  top3List.innerHTML = "";
  resultSection.style.display = "block";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await resp.json();

    if (!resp.ok) {
      message.textContent = (data.error || "CLASSIFICATION FAILED").toUpperCase();
      predictionText.textContent = "\u2014";
      predictBtn.disabled = false;
      return;
    }

    message.textContent = "";
    predictionText.textContent = data.prediction.toUpperCase();
    confidenceValue.textContent = `${data.confidence.toFixed(2)}%`;
    progressBar.style.width = `${data.confidence}%`;

    top3List.innerHTML = "";
    if (Array.isArray(data.top3)) {
      data.top3.forEach((item, i) => {
        const row = document.createElement("div");
        row.className = "top3-row";
        row.innerHTML = `
          <span class="t3-rank">${String(i + 1).padStart(2, "0")}</span>
          <span class="t3-label">${item.label}</span>
          <span class="t3-conf">${item.confidence.toFixed(2)}%</span>
        `;
        top3List.appendChild(row);
      });
    }
  } catch (err) {
    console.error(err);
    message.textContent = "NETWORK ERROR \u2014 TRY AGAIN";
  } finally {
    predictBtn.disabled = false;
  }
});

/* --- Reset --- */
resetBtn.addEventListener("click", () => {
  resetUI();
});

/* --- Init --- */
resetUI();
