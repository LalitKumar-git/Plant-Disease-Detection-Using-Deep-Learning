
const uploadForm = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");

const predictBtn = document.getElementById("predictBtn");
const loading = document.getElementById("loading");
const errorMessage = document.getElementById("errorMessage");

const resultSection = document.getElementById("resultSection");
const predictionResult = document.getElementById("predictionResult");

// 1. Image Preview
imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    errorMessage.textContent = "";
    resultSection.hidden = true;

    if (!file) {
        previewContainer.style.display = "none";
        return;
    }

    const allowedTypes = [
        "image/png",
        "image/jpeg",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {
        errorMessage.textContent = "Please select a PNG, JPG, JPEG or WEBP image.";
        imageInput.value = "";
        previewContainer.style.display = "none";
        return;
    }

    const imageURL = URL.createObjectURL(file);

    imagePreview.src = imageURL;
    previewContainer.style.display = "block";
});

// 2. HTML Escape Helper
function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, function (char) {
        return {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;"
        }[char];
    });
}

// 3. Upload Image and Predict Disease
uploadForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const file = imageInput.files[0];

    if (!file) {
        errorMessage.textContent = "Please select a plant image first.";
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    errorMessage.textContent = "";
    resultSection.hidden = true;
    loading.hidden = false;
    predictBtn.disabled = true;
    predictBtn.textContent = "Analyzing...";

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Prediction failed.");
        }

        // 4. Display Prediction Result
        let topFiveHTML = "";

        if (Array.isArray(data.top_5)) {
            topFiveHTML = data.top_5.map(function (item, index) {
                return `
                    <p>
                        ${index + 1}. ${escapeHTML(item.crop)}
                        - ${escapeHTML(item.disease)}
                        (${escapeHTML(item.confidence)}%)
                    </p>
                `;
            }).join("");
        }

        predictionResult.innerHTML = `
            <h3>${escapeHTML(data.crop)}</h3>

            <p>
                <strong>Disease:</strong>
                ${escapeHTML(data.disease)}
            </p>

            <p>
                <strong>Status:</strong>
                ${data.is_healthy ? "Healthy" : "Diseased"}
            </p>

            <p>
                <strong>Confidence:</strong>
                ${escapeHTML(data.confidence)}%
            </p>

            <p>
                <strong>Symptoms:</strong>
                ${escapeHTML(data.symptoms)}
            </p>

            <p>
                <strong>Treatment:</strong>
                ${escapeHTML(data.treatment)}
            </p>

            <hr>

            <h3>Top 5 Predictions</h3>
            ${topFiveHTML}
        `;

        resultSection.hidden = false;

        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (error) {
        errorMessage.textContent = error.message;
    } finally {
        loading.hidden = true;
        predictBtn.disabled = false;
        predictBtn.textContent = "Detect Plant Disease";
    }
});
