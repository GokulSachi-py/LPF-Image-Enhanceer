const form = document.querySelector("#enhance-form");
const submitButton = document.querySelector("#submit-button");
const loadingState = document.querySelector("#loading-state");
const resultsSection = document.querySelector("#results");
const errorBox = document.querySelector("#error-box");

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function displayMetrics(data) {
  const referenceMetrics = document.querySelector("#reference-metrics");
  const metricsNote = document.querySelector("#metrics-note");
  const lpipsNote = document.querySelector("#lpips-note");
  const scores = data.reference_metrics;

  document.querySelector("#brightness-before").textContent = `${data.statistics.original.brightness}%`;
  document.querySelector("#brightness-after").textContent = `${data.statistics.enhanced.brightness}%`;
  document.querySelector("#contrast-before").textContent = `${data.statistics.original.contrast}%`;
  document.querySelector("#contrast-after").textContent = `${data.statistics.enhanced.contrast}%`;

  if (!scores) {
    referenceMetrics.hidden = true;
    lpipsNote.hidden = true;
    metricsNote.textContent =
      "PSNR, SSIM, and LPIPS need a matching well-lit reference image. Brightness and contrast below are descriptive measurements, not quality scores.";
    return;
  }

  referenceMetrics.hidden = false;
  metricsNote.textContent =
    "Compared with your reference image. PSNR and SSIM use the project’s LPF evaluation method.";
  document.querySelector("#metric-psnr").textContent =
    scores.psnr === null ? "∞" : scores.psnr;
  document.querySelector("#metric-ssim").textContent = scores.ssim;
  document.querySelector("#metric-lpips").textContent = scores.lpips ?? "—";
  lpipsNote.hidden = !scores.lpips_note;
  if (scores.lpips_note) lpipsNote.textContent = scores.lpips_note;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  resultsSection.hidden = true;
  loadingState.hidden = false;
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/enhance", {
      method: "POST",
      body: new FormData(form),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not enhance this image.");

    document.querySelector("#original-image").src = data.images.original;
    document.querySelector("#enhanced-image").src = data.images.enhanced;
    document.querySelector("#download-link").href = data.images.enhanced;
    displayMetrics(data);
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    if (error instanceof TypeError) {
      showError("Cannot reach the enhancement server. Start it with `python app.py`, keep that terminal open, then try again.");
    } else {
      showError(error.message);
    }
  } finally {
    loadingState.hidden = true;
    submitButton.disabled = false;
  }
});
