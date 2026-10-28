// ---------------------------------------------------------------
// فرانت‌اند ساده: موضوع را به بک‌اند می‌فرستد و مراحل کار را
// به‌صورت زنده (Server-Sent Events) از بک‌اند دریافت می‌کند.
// ---------------------------------------------------------------

// عنوان مراحل — همان ترتیبی که بک‌اند اجرا می‌کند
const STEP_DEFINITIONS = [
  { key: "topic", title: "دریافت موضوع پژوهش" },
  { key: "decide", title: "تصمیم مدل برای استفاده از ابزار جست‌وجو" },
  { key: "search", title: "جست‌وجوی اینترنت" },
  { key: "send", title: "ارسال نتایج به مدل" },
  { key: "article", title: "تولید مقاله" },
  { key: "save", title: "ذخیره مقاله روی لپ‌تاپ" },
];

const el = {
  topic: document.getElementById("topic"),
  startBtn: document.getElementById("startBtn"),
  errorBox: document.getElementById("errorBox"),
  steps: document.getElementById("steps"),
  resultCard: document.getElementById("resultCard"),
  summary: document.getElementById("summary"),
  sources: document.getElementById("sources"),
  pathBtn: document.getElementById("pathBtn"),
  filePath: document.getElementById("filePath"),
  article: document.getElementById("article"),
};

let currentFilePath = "";

// ساخت لیست مراحل با وضعیت «در انتظار»
function renderSteps() {
  el.steps.innerHTML = "";
  STEP_DEFINITIONS.forEach((step) => {
    const li = document.createElement("li");
    li.className = "step step-pending";
    li.dataset.key = step.key;
    li.innerHTML =
      '<span class="step-title"></span> ' +
      '<span class="step-status">در انتظار</span>' +
      '<span class="step-detail"></span>';
    li.querySelector(".step-title").textContent = step.title;
    el.steps.appendChild(li);
  });
}

// به‌روزرسانی وضعیت یک مرحله
function updateStep(key, status, detail) {
  const li = el.steps.querySelector('li[data-key="' + key + '"]');
  if (!li) return;
  const labels = { running: "در حال اجرا", done: "انجام شد", skipped: "رد شد", error: "خطا" };
  li.className = "step step-" + status;
  li.querySelector(".step-status").textContent = labels[status] || status;
  if (detail !== undefined && detail !== null) {
    li.querySelector(".step-detail").textContent = detail;
  }
}

function showError(message) {
  el.errorBox.textContent = message;
  el.errorBox.hidden = false;
}

function clearError() {
  el.errorBox.textContent = "";
  el.errorBox.hidden = true;
}

// نمایش نتیجه نهایی: خلاصه، منابع و متن مقاله
function showResult(result) {
  el.summary.textContent = result.summary || "(خلاصه‌ای تولید نشد)";

  el.sources.innerHTML = "";
  (result.sources || []).forEach((source) => {
    const li = document.createElement("li");
    const link = document.createElement("a");
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = source.title || source.url;
    li.appendChild(link);
    if (source.snippet) {
      const small = document.createElement("span");
      small.className = "step-detail";
      small.textContent = " — " + source.snippet;
      li.appendChild(small);
    }
    el.sources.appendChild(li);
  });
  if (!el.sources.children.length) {
    const li = document.createElement("li");
    li.textContent = "منبعی از جست‌وجو در دسترس نبود.";
    el.sources.appendChild(li);
  }

  el.article.textContent = result.article_markdown || "";
  currentFilePath = result.file_path || "";
  el.filePath.textContent = currentFilePath;
  el.filePath.hidden = true;
  el.resultCard.hidden = false;
}

// پردازش هر رویداد دریافتی از بک‌اند
function handleEvent(event) {
  if (event.type === "step") {
    updateStep(event.key, event.status, event.detail);
  } else if (event.type === "done") {
    showResult(event.result);
  } else if (event.type === "error") {
    showError(event.message);
  }
}

// خواندن جریان SSE با fetch و پردازش آن
async function readEventStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // رویدادهای SSE با یک خط خالی از هم جدا می‌شوند
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop();
    chunks.forEach((chunk) => {
      const line = chunk.split("\n").find((part) => part.startsWith("data:"));
      if (!line) return;
      try {
        handleEvent(JSON.parse(line.slice(5).trim()));
      } catch (err) {
        console.error("رویداد نامعتبر از سرور:", line, err);
      }
    });
  }
}

// شروع پژوهش
async function startResearch() {
  const topic = el.topic.value.trim();
  clearError();
  el.resultCard.hidden = true;

  if (!topic) {
    showError("موضوع پژوهش را وارد کن.");
    el.topic.focus();
    return;
  }

  renderSteps();
  el.startBtn.disabled = true;
  el.startBtn.textContent = "در حال پژوهش...";

  try {
    const response = await fetch("/api/research/stream?topic=" + encodeURIComponent(topic));
    if (!response.ok) {
      throw new Error("پاسخ نامعتبر از سرور (کد " + response.status + ")");
    }
    if (!response.body) {
      throw new Error("مرورگر از خواندن جریان پاسخ پشتیبانی نمی‌کند.");
    }
    await readEventStream(response);
  } catch (err) {
    showError("خطا در ارتباط با سرور: " + err.message);
  } finally {
    el.startBtn.disabled = false;
    el.startBtn.textContent = "شروع پژوهش";
  }
}

el.startBtn.addEventListener("click", startResearch);
el.topic.addEventListener("keydown", (event) => {
  if (event.key === "Enter") startResearch();
});
el.pathBtn.addEventListener("click", () => {
  el.filePath.hidden = !el.filePath.hidden;
  if (!el.filePath.hidden && !currentFilePath) {
    el.filePath.textContent = "فایلی ذخیره نشده است.";
  }
});

renderSteps();
