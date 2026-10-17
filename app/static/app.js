"use strict";

// Браузер не получает API-ключ: интерфейс работает через /ui/analyze и /ui/jobs/{id},
// которые используют ту же очередь и те же задания, что и /api/v1.

const MAX_BYTES = 200 * 1024 * 1024;
const RECOMMENDED_SECONDS = 145;
const CLIENT_LIMIT_MS = 5 * 60 * 1000;
const UPLOAD_TIMEOUT_MS = 270 * 1000;
const REQUEST_TIMEOUT_MS = 15 * 1000;
const MAX_NETWORK_FAILURES = 3;

const $ = (id) => document.getElementById(id);
const player = $("video-player");
const input = $("file-input");
const dropzone = $("dropzone");
const states = ["ready-state", "processing-state", "error-state", "result-state"];

let selectedFile = null;
let objectUrl = null;
let busy = false;
let generation = 0;
let pollTimer = null;
let clockTimer = null;
let activeXhr = null;
let startedAt = 0;
let violations = [];
let retryAction = null;
let dialogFrames = [];
let dialogIndex = 0;
let dialogTimecode = 0;

// ---------- вспомогательные функции ----------

function timecode(seconds) {
  const whole = Math.floor(Math.max(0, Number(seconds) || 0));
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = whole % 60;
  const mmss = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return h ? `${h}:${mmss}` : mmss;
}

function formatSize(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1).replace(".", ",")} МБ`;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function announce(message) {
  const live = $("live-status");
  live.textContent = "";
  // Повторная запись после очистки заставляет экранные дикторы прочитать текст снова.
  window.setTimeout(() => { live.textContent = message; }, 50);
}

function showState(id) {
  for (const state of states) $(state).hidden = state !== id;
  document.querySelector(".status-card").setAttribute("aria-busy", String(id === "processing-state"));
}

function setBusy(value) {
  busy = value;
  $("analyze-button").disabled = value;
  $("change-file").disabled = value;
  input.disabled = value;
}

function stopTimers() {
  window.clearTimeout(pollTimer);
  window.clearInterval(clockTimer);
  pollTimer = null;
  clockTimer = null;
  if (activeXhr) {
    activeXhr.abort();
    activeXhr = null;
  }
}

function elapsedMs() {
  return startedAt ? Date.now() - startedAt : 0;
}

// ---------- состояние сервера ----------

function setServerStatus(state) {
  const node = $("server-status");
  node.dataset.state = state;
  $("server-status-text").textContent = state === "ok" ? "Сервер доступен" : state === "down" ? "Сервер недоступен" : "Сервер: проверка…";
}

async function requestJson(path) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(path, { signal: controller.signal, credentials: "same-origin", cache: "no-store" });
    let data = null;
    try { data = await response.json(); } catch { data = null; }
    if (!response.ok) {
      const failure = new Error((data && data.error) || `Ошибка сервера (${response.status}).`);
      failure.kind = response.status === 404 ? "not-found" : "http";
      failure.status = response.status;
      throw failure;
    }
    if (!data) {
      const failure = new Error("Сервер вернул некорректный ответ.");
      failure.kind = "http";
      throw failure;
    }
    return data;
  } catch (failure) {
    if (failure.kind) throw failure;
    const network = new Error("Нет связи с сервером.");
    network.kind = "network";
    throw network;
  } finally {
    window.clearTimeout(timer);
  }
}

async function checkServer() {
  try {
    await requestJson("/health");
    setServerStatus("ok");
    return true;
  } catch {
    setServerStatus("down");
    return false;
  }
}

// ---------- ошибки ----------

const MESSAGES = {
  network: ["Сервер недоступен", "Не удалось связаться с сервером. Проверьте, что сервис запущен и сеть доступна, затем нажмите «Повторить»."],
  timeout: ["Превышено время ожидания", "Проверка не завершилась за 5 минут. Нажмите «Повторить», чтобы отправить видео ещё раз. Если ошибка повторяется, обратитесь к администратору сервиса."],
  "not-found": ["Задание не найдено", "Сервер не нашёл задание: возможно, сервис был перезапущен или истёк срок хранения. Отправьте видео ещё раз."],
};

function showError(title, message, retry) {
  stopTimers();
  setBusy(false);
  retryAction = retry;
  $("error-title").textContent = title;
  $("error-message").textContent = message;
  $("retry-button").hidden = !retry;
  showState("error-state");
  $("error-title").focus({ preventScroll: true });
}

function showFailure(failure) {
  const preset = MESSAGES[failure.kind];
  if (failure.kind === "network") setServerStatus("down");
  if (preset) showError(preset[0], preset[1], analyze);
  else showError("Сервер не смог обработать запрос", failure.message || "Неизвестная ошибка.", analyze);
}

function fileError(title, message) {
  if ($("workspace").hidden) {
    $("upload-error-title").textContent = title;
    $("upload-error-message").textContent = message;
    $("upload-error").hidden = false;
  } else {
    showError(title, message, openPicker);
  }
}

// ---------- выбор файла ----------

function openPicker() {
  if (!busy) input.click();
}

function validate(file) {
  if (!/\.mov$/i.test(file.name)) {
    return ["Неподдерживаемый формат", `Файл «${file.name}» не подходит. Выберите видеозапись в формате .mov.`];
  }
  if (!file.size) return ["Файл пустой", `Файл «${file.name}» не содержит данных. Выберите другую запись.`];
  if (file.size > MAX_BYTES) {
    return ["Файл слишком большой", `Размер файла ${formatSize(file.size)}, допустимо не более 200 МБ. Выберите запись меньшего размера.`];
  }
  return null;
}

function chooseFile(file) {
  if (busy || !file) return;
  const problem = validate(file);
  if (problem) {
    fileError(problem[0], problem[1]);
    return;
  }
  generation += 1;
  stopTimers();
  selectedFile = file;
  violations = [];
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  player.src = objectUrl;
  $("upload-error").hidden = true;
  $("playback-note").hidden = true;
  $("duration-warning").hidden = true;
  $("timeline").hidden = true;
  $("file-name").textContent = file.name;
  $("file-meta").textContent = `· ${formatSize(file.size)}`;
  $("upload-screen").hidden = true;
  $("workspace").hidden = false;
  showState("ready-state");
  $("analyze-button").focus({ preventScroll: true });
}

function resetToUpload() {
  generation += 1;
  stopTimers();
  setBusy(false);
  selectedFile = null;
  violations = [];
  player.removeAttribute("src");
  player.load();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  $("violations-list").replaceChildren();
  $("upload-error").hidden = true;
  $("workspace").hidden = true;
  $("upload-screen").hidden = false;
  window.scrollTo(0, 0);
  dropzone.focus();
}

// ---------- обработка ----------

function setStep(id, state, detail) {
  const step = $(id);
  step.dataset.state = state;
  if (state === "active") step.setAttribute("aria-current", "step");
  else step.removeAttribute("aria-current");
  if (detail !== undefined) $(`${id}-detail`).textContent = detail;
}

function startClock() {
  startedAt = Date.now();
  $("elapsed").textContent = "00:00";
  clockTimer = window.setInterval(() => {
    $("elapsed").textContent = timecode(elapsedMs() / 1000);
  }, 1000);
}

function upload(file, current) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    activeXhr = xhr;
    const form = new FormData();
    form.append("file", file);
    xhr.open("POST", "/ui/analyze");
    xhr.timeout = UPLOAD_TIMEOUT_MS;
    xhr.responseType = "text";
    xhr.upload.addEventListener("progress", (event) => {
      if (current !== generation || !event.lengthComputable) return;
      const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
      $("upload-progress").value = percent;
      setStep("step-upload", "active", `Загрузка: ${percent}%`);
    });
    const fail = (kind, message, status) => {
      const failure = new Error(message);
      failure.kind = kind;
      failure.status = status;
      reject(failure);
    };
    xhr.addEventListener("load", () => {
      activeXhr = null;
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { data = null; }
      if (xhr.status >= 200 && xhr.status < 300 && data && data.job_id) resolve(data.job_id);
      else if (xhr.status === 413) fail("too-large", (data && data.error) || "Файл превышает лимит сервера.", 413);
      else if (xhr.status >= 400 && xhr.status < 500) fail("rejected", (data && data.error) || `Сервер отклонил файл (${xhr.status}).`, xhr.status);
      else fail("http", (data && data.error) || `Ошибка сервера (${xhr.status}).`, xhr.status);
    });
    xhr.addEventListener("error", () => { activeXhr = null; fail("network", "Нет связи с сервером."); });
    xhr.addEventListener("timeout", () => { activeXhr = null; fail("network", "Сервер не ответил вовремя."); });
    xhr.send(form);
  });
}

async function analyze() {
  if (!selectedFile || busy) return;
  const current = ++generation;
  stopTimers();
  setBusy(true);
  violations = [];
  $("timeline").hidden = true;
  $("upload-progress").hidden = false;
  $("upload-progress").value = 0;
  $("indeterminate").hidden = true;
  setStep("step-upload", "active", "Загрузка: 0%");
  setStep("step-queue", "pending", "");
  setStep("step-analysis", "pending", "");
  showState("processing-state");
  startClock();
  announce("Идёт загрузка видео на сервер.");

  let jobId;
  try {
    jobId = await upload(selectedFile, current);
  } catch (failure) {
    if (current !== generation) return;
    if (failure.kind === "too-large") showError("Файл слишком большой", failure.message, openPicker);
    else if (failure.kind === "rejected") showError("Сервер отклонил файл", failure.message, openPicker);
    else showFailure(failure);
    return;
  }
  if (current !== generation) return;
  setServerStatus("ok");
  setStep("step-upload", "done", "Готово");
  setStep("step-queue", "active", "");
  $("upload-progress").hidden = true;
  $("indeterminate").hidden = false;
  announce("Видео загружено. Задание в очереди.");
  poll(jobId, current, 0, false);
}

async function updateQueueStage(current) {
  // /ui/jobs возвращает только «processing», поэтому этап уточняется по длине
  // очереди из /health: при пустой очереди наше задание уже обрабатывается.
  try {
    const health = await requestJson("/health");
    if (current !== generation) return false;
    const waiting = Number(health.queue_length) || 0;
    if (waiting === 0) return true;
    setStep("step-queue", "active", `Заданий в очереди: ${waiting}`);
  } catch {
    // Состояние очереди — вспомогательная информация; ошибку покажет основной опрос.
  }
  return false;
}

function enterAnalysis() {
  setStep("step-queue", "done", "");
  setStep("step-analysis", "active", "Обработка видео");
  announce("Идёт анализ видео.");
}

async function poll(jobId, current, failures, analyzing) {
  if (current !== generation) return;
  if (elapsedMs() > CLIENT_LIMIT_MS) {
    showFailure({ kind: "timeout" });
    return;
  }
  let job;
  try {
    job = await requestJson(`/ui/jobs/${encodeURIComponent(jobId)}`);
  } catch (failure) {
    if (current !== generation) return;
    if (failure.kind === "network" && failures + 1 < MAX_NETWORK_FAILURES) {
      pollTimer = window.setTimeout(() => poll(jobId, current, failures + 1, analyzing), 3000);
      return;
    }
    showFailure(failure);
    return;
  }
  if (current !== generation) return;
  setServerStatus("ok");

  if (job.status === "done") {
    stopTimers();
    setBusy(false);
    renderResult(job);
    return;
  }
  if (job.status === "failed") {
    showError("Анализ завершился с ошибкой", job.error || "Сервер не смог обработать видео.", analyze);
    return;
  }
  if (!analyzing && await updateQueueStage(current)) {
    analyzing = true;
    enterAnalysis();
  }
  if (current !== generation) return;
  const delay = elapsedMs() < 60000 ? 2000 : 3000;
  pollTimer = window.setTimeout(() => poll(jobId, current, 0, analyzing), delay);
}

// ---------- результат ----------

const RULE_FIELDS = [
  ["Правило", ["rule_title", "rule_name", "rule"]],
  ["Источник", ["source_document", "rule_source", "source", "document"]],
  ["Пункт", ["source_clause", "rule_clause", "clause", "paragraph"]],
];

function text(value) {
  return typeof value === "string" ? value.trim() : (typeof value === "number" ? String(value) : "");
}

function ruleBlock(violation) {
  const list = element("dl", "rule-block");
  const add = (label, value) => {
    list.append(element("dt", "", label));
    const dd = element("dd", value ? "" : "is-empty", value || "источник не указан");
    list.append(dd);
  };
  const extra = RULE_FIELDS
    .map(([label, keys]) => [label, keys.map((key) => text(violation[key])).find(Boolean) || ""])
    .filter(([, value]) => value);
  if (extra.length) {
    for (const [label, value] of extra) add(label, value);
    const ref = text(violation.rule_ref);
    if (ref && !extra.some(([, value]) => value === ref)) add("Ссылка", ref);
    if (!extra.some(([label]) => label === "Источник")) add("Источник", "");
  } else {
    add("Правило и источник", text(violation.rule_ref));
  }
  return list;
}

function frameSrc(frame) {
  return `data:image/jpeg;base64,${frame.image_base64}`;
}

function renderCard(violation, index) {
  const time = timecode(violation.timecode_sec);
  const item = element("li", "violation-card");
  item.dataset.index = String(index);

  const head = element("div", "violation-head");
  head.append(element("span", "violation-index", String(index + 1)));
  head.append(element("h4", "violation-type", text(violation.type) || "Тип не указан"));
  const jump = element("button", "timecode", time);
  jump.type = "button";
  jump.setAttribute("aria-label", `Перейти к моменту ${time} в видео`);
  jump.addEventListener("click", (event) => { event.stopPropagation(); seekTo(violation.timecode_sec, index); });
  head.append(jump);
  item.append(head);

  item.append(element("p", "violation-description", text(violation.description) || "Описание не предоставлено."));
  item.append(ruleBlock(violation));

  const score = Number(violation.score);
  if (Number.isFinite(score)) {
    item.append(element("p", "score-line", `Оценка модели: ${Math.round(Math.min(100, Math.max(0, score)))} из 100 (не вероятность и не юридическая оценка)`));
  }

  const frames = Array.isArray(violation.evidence_frames) ? violation.evidence_frames : [];
  if (frames.length) {
    const evidence = element("div", "evidence");
    evidence.append(element("p", "evidence-title", `Кадры-доказательства (${frames.length})`));
    const grid = element("ul", "evidence-grid");
    frames.forEach((frame, frameIndex) => {
      const li = element("li");
      const figure = element("figure");
      const button = element("button", "frame-button");
      button.type = "button";
      const at = timecode(frame.timestamp_sec);
      button.setAttribute("aria-label", `Увеличить кадр ${frameIndex + 1} из ${frames.length}, время ${at}`);
      const image = element("img");
      image.src = frameSrc(frame);
      image.alt = `Кадр ${at}`;
      button.append(image);
      button.addEventListener("click", (event) => { event.stopPropagation(); openFrame(violation, frames, frameIndex); });
      figure.append(button, element("figcaption", "", at));
      li.append(figure);
      grid.append(li);
    });
    evidence.append(grid);
    item.append(evidence);
  }

  item.addEventListener("click", () => seekTo(violation.timecode_sec, index));
  return item;
}

function renderResult(job) {
  violations = Array.isArray(job.violations) ? job.violations : [];
  const found = job.verdict === "violation";
  const verdict = $("verdict");
  verdict.classList.toggle("is-found", found);
  verdict.classList.toggle("is-clean", !found);
  $("verdict-text").textContent = found ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено";
  $("verdict-count").hidden = !found;
  $("verdict-count").textContent = String(violations.length);
  $("verdict-count").setAttribute("aria-label", `Отметок: ${violations.length}`);
  const duration = timecode(elapsedMs() / 1000);
  $("verdict-meta").textContent = found
    ? `Отметок для проверки: ${violations.length} · время проверки ${duration}`
    : `Время проверки ${duration}`;

  const list = $("violations-list");
  list.replaceChildren(...violations.map(renderCard));
  $("violations-section").hidden = !found;
  $("clean-text").hidden = found;
  renderTimeline();
  showState("result-state");
  announce(found
    ? `Проверка завершена. Обнаружены признаки возможных нарушений: ${violations.length}.`
    : "Проверка завершена. Нарушение не обнаружено.");
  $("verdict-text").focus({ preventScroll: true });
  const panel = document.querySelector(".status-card");
  if (panel.getBoundingClientRect().top < 0) panel.scrollIntoView({ block: "start" });
}

// ---------- плеер и шкала ----------

function knownDuration() {
  return Number.isFinite(player.duration) && player.duration > 0 ? player.duration : 0;
}

function renderTimeline() {
  const track = $("timeline-track");
  for (const marker of track.querySelectorAll(".timeline-marker")) marker.remove();
  const duration = knownDuration();
  if (!violations.length || !duration) {
    $("timeline").hidden = true;
    return;
  }
  $("timeline-end").textContent = timecode(duration);
  violations.forEach((violation, index) => {
    const at = Math.min(Math.max(Number(violation.timecode_sec) || 0, 0), duration);
    const marker = element("button", "timeline-marker");
    marker.type = "button";
    marker.dataset.index = String(index);
    marker.style.left = `${(at / duration) * 100}%`;
    marker.setAttribute("aria-label", `Отметка ${index + 1}: ${text(violation.type)}, ${timecode(at)}`);
    marker.title = `${index + 1}. ${text(violation.type)} — ${timecode(at)}`;
    marker.append(element("span", "", String(index + 1)));
    marker.addEventListener("click", () => seekTo(violation.timecode_sec, index, true));
    track.append(marker);
  });
  updatePlayhead();
  $("timeline").hidden = false;
}

function updatePlayhead() {
  const duration = knownDuration();
  if (!duration) return;
  $("timeline-playhead").style.left = `${Math.min(100, (player.currentTime / duration) * 100)}%`;
}

function highlight(index, scrollCard) {
  for (const card of document.querySelectorAll(".violation-card")) {
    card.classList.toggle("is-active", card.dataset.index === String(index));
  }
  for (const marker of document.querySelectorAll(".timeline-marker")) {
    marker.classList.toggle("is-active", marker.dataset.index === String(index));
  }
  if (scrollCard) {
    const card = document.querySelector(`.violation-card[data-index="${index}"]`);
    if (card) card.scrollIntoView({ block: "nearest" });
  }
}

function seekTo(seconds, index, scrollCard = false) {
  const target = Number(seconds);
  if (!Number.isFinite(target)) return;
  const duration = knownDuration();
  player.currentTime = Math.max(0, duration ? Math.min(target, duration) : target);
  updatePlayhead();
  if (index !== undefined) highlight(index, scrollCard);
  const rect = player.getBoundingClientRect();
  if (!scrollCard && (rect.bottom < 0 || rect.top > window.innerHeight)) player.scrollIntoView({ block: "nearest" });
  announce(`Видео перемотано на ${timecode(target)}.`);
}

// ---------- увеличенный кадр ----------

function showDialogFrame() {
  const frame = dialogFrames[dialogIndex];
  const at = timecode(frame.timestamp_sec);
  $("enlarged-frame").src = frameSrc(frame);
  $("enlarged-frame").alt = `Кадр ${at}`;
  $("frame-caption").textContent = `Кадр ${dialogIndex + 1} из ${dialogFrames.length} · ${at} · ${dialogCaption}`;
  $("prev-frame").disabled = dialogIndex === 0;
  $("next-frame").disabled = dialogIndex === dialogFrames.length - 1;
  $("prev-frame").hidden = $("next-frame").hidden = dialogFrames.length < 2;
}

let dialogCaption = "";
let dialogViolationIndex = 0;

function openFrame(violation, frames, index) {
  dialogFrames = frames;
  dialogIndex = index;
  dialogCaption = text(violation.type);
  dialogTimecode = violation.timecode_sec;
  dialogViolationIndex = violations.indexOf(violation);
  showDialogFrame();
  $("frame-dialog").showModal();
}

function stepFrame(delta) {
  const next = dialogIndex + delta;
  if (next < 0 || next >= dialogFrames.length) return;
  dialogIndex = next;
  showDialogFrame();
}

// ---------- события ----------

dropzone.addEventListener("click", openPicker);
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); openPicker(); }
});
input.addEventListener("change", () => { chooseFile(input.files[0]); input.value = ""; });
for (const name of ["dragenter", "dragover"]) {
  dropzone.addEventListener(name, (event) => { event.preventDefault(); if (!busy) dropzone.classList.add("dragging"); });
}
for (const name of ["dragleave", "drop"]) {
  dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); });
}
dropzone.addEventListener("drop", (event) => {
  const files = event.dataTransfer.files;
  if (files.length !== 1) {
    fileError("Нужен один файл", "Перетащите одну видеозапись в формате .mov.");
    return;
  }
  chooseFile(files[0]);
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

$("upload-retry").addEventListener("click", openPicker);
$("change-file").addEventListener("click", openPicker);
$("analyze-button").addEventListener("click", analyze);
$("retry-button").addEventListener("click", () => { if (retryAction) retryAction(); });
$("error-new-button").addEventListener("click", resetToUpload);
$("new-check-button").addEventListener("click", resetToUpload);

player.addEventListener("error", () => { if (selectedFile) $("playback-note").hidden = false; });
player.addEventListener("loadedmetadata", () => {
  if (!selectedFile) return;
  const duration = knownDuration();
  $("file-meta").textContent = `· ${formatSize(selectedFile.size)} · ${timecode(duration)}`;
  const warning = $("duration-warning");
  warning.hidden = duration <= RECOMMENDED_SECONDS;
  warning.textContent = `Длительность записи ${timecode(duration)} больше рекомендованных 145 секунд. Анализ может не уложиться в лимит 5 минут.`;
  if (violations.length) renderTimeline();
});
player.addEventListener("timeupdate", updatePlayhead);
player.addEventListener("seeked", updatePlayhead);

$("close-dialog").addEventListener("click", () => $("frame-dialog").close());
$("prev-frame").addEventListener("click", () => stepFrame(-1));
$("next-frame").addEventListener("click", () => stepFrame(1));
$("seek-frame").addEventListener("click", () => {
  const frame = dialogFrames[dialogIndex];
  $("frame-dialog").close();
  seekTo(frame ? frame.timestamp_sec : dialogTimecode, dialogViolationIndex >= 0 ? dialogViolationIndex : undefined);
});
$("frame-dialog").addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") { event.preventDefault(); stepFrame(-1); }
  if (event.key === "ArrowRight") { event.preventDefault(); stepFrame(1); }
});
$("frame-dialog").addEventListener("click", (event) => { if (event.target === $("frame-dialog")) $("frame-dialog").close(); });
$("frame-dialog").addEventListener("close", () => $("enlarged-frame").removeAttribute("src"));

window.addEventListener("pagehide", () => { stopTimers(); if (objectUrl) URL.revokeObjectURL(objectUrl); });
window.addEventListener("beforeunload", (event) => { if (busy) event.preventDefault(); });

checkServer();
requestJson("/ui/config").then((config) => { $("mock-notice").hidden = !config.mock_model; }).catch(() => {});
