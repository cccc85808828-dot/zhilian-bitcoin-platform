(function () {
  "use strict";

  const STORAGE_KEY = "tthgnn_qif_context";

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function percent(value, digits = 1) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
  }

  function score(value, digits = 1) {
    const number = Number(value);
    return Number.isFinite(number) ? (number * 100).toFixed(digits) : "—";
  }

  function eventTime(item) {
    const raw = item && item.event_time;
    if (raw) {
      const date = new Date(raw);
      if (!Number.isNaN(date.getTime())) {
        return new Intl.DateTimeFormat("zh-CN", {
          year: "numeric", month: "2-digit", day: "2-digit",
          hour: "2-digit", minute: "2-digit", second: "2-digit",
          hour12: false,
        }).format(date);
      }
    }
    if (item && item.confirmed === false) return "等待链上确认";
    if (item && item.confirmation && item.confirmation.confirmed === false) return "等待链上确认";
    const order = item && item.sequence_order;
    return Number.isFinite(Number(order)) ? `历史顺序 #${order}` : "时间信息待接入";
  }

  function shortHash(value, front = 10, back = 8) {
    const text = String(value ?? "");
    return text.length > front + back + 3
      ? `${text.slice(0, front)}…${text.slice(-back)}`
      : text;
  }

  function btc(value, digits = 8) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return `${number.toLocaleString("zh-CN", { maximumFractionDigits: digits })} BTC`;
  }

  async function requestJson(url, options) {
    const response = await fetch(url, options);
    let data = {};
    try {
      data = await response.json();
    } catch (_) {
      data = { error: "服务返回内容无法解析" };
    }
    if (!response.ok) {
      const error = new Error(data.error || "请求未完成");
      error.hint = data.hint || "";
      error.fallback = data.fallback || "";
      throw error;
    }
    return data;
  }

  let serviceStatusPromise = null;

  function serviceStatus() {
    if (!serviceStatusPromise) {
      serviceStatusPromise = requestJson("/api/health");
    }
    return serviceStatusPromise;
  }

  function applyServiceMode(status) {
    if (!status || status.data_mode !== "offline_snapshot") return;
    document.body.dataset.dataMode = "offline_snapshot";
    const systemStatus = document.querySelector(".system-status");
    if (systemStatus) systemStatus.innerHTML = "<i></i>近期真实链上快照已载入";
  }

  function getContext() {
    try {
      return JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "{}") || {};
    } catch (_) {
      return {};
    }
  }

  function setContext(next) {
    const merged = { ...getContext(), ...next };
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
    return merged;
  }

  function getQuery(name) {
    return new URLSearchParams(window.location.search).get(name) || "";
  }

  function currentContext() {
    const stored = getContext();
    return {
      address: getQuery("address") || stored.address || "",
      transaction_id: getQuery("transaction") || stored.transactionId || "",
    };
  }

  function renderAnswer(text) {
    const inline = (value) => escapeHtml(value)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.+?)`/g, "<code>$1</code>");
    const lines = String(text || "").split(/\r?\n/);
    const parts = [];
    let listOpen = false;
    const closeList = () => {
      if (listOpen) { parts.push("</ul>"); listOpen = false; }
    };
    for (const raw of lines) {
      const line = raw.trim();
      if (!line) { closeList(); continue; }
      if (line.startsWith("### ")) { closeList(); parts.push(`<h3>${inline(line.slice(4))}</h3>`); continue; }
      if (line.startsWith("## ")) { closeList(); parts.push(`<h3>${inline(line.slice(3))}</h3>`); continue; }
      if (line.startsWith("# ")) { closeList(); parts.push(`<h3>${inline(line.slice(2))}</h3>`); continue; }
      if (/^[-•]\s+/.test(line)) {
        if (!listOpen) { parts.push("<ul>"); listOpen = true; }
        parts.push(`<li>${inline(line.replace(/^[-•]\s+/, ""))}</li>`);
        continue;
      }
      closeList();
      parts.push(`<p>${inline(line)}</p>`);
    }
    closeList();
    return parts.join("");
  }

  async function mountAssistant() {
    const root = document.getElementById("assistant-root");
    if (!root) return;
    root.innerHTML = `
      <button class="assistant-launcher" id="assistant-launcher" aria-label="打开智能助手">
        <span class="assistant-icon">AI</span>
        <span><b>智链助手</b><small>研判解释 · 技术问答</small></span>
        <i></i>
      </button>
      <aside class="assistant-panel" id="assistant-panel" aria-hidden="true">
        <div class="assistant-head">
          <div class="assistant-avatar">智</div>
          <div><b>智链助手</b><small id="assistant-status"><i></i>智能知识服务在线</small></div>
          <button id="assistant-close" aria-label="关闭">×</button>
        </div>
        <div class="assistant-context" id="assistant-context">我会结合当前页面的地址与交易证据回答。</div>
        <div class="assistant-messages" id="assistant-messages">
          <div class="assistant-message bot">你好，我会结合当前页面证据进行多轮问答。你可以追问本笔交易呈现的具体行为、对应链上事实，以及关联网络行为研判和单笔交易属性研判如何共同支持结论。</div>
        </div>
        <div class="assistant-prompts">
          <button data-prompt="请生成当前对象的可解释性研判报告">生成研判报告</button>
          <button data-prompt="请先说明当前交易具体呈现了哪些可核验行为，再解释关联网络、单笔属性和融合证据">解释异常原因</button>
          <button data-prompt="请说明TTHGNN-QIF的技术原理">说明技术原理</button>
        </div>
        <form class="assistant-form" id="assistant-form">
          <textarea id="assistant-input" rows="2" placeholder="输入想了解的问题…"></textarea>
          <button type="submit">发送</button>
        </form>
      </aside>`;

    const launcher = document.getElementById("assistant-launcher");
    const panel = document.getElementById("assistant-panel");
    const close = document.getElementById("assistant-close");
    const form = document.getElementById("assistant-form");
    const input = document.getElementById("assistant-input");
    const messages = document.getElementById("assistant-messages");
    const contextLabel = document.getElementById("assistant-context");
    const status = document.getElementById("assistant-status");
    const sendButton = form.querySelector("button[type=submit]");
    const conversation = [];
    let sending = false;

    function refreshContextLabel() {
      const context = currentContext();
      if (context.transaction_id) {
        contextLabel.textContent = `已载入交易 ${shortHash(context.transaction_id, 14, 12)} 的结构、属性与地址证据`;
        contextLabel.title = context.transaction_id;
      } else if (context.address) {
        contextLabel.textContent = `已载入地址 ${shortHash(context.address, 14, 10)} 的关联交易证据`;
        contextLabel.title = context.address;
      } else {
        contextLabel.textContent = "可随时咨询地址研判、风险证据与模型技术原理";
        contextLabel.removeAttribute("title");
      }
    }

    function openPanel() {
      panel.classList.add("open");
      panel.setAttribute("aria-hidden", "false");
      refreshContextLabel();
      setTimeout(() => input.focus(), 80);
    }

    function closePanel() {
      panel.classList.remove("open");
      panel.setAttribute("aria-hidden", "true");
    }

    launcher.addEventListener("click", openPanel);
    close.addEventListener("click", closePanel);

    try {
      const assistant = await requestJson("/api/assistant/status");
      status.innerHTML = assistant.configured
        ? `<i></i>${escapeHtml(assistant.model || "大模型")} 已连接`
        : "<i></i>本地研判知识库在线";
      status.title = assistant.configured
        ? "已连接服务端大模型，支持证据约束多轮交互"
        : "配置服务端API密钥后自动切换为大模型多轮交互";
    } catch (_) {
      status.innerHTML = "<i></i>智能助手已就绪";
    }

    async function ask(question) {
      const clean = String(question || "").trim();
      if (!clean || sending) return;
      sending = true;
      sendButton.disabled = true;
      openPanel();
      messages.insertAdjacentHTML("beforeend", `<div class="assistant-message user">${escapeHtml(clean)}</div>`);
      const pendingId = `pending-${Date.now()}`;
      messages.insertAdjacentHTML("beforeend", `<div class="assistant-message bot thinking" id="${pendingId}"><span></span><span></span><span></span></div>`);
      messages.scrollTop = messages.scrollHeight;
      input.value = "";
      input.style.height = "auto";
      const pending = document.getElementById(pendingId);
      const priorHistory = conversation.slice(-6).map((message) => ({
        role: message.role,
        content: String(message.content || "").slice(0, 1200),
      }));
      conversation.push({ role: "user", content: clean });
      try {
        const context = currentContext();
        const result = await requestJson("/api/assistant", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: clean, history: priorHistory, ...context }),
        });
        pending.className = "assistant-message bot";
        pending.innerHTML = renderAnswer(result.answer);
        conversation.push({ role: "assistant", content: result.answer });
      } catch (error) {
        pending.className = "assistant-message bot";
        const fallback = error.fallback || "智能助手正在整理证据，请稍后再次提问。";
        pending.innerHTML = renderAnswer(fallback);
        conversation.push({ role: "assistant", content: fallback });
      }
      sending = false;
      sendButton.disabled = false;
      messages.scrollTop = messages.scrollHeight;
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      ask(input.value);
    });
    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = `${Math.min(input.scrollHeight, 112)}px`;
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
    root.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => ask(button.dataset.prompt));
    });
    window.addEventListener("tq:assistant", (event) => ask(event.detail || "请解释当前研判结果"));
    if (getQuery("assistant") === "1") openPanel();
  }

  window.TQ = {
    escapeHtml,
    percent,
    score,
    eventTime,
    shortHash,
    btc,
    requestJson,
    serviceStatus,
    getContext,
    setContext,
    getQuery,
    currentContext,
  };

  document.addEventListener("DOMContentLoaded", () => {
    serviceStatus().then(applyServiceMode).catch(() => {});
    mountAssistant();
  });
})();
