document.addEventListener("DOMContentLoaded", async () => {
  "use strict";
  const form = document.getElementById("address-form");
  const input = document.getElementById("address-input");
  const examples = document.getElementById("address-examples");
  const message = document.getElementById("address-message");
  let offlineMode = false;

  function setMessage(text, type = "info") {
    message.className = `form-message ${type}`;
    message.textContent = text;
  }

  try {
    const status = await TQ.serviceStatus();
    offlineMode = status.data_mode === "offline_snapshot";
    if (offlineMode) {
      document.getElementById("hero-service-copy").textContent = "已固化近期比特币主网公开交易数据，自动载入地址历史、区块时间、资金金额与输入输出关系，支持关联网络查看和单笔交易深入研判。";
      document.getElementById("query-service-title").textContent = "近期真实链上地址查询";
      document.getElementById("query-service-note").textContent = "公开主网数据快照 · 无需外网即可完整研判";
      document.getElementById("query-process-stage").textContent = "快照载入";
    }
  } catch (_) {}

  try {
    const samples = await TQ.requestJson("/api/address-samples");
    examples.innerHTML = samples
      .map((sample) => `<button type="button" data-address="${TQ.escapeHtml(sample.address)}">${TQ.escapeHtml(sample.label)}</button>`)
      .join("");
    examples.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        input.value = button.dataset.address;
        input.focus();
      });
    });
  } catch (_) {
    examples.innerHTML = "<span>输入完整地址即可查询</span>";
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const address = input.value.trim();
    if (!address) {
      setMessage("请输入需要研判的交易地址", "error");
      input.focus();
      return;
    }
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    button.innerHTML = offlineMode
      ? "正在载入真实链上快照 <b>···</b>"
      : "正在读取比特币主网 <b>···</b>";
    setMessage(
      offlineMode
        ? "正在载入近期真实链上记录、构建关联网络并呈现智链研判结果…"
        : "正在读取链上历史、构建关联网络并执行智链智能研判…",
      "loading"
    );
    try {
      await TQ.requestJson(`/api/address?value=${encodeURIComponent(address)}`);
      TQ.setContext({ address, transactionId: "" });
      setMessage(
        offlineMode
          ? "近期真实主网快照与模型研判结果已载入，正在进入关联分析"
          : "比特币主网数据与模型研判均已完成，正在进入关联分析",
        "success"
      );
      window.setTimeout(() => {
        window.location.href = `/relations.html?address=${encodeURIComponent(address)}`;
      }, 280);
    } catch (error) {
      setMessage(`${error.message}${error.hint ? `。${error.hint}` : ""}`, "error");
      button.disabled = false;
      button.innerHTML = "开始查询 <b>→</b>";
    }
  });
});
