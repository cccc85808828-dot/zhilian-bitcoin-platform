document.addEventListener("DOMContentLoaded", async () => {
  "use strict";
  const stored = TQ.getContext();
  const transactionId = TQ.getQuery("transaction") || stored.transactionId || "";
  const address = TQ.getQuery("address") || stored.address || "";
  const transactionLabel = document.getElementById("analysis-transaction");
  const addressLabel = document.getElementById("analysis-address");
  const content = document.getElementById("analysis-content");
  const back = document.getElementById("back-relations");

  back.addEventListener("click", () => {
    window.location.href = address ? `/relations.html?address=${encodeURIComponent(address)}` : "/index.html";
  });
  addressLabel.textContent = address || "独立交易研判";

  if (!transactionId) {
    transactionLabel.textContent = "尚未选择交易";
    content.innerHTML = `<section class="tech-card empty-analysis"><b>请先从关联交易中选择一笔交易</b><a class="primary-button" href="${address ? `/relations.html?address=${encodeURIComponent(address)}` : "/index.html"}">前往选择 →</a></section>`;
    document.getElementById("pipeline-status").innerHTML = "等待选择";
    return;
  }
  transactionLabel.textContent = TQ.shortHash(transactionId, 16, 14);
  transactionLabel.title = transactionId;
  TQ.setContext({ address, transactionId });

  function addressItems(group, role) {
    const items = group.items || [];
    const visible = items.slice(0, 8);
    const list = visible.map((item) => `<li><span>${role}</span><code title="${TQ.escapeHtml(item.address)}">${TQ.escapeHtml(TQ.shortHash(item.address, 13, 10))}</code><small>${TQ.escapeHtml(TQ.btc(item.amount_btc))}</small></li>`).join("");
    const remaining = group.count - visible.length;
    return `${list}${remaining > 0 ? `<li class="more-addresses">另有 ${remaining} 个关联地址已纳入结构分析</li>` : ""}`;
  }

  function attributeRows(attributes) {
    if (!attributes || !attributes.length) return "<p>暂未读取到可展示的链上属性</p>";

    function displayValue(item) {
      const value = Number(item.value ?? item.raw_value ?? 0);
      if (!Number.isFinite(value)) return "—";
      if (item.unit === "btc") return TQ.btc(value);
      if (item.unit === "bytes") return `${Math.round(value).toLocaleString("zh-CN")} 字节`;
      if (item.unit === "count") return `${Math.round(value).toLocaleString("zh-CN")} 项`;
      return value.toLocaleString("zh-CN", { maximumFractionDigits: 8 });
    }

    return attributes.slice(0, 8).map((item, index) => {
      const strength = Number(item.signal_strength);
      const normalized = Number.isFinite(strength)
        ? Math.min(100, Math.max(4, strength))
        : 4;
      return `<div><p><span>${TQ.escapeHtml(item.name || `链上属性 ${String(index + 1).padStart(2, "0")}`)}</span><b>${TQ.escapeHtml(displayValue(item))}</b></p><i title="相对参考范围偏离度"><em style="width:${normalized}%"></em></i></div>`;
    }).join("");
  }

  function render(analysis) {
    const item = analysis.result;
    const high = item.predicted_label === 1;
    const verdict = high ? "重点关注交易" : "常规关注交易";
    const explanation = item.decision_explanation || {};
    const behaviorExplanation = explanation.behavior || {
      primary_label: high ? "结构—属性复合异常" : "未见典型异常行为组合",
      patterns: [],
      summary: high ? "模型识别到结构与属性的复合异常组合。" : "当前未识别到典型异常行为组合。",
    };
    const structureExplanation = explanation.structure || {
      summary: `结构分支得到 ${TQ.score(item.graph_score)} 分，并结合输入输出地址角色、交易连接形态和地址历史记忆形成判断。`,
    };
    const attributeExplanation = explanation.attribute || {
      summary: `属性分支得到 ${TQ.score(item.qif_score)} 分，并联合评估金额、手续费、交易大小等属性及其二阶交互。`,
      top_features: [],
    };
    const fusionExplanation = explanation.fusion || {
      agreement: "结构与属性证据已完成统一融合",
      summary: `融合结果为 ${TQ.score(item.risk_score)} 分，对应“${verdict}”。`,
    };
    const conclusion = explanation.conclusion || (high
      ? "融合结果已越过重点关注阈值，建议结合相邻交易、关联地址及资金后续流向继续研判。"
      : "融合结果当前未越过重点关注阈值，可按常规关注流程保留记录并持续观察关联变化。");
    const mainEvidence = `${behaviorExplanation.summary} ${fusionExplanation.agreement}。`;
    const behaviorPatterns = (behaviorExplanation.patterns || []).map((pattern) => `
      <span class="behavior-pattern">
        <span class="behavior-strength">${TQ.escapeHtml(pattern.strength || "证据")}</span>
        <span class="behavior-pattern-copy"><b>${TQ.escapeHtml(pattern.label || "行为模式")}</b><em>${TQ.escapeHtml(pattern.evidence || "")}</em></span>
      </span>`).join("");
    const attributeChips = (attributeExplanation.top_features || []).map((feature) =>
      `<span><b>${TQ.escapeHtml(feature.name)}</b>${TQ.escapeHtml(feature.value)}</span>`
    ).join("");
    const confirmation = item.confirmation || {};
    const isLive = item.data_source === "bitcoin_mainnet";
    const chainStatus = confirmation.confirmed
      ? `区块 #${confirmation.block_height}`
      : "等待链上确认";
    const chainFacts = isLive
      ? `<span>转账金额：${TQ.escapeHtml(TQ.btc(item.amount_btc))}</span><span>手续费：${TQ.escapeHtml(TQ.btc(item.fee_btc))}</span><span>${TQ.escapeHtml(chainStatus)}</span>`
      : "";
    const firstExplanation = isLive
      ? `交易发生于 ${TQ.escapeHtml(TQ.eventTime(item))}，转账 ${TQ.escapeHtml(TQ.btc(item.amount_btc))}，手续费 ${TQ.escapeHtml(TQ.btc(item.fee_btc))}，状态为“${TQ.escapeHtml(chainStatus)}”。`
      : `系统已读取当前交易记录，并完成输入输出关系与交易属性解析。`;

    content.className = "analysis-content";
    content.innerHTML = `
      <section class="decision-banner ${high ? "high" : "low"}">
        <div class="decision-orb"><span>${TQ.score(item.risk_score, 0)}</span><small>风险指数 / 100</small></div>
        <div class="decision-copy"><small>智链智能研判结论${isLive ? " · 比特币主网" : ""}</small><h2>${verdict}</h2><h3>${TQ.escapeHtml(behaviorExplanation.primary_label || "综合行为研判")}</h3><p>${TQ.escapeHtml(conclusion)}</p><div><span>交易时间：${TQ.escapeHtml(TQ.eventTime(item))}</span>${chainFacts}<span>输入地址：${item.input_addresses.count} 个</span><span>输出地址：${item.output_addresses.count} 个</span></div></div>
        <div class="decision-actions"><button class="primary-button" id="ask-report">生成研判报告</button><button class="outline-button" id="print-result">打印结论</button></div>
      </section>

      <section class="evidence-grid">
        <article class="tech-card score-card">
          <div class="card-head"><div><span>02</span><h2>风险证据融合</h2></div></div>
          <div class="score-comparison">
            <div><span>关联网络行为研判</span><b>${TQ.percent(item.graph_score)}</b><i><em style="width:${Math.round(item.graph_score * 100)}%"></em></i><small>时序超图 · 地址角色 · 历史关联</small></div>
            <div><span>单笔交易属性研判</span><b>${TQ.percent(item.qif_score)}</b><i><em style="width:${Math.round(item.qif_score * 100)}%"></em></i><small>分位编码 · 金额分布 · 属性交互</small></div>
            <div class="fusion-score"><span>综合风险指数</span><b>${TQ.score(item.risk_score)}分</b><i><em style="width:${Math.round(item.risk_score * 100)}%"></em></i><small>结构与属性证据统一标尺映射结果</small></div>
          </div>
          <div class="evidence-summary"><i>✦</i><p><b>核心证据</b>${mainEvidence}</p></div>
        </article>

        <article class="tech-card explanation-card">
          <div class="card-head"><div><span>03</span><h2>本笔交易异常原因与判定依据</h2></div><small>行为事实优先 · 模型证据补充</small></div>
          <ul class="explanation-list">
            <li><i>1</i><p><b>${isLive ? "链上交易事实" : "交易记录读取"}</b>${firstExplanation}</p></li>
            <li class="behavior-reason"><i>2</i><p><b>行为原因 · ${TQ.escapeHtml(behaviorExplanation.primary_label || "综合行为研判")}</b>${TQ.escapeHtml(behaviorExplanation.summary || "")}${behaviorPatterns ? `<span class="behavior-patterns">${behaviorPatterns}</span>` : ""}</p></li>
            <li><i>3</i><p><b>关联网络行为研判如何支持上述判断</b>${TQ.escapeHtml(structureExplanation.summary)}</p></li>
            <li><i>4</i><p><b>单笔交易属性研判如何支持上述判断</b>${TQ.escapeHtml(attributeExplanation.summary)}${attributeChips ? `<span class="explanation-features">${attributeChips}</span>` : ""}</p></li>
            <li><i>5</i><p><b>综合研判 · 风险指数作为辅助</b>${TQ.escapeHtml(fusionExplanation.summary)}<strong>${TQ.escapeHtml(conclusion)}</strong></p></li>
          </ul>
        </article>
      </section>

      <section class="evidence-grid address-evidence">
        <article class="tech-card address-list-card">
          <div class="card-head"><div><span>04</span><h2>关联地址证据</h2></div><small>${item.input_addresses.count + item.output_addresses.count} 个地址参与本次交易</small></div>
          <div class="address-columns"><div><h3>输入地址 <span>${item.input_addresses.count}</span></h3><ul>${addressItems(item.input_addresses, "IN")}</ul></div><div><h3>输出地址 <span>${item.output_addresses.count}</span></h3><ul>${addressItems(item.output_addresses, "OUT")}</ul></div></div>
        </article>
        <article class="tech-card attribute-card">
          <div class="card-head"><div><span>05</span><h2>链上交易属性</h2></div><small>原始值 · 横条表示相对参考范围偏离度</small></div>
          <div class="attribute-bars">${attributeRows(item.attribute_profile)}</div>
        </article>
      </section>`;

    document.getElementById("analysis-id").textContent = `研判编号 ${analysis.analysis_id}`;
    document.getElementById("pipeline-status").innerHTML = "<i></i>全部环节已完成";
    document.querySelectorAll("#analysis-pipeline > div").forEach((step) => step.classList.add("done"));
    document.querySelectorAll("#analysis-pipeline > em").forEach((line) => line.classList.add("done"));
    document.getElementById("ask-report").addEventListener("click", () => window.dispatchEvent(new CustomEvent("tq:assistant", { detail: "请先用清晰的业务语言说明当前交易具体呈现了哪些可核验行为；再说明这些行为如何得到关联网络行为研判和单笔交易属性研判支持。风险指数只作为辅助，不要把越过阈值本身当作异常原因。" })));
    document.getElementById("print-result").addEventListener("click", () => window.print());
  }

  try {
    const analysis = await TQ.requestJson("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transaction_id: transactionId }),
    });
    window.setTimeout(() => render(analysis), 450);
  } catch (error) {
    document.getElementById("pipeline-status").textContent = "请重新选择交易";
    content.className = "analysis-content";
    content.innerHTML = `<section class="tech-card empty-analysis"><b>${TQ.escapeHtml(error.message)}</b><button class="primary-button" id="retry-selection">返回关联交易 →</button></section>`;
    document.getElementById("retry-selection").addEventListener("click", () => back.click());
  }
});
