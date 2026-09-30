document.addEventListener("DOMContentLoaded", async () => {
  "use strict";
  const address = TQ.getQuery("address") || TQ.getContext().address || "";
  const addressLabel = document.getElementById("current-address");
  const body = document.getElementById("transaction-body");
  const search = document.getElementById("transaction-search");
  const filters = document.getElementById("risk-filter");
  let profile = null;
  let currentFilter = "all";

  if (!address) {
    addressLabel.textContent = "请先输入需要查询的地址";
    body.innerHTML = `<tr><td colspan="7" class="empty-cell"><a href="/index.html">返回地址查询页</a></td></tr>`;
    return;
  }
  addressLabel.textContent = address;
  TQ.setContext({ address, transactionId: "" });

  function roleText(roles) {
    const labels = [];
    if (roles.includes("input")) labels.push("转出方");
    if (roles.includes("output")) labels.push("接收方");
    return labels.join(" / ") || "关联地址";
  }

  function renderMetrics() {
    const values = [
      [profile.related_transaction_count, "笔"],
      [profile.high_risk_transactions, "笔"],
      [profile.input_role_transactions, "笔"],
      [profile.output_role_transactions, "笔"],
      [TQ.score(profile.maximum_risk_score), "分"],
    ];
    document.querySelectorAll("#address-metrics article").forEach((card, index) => {
      card.querySelector("b").textContent = values[index][0];
      card.querySelector("span").textContent = values[index][1];
    });
    const range = profile.first_event_time && profile.last_event_time
      ? `${TQ.eventTime({ event_time: profile.first_event_time })} — ${TQ.eventTime({ event_time: profile.last_event_time })}`
      : `历史顺序 #${profile.first_order} — #${profile.last_order}`;
    document.getElementById("sequence-range").textContent = range;
    const badge = document.getElementById("coverage-badge");
    const completeText = profile.history_complete
      ? `已完整读取并研判 ${profile.loaded_transaction_count} 笔链上交易`
      : `已读取并研判最近 ${profile.loaded_transaction_count} 笔，地址链上共 ${profile.total_transaction_count} 笔`;
    badge.querySelector("b").textContent = "比特币主网交易研判已完成";
    badge.querySelector("small").textContent = completeText;
  }

  function renderRoles() {
    const total = Math.max(profile.related_transaction_count, 1);
    const inputPct = Math.round((profile.input_role_transactions / total) * 100);
    const outputPct = Math.round((profile.output_role_transactions / total) * 100);
    const ring = document.querySelector(".role-ring");
    ring.style.setProperty("--risk", `${Math.round((profile.high_risk_transactions / total) * 100)}%`);
    ring.innerHTML = `<div><b>${profile.related_transaction_count}</b><small>关联交易</small></div>`;
    document.querySelector(".role-bars").innerHTML = `
      <div><p><span>作为转出方</span><b>${profile.input_role_transactions} 笔</b></p><i><em style="width:${inputPct}%"></em></i></div>
      <div><p><span>作为接收方</span><b>${profile.output_role_transactions} 笔</b></p><i><em style="width:${outputPct}%"></em></i></div>
      <div><p><span>平均风险指数</span><b>${TQ.score(profile.average_risk_score)}分</b></p><i><em style="width:${Math.round(profile.average_risk_score * 100)}%"></em></i></div>`;
  }

  async function renderMap() {
    let graph = profile.relation_graph || {};
    if (!Array.isArray(graph.nodes) || !graph.nodes.length) {
      const cases = await Promise.all(profile.transactions.map((item) =>
        TQ.requestJson(`/api/case/${encodeURIComponent(item.transaction_id)}`)
      ));
      const nodeMap = new Map();
      const edges = [];
      const ensureAddress = (value, focal = false) => {
        const id = `address:${value}`;
        if (!nodeMap.has(id)) nodeMap.set(id, {
          id, kind: focal ? "focal" : "address", address: value,
          label: value, roles: focal ? ["focus"] : [], degree: 0,
        });
        return nodeMap.get(id);
      };
      ensureAddress(address, true);
      for (const item of cases) {
        const txid = String(item.transaction_id);
        const txNode = {
          id: `transaction:${txid}`, kind: "transaction", transaction_id: txid,
          label: txid, risk_score: item.risk_score,
          predicted_label: item.predicted_label, event_time: item.event_time, degree: 0,
        };
        nodeMap.set(txNode.id, txNode);
        for (const [role, group] of [["input", item.input_addresses], ["output", item.output_addresses]]) {
          for (const party of (group && group.items) || []) {
            const addressNode = ensureAddress(String(party.address));
            if (!addressNode.roles.includes(role)) addressNode.roles.push(role);
            addressNode.degree += 1;
            txNode.degree += 1;
            edges.push({
              id: `edge:${edges.length}`,
              source: role === "input" ? addressNode.id : txNode.id,
              target: role === "input" ? txNode.id : addressNode.id,
              role,
              amount_btc: party.amount_btc,
              transaction_id: txid,
              risk_score: item.risk_score,
              predicted_label: item.predicted_label,
            });
          }
        }
      }
      graph = {
        focal_node_id: `address:${address}`,
        nodes: [...nodeMap.values()],
        edges,
      };
    }
    const rawNodes = Array.isArray(graph.nodes) ? graph.nodes : [];
    const rawEdges = Array.isArray(graph.edges) ? graph.edges : [];
    const canvas = document.getElementById("relation-canvas");
    const container = document.getElementById("relation-map");
    const tooltip = document.getElementById("graph-tooltip");
    const loading = document.getElementById("graph-loading");
    const count = document.getElementById("graph-count");
    const context = canvas.getContext("2d");
    const focalId = graph.focal_node_id || `address:${address}`;
    let frame = 0;
    let simulationFrames = 360;
    let hovered = null;
    let selected = null;
    let dragNode = null;
    let pointerDown = null;
    let panning = false;
    let moved = false;
    const view = { x: 0, y: 0, scale: 1 };

    if (!rawNodes.length) {
      loading.textContent = "当前地址暂未形成可展示的关联网络";
      count.textContent = "0 个节点 · 0 条连接";
      return;
    }
    loading.hidden = true;
    count.textContent = `${rawNodes.length.toLocaleString("zh-CN")} 个节点 · ${rawEdges.length.toLocaleString("zh-CN")} 条连接`;

    function stableNumber(text) {
      let value = 2166136261;
      for (const char of String(text)) value = Math.imul(value ^ char.charCodeAt(0), 16777619);
      return Math.abs(value >>> 0);
    }

    const transactionNodes = rawNodes.filter((node) => node.kind === "transaction");
    const nodes = rawNodes.map((raw, index) => {
      const txIndex = raw.kind === "transaction" ? transactionNodes.findIndex((item) => item.id === raw.id) : -1;
      const seed = stableNumber(raw.id);
      const angle = raw.kind === "transaction"
        ? (Math.PI * 2 * txIndex) / Math.max(transactionNodes.length, 1)
        : (seed % 6283) / 1000;
      const radius = raw.kind === "focal" ? 0 : raw.kind === "transaction"
        ? 190 + (txIndex % 3) * 75
        : 360 + (seed % 260);
      return {
        ...raw,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
        radius: raw.kind === "focal" ? 18 : raw.kind === "transaction" ? 10 : 6 + Math.min(5, Math.sqrt(Number(raw.degree || 1))),
        fixed: raw.id === focalId,
        index,
      };
    });
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const edges = rawEdges.map((edge) => ({
      ...edge,
      sourceNode: nodeById.get(edge.source),
      targetNode: nodeById.get(edge.target),
    })).filter((edge) => edge.sourceNode && edge.targetNode);

    // Start each counterparty near its first connected transaction, then let the
    // lightweight force layout separate overlapping nodes.
    for (const edge of edges) {
      const addressNode = edge.sourceNode.kind === "address" || edge.sourceNode.kind === "focal"
        ? edge.sourceNode : edge.targetNode;
      const txNode = edge.sourceNode.kind === "transaction" ? edge.sourceNode : edge.targetNode;
      if (addressNode.kind === "address") {
        const seed = stableNumber(`${addressNode.id}:${txNode.id}`);
        const angle = (seed % 6283) / 1000;
        addressNode.x = txNode.x + Math.cos(angle) * (70 + seed % 65);
        addressNode.y = txNode.y + Math.sin(angle) * (70 + seed % 65);
      }
    }

    function resize() {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const rect = container.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }

    function fitGraph() {
      const rect = container.getBoundingClientRect();
      const xs = nodes.map((node) => node.x);
      const ys = nodes.map((node) => node.y);
      const minX = Math.min(...xs) - 70;
      const maxX = Math.max(...xs) + 70;
      const minY = Math.min(...ys) - 70;
      const maxY = Math.max(...ys) + 70;
      view.scale = Math.max(0.08, Math.min(1.35, Math.min(rect.width / (maxX - minX), rect.height / (maxY - minY))));
      view.x = rect.width / 2 - ((minX + maxX) / 2) * view.scale;
      view.y = rect.height / 2 - ((minY + maxY) / 2) * view.scale;
      draw();
    }

    function worldToScreen(node) {
      return { x: node.x * view.scale + view.x, y: node.y * view.scale + view.y };
    }

    function screenToWorld(x, y) {
      return { x: (x - view.x) / view.scale, y: (y - view.y) / view.scale };
    }

    function simulate() {
      if (simulationFrames <= 0 || dragNode) return;
      for (const edge of edges) {
        const source = edge.sourceNode;
        const target = edge.targetNode;
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const distance = Math.max(1, Math.hypot(dx, dy));
        const desired = source.kind === "transaction" && target.kind === "transaction" ? 90 : 105;
        const force = (distance - desired) * 0.0018;
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        if (!source.fixed) { source.vx += fx; source.vy += fy; }
        if (!target.fixed) { target.vx -= fx; target.vy -= fy; }
      }
      if (nodes.length <= 320) {
        for (let i = 0; i < nodes.length; i += 1) {
          for (let j = i + 1; j < nodes.length; j += 1) {
            const a = nodes[i];
            const b = nodes[j];
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const distance2 = Math.max(100, dx * dx + dy * dy);
            if (distance2 > 32000) continue;
            const force = 28 / distance2;
            if (!a.fixed) { a.vx -= dx * force; a.vy -= dy * force; }
            if (!b.fixed) { b.vx += dx * force; b.vy += dy * force; }
          }
        }
      }
      for (const node of nodes) {
        if (node.fixed) { node.x = 0; node.y = 0; continue; }
        node.vx += -node.x * 0.000015;
        node.vy += -node.y * 0.000015;
        node.vx *= 0.9;
        node.vy *= 0.9;
        node.x += node.vx;
        node.y += node.vy;
      }
      simulationFrames -= 1;
    }

    function drawArrow(edge, alpha, now) {
      const source = edge.sourceNode;
      const target = edge.targetNode;
      const start = worldToScreen(source);
      const end = worldToScreen(target);
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const ux = dx / distance;
      const uy = dy / distance;
      const startRadius = source.radius * view.scale + 2;
      const endRadius = target.radius * view.scale + 3;
      const sx = start.x + ux * startRadius;
      const sy = start.y + uy * startRadius;
      const ex = end.x - ux * endRadius;
      const ey = end.y - uy * endRadius;
      const inputFlow = edge.role === "input";
      const color = inputFlow ? "#788cff" : "#25e3af";
      const bright = inputFlow ? "#b7c1ff" : "#9affdf";
      const phase = (stableNumber(edge.id || `${edge.source}:${edge.target}`) % 1000) / 1000;
      const gradient = context.createLinearGradient(sx, sy, ex, ey);
      gradient.addColorStop(0, inputFlow ? "rgba(88,105,255,.18)" : "rgba(27,180,140,.18)");
      gradient.addColorStop(0.55, color);
      gradient.addColorStop(1, bright);
      context.save();
      context.globalAlpha = alpha;
      context.strokeStyle = gradient;
      context.lineWidth = Math.max(0.8, view.scale * (edge.predicted_label === 1 ? 1.35 : 1.05));
      context.setLineDash([3, 8]);
      context.lineDashOffset = -((now || 0) * 0.025 + phase * 24);
      context.shadowBlur = edge.predicted_label === 1 ? 10 : 6;
      context.shadowColor = edge.predicted_label === 1 ? "#ffad4b" : color;
      context.beginPath();
      context.moveTo(sx, sy);
      context.lineTo(ex, ey);
      context.stroke();
      context.setLineDash([]);
      if (view.scale > 0.45) {
        const size = Math.min(6, 3 + view.scale * 2);
        context.fillStyle = color;
        context.beginPath();
        context.moveTo(ex, ey);
        context.lineTo(ex - ux * size - uy * size * 0.55, ey - uy * size + ux * size * 0.55);
        context.lineTo(ex - ux * size + uy * size * 0.55, ey - uy * size - ux * size * 0.55);
        context.closePath();
        context.fill();
      }
      const travel = (((now || 0) * (inputFlow ? 0.00017 : 0.0002)) + phase) % 1;
      const particleCount = distance > 170 ? 2 : 1;
      for (let index = 0; index < particleCount; index += 1) {
        const position = (travel + index / particleCount) % 1;
        const px = sx + (ex - sx) * position;
        const py = sy + (ey - sy) * position;
        const particleRadius = Math.max(1.7, Math.min(3.8, 1.8 + view.scale));
        context.globalAlpha = Math.min(1, alpha + 0.28);
        context.fillStyle = bright;
        context.shadowBlur = 16;
        context.shadowColor = color;
        context.beginPath();
        context.arc(px, py, particleRadius, 0, Math.PI * 2);
        context.fill();
      }
      context.restore();
    }

    function drawNode(node, now) {
      const point = worldToScreen(node);
      const radius = Math.max(2.5, node.radius * view.scale);
      const active = node === hovered || node === selected;
      let fill = "#3e83c7";
      if (node.kind === "focal") fill = "#27d7ff";
      else if (node.kind === "address") fill = "#42caa1";
      else if (node.predicted_label === 1) fill = "#ffb545";
      const pulse = 0.5 + 0.5 * Math.sin((now || 0) / 520 + (stableNumber(node.id) % 628) / 100);
      const glowAllowed = nodes.length <= 350 || active || node.kind !== "address";
      context.save();
      if (glowAllowed) {
        context.globalAlpha = active ? 0.34 : 0.12 + pulse * 0.12;
        context.fillStyle = fill;
        context.shadowBlur = active ? 32 : 18 + pulse * 10;
        context.shadowColor = fill;
        context.beginPath();
        if (node.kind === "transaction") {
          context.rect(point.x - radius * 1.75, point.y - radius * 1.75, radius * 3.5, radius * 3.5);
        } else {
          context.arc(point.x, point.y, radius * (1.8 + pulse * 0.45), 0, Math.PI * 2);
        }
        context.fill();
      }
      context.globalAlpha = 1;
      context.shadowBlur = active ? 24 : node.kind === "focal" ? 18 : 10;
      context.shadowColor = fill;
      context.fillStyle = fill;
      context.strokeStyle = active ? "#ffffff" : "#092342";
      context.lineWidth = active ? 2 : 1;
      context.beginPath();
      if (node.kind === "transaction") {
        context.rect(point.x - radius, point.y - radius, radius * 2, radius * 2);
      } else {
        context.arc(point.x, point.y, radius, 0, Math.PI * 2);
      }
      context.fill();
      context.stroke();
      context.restore();

      const showLabel = active || node.kind === "focal" || (view.scale > 1.05 && nodes.length < 500);
      if (showLabel) {
        const value = node.kind === "transaction" ? node.transaction_id : node.address;
        context.font = `${active ? 11 : 9}px Consolas, monospace`;
        context.textAlign = "center";
        context.textBaseline = "top";
        context.fillStyle = active ? "#eaf8ff" : "#7fa4c2";
        context.fillText(TQ.shortHash(value, 7, 5), point.x, point.y + radius + 5);
      }
    }

    function draw(now = performance.now()) {
      const rect = container.getBoundingClientRect();
      context.clearRect(0, 0, rect.width, rect.height);
      for (const edge of edges) {
        const highlighted = selected && (edge.sourceNode === selected || edge.targetNode === selected);
        drawArrow(edge, highlighted ? 0.98 : selected ? 0.15 : 0.52, now);
      }
      for (const node of nodes) drawNode(node, now);
    }

    function tick(now) {
      simulate();
      draw(now);
      frame = window.requestAnimationFrame(tick);
    }

    function nodeAt(x, y) {
      let winner = null;
      let best = Infinity;
      for (const node of nodes) {
        const point = worldToScreen(node);
        const distance = Math.hypot(point.x - x, point.y - y);
        const hitRadius = Math.max(9, node.radius * view.scale + 5);
        if (distance <= hitRadius && distance < best) { winner = node; best = distance; }
      }
      return winner;
    }

    function tooltipText(node) {
      if (node.kind === "transaction") {
        return `<b>${node.predicted_label === 1 ? "重点关注交易" : "常规交易"}</b><code>${TQ.escapeHtml(node.transaction_id)}</code><span>风险指数 ${TQ.score(node.risk_score)} 分 · 点击进入研判</span>`;
      }
      return `<b>${node.kind === "focal" ? "当前查询地址" : "关联地址"}</b><code>${TQ.escapeHtml(node.address)}</code><span>${Number(node.degree || 0)} 条连接 · 双击以该地址继续查询</span>`;
    }

    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      pointerDown = { x, y, viewX: view.x, viewY: view.y };
      dragNode = nodeAt(x, y);
      panning = !dragNode;
      moved = false;
      canvas.classList.toggle("grabbing", true);
    });
    canvas.addEventListener("pointermove", (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      if (pointerDown) {
        const dx = x - pointerDown.x;
        const dy = y - pointerDown.y;
        moved ||= Math.hypot(dx, dy) > 3;
        if (dragNode && !dragNode.fixed) {
          const world = screenToWorld(x, y);
          dragNode.x = world.x;
          dragNode.y = world.y;
          dragNode.vx = 0;
          dragNode.vy = 0;
          simulationFrames = Math.max(simulationFrames, 80);
        } else if (panning) {
          view.x = pointerDown.viewX + dx;
          view.y = pointerDown.viewY + dy;
        }
        return;
      }
      hovered = nodeAt(x, y);
      canvas.style.cursor = hovered ? "pointer" : "grab";
      if (hovered) {
        tooltip.hidden = false;
        tooltip.innerHTML = tooltipText(hovered);
        tooltip.style.left = `${Math.min(rect.width - 310, Math.max(8, x + 14))}px`;
        tooltip.style.top = `${Math.min(rect.height - 100, Math.max(8, y + 14))}px`;
      } else {
        tooltip.hidden = true;
      }
    });
    function releasePointer(event) {
      if (!pointerDown) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const clicked = !moved ? nodeAt(x, y) : null;
      if (clicked) {
        selected = clicked;
        if (clicked.kind === "transaction") openTransaction(clicked.transaction_id);
      }
      pointerDown = null;
      dragNode = null;
      panning = false;
      canvas.classList.remove("grabbing");
    }
    canvas.addEventListener("pointerup", releasePointer);
    canvas.addEventListener("pointercancel", releasePointer);
    canvas.addEventListener("pointerleave", () => { if (!pointerDown) { hovered = null; tooltip.hidden = true; } });
    canvas.addEventListener("dblclick", (event) => {
      const rect = canvas.getBoundingClientRect();
      const node = nodeAt(event.clientX - rect.left, event.clientY - rect.top);
      if (node && node.kind !== "transaction" && node.address && node.address !== "coinbase" && !node.address.startsWith("script:")) {
        window.location.href = `/relations.html?address=${encodeURIComponent(node.address)}`;
      }
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const world = screenToWorld(x, y);
      const factor = event.deltaY < 0 ? 1.14 : 0.88;
      view.scale = Math.min(4.5, Math.max(0.05, view.scale * factor));
      view.x = x - world.x * view.scale;
      view.y = y - world.y * view.scale;
    }, { passive: false });

    function zoom(factor) {
      const rect = container.getBoundingClientRect();
      const world = screenToWorld(rect.width / 2, rect.height / 2);
      view.scale = Math.min(4.5, Math.max(0.05, view.scale * factor));
      view.x = rect.width / 2 - world.x * view.scale;
      view.y = rect.height / 2 - world.y * view.scale;
    }
    document.getElementById("graph-zoom-in").addEventListener("click", () => zoom(1.25));
    document.getElementById("graph-zoom-out").addEventListener("click", () => zoom(0.8));
    document.getElementById("graph-reset").addEventListener("click", fitGraph);

    function locateNode() {
      const query = document.getElementById("graph-search").value.trim().toLowerCase();
      if (!query) return;
      const match = nodes.find((node) => String(node.address || node.transaction_id || "").toLowerCase().includes(query));
      if (!match) {
        tooltip.hidden = false;
        tooltip.innerHTML = "<b>未找到匹配节点</b><span>请检查地址或交易哈希字符</span>";
        tooltip.style.left = "16px";
        tooltip.style.top = "16px";
        return;
      }
      selected = match;
      view.scale = Math.max(view.scale, 1.1);
      const rect = container.getBoundingClientRect();
      view.x = rect.width / 2 - match.x * view.scale;
      view.y = rect.height / 2 - match.y * view.scale;
      tooltip.hidden = true;
    }
    document.getElementById("graph-search-button").addEventListener("click", locateNode);
    document.getElementById("graph-search").addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); locateNode(); }
    });

    const observer = new ResizeObserver(() => resize());
    observer.observe(container);
    resize();
    window.setTimeout(fitGraph, 80);
    tick(performance.now());
    window.addEventListener("pagehide", () => {
      observer.disconnect();
      window.cancelAnimationFrame(frame);
    }, { once: true });
  }

  function openTransaction(transactionId) {
    TQ.setContext({ address, transactionId });
    window.location.href = `/analysis.html?transaction=${encodeURIComponent(transactionId)}&address=${encodeURIComponent(address)}`;
  }

  function renderTable() {
    const query = search.value.trim();
    const rows = profile.transactions.filter((item) => {
      if (currentFilter === "high" && item.predicted_label !== 1) return false;
      if (currentFilter === "low" && item.predicted_label !== 0) return false;
      return !query || String(item.transaction_id).includes(query);
    });
    document.getElementById("transaction-count").textContent = `${rows.length} 笔`;
    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="7" class="empty-cell">没有符合当前筛选条件的交易</td></tr>`;
      return;
    }
    body.innerHTML = rows.map((item) => {
      const high = item.predicted_label === 1;
      return `<tr data-transaction="${item.transaction_id}">
        <td><b class="tx-id" title="${TQ.escapeHtml(item.transaction_id)}">${TQ.escapeHtml(TQ.shortHash(item.transaction_id))}</b></td>
        <td><span class="sequence-tag">${TQ.escapeHtml(TQ.eventTime(item))}</span></td>
        <td>${TQ.escapeHtml(roleText(item.roles))}</td>
        <td><b>${TQ.btc(item.amount_btc)}</b></td>
        <td><div class="risk-meter"><i><em style="width:${Math.round(item.risk_score * 100)}%"></em></i><span>${TQ.score(item.risk_score)}分</span></div></td>
        <td><span class="verdict ${high ? "high" : "low"}"><i></i>${high ? "重点关注" : "常规关注"}</span></td>
        <td><button class="row-action" data-open="${item.transaction_id}">查看研判 →</button></td>
      </tr>`;
    }).join("");
    body.querySelectorAll("[data-open]").forEach((button) => button.addEventListener("click", () => openTransaction(button.dataset.open)));
  }

  search.addEventListener("input", renderTable);
  filters.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    filters.querySelectorAll("button").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    currentFilter = button.dataset.filter;
    renderTable();
  }));

  try {
    profile = await TQ.requestJson(`/api/address?value=${encodeURIComponent(address)}`);
    renderMetrics();
    renderRoles();
    await renderMap();
    renderTable();
  } catch (error) {
    addressLabel.textContent = error.message;
    body.innerHTML = `<tr><td colspan="7" class="empty-cell"><a href="/index.html">返回并重新输入地址</a></td></tr>`;
  }
});
