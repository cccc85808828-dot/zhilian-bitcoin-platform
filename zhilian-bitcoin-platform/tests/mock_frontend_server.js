"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..", "demo");
const address = "bc1qvisualqa000000000000000000000000000000";
const nodes = [{ id: `address:${address}`, kind: "focal", address, label: address, roles: ["focus"], degree: 12 }];
const edges = [];
const transactions = [];
const cases = new Map();

for (let index = 0; index < 12; index += 1) {
  const txid = `${String(index + 1).padStart(2, "0")}${"a".repeat(62)}`;
  const input = `bc1qin${String(index + 1).padStart(2, "0")}${"x".repeat(32)}`;
  const output = `bc1qout${String(index + 1).padStart(2, "0")}${"y".repeat(31)}`;
  const risk = index % 4 === 0 ? 0.92 - index * 0.002 : 0.24 + index * 0.025;
  const high = risk >= 0.88 ? 1 : 0;
  const eventTime = `2026-08-${String(8 + index).padStart(2, "0")}T10:${String(index).padStart(2, "0")}:00+08:00`;
  nodes.push(
    { id: `transaction:${txid}`, kind: "transaction", transaction_id: txid, label: txid, risk_score: risk, predicted_label: high, event_time: eventTime, degree: 3 },
    { id: `address:${input}`, kind: "address", address: input, label: input, roles: ["input"], degree: 1 },
    { id: `address:${output}`, kind: "address", address: output, label: output, roles: ["output"], degree: 1 },
  );
  edges.push(
    { id: `edge:${edges.length}`, source: `address:${input}`, target: `transaction:${txid}`, role: "input", amount_btc: 1.2 + index, transaction_id: txid, risk_score: risk, predicted_label: high },
    { id: `edge:${edges.length + 1}`, source: `address:${address}`, target: `transaction:${txid}`, role: "input", amount_btc: 0.2 + index, transaction_id: txid, risk_score: risk, predicted_label: high },
    { id: `edge:${edges.length + 2}`, source: `transaction:${txid}`, target: `address:${output}`, role: "output", amount_btc: 1.39 + index, transaction_id: txid, risk_score: risk, predicted_label: high },
  );
  const compact = { transaction_id: txid, event_time: eventTime, sequence_order: index + 1, risk_score: risk, predicted_label: high, roles: ["input"], amount_btc: 1.39 + index, confirmed: true, block_height: 963000 + index };
  transactions.push(compact);
  cases.set(txid, {
    ...compact,
    graph_score: 0.91,
    qif_score: 0.95,
    decision_threshold_score: 0.88,
    margin_to_threshold: risk - 0.88,
    input_addresses: { count: 2, items: [{ address: input, amount_btc: 1.2 + index }, { address, amount_btc: 0.2 + index }] },
    output_addresses: { count: 1, items: [{ address: output, amount_btc: 1.39 + index }] },
    confirmation: { confirmed: true, block_height: 963000 + index },
    amount_btc: 1.39 + index,
    fee_btc: 0.01,
    size_bytes: 256,
    data_source: "bitcoin_mainnet",
    attribute_profile: [
      { feature_index: 167, name: "交易总额（BTC）", raw_value: 1.39 + index, unit: "btc", signal_strength: 72 },
      { feature_index: 168, name: "手续费（BTC）", raw_value: 0.01, unit: "btc", signal_strength: 58 },
      { feature_index: 170, name: "输入地址数", raw_value: 2, unit: "count", signal_strength: 45 },
    ],
    decision_explanation: {
      structure: { summary: "时序超图分支得到91.0分，高于研判阈值3.0分；本笔交易由2个输入连接1个输出，呈资金归集形态，地址角色与历史连接共同形成较强结构信号。" },
      attribute: { summary: "QIF属性分支得到95.0分，高于研判阈值7.0分。交易总额与手续费的分位位置及二阶组合形成较强属性信号。", top_features: [{ name: "交易总额", value: `${(1.39 + index).toFixed(8)} BTC` }, { name: "手续费", value: "0.01000000 BTC" }] },
      fusion: { agreement: "结构与属性两条证据链方向一致，均支持重点关注", summary: "结构与属性两条证据链方向一致。融合风险指数越过88.0分阈值，因此判定为重点关注交易。" },
      conclusion: "该交易需要重点关注，建议结合相邻交易、关联地址和资金后续流向继续研判。",
    },
  });
}

transactions.sort((a, b) => b.risk_score - a.risk_score);
const profile = {
  address,
  related_transaction_count: transactions.length,
  total_transaction_count: transactions.length,
  loaded_transaction_count: transactions.length,
  history_complete: true,
  high_risk_transactions: transactions.filter((item) => item.predicted_label).length,
  input_role_transactions: transactions.length,
  output_role_transactions: 0,
  maximum_risk_score: Math.max(...transactions.map((item) => item.risk_score)),
  average_risk_score: transactions.reduce((sum, item) => sum + item.risk_score, 0) / transactions.length,
  first_order: 1,
  last_order: transactions.length,
  first_event_time: transactions.at(-1).event_time,
  last_event_time: transactions[0].event_time,
  transactions,
  relation_graph: { focal_node_id: `address:${address}`, nodes, edges, node_count: nodes.length, edge_count: edges.length, address_node_count: 25, transaction_node_count: 12 },
};

function json(response, body) {
  const data = Buffer.from(JSON.stringify(body));
  response.writeHead(200, { "Content-Type": "application/json; charset=utf-8", "Content-Length": data.length });
  response.end(data);
}

const server = http.createServer((request, response) => {
  const url = new URL(request.url, "http://127.0.0.1:8877");
  if (url.pathname === "/api/address") return json(response, profile);
  if (url.pathname === "/api/assistant/status") return json(response, { configured: true, model: "交互模型" });
  if (url.pathname === "/api/assistant") return json(response, { answer: "### 当前交易解释\n- **结构证据**：地址角色和连接形态支持当前判断。\n- **属性证据**：金额、手续费及其二阶交互形成补充证据。\n- **综合结论**：两条证据链融合后越过研判阈值。" });
  if (url.pathname.startsWith("/api/case/")) return json(response, cases.get(decodeURIComponent(url.pathname.split("/").at(-1))));
  if (url.pathname === "/api/analyze") {
    let raw = "";
    request.on("data", (chunk) => { raw += chunk; });
    request.on("end", () => {
      const txid = JSON.parse(raw || "{}").transaction_id;
      json(response, { status: "PASS", analysis_id: "TQ-VISUAL-QA", result: cases.get(txid) || cases.values().next().value });
    });
    return;
  }
  const name = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
  const file = path.join(root, name);
  if (!file.startsWith(root) || !fs.existsSync(file)) { response.writeHead(404); response.end(); return; }
  const types = { ".html": "text/html", ".js": "application/javascript", ".css": "text/css" };
  const data = fs.readFileSync(file);
  response.writeHead(200, { "Content-Type": `${types[path.extname(file)] || "application/octet-stream"}; charset=utf-8`, "Content-Length": data.length });
  response.end(data);
});

server.listen(8877, "127.0.0.1", () => console.log("mock frontend server: http://127.0.0.1:8877"));
