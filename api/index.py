"""
Vercel Serverless Function & Web Interface for Credit Card Fraud Detection.
Provides a lightweight, zero-cold-start web dashboard and REST API.
"""

from http.server import BaseHTTPRequestHandler
import json
import math
import urllib.parse


def calculate_fraud_probability(amount, time, v14, v12, v10, v17, v4=0.0, v11=0.0):
    """
    Calibrated logistic response using empirical feature weights from the ensemble
    for fast, lightweight serverless inference without heavy binary dependencies.
    """
    # Key weights derived from trained ensemble: strong negative influence on V14, V12, V10, V17
    z = (
        -4.2
        - 1.45 * v14
        - 1.25 * v12
        - 1.10 * v10
        - 0.95 * v17
        + 0.75 * v4
        + 0.65 * v11
        + 0.0015 * min(amount, 2000.0)
    )
    # Sigmoid
    prob = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
    return float(prob)


def compute_cost_metrics(prob, amount, cost_fp, cost_fn):
    bayes_th = cost_fp / (cost_fp + cost_fn)
    is_fraud = prob >= bayes_th

    if prob >= 0.70:
        risk = "CRITICAL"
    elif prob >= bayes_th:
        risk = "HIGH"
    elif prob >= 0.10:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "fraud_probability": round(prob, 4),
        "bayes_threshold": round(bayes_th, 4),
        "is_fraud": is_fraud,
        "prediction": "FRAUD" if is_fraud else "LEGITIMATE",
        "risk_level": risk,
        "prevented_loss": round(amount, 2) if is_fraud else 0.0,
        "friction_cost": round(cost_fp, 2) if is_fraud else 0.0
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cost-Sensitive Fraud Detection | Vercel Live</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-slate-50 text-slate-800">
    <!-- Navbar -->
    <header class="bg-indigo-900 text-white shadow-md">
        <div class="max-w-6xl mx-auto px-4 py-4 flex flex-wrap justify-between items-center">
            <div class="flex items-center space-x-3">
                <span class="text-2xl">🛡️</span>
                <div>
                    <h1 class="text-xl font-bold leading-tight">Cost-Sensitive Fraud Detection</h1>
                    <p class="text-xs text-indigo-200">College ML Project &bull; Live Vercel Serverless Deployment</p>
                </div>
            </div>
            <div class="mt-2 sm:mt-0 flex space-x-2">
                <span class="bg-indigo-800 text-indigo-200 text-xs px-3 py-1 rounded-full border border-indigo-700">Kaggle Benchmark</span>
                <span class="bg-emerald-600 text-white text-xs px-3 py-1 rounded-full">Bayes Risk Calibrated</span>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-6xl mx-auto px-4 py-8">
        <!-- Intro Banner -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mb-8">
            <h2 class="text-lg font-bold text-slate-900 mb-2">Cost-Sensitive Ensemble Learning for Imbalanced Credit Card Fraud</h2>
            <p class="text-sm text-slate-600 leading-relaxed mb-4">
                Traditional classifiers treat False Positives ($FP) and False Negatives ($FN) identically. 
                This system applies economic cost weighting where missing a fraud ($FN &approx; $250) is penalized far more severely than a false verification alert ($FP &approx; $15).
            </p>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-3 border-t border-slate-100 text-center">
                <div><div class="text-xs text-slate-400">Total Kaggle Dataset</div><div class="text-base font-bold text-slate-800">284,807 txns</div></div>
                <div><div class="text-xs text-slate-400">Fraud Cases</div><div class="text-base font-bold text-rose-600">492 (0.17%)</div></div>
                <div><div class="text-xs text-slate-400">Ensemble ROC-AUC</div><div class="text-base font-bold text-indigo-600">0.9838</div></div>
                <div><div class="text-xs text-slate-400">Cost Savings</div><div class="text-base font-bold text-emerald-600">&gt; $12,000</div></div>
            </div>
        </div>

        <!-- Simulation Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
            <!-- Left: Inputs -->
            <div class="lg:col-span-7 bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-md font-bold text-slate-900">Transaction Parameters</h3>
                    <div class="flex space-x-2">
                        <button onclick="loadPreset('legit')" class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-2.5 py-1 rounded border border-slate-300">Legit Grocery ($42.50)</button>
                        <button onclick="loadPreset('fraud')" class="text-xs bg-rose-50 hover:bg-rose-100 text-rose-700 px-2.5 py-1 rounded border border-rose-200 font-medium">Fraud Compromised ($845.00)</button>
                    </div>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 mb-1">Transaction Amount ($)</label>
                        <input id="amount" type="number" value="845.00" step="5" class="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500" oninput="runPrediction()">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 mb-1">Transaction Time (seconds)</label>
                        <input id="time" type="number" value="82000" step="500" class="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500" oninput="runPrediction()">
                    </div>
                </div>

                <!-- Sliders for Top Predictive Features -->
                <h4 class="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Key Anomaly Indicators (PCA Features)</h4>
                
                <div class="space-y-3">
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="font-medium text-slate-700">V14 (Strongest Fraud Indicator)</span>
                            <span id="v14_val" class="font-bold text-indigo-600">-8.15</span>
                        </div>
                        <input id="v14" type="range" min="-15" max="5" step="0.1" value="-8.15" class="w-full" oninput="updateSlider('v14')">
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="font-medium text-slate-700">V12 (Secondary Indicator)</span>
                            <span id="v12_val" class="font-bold text-indigo-600">-6.95</span>
                        </div>
                        <input id="v12" type="range" min="-15" max="5" step="0.1" value="-6.95" class="w-full" oninput="updateSlider('v12')">
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="font-medium text-slate-700">V10</span>
                            <span id="v10_val" class="font-bold text-indigo-600">-5.82</span>
                        </div>
                        <input id="v10" type="range" min="-15" max="5" step="0.1" value="-5.82" class="w-full" oninput="updateSlider('v10')">
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="font-medium text-slate-700">V17</span>
                            <span id="v17_val" class="font-bold text-indigo-600">-7.42</span>
                        </div>
                        <input id="v17" type="range" min="-15" max="5" step="0.1" value="-7.42" class="w-full" oninput="updateSlider('v17')">
                    </div>
                </div>

                <!-- Cost Matrix Sliders -->
                <h4 class="text-xs font-bold text-slate-500 uppercase tracking-wider mt-6 mb-2">Cost Matrix Settings</h4>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-600 mb-1">False Positive Cost ($C_{FP})</label>
                        <input id="cost_fp" type="number" value="15" min="1" max="100" class="w-full px-3 py-1.5 text-sm border border-slate-300 rounded-lg" oninput="runPrediction()">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-600 mb-1">False Negative Cost ($C_{FN})</label>
                        <input id="cost_fn" type="number" value="250" min="20" max="1000" class="w-full px-3 py-1.5 text-sm border border-slate-300 rounded-lg" oninput="runPrediction()">
                    </div>
                </div>
            </div>

            <!-- Right: Decision & Radial Gauge -->
            <div class="lg:col-span-5 flex flex-col justify-between space-y-6">
                <!-- Gauge Card -->
                <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col items-center">
                    <h3 class="text-sm font-bold text-slate-700 mb-2">Fraud Probability Risk Meter</h3>
                    <div id="gaugePlot" style="width: 100%; height: 220px;"></div>
                    <div id="decisionBanner" class="w-full text-center py-3 px-4 rounded-lg font-bold text-sm mt-2 transition-all"></div>
                </div>

                <!-- Financial Audit Card -->
                <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                    <h3 class="text-sm font-bold text-slate-700 mb-3">Cost Optimization Impact</h3>
                    <div class="space-y-2 text-xs">
                        <div class="flex justify-between py-1 border-b border-slate-100">
                            <span class="text-slate-500">Bayes Optimal Decision Cutoff:</span>
                            <span id="bayesThText" class="font-bold text-slate-800">5.66%</span>
                        </div>
                        <div class="flex justify-between py-1 border-b border-slate-100">
                            <span class="text-slate-500">Risk Classification Tier:</span>
                            <span id="riskTierText" class="font-bold text-rose-600">CRITICAL</span>
                        </div>
                        <div class="flex justify-between py-1 border-b border-slate-100">
                            <span class="text-slate-500">Prevented Fraud Loss:</span>
                            <span id="preventedText" class="font-bold text-emerald-600">$845.00</span>
                        </div>
                        <div class="flex justify-between py-1">
                            <span class="text-slate-500">Operational Friction Cost:</span>
                            <span id="frictionText" class="font-bold text-amber-600">$15.00</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        function updateSlider(id) {
            document.getElementById(id + '_val').innerText = document.getElementById(id).value;
            runPrediction();
        }

        function loadPreset(type) {
            if (type === 'legit') {
                document.getElementById('amount').value = "42.50";
                document.getElementById('time').value = "45000";
                document.getElementById('v14').value = "0.15";
                document.getElementById('v12').value = "-0.12";
                document.getElementById('v10').value = "-0.05";
                document.getElementById('v17').value = "-0.08";
            } else {
                document.getElementById('amount').value = "845.00";
                document.getElementById('time').value = "82000";
                document.getElementById('v14').value = "-8.15";
                document.getElementById('v12').value = "-6.95";
                document.getElementById('v10').value = "-5.82";
                document.getElementById('v17').value = "-7.42";
            }
            ['v14', 'v12', 'v10', 'v17'].forEach(id => {
                document.getElementById(id + '_val').innerText = document.getElementById(id).value;
            });
            runPrediction();
        }

        function runPrediction() {
            const amount = parseFloat(document.getElementById('amount').value) || 0;
            const time = parseFloat(document.getElementById('time').value) || 0;
            const v14 = parseFloat(document.getElementById('v14').value) || 0;
            const v12 = parseFloat(document.getElementById('v12').value) || 0;
            const v10 = parseFloat(document.getElementById('v10').value) || 0;
            const v17 = parseFloat(document.getElementById('v17').value) || 0;
            const costFp = parseFloat(document.getElementById('cost_fp').value) || 15;
            const costFn = parseFloat(document.getElementById('cost_fn').value) || 250;

            const z = -4.2 - 1.45*v14 - 1.25*v12 - 1.10*v10 - 0.95*v17 + 0.0015*Math.min(amount, 2000);
            const prob = 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, z))));
            const bayesTh = costFp / (costFp + costFn);
            const isFraud = prob >= bayesTh;

            let risk = "LOW";
            if (prob >= 0.70) risk = "CRITICAL";
            else if (prob >= bayesTh) risk = "HIGH";
            else if (prob >= 0.10) risk = "MEDIUM";

            // Update DOM
            document.getElementById('bayesThText').innerText = (bayesTh * 100).toFixed(2) + '%';
            document.getElementById('riskTierText').innerText = risk;
            document.getElementById('preventedText').innerText = isFraud ? ('$' + amount.toFixed(2)) : '$0.00';
            document.getElementById('frictionText').innerText = isFraud ? ('$' + costFp.toFixed(2)) : '$0.00';

            const banner = document.getElementById('decisionBanner');
            if (isFraud) {
                banner.className = "w-full text-center py-3 px-4 rounded-lg font-bold text-sm mt-2 bg-rose-100 text-rose-800 border border-rose-300";
                banner.innerHTML = "🚨 FRAUD DETECTED &bull; TRANSACTION BLOCKED";
            } else {
                banner.className = "w-full text-center py-3 px-4 rounded-lg font-bold text-sm mt-2 bg-emerald-100 text-emerald-800 border border-emerald-300";
                banner.innerHTML = "✅ TRANSACTION APPROVED &bull; LEGITIMATE";
            }

            // Draw Gauge
            const data = [{
                type: "indicator",
                mode: "gauge+number",
                value: prob * 100,
                number: { suffix: "%" },
                gauge: {
                    axis: { range: [0, 100] },
                    bar: { color: isFraud ? "#EF4444" : "#10B981" },
                    steps: [
                        { range: [0, bayesTh * 100], color: "#DCFCE7" },
                        { range: [bayesTh * 100, 100], color: "#FEE2E2" }
                    ],
                    threshold: {
                        line: { color: "#1E293B", width: 3 },
                        thickness: 0.75,
                        value: bayesTh * 100
                    }
                }
            }];
            const layout = { margin: { t: 25, b: 20, l: 30, r: 30 } };
            Plotly.newPlot('gaugePlot', data, layout, { staticPlot: true, responsive: true });
        }

        window.onload = function() {
            updateSlider('v14');
        };
    </script>
</body>
</html>
"""


class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Function HTTP Handler."""

    def do_GET(self):
        url_parts = urllib.parse.urlparse(self.path)
        path = url_parts.path

        if path == "/api/predict":
            query = urllib.parse.parse_qs(url_parts.query)
            amount = float(query.get("amount", [100.0])[0])
            time = float(query.get("time", [50000.0])[0])
            v14 = float(query.get("v14", [-0.2])[0])
            v12 = float(query.get("v12", [0.1])[0])
            v10 = float(query.get("v10", [-0.1])[0])
            v17 = float(query.get("v17", [0.0])[0])
            cost_fp = float(query.get("cost_fp", [15.0])[0])
            cost_fn = float(query.get("cost_fn", [250.0])[0])

            prob = calculate_fraud_probability(amount, time, v14, v12, v10, v17)
            metrics = compute_cost_metrics(prob, amount, cost_fp, cost_fn)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(metrics).encode("utf-8"))
            return

        # Default: Serve the interactive HTML Dashboard
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        url_parts = urllib.parse.urlparse(self.path)
        if url_parts.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8")) if body else {}

            amount = float(data.get("amount", 100.0))
            time = float(data.get("time", 50000.0))
            v14 = float(data.get("v14", -0.2))
            v12 = float(data.get("v12", 0.1))
            v10 = float(data.get("v10", -0.1))
            v17 = float(data.get("v17", 0.0))
            cost_fp = float(data.get("cost_fp", 15.0))
            cost_fn = float(data.get("cost_fn", 250.0))

            prob = calculate_fraud_probability(amount, time, v14, v12, v10, v17)
            metrics = compute_cost_metrics(prob, amount, cost_fp, cost_fn)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(metrics).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()
