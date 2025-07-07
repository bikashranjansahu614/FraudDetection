from flask import Flask, request, jsonify
import os
from datetime import datetime
import numpy as np
from threading import Thread
from IPython.display import display, HTML
import pickle
import warnings
warnings.filterwarnings('ignore')
import csv
import pandas as pd

app = Flask(__name__)

# Load model
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("iso_model.pkl", "rb") as f:
    iso_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

@app.route('/')
def home():
    return "<h1>Fraud Detection API</h1><p>POST to /predict with transaction data</p>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        tx_type = data['type']
        amount = float(data['amount'])
        origin_flag = int(data['origin_flag'])
        dest_flag = int(data['dest_flag'])
        hour = int(data.get('hour', 12))
        day = int(data.get('day', 2))
        is_internal = int(origin_flag == 1 and dest_flag == 1)
        is_suspicious_transfer = int(tx_type == "TRANSFER" and origin_flag == 1 and dest_flag == 0)


        tx_type_encoded = le.transform([tx_type])[0]

        features = np.array([[tx_type_encoded, amount, origin_flag, dest_flag,
                              hour, day, is_internal, is_suspicious_transfer]])
        features_scaled = scaler.transform(features)

        rf_pred = rf_model.predict(features_scaled)[0]
        rf_prob = rf_model.predict_proba(features_scaled)[0][1]  # risk score

        # Unsupervised anomaly
        iso_pred = iso_model.predict(features_scaled)[0]  # -1 = anomaly
        is_anomaly = int(iso_pred == -1)
        
        is_fraud = int(rf_pred == 1 or is_anomaly == 1)
        
        explanation = []
        if rf_pred == 1:
            explanation.append("Matched historical fraud patterns (RF)")
        if is_anomaly:
            explanation.append("Unusual pattern detected (IsoForest)")
        if is_suspicious_transfer:
            explanation.append("Suspicious shell transfer")
        if origin_flag == 1 and dest_flag == 1:
            explanation.append("Both origin and destination are shell accounts")

        if is_fraud:
            log_row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tx_type, amount,
                       origin_flag, dest_flag, hour, day, rf_prob]
            with open("flagged_txns.csv", "a", newline='') as f:
                writer = csv.writer(f)
                if os.stat("flagged_txns.csv").st_size == 0:
                    writer.writerow(["timestamp", "type", "amount", "origin_flag", "dest_flag", "hour", "day", "risk_score"])
                writer.writerow(log_row)

        return jsonify({
            "fraud": bool(is_fraud),
            "risk_score": round(rf_prob, 3),
            "explanation": explanation or ["Low risk"]
        })

        

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dashboard')
def dashboard():
    if not os.path.exists("flagged_txns.csv"):
        return "<h3>No transactions logged yet.</h3>"

    df = pd.read_csv("flagged_txns.csv")
    latest = df.sort_values(by='risk_score', ascending=False).head(10)
    table_html = latest.to_html(index=False, classes="table table-bordered")

    chart_data = df['risk_score'].round(2).value_counts().sort_index()
    js_data = ','.join(str(v) for v in chart_data.values)
    js_labels = ','.join(str(k) for k in chart_data.index)

    html = f"""
    <html><head><title>Dashboard</title>
    <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
    <style>body {{ font-family: Arial; }} .table {{ border-collapse: collapse; }}
    .table td, .table th {{ border: 1px solid #ddd; padding: 8px; }}</style></head>
    <body>
    <h1>Fraud Risk Dashboard</h1>
    <h3>Top Flagged Transactions</h3>{table_html}
    <h3>Risk Score Histogram</h3>
    <canvas id="riskChart" width="500" height="200"></canvas>
    <script>
    new Chart(document.getElementById('riskChart').getContext('2d'), {{
        type: 'bar',
        data: {{ labels: [{js_labels}], datasets: [{{ label: 'Counts', data: [{js_data}], backgroundColor: 'rgba(255,99,132,0.6)' }}] }},
        options: {{ scales: {{ y: {{ beginAtZero: true }} }} }}
    }});
    </script>
    </body></html>"""
    return html

if __name__ == '__main__':
    app.run(port=5000, debug=False)