from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import requests
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def create_95percent_recall_justifications(df, alerts):
    """ULTRA-AGGRESSIVE for 90%+ recall"""
    print("🔍 ULTRA-AGGRESSIVE 90%+ SYSTEM")
    
    enhanced_alerts = []
    
    for alert in alerts:
        record_data = alert['raw_data']
        probability = alert['fraud_probability']
        
        justifications = []
        
        # 1. ANY CATEGORY (except most common)
        category = record_data.get('category', '')
        safe_categories = ['es_transportation', 'es_food']  # Only these are "safe"
        if category not in safe_categories:
            justifications.append({
                'category': 'CATEGORY_RISK', 'feature': 'category', 'strength': 0.4,
                'title': '🎯 Suspicious Category', 'description': f"Transaction in {category}",
                'risk_level': 'MEDIUM', 'context': "Category may indicate higher risk"
            })
        
        # 2. ANY AMOUNT ABOVE $30
        amount = record_data.get('amount', 0)
        if amount > 30:
            justifications.append({
                'category': 'AMOUNT_ANOMALY', 'feature': 'amount', 'strength': 0.3,
                'title': '💰 Moderate Amount', 'description': f"Amount ${amount:.2f}",
                'risk_level': 'LOW', 'context': "Amount above minimum threshold"
            })
        
        # 3. ANY RAPID TRANSACTION
        time_since_last = record_data.get('time_since_last_pair_tx', -1)
        if time_since_last >= 0 and time_since_last < 10.0:
            justifications.append({
                'category': 'VELOCITY_RISK', 'feature': 'time_since_last_pair_tx', 'strength': 0.3,
                'title': '⚡ Any Rapid Transaction', 'description': f"Within {time_since_last:.1f} units",
                'risk_level': 'LOW', 'context': "Rapid transaction pattern"
            })
        
        # 4. ANY FIRST-TIME PAIR
        if record_data.get('first_time_pair', 0) == 1:
            justifications.append({
                'category': 'RELATIONSHIP_RISK', 'feature': 'first_time_pair', 'strength': 0.3,
                'title': '🆕 Any First-Time', 'description': "New relationship",
                'risk_level': 'LOW', 'context': "New customer-merchant pair"
            })
        
        # 5. ANY FREQUENCY
        daily_tx = record_data.get('cust_tx_count_1d', 0)
        if daily_tx > 1:
            justifications.append({
                'category': 'FREQUENCY_ANOMALY', 'feature': 'cust_tx_count_1d', 'strength': 0.2,
                'title': '📈 Any Elevated Frequency', 'description': f"{daily_tx} transactions",
                'risk_level': 'LOW', 'context': "Multiple transactions today"
            })
        
        # CATCH EVERYTHING
        if len(justifications) >= 1 or probability > 0.15:
            enhanced_alert = alert.copy()
            enhanced_alert['advanced_justifications'] = justifications[:3]
            enhanced_alert['risk_factors'] = len(justifications)
            enhanced_alert['confidence_score'] = probability
            enhanced_alerts.append(enhanced_alert)
    
    print(f"✅ ULTRA-AGGRESSIVE: {len(alerts)} → {len(enhanced_alerts)} alerts")
    return enhanced_alerts, {}

def get_datarobot_predictions(test_df):
    """Get predictions from DataRobot API"""
    API_KEY = "NjhlYzZiODMxNDNkZGRiNzBkMGZmNDBkOjV6S3cxelZGUDRxZUY4MWpiNXR0SkFCSWpKMVRtQVR0cENSdVJwMFZWTWM9"
    DEPLOYMENT_ID = "68ec56909821c3a55f1c04aa"
    
    url = f"https://app.datarobot.com/api/v2/deployments/{DEPLOYMENT_ID}/predictions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    required_columns = ['age', 'amount', 'amount_over_cust_median_7d', 'category', 'cust_median_amt_7d', 'cust_tx_count_1d', 'cust_tx_count_7d', 'cust_unique_merchants_30d', 'customer', 'first_time_pair', 'gender', 'log_amount', 'mch_tx_count_1d', 'mch_unique_customers_7d', 'step', 'time_since_last_pair_tx']
    
    payload = []
    for _, row in test_df.iterrows():
        record = {}
        for col in required_columns:
            if col in test_df.columns:
                record[col] = row[col] if not pd.isna(row[col]) else 0
            else:
                if col in ['amount', 'cust_tx_count_1d', 'first_time_pair', 'time_since_last_pair_tx']:
                    record[col] = 0
                elif col in ['age', 'cust_tx_count_7d', 'step']:
                    record[col] = 1
                elif col in ['amount_over_cust_median_7d', 'cust_median_amt_7d', 'log_amount']:
                    record[col] = 0.0
                elif col in ['customer', 'gender']:
                    record[col] = "unknown"
                else:
                    record[col] = ""
        payload.append(record)
    
    print(f"📡 Getting predictions for {len(payload)} records...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            results = response.json()["data"]
            alerts = []
            all_predictions = []
            
            for i, item in enumerate(results):
                fraud_prob = None
                for pred_val in item["predictionValues"]:
                    if pred_val["label"] == 1:
                        fraud_prob = pred_val["value"]
                        break
                
                all_predictions.append({
                    'record_id': i,
                    'fraud_probability': fraud_prob,
                    'actual_fraud': test_df.iloc[i]['fraud'] if 'fraud' in test_df.columns else 0
                })
                
                if fraud_prob and fraud_prob > 0.15:
                    alerts.append({
                        'record_id': i,
                        'fraud_probability': fraud_prob,
                        'raw_data': test_df.iloc[i].to_dict(),
                        'risk_level': 'CRITICAL' if fraud_prob > 0.95 else 'HIGH' if fraud_prob > 0.8 else 'MEDIUM',
                        'customer_id': test_df.iloc[i].get('customer', 'Unknown'),
                        'merchant_id': test_df.iloc[i].get('merchant', 'Unknown'),
                        'step': test_df.iloc[i].get('step', 'Unknown')
                    })
            
            print(f"✅ Received {len(alerts)} alerts from {len(results)} predictions")
            return alerts, all_predictions
        else:
            print(f"❌ API Error: {response.status_code}")
            return [], []
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return [], []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: x.strip("'") if isinstance(x, str) else x)
                if col in ['amount', 'cust_tx_count_1d', 'first_time_pair', 'time_since_last_pair_tx']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df = df.reset_index().rename(columns={'index': 'original_index'})
            
            alerts, predictions = get_datarobot_predictions(df)
            enhanced_alerts, stats = create_95percent_recall_justifications(df, alerts)
            
            y_true = [pred['actual_fraud'] for pred in predictions]
            enhanced_alert_ids = set([alert['record_id'] for alert in enhanced_alerts])
            y_pred = [1 if i in enhanced_alert_ids else 0 for i in range(len(predictions))]
            
            tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
            fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
            fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            alerts_data = []
            for alert in enhanced_alerts[:50]:
                alert_data = {
                    'record_id': int(alert['record_id']),
                    'probability': f"{alert['fraud_probability']:.1%}",
                    'amount': f"${alert['raw_data'].get('amount', 0):.2f}",
                    'category': alert['raw_data'].get('category', 'Unknown'),
                    'risk_factors': int(alert['risk_factors']),
                    'confidence': f"{alert['confidence_score']:.1%}",
                    'justifications': alert['advanced_justifications'],
                    'customer_id': alert['raw_data'].get('customer', 'Unknown'),
                    'merchant_id': alert['raw_data'].get('merchant', 'Unknown'),
                    'step': str(alert['raw_data'].get('step', 'Unknown'))
                }
                alerts_data.append(alert_data)
            
            dashboard_data = {
                'total_transactions': int(len(df)),
                'fraud_cases': int(sum(y_true)),
                'fraud_rate': f"{sum(y_true)/len(df):.2%}",
                'alerts_generated': int(len(enhanced_alerts)),
                'fraud_caught': int(tp),
                'false_alerts': int(fp),
                'missed_fraud': int(fn),
                'precision': f"{precision:.1%}",
                'recall': f"{recall:.1%}",
                'alert_efficiency': f"{tp/len(enhanced_alerts):.1%}" if enhanced_alerts else "0%"
            }
            
            return jsonify({'success': True, 'alerts': alerts_data, 'dashboard': dashboard_data})
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Get port from environment variable (for Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)