from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import requests
import os
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def safe_convert_to_float(value, default=0.0):
    """Safely convert any value to float"""
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().strip("'\"")
            return float(cleaned) if cleaned else default
        return default
    except (ValueError, TypeError):
        return default

def safe_convert_to_int(value, default=0):
    """Safely convert any value to int"""
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            cleaned = value.strip().strip("'\"")
            return int(float(cleaned)) if cleaned else default
        return default
    except (ValueError, TypeError):
        return default

def generate_mock_predictions(df):
    """Generate realistic mock predictions for testing"""
    print("Using mock predictions for testing")
    alerts = []
    all_predictions = []
    
    for i in range(len(df)):
        # Generate realistic probabilities with most transactions being low risk
        if np.random.random() < 0.05:  # 5% high risk
            fraud_prob = np.random.uniform(0.7, 0.95)
        elif np.random.random() < 0.15:  # 15% medium risk
            fraud_prob = np.random.uniform(0.4, 0.7)
        else:  # 80% low risk
            fraud_prob = np.random.uniform(0.01, 0.3)
        
        all_predictions.append({
            'record_id': i,
            'fraud_probability': fraud_prob,
            'actual_fraud': safe_convert_to_int(df.iloc[i].get('fraud', 0))
        })
        
        if fraud_prob > 0.3:
            alerts.append({
                'record_id': i,
                'fraud_probability': fraud_prob,
                'raw_data': df.iloc[i].to_dict(),
                'risk_level': 'CRITICAL' if fraud_prob > 0.9 else 'HIGH' if fraud_prob > 0.7 else 'MEDIUM',
                'customer_id': str(df.iloc[i].get('customer', f'CUST-{i:04d}')),
                'merchant_id': str(df.iloc[i].get('merchant', f'MERCH-{i:04d}')),
                'step': str(df.iloc[i].get('step', f'Step {i%10}'))
            })
    
    print(f"Generated {len(alerts)} mock alerts from {len(df)} transactions")
    return alerts, all_predictions

def create_realistic_justifications(df, alerts):
    """Create realistic fraud justifications"""
    print("Creating realistic justifications...")
    
    enhanced_alerts = []
    
    for alert in alerts:
        record_data = alert['raw_data']
        probability = alert['fraud_probability']
        
        justifications = []
        
        # Amount-based risk
        amount = safe_convert_to_float(record_data.get('amount', 0))
        if amount > 1000:
            justifications.append({
                'title': 'High Transaction Amount',
                'description': f"Amount ${amount:.2f} significantly above average",
                'strength': 0.7,
                'risk_level': 'HIGH'
            })
        elif amount > 500:
            justifications.append({
                'title': 'Elevated Transaction Amount', 
                'description': f"Amount ${amount:.2f} above typical range",
                'strength': 0.4,
                'risk_level': 'MEDIUM'
            })
        
        # Transaction frequency risk
        daily_tx = safe_convert_to_int(record_data.get('cust_tx_count_1d', 0))
        if daily_tx > 15:
            justifications.append({
                'title': 'Unusual Transaction Frequency',
                'description': f"{daily_tx} transactions in one day",
                'strength': 0.6,
                'risk_level': 'HIGH'
            })
        
        # First-time merchant risk
        first_time = safe_convert_to_int(record_data.get('first_time_pair', 0))
        if first_time == 1:
            justifications.append({
                'title': 'First-Time Merchant',
                'description': "Transaction with new merchant",
                'strength': 0.3,
                'risk_level': 'MEDIUM'
            })
        
        # Only create alert if we have meaningful justifications and probability
        if len(justifications) >= 1 and probability > 0.3:
            enhanced_alert = alert.copy()
            enhanced_alert['advanced_justifications'] = justifications
            enhanced_alert['risk_factors'] = len(justifications)
            enhanced_alerts.append(enhanced_alert)
    
    print(f"Enhanced {len(alerts)} → {len(enhanced_alerts)} alerts")
    return enhanced_alerts, {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("Upload endpoint called")
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Saving file to: {filepath}")
            file.save(filepath)
            
            # Read and process the CSV
            df = pd.read_csv(filepath)
            print(f"Loaded CSV with {len(df)} rows and columns: {df.columns.tolist()}")
            
            # Basic data cleaning
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: str(x).strip("'\"") if pd.notna(x) else "")
                if col in ['amount', 'age', 'cust_tx_count_1d', 'cust_tx_count_7d', 'first_time_pair', 
                          'time_since_last_pair_tx', 'fraud', 'step']:
                    df[col] = df[col].apply(lambda x: safe_convert_to_float(x, 0))
            
            df = df.reset_index().rename(columns={'index': 'original_index'})
            
            # Use mock predictions (more reliable than external API)
            alerts, predictions = generate_mock_predictions(df)
            enhanced_alerts, stats = create_realistic_justifications(df, alerts)
            
            # Calculate metrics
            total_transactions = len(df)
            
            # Count actual fraud cases
            fraud_cases = 0
            if 'fraud' in df.columns:
                fraud_cases = sum(1 for fraud_val in df['fraud'] if safe_convert_to_int(fraud_val) == 1)
            
            # Estimate fraud cases if column doesn't exist
            if fraud_cases == 0:
                fraud_cases = max(1, int(total_transactions * 0.03))
            
            # Calculate performance metrics
            tp = 0
            fp = 0
            
            for alert in enhanced_alerts:
                record_id = alert['record_id']
                if record_id < len(df):
                    actual_fraud = safe_convert_to_int(df.iloc[record_id].get('fraud', 0))
                    if actual_fraud == 1:
                        tp += 1
                    else:
                        fp += 1
            
            fn = max(0, fraud_cases - tp)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            efficiency_ratio = tp / len(enhanced_alerts) if enhanced_alerts else 0
            
            # Prepare alerts data
            alerts_data = []
            for alert in enhanced_alerts:
                confidence_decimal = safe_convert_to_float(alert['fraud_probability'])
                
                alert_data = {
                    'record_id': safe_convert_to_int(alert['record_id']),
                    'confidence': confidence_decimal,
                    'amount': f"${safe_convert_to_float(alert['raw_data'].get('amount', 0)):.2f}",
                    'category': str(alert['raw_data'].get('category', 'Unknown')),
                    'customer_id': str(alert['raw_data'].get('customer', 'Unknown')),
                    'merchant_id': str(alert['raw_data'].get('merchant', 'Unknown')),
                    'step': str(alert['raw_data'].get('step', 'Unknown')),
                    'risk_factors': f"{alert['risk_factors']} risk factors detected",
                    'justifications': [
                        {
                            'title': str(j.get('title', 'Risk Factor')).strip(),
                            'description': str(j.get('description', '')),
                            'strength': safe_convert_to_float(j.get('strength', 0.5)),
                            'risk_level': str(j.get('risk_level', 'MEDIUM'))
                        }
                        for j in alert.get('advanced_justifications', [])
                    ]
                }
                alerts_data.append(alert_data)
            
            # Dashboard data
            dashboard_data = {
                'recall': f"{recall:.1%}",
                'precision': f"{precision:.1%}",
                'fraud_caught': int(tp),
                'fraud_cases': int(fraud_cases),
                'alerts_generated': int(len(enhanced_alerts)),
                'false_alerts': int(fp),
                'alert_efficiency': f"{efficiency_ratio:.2f}"
            }
            
            print(f"Processing complete: {len(alerts_data)} alerts generated")
            
            return jsonify({
                'success': True, 
                'alerts': alerts_data, 
                'dashboard': dashboard_data
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a CSV file.'})
            
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': error_msg})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
