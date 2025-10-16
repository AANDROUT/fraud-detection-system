from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global model variables
model = None
scaler = None
feature_columns = None
label_encoders = {}

def initialize_model():
    """Initialize the model using pre-engineered features from BankSim parts"""
    global model, scaler, feature_columns
    
    try:
        print("🔄 Initializing Fraud Detection Model...")
        
        # Load and combine all BankSim parts (which already have engineered features)
        parts = []
        for i in range(1, 7):
            try:
                part_df = pd.read_csv(f"BankSim_part_{i}.csv")
                parts.append(part_df)
                print(f"✅ Loaded BankSim_part_{i}.csv with {len(part_df)} records")
                print(f"   Columns: {list(part_df.columns)}")
            except FileNotFoundError:
                print(f"⚠️ BankSim_part_{i}.csv not found, skipping...")
                continue
        
        if not parts:
            raise Exception("No BankSim data files found")
        
        # Combine datasets
        df = pd.concat(parts, ignore_index=True)
        print(f"📊 Total dataset: {len(df)} records")
        
        # Define feature columns (same as your DataRobot requirements)
        feature_columns = [
            'age', 'amount', 'amount_over_cust_median_7d', 'category', 
            'cust_median_amt_7d', 'cust_tx_count_1d', 'cust_tx_count_7d', 
            'cust_unique_merchants_30d', 'customer', 'first_time_pair', 'gender', 
            'log_amount', 'mch_tx_count_1d', 'mch_unique_customers_7d', 
            'step', 'time_since_last_pair_tx'
        ]
        
        # Check which features actually exist in the data
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"🔍 Available features: {available_features}")
        
        # Prepare features
        X = df[available_features].copy()
        
        # Handle categorical variables
        categorical_cols = ['category', 'gender', 'customer']
        for col in categorical_cols:
            if col in X.columns:
                label_encoders[col] = LabelEncoder()
                X[col] = label_encoders[col].fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(0)
        
        # Create target variable (assuming 'fraud' column exists)
        if 'fraud' not in df.columns:
            raise Exception("'fraud' column not found in dataset")
        
        y = df['fraud'].copy()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest model (your 3rd code model)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_scaled, y)
        
        print(f"✅ Model trained successfully! Fraud rate in training: {y.mean():.3f}")
        feature_columns = available_features  # Update with actual available features
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_local_predictions(test_df):
    """Get predictions using the local Random Forest model (replaces DataRobot API)"""
    global model, scaler, feature_columns
    
    if model is None:
        print("❌ Model not loaded")
        return [], []
    
    try:
        print(f"🔍 Getting local predictions for {len(test_df)} records...")
        
        # Use available features that exist in test data
        available_features = [col for col in feature_columns if col in test_df.columns]
        
        # Prepare features
        X_test = test_df[available_features].copy()
        
        # Handle categorical variables
        for col in ['category', 'gender', 'customer']:
            if col in X_test.columns and col in label_encoders:
                # Transform using pre-fitted label encoders
                X_test[col] = label_encoders[col].transform(X_test[col].astype(str))
        
        # Handle missing values
        X_test = X_test.fillna(0)
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Get predictions
        fraud_probabilities = model.predict_proba(X_test_scaled)[:, 1]
        
        alerts = []
        all_predictions = []
        
        for i, prob in enumerate(fraud_probabilities):
            all_predictions.append({
                'record_id': i,
                'fraud_probability': prob,
                'actual_fraud': test_df.iloc[i]['fraud'] if 'fraud' in test_df.columns else 0
            })
            
            if prob > 0.15:  # Same threshold as your original code
                alerts.append({
                    'record_id': i,
                    'fraud_probability': prob,
                    'raw_data': test_df.iloc[i].to_dict(),
                    'risk_level': 'CRITICAL' if prob > 0.95 else 'HIGH' if prob > 0.8 else 'MEDIUM',
                    'customer_id': test_df.iloc[i].get('customer', 'Unknown'),
                    'merchant_id': test_df.iloc[i].get('merchant', 'Unknown'),
                    'step': test_df.iloc[i].get('step', 'Unknown')
                })
        
        print(f"✅ Generated {len(alerts)} alerts from {len(fraud_probabilities)} predictions")
        return alerts, all_predictions
        
    except Exception as e:
        print(f"❌ Local prediction error: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def create_95percent_recall_justifications(df, alerts):
    """ULTRA-AGGRESSIVE for 90%+ recall - EXACT SAME AS YOUR ORIGINAL CODE"""
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
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
            
            # USE LOCAL MODEL INSTEAD OF DATAROBOT API
            alerts, predictions = get_local_predictions(df)
            enhanced_alerts, stats = create_95percent_recall_justifications(df, alerts)
            
            y_true = [pred['actual_fraud'] for pred in predictions]
            enhanced_alert_ids = set([alert['record_id'] for alert in enhanced_alerts])
            y_pred = [1 if i in enhanced_alert_ids else 0 for i in range(len(predictions))]
            
            tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
            fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
            fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Convert alerts to match HTML expected structure
            alerts_data = []
            for alert in enhanced_alerts[:50]:  # Limit to 50 alerts
                confidence_decimal = float(alert['fraud_probability'])
                
                alert_data = {
                    'record_id': int(alert['record_id']),
                    'confidence': confidence_decimal,
                    'amount': f"${alert['raw_data'].get('amount', 0):.2f}",
                    'category': alert['raw_data'].get('category', 'Unknown'),
                    'customer_id': alert['raw_data'].get('customer', 'Unknown'),
                    'merchant_id': alert['raw_data'].get('merchant', 'Unknown'),
                    'step': str(alert['raw_data'].get('step', 'Unknown')),
                    'risk_factors': f"{alert['risk_factors']} risk factors detected",
                    'justifications': [
                        {
                            'title': j.get('title', 'Risk Factor').replace('🎯', '').replace('💰', '').replace('⚡', '').replace('🆕', '').replace('📈', '').strip(),
                            'description': j.get('description', ''),
                            'strength': j.get('strength', 0.5),
                            'risk_level': j.get('risk_level', 'MEDIUM')
                        }
                        for j in alert.get('advanced_justifications', [])
                    ]
                }
                alerts_data.append(alert_data)
            
            # Dashboard data with correct field names
            dashboard_data = {
                'recall': f"{recall:.1%}",
                'precision': f"{precision:.1%}",
                'fraud_caught': int(tp),
                'fraud_cases': int(sum(y_true)),
                'alerts_generated': int(len(enhanced_alerts)),
                'false_alerts': int(fp),
                'alert_efficiency': f"{tp/len(enhanced_alerts):.1%}" if enhanced_alerts else "0%"
            }
            
            return jsonify({
                'success': True, 
                'alerts': alerts_data, 
                'dashboard': dashboard_data
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Processing error: {str(e)}'})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Initialize the model on startup
    print("🚀 Starting Fraud Alert System...")
    if initialize_model():
        print("✅ System ready! Model loaded successfully.")
    else:
        print("⚠️ System started with limited functionality (no model)")
    
    # Get port from environment variable (for Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
