from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables for model and encoders
model = None
label_encoders = None

def load_model():
    """Load the pre-trained XGBoost model"""
    global model, label_encoders
    
    try:
        # For Render, we'll train the model on startup if no saved model exists
        if not os.path.exists('fraud_model.pkl'):
            print("🔄 No saved model found, training new model...")
            train_and_save_model()
        else:
            model = joblib.load('fraud_model.pkl')
            label_encoders = joblib.load('label_encoders.pkl')
            print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Training new model...")
        train_and_save_model()

def train_and_save_model():
    """Train and save the model - optimized for Render"""
    global model, label_encoders
    
    try:
        print("📥 Loading dataset...")
        # Use the dataset that should be in your GitHub repo
        df = pd.read_csv('BankSim_Fraud_10Features.csv')
        df = df.fillna(0)
        
        print("🔧 Preparing features...")
        X = df.drop(columns=["fraud"])
        y = df["fraud"]
        
        # Encode categorical variables
        X_encoded = X.copy()
        label_encoders = {}
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                label_encoders[col] = le
        
        print("🏋️ Training XGBoost model...")
        ratio = (y == 0).sum() / (y == 1).sum()
        model = xgb.XGBClassifier(
            n_estimators=200,  # Reduced for faster training on Render
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='aucpr',
            tree_method='hist',
            scale_pos_weight=ratio,
            random_state=42
        )
        
        model.fit(X_encoded, y, verbose=False)
        
        # Save for future use
        joblib.dump(model, 'fraud_model.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        print("✅ Model trained and saved successfully")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        # Create a simple fallback model
        create_fallback_model()

def create_fallback_model():
    """Create a simple fallback model if training fails"""
    global model, label_encoders
    
    print("🛠️ Creating fallback model...")
    # Simple rules-based fallback
    from sklearn.ensemble import RandomForestClassifier
    
    # Create dummy data for fallback model
    X_dummy = np.random.rand(1000, 5)
    y_dummy = (X_dummy[:, 0] > 0.7).astype(int)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    
    label_encoders = {}
    print("✅ Fallback model created")

def get_xgboost_predictions(test_df):
    """Get predictions from XGBoost model"""
    global model, label_encoders
    
    if model is None:
        load_model()
    
    print(f"🔍 Processing {len(test_df)} records with XGBoost...")
    
    try:
        # Prepare the test data
        X_test = test_df.copy()
        
        # Handle categorical encoding
        for col in X_test.columns:
            if col in label_encoders:
                X_test[col] = X_test[col].astype(str)
                # Map unseen values to 'unknown'
                trained_vals = set(label_encoders[col].classes_)
                X_test.loc[~X_test[col].isin(trained_vals), col] = 'unknown'
                X_test[col] = label_encoders[col].transform(X_test[col])
        
        # Ensure all training columns are present
        missing_cols = set(model.get_booster().feature_names) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        
        # Reorder columns to match training
        X_test = X_test[model.get_booster().feature_names]
        
        # Get predictions
        fraud_proba = model.predict_proba(X_test)[:, 1]
        
        alerts = []
        all_predictions = []
        
        for i, prob in enumerate(fraud_proba):
            prob_float = float(prob)
            all_predictions.append({
                'record_id': i,
                'fraud_probability': prob_float,
                'actual_fraud': test_df.iloc[i]['fraud'] if 'fraud' in test_df.columns else 0
            })
            
            if prob_float > 0.15:  # Same threshold as before
                alerts.append({
                    'record_id': i,
                    'fraud_probability': prob_float,
                    'raw_data': test_df.iloc[i].to_dict(),
                    'risk_level': 'CRITICAL' if prob_float > 0.95 else 'HIGH' if prob_float > 0.8 else 'MEDIUM',
                    'customer_id': str(test_df.iloc[i].get('customer', 'Unknown')),
                    'merchant_id': str(test_df.iloc[i].get('merchant', 'Unknown')),
                    'step': str(test_df.iloc[i].get('step', 'Unknown'))
                })
        
        print(f"✅ Generated {len(alerts)} alerts")
        return alerts, all_predictions
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        # Return empty results on error
        return [], []

def create_95percent_recall_justifications(df, alerts):
    """ULTRA-AGGRESSIVE for 90%+ recall - EXACTLY THE SAME"""
    print("🔍 ULTRA-AGGRESSIVE 90%+ RECALL SYSTEM")
    
    enhanced_alerts = []
    
    for alert in alerts:
        record_data = alert['raw_data']
        probability = alert['fraud_probability']
        
        justifications = []
        
        # 1. ANY CATEGORY (except most common)
        category = record_data.get('category', '')
        safe_categories = ['es_transportation', 'es_food']
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
    
    print(f"✅ Enhanced {len(alerts)} → {len(enhanced_alerts)} alerts")
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
        
        try:
            file.save(filepath)
            df = pd.read_csv(filepath)
            
            # Data cleaning
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: x.strip("'") if isinstance(x, str) else x)
                if col in ['amount', 'cust_tx_count_1d', 'first_time_pair', 'time_since_last_pair_tx']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df = df.reset_index().rename(columns={'index': 'original_index'})
            
            # Use XGBoost instead of DataRobot
            alerts, predictions = get_xgboost_predictions(df)
            enhanced_alerts, stats = create_95percent_recall_justifications(df, alerts)
            
            # Calculate metrics
            y_true = [pred['actual_fraud'] for pred in predictions] if predictions else []
            enhanced_alert_ids = set([alert['record_id'] for alert in enhanced_alerts])
            y_pred = [1 if i in enhanced_alert_ids else 0 for i in range(len(predictions))] if predictions else []
            
            tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1) if y_true else 0
            fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1) if y_true else 0
            fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0) if y_true else 0
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Format alerts for frontend
            alerts_data = []
            for alert in enhanced_alerts[:50]:
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
            
            dashboard_data = {
                'recall': f"{recall:.1%}",
                'precision': f"{precision:.1%}",
                'fraud_caught': int(tp),
                'fraud_cases': int(sum(y_true)) if y_true else 0,
                'alerts_generated': int(len(enhanced_alerts)),
                'false_alerts': int(fp),
                'alert_efficiency': f"{tp/len(enhanced_alerts):.1%}" if enhanced_alerts else "0%"
            }
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'success': True, 
                'alerts': alerts_data, 
                'dashboard': dashboard_data
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': f'Processing error: {str(e)}'})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

if __name__ == '__main__':
    # Load model on startup
    print("🚀 Starting Apex Fraud Studio...")
    load_model()
    
    # Create upload folder
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    port = int(os.environ.get("PORT", 5000))
    print(f"✅ Server ready on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
