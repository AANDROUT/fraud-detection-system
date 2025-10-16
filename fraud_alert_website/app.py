from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables for the model and threshold
fraud_model = None
model_threshold = 0.15  # Default threshold

def load_or_train_model():
    """Load or train the fraud detection model"""
    global fraud_model, model_threshold
    
    model_path = 'fraud_model.pkl'
    threshold_path = 'model_threshold.pkl'
    
    try:
        # Try to load existing model
        if os.path.exists(model_path) and os.path.exists(threshold_path):
            fraud_model = joblib.load(model_path)
            model_threshold = joblib.load(threshold_path)
            print("✅ Model loaded successfully")
            return
    except Exception as e:
        print(f"❌ Could not load existing model: {e}")
    
    # Train new model
    try:
        print("🔄 Training new fraud detection model...")
        
        # Try to load your actual training data
        training_file = "BankSim_Fraud_10Features_Sample.csv"
        if os.path.exists(training_file):
            print(f"📊 Loading training data from {training_file}...")
            df = pd.read_csv(training_file)
            df = df.fillna(0)
        else:
            # Fallback to synthetic data if file not found
            print("❌ Training file not found, creating sample data...")
            n_samples = 5000  # Smaller sample for faster training
            np.random.seed(42)
            
            training_data = {
                'age': np.random.randint(18, 80, n_samples),
                'amount': np.random.exponential(50, n_samples),
                'amount_over_cust_median_7d': np.random.normal(0, 10, n_samples),
                'category': np.random.choice(['es_transportation', 'es_food', 'es_other'], n_samples),
                'cust_median_amt_7d': np.random.exponential(30, n_samples),
                'cust_tx_count_1d': np.random.poisson(1, n_samples),
                'cust_tx_count_7d': np.random.poisson(5, n_samples),
                'cust_unique_merchants_30d': np.random.poisson(3, n_samples),
                'customer': [f'cust_{i}' for i in range(n_samples)],
                'first_time_pair': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                'gender': np.random.choice(['M', 'F'], n_samples),
                'log_amount': np.log(np.random.exponential(50, n_samples) + 1),
                'mch_tx_count_1d': np.random.poisson(10, n_samples),
                'mch_unique_customers_7d': np.random.poisson(15, n_samples),
                'step': np.random.randint(1, 100, n_samples),
                'time_since_last_pair_tx': np.random.exponential(20, n_samples),
                'fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])  # 2% fraud rate
            }
            
            df = pd.DataFrame(training_data)
        
        # MAKE SURE THIS LINE AND BELOW ARE PROPERLY INDENTED
        print("Dataset shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Fraud distribution:", df['fraud'].value_counts())
        
        # Split features and target
        X = df.drop(columns=["fraud"])
        y = df["fraud"]
        
        # Convert categorical variables to numeric codes
        categorical_cols = ['category', 'customer', 'gender', 'merchant']
        for col in categorical_cols:
            if col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    X[col] = X[col].astype('category').cat.codes
        
        # Convert age to numeric if it's object type
        if 'age' in X.columns and X['age'].dtype == 'object':
            X['age'] = pd.to_numeric(X['age'], errors='coerce').fillna(0)
        
        # Convert all columns to numeric to avoid type issues
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Train/test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )
        
        print("Train size:", X_train.shape, "Validation size:", X_val.shape)
        
        # Calculate class imbalance ratio
        ratio = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
        print(f"Class imbalance ratio: {ratio:.2f}")
        
        # Train XGBoost model with categorical support
        fraud_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='aucpr',
            tree_method='hist',
            scale_pos_weight=ratio,
            random_state=42,
            enable_categorical=True  # Add this line for categorical support
        )
        
        fraud_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Find optimal threshold
        val_proba = fraud_model.predict_proba(X_val)[:, 1]
        prec, rec, thr = precision_recall_curve(y_val, val_proba)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best = f1.argmax()
        model_threshold = float(thr[max(best - 1, 0)]) if len(thr) > 0 else 0.15
        
        print(f"✅ Model trained successfully with threshold: {model_threshold:.3f}")
        
        # Save model and threshold
        joblib.dump(fraud_model, model_path)
        joblib.dump(model_threshold, threshold_path)
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        # Fallback to logistic regression
        print("🔄 Falling back to logistic regression...")
        fraud_model = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        fraud_model.fit(X_train, y_train)
        model_threshold = 0.15
        
def get_local_predictions(test_df):
    """Get predictions using local XGBoost model"""
    global fraud_model, model_threshold
    
    if fraud_model is None:
        load_or_train_model()
    
    print(f"🔍 Getting local predictions for {len(test_df)} records...")
    
    try:
        # Prepare features (ensure we have the right columns)
        expected_features = ['age', 'amount', 'amount_over_cust_median_7d', 'category', 
                           'cust_median_amt_7d', 'cust_tx_count_1d', 'cust_tx_count_7d', 
                           'cust_unique_merchants_30d', 'customer', 'first_time_pair', 
                           'gender', 'log_amount', 'mch_tx_count_1d', 'mch_unique_customers_7d', 
                           'step', 'time_since_last_pair_tx']
        
        # Create a copy and fill missing columns with defaults
        prediction_df = test_df.copy()
        
        for col in expected_features:
            if col not in prediction_df.columns:
                if col in ['amount', 'cust_tx_count_1d', 'first_time_pair', 'time_since_last_pair_tx']:
                    prediction_df[col] = 0
                elif col in ['age', 'cust_tx_count_7d', 'step']:
                    prediction_df[col] = 1
                elif col in ['amount_over_cust_median_7d', 'cust_median_amt_7d', 'log_amount']:
                    prediction_df[col] = 0.0
                elif col in ['customer', 'gender']:
                    prediction_df[col] = "unknown"
                elif col == 'category':
                    prediction_df[col] = "es_other"
                else:
                    prediction_df[col] = ""
        
        # Ensure we only use the expected features in the right order
        prediction_df = prediction_df[expected_features]
        
               # Convert categorical variables if needed
        for col in ['category', 'customer', 'gender', 'merchant']:
            if col in prediction_df.columns:
                if prediction_df[col].dtype == 'object' or prediction_df[col].dtype.name == 'category':
                    prediction_df[col] = prediction_df[col].astype('category').cat.codes
        
        # Convert age to numeric if it's object type
        if 'age' in prediction_df.columns and prediction_df['age'].dtype == 'object':
            prediction_df['age'] = pd.to_numeric(prediction_df['age'], errors='coerce').fillna(0)
        
        # Convert all columns to numeric to avoid type issues
        for col in prediction_df.columns:
            if prediction_df[col].dtype == 'object':
                prediction_df[col] = pd.to_numeric(prediction_df[col], errors='coerce').fillna(0)
        
        # Get predictions
        fraud_proba = fraud_model.predict_proba(prediction_df)[:, 1]
        
        alerts = []
        all_predictions = []
        
        for i, prob in enumerate(fraud_proba):
            actual_fraud = test_df.iloc[i]['fraud'] if 'fraud' in test_df.columns else 0
            all_predictions.append({
                'record_id': i,
                'fraud_probability': float(prob),
                'actual_fraud': actual_fraud
            })
            
            if prob > model_threshold:
                alerts.append({
                    'record_id': i,
                    'fraud_probability': float(prob),
                    'raw_data': test_df.iloc[i].to_dict(),
                    'risk_level': 'CRITICAL' if prob > 0.95 else 'HIGH' if prob > 0.8 else 'MEDIUM',
                    'customer_id': test_df.iloc[i].get('customer', 'Unknown'),
                    'merchant_id': test_df.iloc[i].get('merchant', 'Unknown'),
                    'step': test_df.iloc[i].get('step', 'Unknown')
                })
        
        print(f"✅ Local model: {len(alerts)} alerts from {len(fraud_proba)} predictions (threshold: {model_threshold:.3f})")
        return alerts, all_predictions
        
    except Exception as e:
        print(f"❌ Local prediction error: {e}")
        return [], []

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
            
            # Use local model instead of DataRobot
            alerts, predictions = get_local_predictions(df)
            enhanced_alerts, stats = create_95percent_recall_justifications(df, alerts)
            
            # Calculate metrics
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
    
    # Load or train model on startup
    print("🚀 Starting Flask app - loading fraud detection model...")
    load_or_train_model()
    
    # Get port from environment variable (for Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)



