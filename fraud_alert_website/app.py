from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import xgboost as xgb

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables for model and encoders
model = None
label_encoders = None
optimal_threshold = None  # Store the optimal threshold

def load_model():
    """Load the pre-trained XGBoost model"""
    global model, label_encoders, optimal_threshold
    
    try:
        if not os.path.exists('fraud_model.pkl'):
            print("🔄 No saved model found, training new model...")
            train_and_save_model()
        else:
            model = joblib.load('fraud_model.pkl')
            label_encoders = joblib.load('label_encoders.pkl')
            optimal_threshold = joblib.load('optimal_threshold.pkl')
            print("✅ Model loaded successfully")
            print(f"🎯 Using optimal threshold: {optimal_threshold}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Training new model...")
        train_and_save_model()

def load_dataset_from_parts():
    """Load the full dataset from 6 parts"""
    print("📥 Loading dataset from parts...")
    
    parts = []
    for i in range(1, 7):
        filename = f'BankSim_part_{i}.csv'
        try:
            part_df = pd.read_csv(filename)
            parts.append(part_df)
            print(f"✅ Loaded {filename}: {len(part_df)} records")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
    
    if not parts:
        raise Exception("Could not load any dataset parts")
    
    full_df = pd.concat(parts, ignore_index=True)
    print(f"📊 Combined dataset: {len(full_df)} total records")
    return full_df

def train_and_save_model():
    """Train and save the model using the 6-part dataset"""
    global model, label_encoders, optimal_threshold
    
    try:
        print("📥 Loading dataset from parts...")
        df = load_dataset_from_parts()
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
        
        # CRITICAL FIX: Use same hyperparameters as Jupyter notebook
        model = xgb.XGBClassifier(
            n_estimators=1000,        # Changed from 200 to 1000
            learning_rate=0.05,       # Changed from 0.1 to 0.05
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='aucpr',
            tree_method='hist',
            scale_pos_weight=ratio,
            random_state=42
        )
        
        # Use proper train/validation split like Jupyter
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_encoded, y, test_size=0.30, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )
        
        print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}, Test size: {X_test.shape}")
        
        # Train with validation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Find optimal threshold like in Jupyter
        val_proba = model.predict_proba(X_val)[:, 1]
        prec, rec, thr = precision_recall_curve(y_val, val_proba)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best = f1.argmax()
        optimal_threshold = float(thr[max(best - 1, 0)])
        
        print(f"🎯 Optimal threshold found: {optimal_threshold}")
        
        # Save for future use
        joblib.dump(model, 'fraud_model.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        joblib.dump(optimal_threshold, 'optimal_threshold.pkl')
        print("✅ Model trained and saved successfully")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        create_fallback_model()

def create_fallback_model():
    """Create a simple fallback model if training fails"""
    global model, label_encoders, optimal_threshold
    
    print("🛠️ Creating fallback model...")
    from sklearn.ensemble import RandomForestClassifier
    
    # Create dummy data for fallback model
    X_dummy = np.random.rand(1000, 5)
    y_dummy = (X_dummy[:, 0] > 0.7).astype(int)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    
    label_encoders = {}
    optimal_threshold = 0.5
    print("✅ Fallback model created")

def get_xgboost_predictions(test_df):
    """Get predictions from XGBoost model with PROPER encoding"""
    global model, label_encoders, optimal_threshold
    
    if model is None:
        load_model()
    
    print(f"🔍 Processing {len(test_df)} records with XGBoost...")
    
    try:
        # Prepare the test data
        X_test = test_df.copy()
        if 'fraud' in X_test.columns:
            X_test = X_test.drop(columns=['fraud'])
            print("✅ Dropped 'fraud' column from features")
        
        print(f"📊 X_test shape before encoding: {X_test.shape}")
        print(f"🔍 Data types: {X_test.dtypes}")
        
        # CRITICAL FIX: Use EXACT same encoding as training
        for col in X_test.columns:
            if col in label_encoders:
                print(f"🔧 Properly encoding column: {col}")
                # Convert to string and use the trained LabelEncoder
                X_test[col] = X_test[col].astype(str)
                
                # Handle unseen categories by mapping them to most common class
                unseen_mask = ~X_test[col].isin(label_encoders[col].classes_)
                if unseen_mask.any():
                    print(f"⚠️  Mapping {unseen_mask.sum()} unseen values in {col} to default")
                    # Map unseen values to the first class (usually most common)
                    X_test.loc[unseen_mask, col] = label_encoders[col].classes_[0]
                
                X_test[col] = label_encoders[col].transform(X_test[col])
        
        print(f"📊 X_test shape after encoding: {X_test.shape}")
        
        # Ensure all training columns are present
        if hasattr(model, 'get_booster'):
            expected_features = model.get_booster().feature_names
            
            missing_cols = set(expected_features) - set(X_test.columns)
            for col in missing_cols:
                X_test[col] = 0
                print(f"➕ Added missing column: {col}")
            
            X_test = X_test[expected_features]
        
        # Get predictions
        fraud_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"📊 Prediction range: {fraud_proba.min():.4f} to {fraud_proba.max():.4f}")
        print(f"📈 Alerts above optimal threshold ({optimal_threshold}): {np.sum(fraud_proba > optimal_threshold)}")
        print(f"📈 Alerts above 0.1: {np.sum(fraud_proba > 0.1)}")
        print(f"📈 Alerts above 0.05: {np.sum(fraud_proba > 0.05)}")
        
        alerts = []
        all_predictions = []

        # After getting predictions, add:
        print(f"🔍 PROBABILITY ANALYSIS:")
        print(f"   Min: {fraud_proba.min():.4f}")
        print(f"   Max: {fraud_proba.max():.4f}") 
        print(f"   Mean: {fraud_proba.mean():.4f}")
        print(f"   % > 0.9: {np.mean(fraud_proba > 0.9):.2%}")
        print(f"   % > 0.5: {np.mean(fraud_proba > 0.5):.2%}")
        print(f"   % > 0.1: {np.mean(fraud_proba > 0.1):.2%}")
        
        # Use the OPTIMAL threshold from training (not arbitrary low threshold)
        threshold = optimal_threshold
        
        for i, prob in enumerate(fraud_proba):
            prob_float = float(prob)
            all_predictions.append({
                'record_id': i,
                'fraud_probability': prob_float,
                'actual_fraud': test_df.iloc[i]['fraud'] if 'fraud' in test_df.columns else 0
            })
            
            if prob_float > threshold:
                alerts.append({
                    'record_id': i,
                    'fraud_probability': prob_float,
                    'raw_data': test_df.iloc[i].to_dict(),
                    'risk_level': 'CRITICAL' if prob_float > 0.8 else 'HIGH' if prob_float > 0.5 else 'MEDIUM' if prob_float > 0.2 else 'LOW',
                    'customer_id': str(test_df.iloc[i].get('customer', 'Unknown')),
                    'merchant_id': str(test_df.iloc[i].get('merchant', 'Unknown')),
                    'step': str(test_df.iloc[i].get('step', 'Unknown'))
                })
        
        print(f"✅ Generated {len(alerts)} alerts (optimal threshold: {threshold})")
        return alerts, all_predictions
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return [], []

# ... [Keep all the other functions the same: create_individualized_justifications, validate_uploaded_data, upload_file, etc.]

def create_individualized_justifications(df, alerts):
    """Create UNIQUE justifications for each alert based on actual feature values"""
    print(f"Creating INDIVIDUALIZED justifications for {len(alerts)} alerts")
    
    enhanced_alerts = []
    
    # Analyze feature distributions for dynamic thresholds
    feature_stats = {}
    for col in ['amount', 'cust_tx_count_1d', 'amount_over_cust_median_7d', 'time_since_last_pair_tx']:
        if col in df.columns:
            feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'q75': df[col].quantile(0.75),
                'q90': df[col].quantile(0.90)
            }
    
    for alert in alerts:
        record_data = alert['raw_data']
        probability = alert['fraud_probability']
        
        justifications = []
        
        # 1. DYNAMIC AMOUNT ANALYSIS (based on actual values)
        amount = record_data.get('amount', 0)
        if amount > feature_stats.get('amount', {}).get('q90', 500):
            strength = min(0.95, max(0.1, (amount - 200) / 1000))
            justifications.append({
                'category': 'EXTREME_AMOUNT',
                'feature': 'amount',
                'strength': strength,
                'title': 'Extreme Amount',
                'description': f"Amount ${amount:.2f} in top 10% of all transactions",
                'risk_level': 'CRITICAL',
                'context': f"Extremely high amount - ${amount:.2f} is unusually large"
            })
        elif amount > feature_stats.get('amount', {}).get('q75', 200):
            strength = min(0.7, max(0.1, (amount - 100) / 500))
            justifications.append({
                'category': 'HIGH_AMOUNT',
                'feature': 'amount',
                'strength': strength,
                'title': 'High Amount',
                'description': f"Amount ${amount:.2f} in top 25% of transactions",
                'risk_level': 'HIGH',
                'context': f"Above-average transaction amount of ${amount:.2f}"
            })
        elif amount > 50:
            justifications.append({
                'category': 'MODERATE_AMOUNT',
                'feature': 'amount',
                'strength': 0.4,
                'title': 'Moderate Amount',
                'description': f"Amount ${amount:.2f}",
                'risk_level': 'MEDIUM',
                'context': "Transaction amount carries moderate risk"
            })
        
        # 2. TRANSACTION FREQUENCY ANALYSIS (1 step = 1 day)
        daily_tx = record_data.get('cust_tx_count_1d', 0)
        if daily_tx > feature_stats.get('cust_tx_count_1d', {}).get('q90', 8):
            strength = min(0.9, max(0.1, daily_tx / 15))
            justifications.append({
                'category': 'EXTREME_FREQUENCY',
                'feature': 'cust_tx_count_1d',
                'strength': strength,
                'title': 'Extreme Daily Frequency',
                'description': f"{daily_tx} transactions in one day - extremely high activity",
                'risk_level': 'CRITICAL',
                'context': f"Customer made {daily_tx} transactions in a single day"
            })
        elif daily_tx > feature_stats.get('cust_tx_count_1d', {}).get('q75', 4):
            strength = min(0.6, max(0.1, daily_tx / 8))
            justifications.append({
                'category': 'HIGH_FREQUENCY',
                'feature': 'cust_tx_count_1d',
                'strength': strength,
                'title': 'High Daily Frequency',
                'description': f"{daily_tx} transactions today - elevated activity",
                'risk_level': 'HIGH',
                'context': f"Unusually high daily transaction count: {daily_tx}"
            })
        
        # 3. BEHAVIORAL ANOMALY DETECTION
        amount_over_median = record_data.get('amount_over_cust_median_7d', 0)
        if amount_over_median > 10:
            strength = min(0.95, max(0.1, amount_over_median / 20))
            justifications.append({
                'category': 'MAJOR_BEHAVIOR_CHANGE',
                'feature': 'amount_over_cust_median_7d',
                'strength': strength,
                'title': 'Major Behavioral Anomaly',
                'description': f"Amount {amount_over_median:.1f}x above customer's 7-day median",
                'risk_level': 'CRITICAL',
                'context': "Massive deviation from customer's typical spending pattern"
            })
        elif amount_over_median > 5:
            strength = min(0.8, max(0.1, amount_over_median / 10))
            justifications.append({
                'category': 'BEHAVIORAL_ANOMALY',
                'feature': 'amount_over_cust_median_7d',
                'strength': strength,
                'title': 'Behavioral Anomaly',
                'description': f"Amount {amount_over_median:.1f}x above 7-day median spending",
                'risk_level': 'HIGH',
                'context': "Significant deviation from customer's spending pattern"
            })
        elif amount_over_median > 2:
            justifications.append({
                'category': 'SPENDING_CHANGE',
                'feature': 'amount_over_cust_median_7d',
                'strength': 0.5,
                'title': 'Spending Pattern Change',
                'description': f"Amount {amount_over_median:.1f}x above 7-day median",
                'risk_level': 'MEDIUM',
                'context': "Different from customer's usual spending pattern"
            })
        
        # 4. TRANSACTION TIMING ANALYSIS (1 step = 1 day)
        time_since_last = record_data.get('time_since_last_pair_tx', -1)
        if 0 <= time_since_last < 0.1:  # Same day repeat (very rapid)
            justifications.append({
                'category': 'SAME_DAY_REPEAT',
                'feature': 'time_since_last_pair_tx',
                'strength': 0.9,
                'title': 'Same Day Repeat',
                'description': f"Repeat transaction within same day (step difference: {time_since_last:.1f})",
                'risk_level': 'CRITICAL',
                'context': "Extremely quick repeat transaction with same merchant on same day"
            })
        elif 0 <= time_since_last < 1.0:  # Within 1 day
            strength = 0.8 if time_since_last < 0.5 else 0.6
            justifications.append({
                'category': 'RAPID_REPEAT',
                'feature': 'time_since_last_pair_tx',
                'strength': strength,
                'title': 'Rapid Repeat',
                'description': f"Repeat within {time_since_last:.1f} days",
                'risk_level': 'HIGH',
                'context': "Quick repeat transaction pattern with same merchant"
            })
        elif 0 <= time_since_last < 3.0:  # Within 3 days
            justifications.append({
                'category': 'RECENT_REPEAT',
                'feature': 'time_since_last_pair_tx',
                'strength': 0.4,
                'title': 'Recent Repeat',
                'description': f"Repeat within {time_since_last:.1f} days",
                'risk_level': 'MEDIUM',
                'context': "Recent repeat transaction with same merchant"
            })
        
        # 5. CATEGORY-SPECIFIC RISK
        category = record_data.get('category', '')
        category_strength = {
            'es_sportsandtoys': 0.8, 'es_health': 0.7, 'es_wellnessandbeauty': 0.7,
            'es_fashion': 0.6, 'es_travel': 0.6, 'es_home': 0.5,
            'es_leisure': 0.5, 'es_otherservices': 0.4,
            'es_transportation': 0.3, 'es_food': 0.2, 'es_barsandrestaurants': 0.2
        }
        if category in category_strength:
            justifications.append({
                'category': 'CATEGORY_RISK',
                'feature': 'category',
                'strength': category_strength[category],
                'title': 'Category Risk',
                'description': f"Transaction in {category} category",
                'risk_level': 'HIGH' if category_strength[category] > 0.6 else 'MEDIUM',
                'context': f"This category has specific fraud patterns"
            })
        
        # 6. RELATIONSHIP RISK
        if record_data.get('first_time_pair', 0) == 1:
            justifications.append({
                'category': 'NEW_RELATIONSHIP',
                'feature': 'first_time_pair',
                'strength': 0.6,
                'title': 'First-Time Merchant',
                'description': "First transaction with this merchant",
                'risk_level': 'MEDIUM',
                'context': "New customer-merchant relationship"
            })
        
        # 7. CUSTOMER HISTORY ANALYSIS (30 days = 30 steps)
        unique_merchants = record_data.get('cust_unique_merchants_30d', 0)
        if unique_merchants > 20:
            justifications.append({
                'category': 'BROAD_MERCHANT_USAGE',
                'feature': 'cust_unique_merchants_30d',
                'strength': 0.5,
                'title': 'Diverse Merchant Usage',
                'description': f"Used {unique_merchants} different merchants in last 30 days",
                'risk_level': 'MEDIUM',
                'context': "Customer transacts with many different merchants"
            })
        
        # ENHANCE ALERT WITH UNIQUE JUSTIFICATIONS
        enhanced_alert = alert.copy()
        
        # Sort by strength and take top justifications
        justifications.sort(key=lambda x: x['strength'], reverse=True)
        enhanced_alert['advanced_justifications'] = justifications[:5]
        enhanced_alert['risk_factors'] = len(justifications)
        enhanced_alert['primary_risk_factor'] = justifications[0]['category'] if justifications else 'UNKNOWN'
        
        # Boost confidence based on strongest justification
        if justifications:
            max_strength = max(j['strength'] for j in justifications)
            confidence_boost = min(0.3, max_strength * 0.4)
            enhanced_alert['adjusted_confidence'] = min(0.99, probability + confidence_boost)
        else:
            enhanced_alert['adjusted_confidence'] = probability
        
        enhanced_alerts.append(enhanced_alert)
    
    print(f"Enhanced {len(alerts)} alerts with UNIQUE justifications")
    return enhanced_alerts, {}

def validate_uploaded_data(df):
    """Check if uploaded data has the required columns"""
    required_columns = [
        'step', 'customer', 'age', 'gender', 'merchant', 'category', 'amount',
        'log_amount', 'cust_tx_count_1d', 'cust_tx_count_7d', 'cust_median_amt_7d',
        'amount_over_cust_median_7d', 'cust_unique_merchants_30d', 'first_time_pair',
        'time_since_last_pair_tx', 'mch_tx_count_1d', 'mch_unique_customers_7d'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in required_columns]
    
    print(f"📊 Uploaded file columns: {list(df.columns)}")
    print(f"❌ Missing columns: {missing_columns}")
    print(f"📈 Extra columns: {extra_columns}")
    
    return missing_columns, extra_columns

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("🔄 UPLOAD ROUTE CALLED - STARTING PROCESSING")
    
    if 'file' not in request.files:
        print("❌ No file in request")
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        print("❌ Empty filename")
        return jsonify({'success': False, 'error': 'No file selected'})
    
    print(f"📁 Processing file: {file.filename}")
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            print(f"✅ File saved to: {filepath}")
            
            df = pd.read_csv(filepath)
            print(f"📊 Loaded DataFrame shape: {df.shape}")
            
            # Check if fraud column exists
            has_fraud_column = 'fraud' in df.columns
            print(f"🎯 Fraud column present: {has_fraud_column}")
            if has_fraud_column:
                fraud_count = df['fraud'].sum()
                print(f"🎯 Actual fraud cases in data: {fraud_count}/{len(df)}")
            
            # Validate data
            missing_cols, extra_cols = validate_uploaded_data(df)
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                error_msg = f"Missing required columns: {missing_cols}"
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'success': False, 'error': error_msg})
            
            print("✅ All required columns present!")
            
            # FIXED Data cleaning (proper indentation)
            print("🧹 Cleaning data...")
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Only clean string columns, don't convert to numeric!
                    df[col] = df[col].apply(lambda x: x.strip("'") if isinstance(x, str) else x)
                
                # Only convert these specific numeric columns
                if col in ['amount', 'log_amount', 'cust_tx_count_1d', 'cust_tx_count_7d', 
                       'cust_median_amt_7d', 'amount_over_cust_median_7d', 
                       'cust_unique_merchants_30d', 'first_time_pair', 
                       'time_since_last_pair_tx', 'mch_tx_count_1d', 'mch_unique_customers_7d']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # DO NOT convert categorical columns like 'customer', 'merchant', 'category' to numeric here!
            # Let the LabelEncoder handle them in get_xgboost_predictions
            
            df = df.reset_index().rename(columns={'index': 'original_index'})
            
            # Use XGBoost for predictions
            alerts, predictions = get_xgboost_predictions(df)
            
            print(f"🔍 DEBUG: Got {len(alerts)} alerts and {len(predictions)} predictions")
            
            # Create individualized justifications
            enhanced_alerts, stats = create_individualized_justifications(df, alerts)
            
            # Calculate metrics
            y_true = [pred['actual_fraud'] for pred in predictions] if predictions else []
            enhanced_alert_ids = set([alert['record_id'] for alert in enhanced_alerts])
            y_pred = [1 if i in enhanced_alert_ids else 0 for i in range(len(predictions))] if predictions else []
            
            tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1) if y_true else 0
            fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1) if y_true else 0
            fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0) if y_true else 0
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"📊 METRICS - TP: {tp}, FP: {fp}, FN: {fn}")
            print(f"📊 METRICS - Precision: {precision:.3f}, Recall: {recall:.3f}")
            
            # Format alerts for frontend
            alerts_data = []
            for alert in enhanced_alerts[:50]:
                # Use adjusted confidence if available
                confidence_decimal = float(alert.get('adjusted_confidence', alert['fraud_probability']))
                
                alert_data = {
                    'record_id': int(alert['record_id']),
                    'confidence': confidence_decimal,
                    'amount': f"${alert['raw_data'].get('amount', 0):.2f}",
                    'category': alert['raw_data'].get('category', 'Unknown'),
                    'customer_id': alert['raw_data'].get('customer', 'Unknown'),
                    'merchant_id': alert['raw_data'].get('merchant', 'Unknown'),
                    'step': str(alert['raw_data'].get('step', 'Unknown')),
                    'risk_factors': f"{alert['risk_factors']} unique risk factors",
                    'primary_risk': alert.get('primary_risk_factor', 'Multiple Factors'),
                    'justifications': [
                        {
                            'title': j.get('title', 'Risk Factor'),
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
            print(f"❌ Upload error: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': f'Processing error: {str(e)}'})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

if __name__ == '__main__':
    print("🚀 Starting Apex Fraud Studio...")
    load_model()
    
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Render-specific port binding
    port = int(os.environ.get("PORT", 10000))
    print(f"✅ Server ready on port {port}")
    app.run(host='0.0.0.0', port=port)

