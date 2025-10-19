from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
from werkzeug.utils import secure_filename
import traceback # Import necessary module
from sklearn.metrics import recall_score, precision_score, confusion_matrix

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables for model and encoders
model = None
label_encoders = None
optimal_threshold = None
original_columns = None

def load_model():
    """Load the pre-trained XGBoost model from create_model_files.py"""
    global model, label_encoders, optimal_threshold, original_columns
    
    try:
        # Load all files created by create_model_files.py
        model = joblib.load('fraud_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl') 
        optimal_threshold = joblib.load('optimal_threshold.pkl')
        original_columns = joblib.load('original_columns.pkl')
        
        print("✅ Pre-trained model loaded successfully")
        print(f"🎯 Using threshold: {optimal_threshold}")
        print(f"📋 Original columns ({len(original_columns)}): {original_columns}")
        
    except Exception as e:
        print(f"❌ Error loading pre-trained model: {e}")
        print("💡 Run create_model_files.py first to generate model files")
        raise Exception("Pre-trained model files missing. Run create_model_files.py first")

def get_xgboost_predictions(test_df):
    """Get predictions from XGBoost model with PROPER encoding"""
    global model, label_encoders, optimal_threshold, original_columns
    
    if model is None:
        load_model()
    
    print(f"🔍 Processing {len(test_df)} records with XGBoost...")
    
    try:
        # Prepare the test data
        X_test = test_df.copy()
        has_fraud_column = 'fraud' in X_test.columns
        if has_fraud_column:
            X_test = X_test.drop(columns=['fraud'])
            print("✅ Dropped 'fraud' column from features")
        
        print(f"📊 X_test shape before encoding: {X_test.shape}")
        
        # CRITICAL: Ensure EXACT same column order as training
        missing_cols = set(original_columns) - set(X_test.columns)
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Reorder to match training data exactly
        X_test = X_test[original_columns]
        print("✅ Columns reordered to match training data")
        
        # CRITICAL FIX: Use EXACT same encoding as training
        for col in X_test.columns:
            if col in label_encoders:
                # Convert to string and use the trained LabelEncoder
                X_test[col] = X_test[col].astype(str)
                
                # Handle unseen categories by mapping them to most common class
                unseen_mask = ~X_test[col].isin(label_encoders[col].classes_)
                if unseen_mask.any():
                    # Map unseen values to the first class (usually most common)
                    X_test.loc[unseen_mask, col] = label_encoders[col].classes_[0]
                
                X_test[col] = label_encoders[col].transform(X_test[col])
        
        # Final reordering to match model feature names
        if hasattr(model, 'get_booster'):
            expected_features = model.get_booster().feature_names
            X_test = X_test[expected_features]
        
        # Get predictions
        fraud_proba = model.predict_proba(X_test)[:, 1]
        threshold = optimal_threshold
        
        alerts = []
        all_predictions = []

        for i, prob in enumerate(fraud_proba):
            prob_float = float(prob)
            
            # Record all predictions for dashboard metrics
            all_predictions.append({
                'record_id': i,
                'fraud_probability': prob_float,
                'actual_fraud': test_df.iloc[i]['fraud'] if has_fraud_column else 0
            })
            
            # Use the OPTIMAL threshold from training
            if prob_float > threshold:
                alerts.append({
                    'record_id': i,
                    'fraud_probability': prob_float, # <--- FIX 1: Use 'fraud_probability' for confidence
                    'raw_data': test_df.iloc[i].to_dict(),
                    'risk_level': 'CRITICAL' if prob_float > 0.8 else 'HIGH' if prob_float > 0.5 else 'MEDIUM' if prob_float > 0.2 else 'LOW',
                    # Include the fields the HTML uses directly
                    'amount': f"${test_df.iloc[i].get('amount', 0.0):,.2f}", 
                    'category': str(test_df.iloc[i].get('category', 'Unknown')),
                    'customer_id': str(test_df.iloc[i].get('customer', 'Unknown')),
                    'merchant_id': str(test_df.iloc[i].get('merchant', 'Unknown')),
                    'step': str(test_df.iloc[i].get('step', 'Unknown'))
                })
        
        print(f"✅ Generated {len(alerts)} alerts (optimal threshold: {threshold})")
        return alerts, all_predictions
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return [], []

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
        if 0 <= time_since_last < 0.1: # Same day repeat (very rapid)
            justifications.append({
                'category': 'SAME_DAY_REPEAT',
                'feature': 'time_since_last_pair_tx',
                'strength': 0.9,
                'title': 'Same Day Repeat',
                'description': f"Repeat transaction within same day (step difference: {time_since_last:.1f})",
                'risk_level': 'CRITICAL',
                'context': "Extremely quick repeat transaction with same merchant on same day"
            })
        elif 0 <= time_since_last < 1.0: # Within 1 day
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
        elif 0 <= time_since_last < 3.0: # Within 3 days
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
        # --- FIX 2: Use 'advanced_justifications' which the HTML expects
        enhanced_alert['advanced_justifications'] = justifications[:5] 
        # --- FIX 3: Include the simple 'risk_factors' count (for display on HTML)
        enhanced_alert['risk_factors'] = len(justifications)
        
        enhanced_alerts.append(enhanced_alert)
    
    print(f"Enhanced {len(alerts)} alerts with UNIQUE justifications")
    return enhanced_alerts

def calculate_dashboard_metrics(all_predictions):
    """Calculates precision, recall, and TP/FP/TN/FN counts"""
    global optimal_threshold

    if not all_predictions:
        return {
            'recall': 'N/A', 'fraud_caught': 0, 'fraud_cases': 0,
            'precision': 'N/A', 'total_alerts': 0, 'false_positives': 0
        }

    # Convert predictions to a DataFrame for easy calculation
    df_preds = pd.DataFrame(all_predictions)
    
    # Calculate binary prediction based on optimal threshold
    df_preds['prediction'] = (df_preds['fraud_probability'] > optimal_threshold).astype(int)
    
    # Filter for cases where 'actual_fraud' column exists (i.e., we have ground truth)
    if 'actual_fraud' not in df_preds.columns:
        # If no ground truth, we can only report alert counts
        alerts_generated = df_preds['prediction'].sum()
        
        # --- FIX 4: Use correct keys for the dashboard JSON
        return {
            'recall': 'N/A', 
            'fraud_caught': 'N/A', 
            'fraud_cases': 'N/A',
            'precision': 'N/A', 
            'total_alerts': int(alerts_generated),
            'false_positives': 'N/A'
        }

    # Calculate metrics (only runs if 'actual_fraud' exists)
    y_true = df_preds['actual_fraud']
    y_pred = df_preds['prediction']

    # Handle case where all true or all predicted are zero to avoid division by zero errors
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # --- FIX 4: Use correct keys for the dashboard JSON
    dashboard_data = {
        'recall': f"{recall * 100:.2f}%",         # Matches JS: dashboard.recall
        'fraud_caught': int(tp),                  # Matches JS: dashboard.fraud_caught
        'fraud_cases': int(tp + fn),              # Matches JS: dashboard.fraud_cases
        'precision': f"{precision * 100:.2f}%",   # Matches JS: dashboard.precision
        'total_alerts': int(tp + fp),             # Matches JS: dashboard.alerts_generated (fixed key in HTML)
        'false_positives': int(fp)                # Matches JS: dashboard.false_alerts (fixed key in HTML)
    }

    print(f"📊 Dashboard Metrics: Recall={dashboard_data['recall']}, Precision={dashboard_data['precision']}")
    print(f"📊 TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    return dashboard_data


def validate_uploaded_data(df):
    """Check if uploaded data has the required columns"""
    required_columns = [
        'step', 'customer', 'age', 'gender', 'merchant', 'category', 'amount',
        'cust_tx_count_1d', 'amount_over_cust_median_7d', 'time_since_last_pair_tx', 
        'first_time_pair'
    ]
    
    # Optional 'fraud' column for metric calculation (it's often in uploaded data for testing)
    required_columns_plus_optional = required_columns + ['fraud']

    missing_cols = [col for col in required_columns_plus_optional if col not in df.columns]

    if len(missing_cols) > 2: # Allow missing 'fraud' and maybe one other feature if you are testing a live stream
        # Check if the missing columns are all "non-essential" features (i.e. those not in the initial required list)
        non_essential_missing = [col for col in missing_cols if col not in required_columns]
        if len(missing_cols) - len(non_essential_missing) > 0:
             return False, f"Missing essential columns: {', '.join(set(required_columns) - set(df.columns))}"

    return True, ""


# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    try:
        load_model() # Attempt to load model on startup
        return render_template('index.html')
    except Exception as e:
        return f"Model Error: {e}", 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the file upload, prediction, and returns the dashboard/alerts data."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        file.save(filepath)
        
        try:
            # Read and process the CSV
            df = pd.read_csv(filepath)
            print(f"📂 Read file with {len(df)} records. Columns: {df.columns.tolist()}")

            # 1. Validation
            is_valid, error_msg = validate_uploaded_data(df)
            if not is_valid:
                return jsonify({'success': False, 'error': error_msg}), 400

            # 2. Get Raw Predictions
            alerts, all_predictions = get_xgboost_predictions(df)
            
            # 3. Create Justifications and Final Alert List
            final_alerts = create_individualized_justifications(df, alerts)
            
            # 4. Calculate Dashboard Metrics
            dashboard_data = calculate_dashboard_metrics(all_predictions)

            # --- CRITICAL FIX 5: Map Python keys to expected HTML keys ---
            final_alerts_output = []
            for alert in final_alerts:
                final_alerts_output.append({
                    # Mapped to alert.confidence in JS
                    'confidence': alert['fraud_probability'], 
                    # Mapped to alert.justifications in JS
                    'justifications': alert['advanced_justifications'], 
                    # Pass through other keys used by JS
                    'record_id': alert['record_id'],
                    'amount': alert['amount'],
                    'category': alert['category'],
                    'customer_id': alert['customer_id'],
                    'merchant_id': alert['merchant_id'],
                    'step': alert['step'],
                    'risk_factors': alert['risk_factors']
                })
            
            # --- CRITICAL FIX 6: Map Python dashboard keys to expected HTML keys ---
            final_dashboard_output = {
                'recall': dashboard_data['recall'],
                'fraud_caught': dashboard_data['fraud_caught'],
                'fraud_cases': dashboard_data['fraud_cases'],
                'precision': dashboard_data['precision'],
                # Mismatched keys in HTML are fixed here
                'alerts_generated': dashboard_data['total_alerts'], # JS: dashboard.alerts_generated
                'false_alerts': dashboard_data['false_positives']   # JS: dashboard.false_alerts
            }

            return jsonify({
                'success': True,
                # --- Send the correctly mapped data ---
                'dashboard': final_dashboard_output,
                'alerts': final_alerts_output
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'An unexpected error occurred during processing: {e}'}), 500
        
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

# Initialize the model on app start
if __name__ == '__main__':
    try:
        load_model()
        # Set host to '0.0.0.0' to be accessible externally (or on repl.it/remote host)
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Failed to start Flask app: {e}")
