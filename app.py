import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import os  
import tempfile 

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="XAI Stress Detection Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS STYLING DESIGN ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-low {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #28a745;
    }
    .prediction-moderate {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #ffc107;
    }
    .prediction-high {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #dc3545;
    }
    .stButton button {
        width: 100%;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        text-align: center;
        justify-content: center;
    }
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================
def get_label_encoder_classes(encoder):
    """Safely get classes from label encoder regardless of format"""
    if hasattr(encoder, 'classes_'):
        return list(encoder.classes_)
    elif isinstance(encoder, dict) and 'classes_' in encoder:
        return encoder['classes_']
    elif isinstance(encoder, list):
        return encoder
    else:
        return ['Low', 'Moderate', 'High']

def safe_inverse_transform(encoder, predictions):
    """Safely inverse transform predictions"""
    try:
        if hasattr(encoder, 'inverse_transform'):
            return encoder.inverse_transform(predictions)
        elif hasattr(encoder, 'classes_'):
            return [encoder.classes_[i] for i in predictions]
        elif isinstance(encoder, list):
            return [encoder[i] for i in predictions]
        elif isinstance(encoder, dict) and 'classes_' in encoder:
            classes = encoder['classes_']
            return [classes[i] for i in predictions]
        else:
            return predictions
    except Exception as e:
        st.error(f"Error in inverse transform: {e}")
        return predictions

def get_unique_display_features(feature_names):
    """Get unique feature names for display (removes .1, .2 duplicates)"""
    unique_features = []
    seen_features = set()
    
    for feature in feature_names:
        # Clean the feature name remove the duplicate columns
        clean_feature = feature.split('.')[0] if '.' in feature else feature
        
        if clean_feature.lower() not in seen_features:
            unique_features.append(clean_feature)
            seen_features.add(clean_feature.lower())
    
    return unique_features

def create_feature_mapping(original_features):
    """Create mapping from display feature to all original feature variants"""
    feature_mapping = {}
    
    for feature in original_features:
        # Get base name 
        base_name = feature.split('.')[0] if '.' in feature else feature
        
        if base_name not in feature_mapping:
            feature_mapping[base_name] = []
        feature_mapping[base_name].append(feature)
    
    return feature_mapping

# ================================= SIDEBAR =========================================
with st.sidebar:
    st.markdown("# XAI Stress Detection")
    st.markdown("---")
    
    # Navigation panel
    st.markdown("### Navigation")
    page = st.sidebar.radio(
        "Select Module:",
        [
            "🏠 Dashboard Overview",
            "📤 Model Files Status",
            "📝 Student Self-Test",
            "🔍 SHAP Explanations",
            "📂 Batch Prediction"
        ]
    )
    
    st.markdown("---")
    
    # System Information
    st.markdown("### System Information")
    with st.expander("How to Use"):
        st.markdown("""
        **Step 1:** Go to 'Student Self-Test' to assess individual stress level
    
        **Step 2:** Use 'Batch Prediction' for multiple students via CSV upload
    
        **Step 3:** View 'SHAP Explanations' to understand predictions
    
        **Step 4:** Check 'Dashboard Overview' to see statistics and trends
        """)
    
    with st.expander("About"):
        st.markdown("""
        - **ML Model:** AdaBoost Classifier
        - **XAI Method:** SHAP (SHapley Additive exPlanations)
        - **Purpose:** Stress detection with XAI Explanation
        - **Output:** Low / Moderate / High Stress Levels
        """)
    
    st.caption("XAI Stress Detection Dashboard")
# ================================= END OF SIDEBAR =========================================

# ================================= AUTO-LOAD MODELS =======================================
@st.cache_resource
def auto_load_models():
    models_path = 'models'
    
    if not os.path.exists(models_path):
        return None, None, None, None, None, None, "Models folder not found"
    
    try:
        # Load model
        model_path = os.path.join(models_path, 'adaboost_model.pkl')
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
            except:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
        else:
            model_path = os.path.join(models_path, 'adaboost_model.joblib')
            if os.path.exists(model_path):
                model = joblib.load(model_path)
            else:
                return None, None, None, None, None, None, "Model file not found"
        
        # Load scaler
        scaler_path = os.path.join(models_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
            except:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
        else:
            scaler_path = os.path.join(models_path, 'scaler.joblib')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                return None, None, None, None, None, None, "Scaler file not found"
        
        # Load label encoder
        label_encoder_path = os.path.join(models_path, 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
        else:
            return None, None, None, None, None, None, "Label encoder file not found"
        
        # Load feature names 
        feature_names_path = os.path.join(models_path, 'feature_names.pkl')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'rb') as f:
                original_feature_names = pickle.load(f)
        else:
            return None, None, None, None, None, None, "Feature names file not found"
        
        # Load SHAP importance
        importance_df = None
        importance_path = os.path.join(models_path, 'shap_global_importance.csv')
        if os.path.exists(importance_path):
            importance_df = pd.read_csv(importance_path)
        
        return model, scaler, label_encoder, original_feature_names, None, importance_df, None
        
    except Exception as e:
        return None, None, None, None, None, None, f"Error: {str(e)}"

# Load models
model, scaler, label_encoder, original_feature_names, shap_values, importance_df, load_error = auto_load_models()

# Initialize session state for models 
if 'model' not in st.session_state:
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.label_encoder = label_encoder
    st.session_state.original_feature_names = original_feature_names  # All features including duplicates
    st.session_state.importance_df = importance_df
    
    # Create display features and mapping if models are loaded
    if original_feature_names is not None:
        st.session_state.display_features = get_unique_display_features(original_feature_names)
        st.session_state.feature_mapping = create_feature_mapping(original_feature_names)

# Initialize history
if 'test_history' not in st.session_state:
    st.session_state.test_history = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# =================================== PREDICTION FUNCTION ===================================================
def predict_stress(features_df, model, scaler, label_encoder, feature_names):
    try:
        # Ensure use the features the model expects
        X = features_df[feature_names].copy()
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        stress_levels = safe_inverse_transform(label_encoder, predictions)
        return stress_levels, probabilities
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None
    
# =============================== PAGE 1: DASHBOARD OVERVIEW ========================================
if page == "🏠 Dashboard Overview":
    st.markdown('<h1 class="main-header">XAI Stress Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explainable AI Dashboard | AdaBoost + SHAP</p>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.info("""
        **Welcome to the XAI Stress Detection Dashboard!**
        
        This dashboard helps you detect stress levels using machine learning with explainable AI.
        
        **To get started:**
        1. Verify that all required models are successfully loaded. If not, please refresh the page.
        2. Navigate to 'Student Self-Test' to conduct an individual stress assessment.
        3. Utilize 'Batch Prediction' for processing multiple student records via CSV file.
        4. Explore 'SHAP Explanations' to understand the factors influencing predictions.
        
        **Note:** Model files are automatically loaded, manual loading files is not allowed.
        """)
    else:
        # ============ SECTION 1: KEY METRICS ============
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_features = len(st.session_state.original_feature_names)
            unique_features = len(st.session_state.display_features)
            st.metric("Total Features", f"{total_features}")
            st.caption(f"({unique_features} unique questions)")
        
        with col2:
            if st.session_state.label_encoder is not None:
                classes = get_label_encoder_classes(st.session_state.label_encoder)
                st.metric("Stress Levels", len(classes))
                st.caption("Low / Moderate / High")
        
        with col3:
            total_predictions = 0
            self_test_count = 0
            batch_count = 0
            
            if 'prediction_history' in st.session_state:
                for batch in st.session_state.prediction_history:
                    batch_count += len(batch['data'])
            
            if 'test_history' in st.session_state:
                self_test_count = len(st.session_state.test_history)
            
            total_predictions = self_test_count + batch_count
            st.metric("Total Predictions", total_predictions)
            st.caption(f"Self-Test: {self_test_count} | Batch: {batch_count}")
        
        with col4:
            st.metric("ML Model", "AdaBoost")
            st.caption("Ensemble Learning Method")
        
        st.markdown("---")
        
        # ============ SECTION 2: DEMOGRAPHIC ANALYSIS ============
        st.subheader("Demographic Analysis")
        st.caption("Age and gender distribution from self-test participants")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age Distribution - Bar Chart
            ages = []
            if 'test_history' in st.session_state:
                for test in st.session_state.test_history:
                    if 'responses' in test:
                        for feature, value in test['responses'].items():
                            if 'age' in feature.lower():
                                if isinstance(value, (int, float)):
                                    ages.append(value)
                                break
            
            if ages:
                age_df = pd.DataFrame({'Age': ages})
                bins = [16, 20, 25, 30, 35, 40, 100]
                labels = ['16-20', '21-25', '26-30', '31-35', '36-40', '40+']
                age_df['Age Group'] = pd.cut(age_df['Age'], bins=bins, labels=labels, right=False)
                age_group_counts = age_df['Age Group'].value_counts().reset_index()
                age_group_counts.columns = ['Age Group', 'Count']
                age_group_counts = age_group_counts.sort_values('Age Group')
                fig_age = px.bar(age_group_counts, x='Age Group', y='Count',
                                title=f"Age Distribution (total={len(ages)})",
                                color='Count', color_continuous_scale='Blues',
                                text='Count')
                fig_age.update_traces(textposition='outside', textfont_size=14)
                fig_age.update_layout(
                    yaxis_title="Number of Students",
                    xaxis_title="Age Group",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("No age data yet. Complete a Self-Test to see age distribution!")
        
        with col2:
            # Gender Distribution - Pie Chart
            gender_counts = {'Male': 0, 'Female': 0, 'Prefer not to say': 0}
            
            if 'test_history' in st.session_state:
                for test in st.session_state.test_history:
                    if 'responses' in test:
                        for feature, value in test['responses'].items():
                            if 'gender' in feature.lower():
                                if value == 0:
                                    gender_counts['Male'] += 1
                                elif value == 1:
                                    gender_counts['Female'] += 1
                                elif value == 2:
                                    gender_counts['Prefer not to say'] += 1
                                break
            
            if sum(gender_counts.values()) > 0:
                gender_df = pd.DataFrame({
                    'Gender': list(gender_counts.keys()),
                    'Count': list(gender_counts.values())
                })
                fig_gender = px.pie(gender_df, values='Count', names='Gender',
                                   title=f"Gender Distribution (total={sum(gender_counts.values())})",
                                   color='Gender',
                                   color_discrete_map={'Male': '#347fdb', 'Female': '#e84343', 'Prefer not to say': '#95a5a6'},
                                   hole=0.4)
                fig_gender.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
                fig_gender.update_layout(height=400)
                st.plotly_chart(fig_gender, use_container_width=True)
            else:
                st.info("No gender data yet. Complete a Self-Test to see gender distribution!")
        
        st.markdown("---")

        # ============ SECTION 3: STRESS ANALYSIS  ============
        st.subheader("Stress Analysis")
        st.caption("Stress level distribution and top contributing factors")

        col1, col2 = st.columns(2)

        with col1:
            # Stress Level Distribution - Bar Chart
            stress_counts = {'Low': 0, 'Moderate': 0, 'High': 0}
            
            if 'prediction_history' in st.session_state:
                for batch in st.session_state.prediction_history:
                    for pred in batch['predictions']:
                        if pred in stress_counts:
                            stress_counts[pred] += 1
            
            if 'test_history' in st.session_state:
                for test in st.session_state.test_history:
                    if test['stress_level'] in stress_counts:
                        stress_counts[test['stress_level']] += 1
            
            if sum(stress_counts.values()) > 0:
                stress_df = pd.DataFrame({
                    'Stress Level': list(stress_counts.keys()),
                    'Count': list(stress_counts.values())
                })
                fig_stress = px.bar(stress_df, x='Stress Level', y='Count', 
                                title="Stress Level Distribution",
                                color='Stress Level',
                                color_discrete_map={'Low': '#28a745', 'Moderate': '#ffc107', 'High': '#dc3545'},
                                text='Count')
                fig_stress.update_traces(textposition='outside', textfont_size=14)
                fig_stress.update_layout(
                    yaxis_title="Number of Students",
                    xaxis_title="Stress Level",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_stress, use_container_width=True)
            else:
                st.info("No prediction data yet.")

        with col2:
            # Stress by Category (Bar chart)
            category_questions = {
                'Physical Health': [
                    'rapid heartbeat or palpitations',
                    'headaches more often',
                    'illness or health issues',
                    'gained/lost weight'
                ],
                'Mental/Emotional': [
                    'anxiety or tension',
                    'get irritated easily',
                    'sadness or low mood',
                    'feel lonely or isolated'
                ],
                'Academic': [
                    'trouble concentrating',
                    'overwhelmed with academic workload',
                    'lack confidence in academic performance',
                    'lack confidence in choice of subjects',
                    'academic and extracurricular activities conflicting',
                    'attend classes regularly'
                ],
                'Social/Environment': [
                    'competition with peers',
                    'relationship causes stress',
                    'difficulties with professors',
                    'working environment unpleasant',
                    'hostel or home environment difficulties'
                ],
                'Lifestyle': [
                    'sleep problems',
                    'struggle to find time for relaxation'
                ]
            }

            if 'test_history' in st.session_state and st.session_state.test_history:
                category_scores = {cat: [] for cat in category_questions.keys()}
                
                for test in st.session_state.test_history:
                    if 'responses' in test:
                        for category, keywords in category_questions.items():
                            category_total = 0
                            count = 0
                            for response_feature, value in test['responses'].items():
                                for keyword in keywords:
                                    if keyword.lower() in response_feature.lower():
                                        if isinstance(value, (int, float)):
                                            category_total += value
                                            count += 1
                                        break
                            if count > 0:
                                category_scores[category].append(category_total / count)
                
                avg_category = {}
                for category, scores in category_scores.items():
                    if scores:
                        avg_category[category] = np.mean(scores)
                    else:
                        avg_category[category] = 0
                
                if any(avg_category.values()):
                    category_df = pd.DataFrame({
                        'Category': list(avg_category.keys()),
                        'Average Stress Score': list(avg_category.values())
                    }).sort_values('Average Stress Score', ascending=False)
                    
                    # Vertical bar chart
                    fig_category = px.bar(category_df, x='Category', y='Average Stress Score',
                                        title="Average Stress Score by Category",
                                        color='Average Stress Score',
                                        color_continuous_scale='RdYlGn_r',
                                        text='Average Stress Score')
                    fig_category.update_traces(texttemplate='%{text:.1f}', textposition='outside', textfont_size=12)
                    fig_category.update_layout(
                        height=450,
                        yaxis_title="Average Stress Score (0-10)",
                        xaxis_title="Category",
                        yaxis=dict(range=[0, 10]),
                        showlegend=False
                    )
                    st.plotly_chart(fig_category, use_container_width=True)
                else:
                    st.info("Complete self-tests to see category breakdown")
            else:
                st.info("Complete self-tests to see category breakdown")
        st.markdown("---")

        # ============ SECTION 4: FEATURE IMPORTANCE ============
        st.subheader("Feature Importance Analysis")
        st.caption("Top factors affecting stress prediction (SHAP-based)")
        
        if st.session_state.importance_df is not None:
            importance_df = st.session_state.importance_df.head(10)
            fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                                   orientation='h', 
                                   title="Top 10 Features Impacting Stress",
                                   color='Importance', 
                                   color_continuous_scale='Reds',
                                   text='Importance')
            fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside', textfont_size=12)
            fig_importance.update_layout(
                height=500,
                xaxis_title="SHAP Importance Score",
                yaxis_title="Feature Name",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("SHAP importance data not available. Please ensure 'shap_global_importance.csv' is present in the models folder.")
        
        st.markdown("---")
        
        # ============ SECTION 5: TRENDS OVER TIME ============
        st.subheader("Stress Trends Over Time")
        st.caption("Track how stress levels change over time")
        
        timeline_data = []
        
        if 'test_history' in st.session_state:
            for test in st.session_state.test_history:
                timeline_data.append({
                    'Date': test['timestamp'],
                    'Stress Level': test['stress_level'],
                    'Type': 'Self-Test'
                })
        
        if 'prediction_history' in st.session_state:
            for batch in st.session_state.prediction_history:
                for i, pred in enumerate(batch['predictions']):
                    timeline_data.append({
                        'Date': batch['timestamp'] + pd.Timedelta(seconds=i),
                        'Stress Level': pred,
                        'Type': 'Batch'
                    })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df = timeline_df.sort_values('Date')
            stress_order = {'Low': 0, 'Moderate': 1, 'High': 2}
            timeline_df['Stress_Numeric'] = timeline_df['Stress Level'].map(stress_order)
            
            fig_timeline = px.line(timeline_df, x='Date', y='Stress_Numeric', 
                                   color='Type', 
                                   title="Stress Level Trends Over Time",
                                   markers=True,
                                   color_discrete_map={'Self-Test': '#2E86AB', 'Batch': '#F39C12'})
            fig_timeline.update_layout(
                yaxis_title="Stress Level",
                xaxis_title="Date",
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2],
                    ticktext=['Low', 'Moderate', 'High']
                ),
                height=450,
                hovermode='x unified'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No prediction history yet. Complete a Self-Test to see your timeline!")
        
        st.markdown("---")
        
        # ============ SECTION 6: RECENT ACTIVITY ============
        st.subheader("Recent Activity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Recent Self-Tests")
            if 'test_history' in st.session_state and st.session_state.test_history:
                recent_tests = st.session_state.test_history[-5:]
                for test in reversed(recent_tests):
                    time_str = test['timestamp'].strftime("%d/%m/%Y %H:%M")
                    if test['stress_level'] == 'Low':
                        st.success(f"{time_str} → {test['stress_level']} Stress")
                    elif test['stress_level'] == 'Moderate':
                        st.warning(f"{time_str} → {test['stress_level']} Stress")
                    else:
                        st.error(f"{time_str} → {test['stress_level']} Stress")
            else:
                st.info("No self-test history yet.")
        
        with col2:
            st.markdown("Recent Batch Predictions")
            if 'prediction_history' in st.session_state and st.session_state.prediction_history:
                recent_batches = st.session_state.prediction_history[-3:]
                for batch in reversed(recent_batches):
                    time_str = batch['timestamp'].strftime("%d/%m/%Y %H:%M")
                    counts = pd.Series(batch['predictions']).value_counts()
                    st.markdown(f"""
                    **{time_str}**  
                    - Total: {len(batch['data'])} students  
                    - Low: {counts.get('Low', 0)}  
                    - Moderate: {counts.get('Moderate', 0)}  
                    - High: {counts.get('High', 0)}
                    """)
                    st.markdown("---")
            else:
                st.info("No batch prediction history yet.")

# ==================================== PAGE 2: MODEL FILES STATUS =========================================
elif page == "📤 Model Files Status":
    st.header("Model Configuration")
    
    # ============ CURRENTLY LOADED STATUS ============
    st.subheader("Current Status")
    
    if st.session_state.model is not None:
        st.markdown("""
        <div class="success-box">
            <strong>Models Loaded Successfully</strong><br>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "Active")
        with col2:
            total = len(st.session_state.original_feature_names)
            unique = len(st.session_state.display_features)
            st.metric("Features", f"{total} total")
        with col3:
            if st.session_state.importance_df is not None:
                st.metric("SHAP", "Available")
            else:
                st.metric("SHAP", "Optional")
        
        st.markdown("---")
        
        # Clear models with confirmation alert box
        if 'confirm_clear' not in st.session_state:
            st.session_state.confirm_clear = False
        
        if not st.session_state.confirm_clear:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Clear loaded models", type="secondary", use_container_width=True):
                    st.session_state.confirm_clear = True
                    st.rerun()
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>Confirm clear models</strong><br>
                This will remove all loaded models. Prediction features will not work until you reload the pages.
            </div>
            """, unsafe_allow_html=True)
            
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, clear models", use_container_width=True):
                    st.session_state.model = None
                    st.session_state.scaler = None
                    st.session_state.label_encoder = None
                    st.session_state.original_feature_names = None
                    st.session_state.display_features = None
                    st.session_state.feature_mapping = None
                    st.session_state.importance_df = None
                    st.session_state.confirm_clear = False
                    st.rerun()
            with col_no:
                if st.button("No, keep models", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()
    
    else:
        # No models loaded
        st.markdown("""
        <div class="warning-box">
        No models loaded. Please refresh the page to enable the auto load models.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

# ==================================== PAGE 3: STUDENT SELF-TEST ===========================================
elif page == "📝 Student Self-Test":
    from datetime import datetime
    from fpdf import FPDF
    
    st.header("Student Stress Assessment")
    st.markdown("Complete the questionnaire to get your personalized stress level assessment.")
    
    if st.session_state.model is None:
        st.warning("No model loaded. Please refresh the page to load the model files first.")
    else:
        st.markdown("""
        <div class="info-box">
            <strong>How to Answer:</strong><br>
            For demographics questions, answer by selecting the correct option, other answer with a number from 0 to 10:<br>
            0 = Never / Not at all / Very Low<br>
            10 = Always / Very severely / Very High
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Stress Assessment Questionnaire")
        st.markdown("Please rate each factor based on your recent experience (past 2 weeks):")
        
        col1, col2 = st.columns(2)
        user_input = {}
        
        # Display only unique features hidden the duplicate features
        for idx, display_feature in enumerate(st.session_state.display_features):
            with col1 if idx % 2 == 0 else col2:
                
                # Gender feature
                if 'gender' in display_feature.lower():
                    selected = st.selectbox(
                        display_feature,
                        options=['Male', 'Female', 'Prefer not to say'],
                        key=f"gender_{idx}"
                    )
                    gender_mapping = {'Male': 0, 'Female': 1, 'Prefer not to say': 2}
                    value = gender_mapping[selected]
                    
                    # Store value for original feature variants
                    for original_feature in st.session_state.feature_mapping[display_feature]:
                        user_input[original_feature] = value
                    st.caption(f"Selected: {selected}")
                
                # Age feature
                elif 'age' in display_feature.lower():
                    value = st.number_input(
                        display_feature,
                        min_value=16,
                        max_value=100,
                        value=22,
                        step=1,
                        key=f"age_{idx}"
                    )
                    
                    # Store value for original feature variants
                    for original_feature in st.session_state.feature_mapping[display_feature]:
                        user_input[original_feature] = value
                    
                    if value < 18:
                        st.caption(f"Age: {value} - Young adult")
                    elif value <= 25:
                        st.caption(f"Age: {value} - University age")
                    elif value <= 35:
                        st.caption(f"Age: {value} - Young professional")
                    else:
                        st.caption(f"Age: {value} - Adult")
                
                # Regular questions
                else:
                    value = st.slider(
                        display_feature,
                        min_value=0,
                        max_value=10,
                        value=5,
                        step=1,
                        key=f"slider_{idx}"
                    )
                    
                    # Store value for original feature variants
                    for original_feature in st.session_state.feature_mapping[display_feature]:
                        user_input[original_feature] = value
                    
                    if value <= 2:
                        st.caption(f"Score: {value} - Low / Never")
                    elif value <= 4:
                        st.caption(f"Score: {value} - Mild / Occasionally")
                    elif value <= 6:
                        st.caption(f"Score: {value} - Moderate / Sometimes")
                    elif value <= 8:
                        st.caption(f"Score: {value} - High / Frequently")
                    else:
                        st.caption(f"Score: {value} - Severe / Always")
        
        st.markdown("---")
        
        col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
        with col_reset2:
            if st.button("Reset All Values", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        
        if st.button("Analyze My Stress Level", type="primary", use_container_width=True):
            with st.spinner("Analyzing your responses..."):
                # Create dataframe with ALL original features
                input_df = pd.DataFrame([user_input])
                
                # Verify all required features are present
                missing_features = [f for f in st.session_state.original_feature_names if f not in input_df.columns]
                if missing_features:
                    st.error(f"Missing features: {missing_features[:3]}")
                else:
                    prediction, probabilities = predict_stress(
                        input_df, 
                        st.session_state.model, 
                        st.session_state.scaler, 
                        st.session_state.label_encoder, 
                        st.session_state.original_feature_names
                    )
                    
                    if prediction is not None:
                        numeric_to_stress = {0: 'Low', 1: 'Moderate', 2: 'High'}
                        
                        if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
                            pred_value = prediction[0]
                        else:
                            pred_value = prediction
                        
                        if isinstance(pred_value, (int, float, np.integer)):
                            stress_level = numeric_to_stress.get(int(pred_value), 'Unknown')
                        else:
                            stress_level = str(pred_value).strip()
                        
                        # Store in history
                        st.session_state.test_history.append({
                            'timestamp': pd.Timestamp.now(),
                            'stress_level': stress_level,
                            'responses': user_input.copy()
                        })
                        
                        # Display result by different stress level
                        if stress_level.lower() == "low":
                            st.markdown("""
                            <div class="prediction-low">
                                <h2>Low Stress Level Detected</h2>
                                <p>Great news! Your responses indicate you're managing stress well.</p>
                                <p>Keep up the healthy habits!</p>
                                <hr>
                                <p><strong>What this means:</strong> Your current coping strategies are working well.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        
                        elif stress_level.lower() == "moderate":
                            st.markdown("""
                            <div class="prediction-moderate">
                                <h2>Moderate Stress Level Detected</h2>
                                <p>You're experiencing some stress, but there are effective ways to manage it.</p>
                                <p>With some adjustments, you can reduce your stress levels.</p>
                                <hr>
                                <p><strong>What this means:</strong> Your stress levels are elevated but manageable.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        elif stress_level.lower() == "high":
                            st.markdown("""
                            <div class="prediction-high">
                                <h2>High Stress Level Detected</h2>
                                <p>Your responses indicate significant stress that may be affecting your well-being.</p>
                                <p>It's important to take action now.</p>
                                <hr>
                                <p><strong>What this means:</strong> Your stress levels are concerning and require attention.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        else:
                            st.error(f"Unexpected prediction result: {stress_level}")
                        
                        # Show confidence chart
                        st.subheader("Model Prediction Confidence")
                        st.markdown("Higher bar = Model is more confident in that prediction")
                        
                        classes = get_label_encoder_classes(st.session_state.label_encoder)
                        prob_df = pd.DataFrame({
                            'Stress Level': classes,
                            'Confidence (%)': probabilities[0] * 100
                        })
                        fig = px.bar(prob_df, x='Stress Level', y='Confidence (%)', 
                                     title="Model's Prediction Confidence",
                                     color='Stress Level',
                                     color_discrete_map={'Low': '#28a745', 'Moderate': '#ffc107', 'High': '#dc3545'})
                        fig.update_layout(yaxis_range=[0, 100])
                        fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)

                        # After confidence chart, before recommendations
                        st.subheader("Why This Prediction?")
                        st.markdown("Understanding which factors most influenced your stress level using XAI techniques:")

                        # Get feature importance from SHAP data
                        shap_importance = {}
                        if st.session_state.importance_df is not None:
                            for _, row in st.session_state.importance_df.iterrows():
                                clean_name = row['Feature'].split('.')[0] if '.' in row['Feature'] else row['Feature']
                                shap_importance[clean_name.lower()] = row['Importance']

                        # Collect user's responses for analysis
                        impact_data = []
                        for feature, value in user_input.items():
                            # Skip gender and age
                            if 'gender' in feature.lower() or 'age' in feature.lower():
                                continue
                                
                            clean = feature.split('.')[0] if '.' in feature else feature
                            
                            if isinstance(value, (int, float)):
                                # Get SHAP importance weight
                                importance_weight = 1.0
                                for shap_feature, weight in shap_importance.items():
                                    if shap_feature in clean.lower() or clean.lower() in shap_feature:
                                        importance_weight = weight
                                        break
                                
                                # Calculate weighted impact
                                distance = abs(value - 5)
                                raw_impact = distance / 5
                                weighted_impact = raw_impact * importance_weight
                                
                                if value >= 7:
                                    effect = "Increases Stress"
                                    color = "#dc3545"
                                elif value <= 3:
                                    effect = "Decreases Stress"
                                    color = "#28a745"
                                else:
                                    effect = "Neutral"
                                    color = "#6c757d"
                                    weighted_impact = 0.05
                                
                                impact_data.append({
                                    "Factor": clean,
                                    "Your Score": value,
                                    "Impact": weighted_impact,
                                    "Effect": effect
                                })

                        if impact_data:
                            # Sort by score (highest first)
                            impact_df = pd.DataFrame(impact_data)
                            impact_df = impact_df.sort_values('Your Score', ascending=False)
                            
                            # Show top 8 factors
                            top_impact_df = impact_df.head(10)
                            chart_df = top_impact_df.sort_values('Your Score', ascending=True)
                            
                            # Full width bar chart
                            fig = px.bar(chart_df, x='Your Score', y='Factor', 
                                        orientation='h',
                                        color='Effect',
                                        color_discrete_map={
                                            'Increases Stress': '#dc3545',
                                            'Decreases Stress': '#28a745',
                                            'Neutral': '#6c757d'
                                        },
                                        title="Top 10 on How Your Responses Affected the Prediction",
                                        text='Your Score',
                                        height=500)
                            
                            fig.update_traces(textposition='outside', textfont_size=11)
                            fig.update_layout(
                                xaxis_title="Your Score (0 = Low Stress, 10 = High Stress)",
                                yaxis_title="",
                                showlegend=True,
                                legend_title="Effect",
                                xaxis=dict(range=[0, 10], tick0=0, dtick=1),
                                height=500,
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Summary stats
                            high_count = len(impact_df[impact_df['Your Score'] >= 7])
                            moderate_count = len(impact_df[(impact_df['Your Score'] >= 4) & (impact_df['Your Score'] <= 6)])
                            low_count = len(impact_df[impact_df['Your Score'] <= 3])

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("High Score (≥7)", high_count, "Increases Stress")
                            with col2:
                                st.metric("Moderate Score (4-6)", moderate_count, "Neutral Effect")
                            with col3:
                                st.metric("Low Score (≤3)", low_count, "Decreases Stress")
                            
                            st.markdown("""
                            <div class="info-box">
                                <strong>How to read this chart:</strong><br>
                                • Red bars = Scores 7-10 that increase stress<br>
                                • Green bars = Scores 0-3 that decrease stress<br>
                                • Gray bars = Scores 4-6 with neutral effect<br>
                                • Longer bars = Higher scores that contribute more to stress<br>
                                • Questions appear in the same order as the questionnaire not ranked
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("---")
                        
                        # ==================== RECOMMENDATIONS SECTION ====================
                        st.subheader("Personalized Recommendations")
                        
                        recommendations_text = ""
                        
                        if stress_level.lower() == "high":
                            recommendations_text = """
                            IMMEDIATE ACTIONS RECOMMENDED
                            
                            1. Professional Support (Priority)
                            - Talk to a counselor or mental health professional
                            - Call a mental health helpline if you need immediate support
                            
                            2. Immediate Self-Care (Today)
                            - Practice deep breathing: Inhale 4 sec, hold 4 sec, exhale 4 sec
                            - Prioritize sleep - aim for 7-9 hours tonight
                            - Take a 10-15 minute walk outside
                            
                            3. Short-term Stress Management
                            - Break large tasks into smaller steps
                            - Set realistic daily goals
                            - Reach out to trusted friends or family
                            """
                            st.error(recommendations_text)
                            
                            st.info("""
                            ### Remember
                            - You are not alone - many students experience high stress
                            - Seeking help is a sign of strength, not weakness
                            - Take one step at a time
                            """)
                        
                        elif stress_level.lower() == "moderate":
                            recommendations_text = """
                            RECOMMENDED ACTIONS
                            
                            1. Daily Habits to Reduce Stress
                            - 5-10 minutes of meditation daily
                            - Maintain consistent sleep schedule (7-9 hours)
                            - 20-30 minutes of physical activity daily
                            
                            2. Work/Life Balance
                            - Create a realistic daily schedule
                            - Take regular breaks (5 min every hour)
                            - Set achievable goals
                            
                            3. Stress Management Techniques
                            - Practice time management
                            - Learn to say no to additional commitments
                            - Connect with supportive friends
                            """
                            st.warning(recommendations_text)
                            
                            st.info("""
                            ### Remember
                            - Small changes today can prevent bigger problems tomorrow
                            - It's okay to ask for help before stress becomes overwhelming
                            """)
                        
                        else:  # Low Stress
                            recommendations_text = """
                            MAINTAIN YOUR HEALTHY HABITS
                            
                            1. Keep Up the Good Work
                            - Continue your current healthy routines
                            - Keep monitoring your stress levels weekly
                            - Practice mindfulness to build resilience
                            
                            2. Prevention Strategies
                            - Practice gratitude journaling
                            - Stay connected with friends and family
                            - Schedule regular self-care activities
                            
                            3. Stay Proactive
                            - Maintain work-life balance
                            - Get adequate sleep and exercise
                            - Eat a balanced diet
                            """
                            st.success(recommendations_text)
                            
                            st.info("""
                            ### Remember
                            Maintaining these habits will help you stay resilient during challenging times!
                            """)
                        
               # ==================== DOWNLOAD RESULTS AS PDF ====================
                st.markdown("---")
                st.subheader("Download Report")

                class PDF(FPDF):
                    def header(self):
                        self.set_fill_color(46, 134, 171)
                        self.rect(0, 0, 210, 40, 'F')
                        
                        # Title
                        self.set_font('Arial', 'B', 20)
                        self.set_text_color(255, 255, 255)
                        self.cell(0, 15, 'STRESS ASSESSMENT REPORT', 0, 1, 'C')
                        
                        # Subtitle
                        self.set_font('Arial', 'I', 10)
                        self.set_text_color(230, 230, 230)
                        self.cell(0, 8, 'XAI Stress Detection System | AdaBoost + SHAP', 0, 1, 'C')
                        self.ln(15)
                    
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 8)
                        self.set_text_color(128, 128, 128)
                        self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.now().strftime("%d/%m/%Y")}', 0, 0, 'C')
                    
                    def section_title(self, title):
                        self.set_font('Arial', 'B', 14)
                        self.set_text_color(46, 134, 171)
                        self.cell(0, 10, title, 0, 1, 'L')
                        self.set_draw_color(46, 134, 171)
                        self.line(10, self.get_y(), 200, self.get_y())
                        self.ln(5)
                    
                    def confidence_bar(self, label, percentage, color_r, color_g, color_b):
                        self.set_font('Arial', 'B', 10)
                        self.set_text_color(0, 0, 0)
                        self.cell(35, 8, label, 0, 0, 'L')
                        
                        # Draw bar background
                        self.set_fill_color(230, 230, 230)
                        self.rect(50, self.get_y() - 2, 100, 6, 'F')
                        
                        # Draw confidence bar
                        self.set_fill_color(color_r, color_g, color_b)
                        self.rect(50, self.get_y() - 2, percentage, 6, 'F')
                        
                        # Draw percentage text to the right of the bar
                        self.set_font('Arial', 'B', 10)
                        self.set_text_color(color_r, color_g, color_b)
                        self.set_x(155)
                        self.cell(0, 8, f'{percentage:.1f}%', 0, 1, 'L')
                        self.ln(2)
                    
                    def add_explanation(self, text):
                        self.set_font('Arial', 'I', 9)
                        self.set_text_color(100, 100, 100)
                        self.multi_cell(0, 5, text, 0, 1)
                        self.ln(3)

                # Create PDF
                pdf = PDF()
                pdf.add_page()

                # Date and Time
                pdf.set_font('Arial', 'B', 10)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 8, f"Report Date: {datetime.now().strftime('%d/%m/%Y at %H:%M:%S')}", 0, 1, 'R')
                pdf.ln(5)

                # ASSESSMENT RESULTS SECTION
                pdf.section_title("ASSESSMENT RESULTS")

                # Stress level 
                stress_color = {
                    'low': (40, 167, 69),
                    'moderate': (255, 193, 7),
                    'high': (220, 53, 69)
                }.get(stress_level.lower(), (100, 100, 100))

                pdf.set_font('Arial', 'B', 16)
                pdf.set_text_color(stress_color[0], stress_color[1], stress_color[2])
                pdf.cell(0, 12, f"Predicted Stress Level: {stress_level.upper()}", 0, 1, 'C')
                pdf.ln(5)

                # CONFIDENCE SCORES SECTION
                pdf.section_title("CONFIDENCE SCORES")
                pdf.add_explanation("The confidence scores below indicate how certain the model is about each stress level prediction. Higher percentages mean the model is more confident in that classification.")

                for i, level in enumerate(classes):
                    percentage = probabilities[0][i] * 100
                    if level.lower() == 'low':
                        pdf.confidence_bar(level, percentage, 40, 167, 69)
                    elif level.lower() == 'moderate':
                        pdf.confidence_bar(level, percentage, 255, 193, 7)
                    else:
                        pdf.confidence_bar(level, percentage, 220, 53, 69)

                # Add interpretation guide
                pdf.add_explanation("Interpretation: Low confidence (<50%) suggests uncertainty, while high confidence (>70%) indicates strong prediction certainty.")
                pdf.ln(5)

                # RESPONSES SUMMARY SECTION
                pdf.section_title("RESPONSES SUMMARY")
                pdf.add_explanation("Your responses to each question are summarized below. Scores range from 0 (Never/Low) to 10 (Always/High).")

                # Get all responses with proper display values
                response_items = []
                gender_display = {0: 'Male', 1: 'Female', 2: 'Prefer not to say'}

                for display_feature in st.session_state.display_features:
                    for orig_feature, val in user_input.items():
                        if display_feature.lower() in orig_feature.lower():
                            # Convert gender code to text 
                            if 'gender' in display_feature.lower():
                                display_value = gender_display.get(val, str(val))
                                interpretation = ""
                                numeric_val = None
                            # Handle age specially
                            elif 'age' in display_feature.lower():
                                display_value = str(val)
                                if val < 18:
                                    interpretation = "(Young adult)"
                                elif val <= 25:
                                    interpretation = "(University age)"
                                elif val <= 35:
                                    interpretation = "(Young professional)"
                                else:
                                    interpretation = "(Adult)"
                                numeric_val = None
                            # Regular questions with score interpretation
                            elif isinstance(val, (int, float)):
                                display_value = str(val)
                                if val <= 2:
                                    interpretation = "(Low/Never)"
                                elif val <= 4:
                                    interpretation = "(Mild/Occasionally)"
                                elif val <= 6:
                                    interpretation = "(Moderate/Sometimes)"
                                elif val <= 8:
                                    interpretation = "(High/Frequently)"
                                else:
                                    interpretation = "(Severe/Always)"
                                numeric_val = val
                            else:
                                display_value = str(val)
                                interpretation = ""
                                numeric_val = None
                            
                            response_items.append((display_feature, display_value, interpretation, numeric_val))
                            break

                # FIXED RESPONSES TABLE - All on same line
                pdf.set_font('Arial', '', 10)

                # Calculate page width for full width table
                page_width = pdf.w - pdf.l_margin - pdf.r_margin

                # Column widths
                width_number = 12      # Question number column
                width_score = 25       # Score column
                width_interpretation = 35  # Interpretation column
                width_feature = page_width - width_number - width_score - width_interpretation - 10  # Remaining for feature

                for idx, (feature, display_value, interpretation, numeric_val) in enumerate(response_items):
                    # Get current Y position
                    start_y = pdf.get_y()
                    
                    # Question number
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_text_color(46, 134, 171)
                    pdf.cell(width_number, 8, f"{idx + 1}.", 0, 0, 'L')
                    
                    # FULL FEATURE NAME (no truncation)
                    pdf.set_font('Arial', '', 9)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(width_feature, 8, feature, 0, 0, 'L')
                    
                    # Score column
                    if 'gender' in feature.lower() or 'age' in feature.lower():
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font('Arial', 'B', 10)
                        pdf.cell(width_score, 8, display_value, 0, 0, 'C')
                    else:
                        if numeric_val is not None:
                            if numeric_val <= 2:
                                pdf.set_text_color(40, 167, 69)
                            elif numeric_val <= 4:
                                pdf.set_text_color(120, 120, 120)
                            elif numeric_val <= 6:
                                pdf.set_text_color(255, 193, 7)
                            elif numeric_val <= 8:
                                pdf.set_text_color(255, 100, 0)
                            else:
                                pdf.set_text_color(220, 53, 69)
                        else:
                            pdf.set_text_color(0, 0, 0)
                        
                        pdf.set_font('Arial', 'B', 10)
                        pdf.cell(width_score, 8, display_value, 0, 0, 'C')
                    
                    # Interpretation column
                    pdf.set_font('Arial', 'I', 8)
                    pdf.set_text_color(100, 100, 100)
                    pdf.cell(width_interpretation, 8, interpretation, 0, 1, 'L')
                    
                    # Add light separator line between rows
                    if idx < len(response_items) - 1:
                        pdf.set_draw_color(230, 230, 230)
                        pdf.line(10, pdf.get_y() + 2, page_width + 10, pdf.get_y() + 2)
                        pdf.ln(4)  # Space after separator line

                # ============ SHAP ANALYSIS SECTION (NO BORDERS) ============
                pdf.ln(8)
                pdf.section_title("SHAP ANALYSIS - KEY STRESS FACTORS")

                # Get SHAP importance data
                shap_importance = {}
                if st.session_state.importance_df is not None:
                    for _, row in st.session_state.importance_df.iterrows():
                        clean_name = row['Feature'].split('.')[0] if '.' in row['Feature'] else row['Feature']
                        shap_importance[clean_name.lower()] = row['Importance']

                # Collect impact data for SHAP analysis
                impact_data = []
                for feature, value in user_input.items():
                    if 'gender' in feature.lower() or 'age' in feature.lower():
                        continue
                        
                    clean = feature.split('.')[0] if '.' in feature else feature
                    
                    if isinstance(value, (int, float)):
                        importance_weight = 1.0
                        for shap_feature, weight in shap_importance.items():
                            if shap_feature in clean.lower() or clean.lower() in shap_feature:
                                importance_weight = weight
                                break
                        
                        if value >= 7:
                            effect = "High (7-10)"
                            category = "Stress Contributor"
                        elif value <= 3:
                            effect = "Low (0-3)"
                            category = "Stress Reducer"
                        else:
                            effect = "Moderate (4-6)"
                            category = "Neutral"
                        
                        impact_data.append({
                            "Factor": clean,
                            "Score": value,
                            "Level": effect,
                            "Category": category
                        })

                if impact_data:
                    # Sort by score (highest first)
                    impact_df = pd.DataFrame(impact_data)
                    impact_df = impact_df.sort_values('Score', ascending=False)
                    
                    # Show top 10 factors
                    top_impact_df = impact_df.head(10)
                    
                    pdf.set_font('Arial', 'I', 10)
                    pdf.set_text_color(100, 100, 100)
                    pdf.cell(0, 8, "The following top 10 factors had the highest scores in your assessment:", 0, 1, 'L')
                    pdf.ln(3)
                    
                    # STANDARDIZED SHAP TABLE - NO BORDERS
                    page_width = pdf.w - pdf.l_margin - pdf.r_margin
                    
                    width_factor = page_width * 0.58    # 58% for Factor (full names)
                    width_score = page_width * 0.12     # 12% for Score
                    width_category = page_width * 0.30  # 30% for Category
                    
                    # Header row (no borders)
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(width_factor, 8, "Factor", 0, 0, 'L')
                    pdf.cell(width_score, 8, "Score", 0, 0, 'C')
                    pdf.cell(width_category, 8, "Category", 0, 1, 'L')
                    
                    # Separator line under header
                    pdf.set_draw_color(200, 200, 200)
                    pdf.line(10, pdf.get_y(), page_width + 10, pdf.get_y())
                    pdf.ln(5)
                    
                    # Data rows (no borders)
                    pdf.set_font('Arial', '', 10)
                    
                    for idx, row in top_impact_df.iterrows():
                        # FULL FEATURE NAME (no truncation)
                        factor_text = row['Factor']
                        
                        # Factor column
                        pdf.cell(width_factor, 7, factor_text, 0, 0, 'L')
                        
                        # Score column - centered
                        pdf.set_font('Arial', 'B', 10)
                        pdf.cell(width_score, 7, str(row['Score']), 0, 0, 'C')
                        
                        # Category column with color coding
                        pdf.set_font('Arial', '', 10)
                        if row['Category'] == "Stress Contributor":
                            pdf.set_text_color(220, 53, 69)
                        elif row['Category'] == "Stress Reducer":
                            pdf.set_text_color(40, 167, 69)
                        else:
                            pdf.set_text_color(100, 100, 100)
                        
                        pdf.cell(width_category, 7, row['Category'], 0, 1, 'L')
                        pdf.set_text_color(0, 0, 0)
                        
                        # Add consistent spacing between rows (3 units)
                        pdf.ln(3)
                    
                    pdf.ln(6)
                    
                    # Summary statistics
                    high_count = len(impact_df[impact_df['Score'] >= 7])
                    moderate_count = len(impact_df[(impact_df['Score'] >= 4) & (impact_df['Score'] <= 6)])
                    low_count = len(impact_df[impact_df['Score'] <= 3])
                    
                    # Separator line before summary
                    pdf.set_draw_color(200, 200, 200)
                    pdf.line(10, pdf.get_y(), page_width + 10, pdf.get_y())
                    pdf.ln(5)
                    
                    pdf.set_font('Arial', 'B', 9)
                    pdf.set_text_color(0, 0, 0)
                    pdf.multi_cell(0, 5, f"Summary: {high_count} factors with scores 7-10 (stress contributors), {low_count} factors with scores 0-3 (stress reducers), {moderate_count} factors with scores 4-6 (neutral).", 0, 'L')
                    
                    pdf.ln(4)
                    pdf.set_font('Arial', 'I', 8)
                    pdf.set_text_color(80, 80, 80)
                    pdf.multi_cell(0, 4, "Note: Scores of 7-10 indicate high stress levels that increase overall stress. Scores of 0-3 indicate low stress levels that decrease overall stress. Scores of 4-6 have minimal impact.", 0, 'L')

                pdf.ln(5)
                # RECOMMENDATIONS SECTION
                pdf.section_title("RECOMMENDATIONS")

                # Add explanation based on stress level
                if stress_level.lower() == "high":
                    pdf.add_explanation("Based on your responses, immediate action is recommended to address your stress levels. The following strategies can help you manage stress effectively:")
                    
                    # 1. Professional Support
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_text_color(220, 53, 69)
                    pdf.cell(0, 6, "1. Professional Support (Priority)", 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    pdf.set_text_color(60, 60, 60)
                    pdf.multi_cell(0, 5, "   - Talk to a counselor or mental health professional\n   - Call a mental health helpline if you need immediate support", 0, 1)
                    pdf.ln(2)
                    
                    # 2. Immediate Self-Care
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_text_color(220, 53, 69)
                    pdf.cell(0, 6, "2. Immediate Self-Care (Today)", 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    pdf.set_text_color(60, 60, 60)
                    pdf.multi_cell(0, 5, "   - Practice deep breathing: Inhale 4 sec, hold 4 sec, exhale 4 sec\n   - Prioritize sleep - aim for 7-9 hours tonight\n   - Take a 10-15 minute walk outside", 0, 1)
                    pdf.ln(5)
                    
                    # 3. Short-term Stress Management
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_text_color(220, 53, 69)
                    pdf.cell(0, 6, "3. Short-term Stress Management", 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    pdf.set_text_color(60, 60, 60)
                    pdf.multi_cell(0, 5, "   - Break large tasks into smaller steps\n   - Set realistic daily goals\n   - Reach out to trusted friends or family", 0, 1)
                    pdf.ln(5)
                    
                    # Remember note
                    pdf.set_fill_color(255, 245, 245)
                    pdf.set_draw_color(220, 53, 69)
                    pdf.rect(10, pdf.get_y(), 190, 25, 'F')
                    pdf.set_font('Arial', 'B', 9)
                    pdf.set_text_color(220, 53, 69)
                    pdf.set_xy(15, pdf.get_y() + 2)
                    pdf.cell(0, 5, "REMEMBER", 0, 1, 'L')
                    pdf.set_font('Arial', 'I', 8)
                    pdf.set_text_color(100, 100, 100)
                    pdf.set_xy(15, pdf.get_y() + 2)
                    pdf.multi_cell(180, 4, "You are not alone - many students experience high stress. Seeking help is a sign of strength, not weakness. Take one step at a time.", 0, 1)

                elif stress_level.lower() == "moderate":
                    pdf.add_explanation("Your responses indicate moderate stress levels. Implementing these recommendations can help prevent stress from escalating:")
                    
                    # 1. Daily Habits to Reduce Stress
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_text_color(255, 140, 0)
                    pdf.cell(0, 6, "1. Daily Habits to Reduce Stress", 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    pdf.set_text_color(60, 60, 60)
                    pdf.multi_cell(0, 5, "   - 5-10 minutes of meditation daily\n   - Maintain consistent sleep schedule (7-9 hours)\n   - 20-30 minutes of physical activity daily", 0, 1)
                    pdf.ln(2)
                    
                    # 2. Work/Life Balance
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_text_color(255, 140, 0)
                    pdf.cell(0, 6, "2. Work/Life Balance", 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    pdf.set_text_color(60, 60, 60)
                    pdf.multi_cell(0, 5, "   - Create a realistic daily schedule\n   - Take regular breaks (5 min every hour)\n   - Set achievable goals", 0, 1)
                    pdf.ln(5)
                    
                    # 3. Stress Management Techniques
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_text_color(255, 140, 0)
                    pdf.cell(0, 6, "3. Stress Management Techniques", 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    pdf.set_text_color(60, 60, 60)
                    pdf.multi_cell(0, 5, "   - Practice time management\n   - Learn to say no to additional commitments\n   - Connect with supportive friends", 0, 1)
                    pdf.ln(5)
                    
                    # Remember note
                    pdf.set_fill_color(255, 252, 235)
                    pdf.set_draw_color(255, 193, 7)
                    pdf.rect(10, pdf.get_y(), 190, 25, 'F')
                    pdf.set_font('Arial', 'B', 9)
                    pdf.set_text_color(255, 140, 0)
                    pdf.set_xy(15, pdf.get_y() + 2)
                    pdf.cell(0, 5, "REMEMBER", 0, 1, 'L')
                    pdf.set_font('Arial', 'I', 8)
                    pdf.set_text_color(100, 100, 100)
                    pdf.set_xy(15, pdf.get_y() + 2)
                    pdf.multi_cell(180, 4, "Small changes today can prevent bigger problems tomorrow. It's okay to ask for help before stress becomes overwhelming.", 0, 1)

                else:  # Low Stress
                    pdf.add_explanation("Your responses indicate good stress management. Continue these healthy habits to maintain your well-being:")
                    
                    # 1. Keep Up the Good Work
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_text_color(40, 167, 69)
                    pdf.cell(0, 6, "1. Keep Up the Good Work", 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    pdf.set_text_color(60, 60, 60)
                    pdf.multi_cell(0, 5, "   - Continue your current healthy routines\n   - Keep monitoring your stress levels weekly\n   - Practice mindfulness to build resilience", 0, 1)
                    pdf.ln(2)
                    
                    # 2. Prevention Strategies
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_text_color(40, 167, 69)
                    pdf.cell(0, 6, "2. Prevention Strategies", 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    pdf.set_text_color(60, 60, 60)
                    pdf.multi_cell(0, 5, "   - Practice gratitude journaling\n   - Stay connected with friends and family\n   - Schedule regular self-care activities", 0, 1)
                    pdf.ln(5)
                    
                    # 3. Stay Proactive
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_text_color(40, 167, 69)
                    pdf.cell(0, 6, "3. Stay Proactive", 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    pdf.set_text_color(60, 60, 60)
                    pdf.multi_cell(0, 5, "   - Maintain work-life balance\n   - Get adequate sleep and exercise\n   - Eat a balanced diet", 0, 1)
                    pdf.ln(5)
                    
                    # Remember note
                    pdf.set_fill_color(240, 255, 240)
                    pdf.set_draw_color(40, 167, 69)
                    pdf.rect(10, pdf.get_y(), 190, 25, 'F')
                    pdf.set_font('Arial', 'B', 9)
                    pdf.set_text_color(40, 167, 69)
                    pdf.set_xy(15, pdf.get_y() + 2)
                    pdf.cell(0, 5, "REMEMBER", 0, 1, 'L')
                    pdf.set_font('Arial', 'I', 8)
                    pdf.set_text_color(100, 100, 100)
                    pdf.set_xy(15, pdf.get_y() + 2)
                    pdf.multi_cell(180, 4, "Maintaining these habits will help you stay resilient during challenging times!", 0, 1)

                # Separate note
                pdf.ln(30)
                pdf.set_font('Arial', 'I', 8)
                pdf.set_text_color(100, 100, 100)
                pdf.multi_cell(0, 4, "Note: This report is generated by the XAI Stress Detection System using an AdaBoost machine learning model with SHAP explanations. It is intended for educational and self-awareness purposes only. For medical concerns, please consult a healthcare professional.", 0, 1)

                # Save to bytes
                pdf_output = pdf.output(dest='S').encode('latin1')

                # Full width download button
                st.download_button(
                    label="Download Results as PDF",
                    data=pdf_output,
                    file_name=f"stress_assessment_{datetime.now().strftime('%d%m%Y')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf_button"
                )

                st.caption("Download your assessment results as a PDF report")

# ==================================== PAGE 4: SHAP EXPLANATIONS ==================================================
elif page == "🔍 SHAP Explanations":
    st.header("Explainable AI - SHAP Analysis")
    st.markdown("Understand why the model predicts a certain stress level.")
    
    if st.session_state.model is None:
        st.warning("No model loaded. Please refresh the page to load the model files first.")
    else:
        # Tab layout for different SHAP views - removed "Your Prediction" tab
        tab1, tab2 = st.tabs(["Global Importance", "Feature Impact"])
        
        with tab1:  # Global feature importance
            st.subheader("Global Feature Importance")
            st.markdown("""
            <div class="info-box">
                <strong>What this shows:</strong> Which factors most influence stress predictions across all students.
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.importance_df is not None:
                # Use actual SHAP data from models folder
                importance_df = st.session_state.importance_df.head(15)
                
                display_names = []
                for f in importance_df['Feature']:
                    clean = f.split('.')[0] if '.' in f else f
                    # Keep the full sentence, no truncation
                    display_names.append(clean)

                importance_df['Display Name'] = display_names
                
                # Show top factor
                top_feature = importance_df.iloc[0]['Display Name']
                top_importance = importance_df.iloc[0]['Importance']
                st.metric("Most Important Factor", top_feature, f"Score: {top_importance:.4f}")
                
                st.markdown("---")
                
                # Bar chart
                fig_importance = px.bar(importance_df, x='Importance', y='Display Name', 
                                    orientation='h', 
                                    title="Top 15 Features Impacting Stress",
                                    color='Importance', 
                                    color_continuous_scale='Reds',
                                    text='Importance',
                                    height=550)
                fig_importance.update_traces(texttemplate='%{text:.4f}', textposition='outside', textfont_size=10)
                fig_importance.update_layout(
                    xaxis_title="Importance Score",
                    yaxis_title="",
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Guide at bottom
                st.markdown("""
                <div class="info-box">
                    <strong>How to read:</strong><br>
                    • Higher importance score = Stronger influence on stress prediction<br>
                    • Top factors should be prioritized for intervention<br>
                    • These results are based on SHAP analysis from the trained model
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("SHAP data not found. Please ensure 'shap_global_importance.csv' is in the models folder.")
        
        with tab2:  # Feature Impact Analysis
            st.subheader("Feature Impact Analysis")
            st.markdown("""
            <div class="info-box">
                <strong>What this shows:</strong> How different scores affect stress prediction for each question.
                Select a feature below to see its impact pattern.
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.importance_df is not None:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Feature selector with cleaned names
                    available = []
                    for f in st.session_state.importance_df['Feature'].tolist()[:10]:
                        clean = f.split('.')[0] if '.' in f else f
                        if len(clean) > 35:
                            clean = clean[:32] + "..."
                        available.append(clean)
                    
                    selected = st.selectbox("Select a factor:", available)
                    
                    st.markdown("---")
                    st.markdown("**Impact Pattern**")
                    
                    # Determine pattern based on keyword
                    if 'sleep' in selected.lower():
                        st.info("Low scores (0-3): Increases stress\nGood scores (7-9): Reduces stress")
                    elif 'anxiety' in selected.lower() or 'tension' in selected.lower():
                        st.info("Higher scores = Higher stress")
                    elif 'academic' in selected.lower() or 'workload' in selected.lower():
                        st.info("Scores above 6 show increasing stress impact")
                    elif 'activity' in selected.lower() or 'exercise' in selected.lower():
                        st.info("Higher scores (7-10) reduce stress")
                    else:
                        st.info("Scores above 7 typically increase stress")
                
                with col2:
                    # Impact chart
                    x_vals = list(range(0, 11))
                    
                    if 'sleep' in selected.lower():
                        y_vals = [0.8, 0.6, 0.4, 0.2, 0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9]
                        note = "Low sleep = Higher stress | Good sleep = Lower stress"
                    elif 'anxiety' in selected.lower() or 'tension' in selected.lower():
                        y_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
                        note = "Higher anxiety = Higher stress"
                    elif 'academic' in selected.lower() or 'workload' in selected.lower():
                        y_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
                        note = "Academic pressure increases stress"
                    elif 'activity' in selected.lower():
                        y_vals = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1]
                        note = "More activity = Lower stress"
                    else:
                        y_vals = [0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8]
                        note = "Higher scores increase stress"
                    
                    impact_df = pd.DataFrame({'Score': x_vals, 'Impact': y_vals})
                    
                    fig = px.line(impact_df, x='Score', y='Impact',
                                 title=f"Impact of '{selected}' on Stress",
                                 markers=True,
                                 color_discrete_sequence=['#2E86AB'])
                    fig.update_traces(marker=dict(size=8), line=dict(width=2))
                    fig.update_layout(
                        xaxis_title="Your Score (0-10)",
                        yaxis_title="Influence Level",
                        height=350,
                        xaxis=dict(tick0=0, dtick=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"💡 {note}")
            else:
                st.info("SHAP data not available. Please upload 'shap_global_importance.csv' to enable this analysis.")

# ==================================== PAGE 5: BATCH PREDICTION =========================================================
elif page == "📂 Batch Prediction":
    st.header("Batch Prediction")
    st.markdown("Upload a CSV file with multiple student records to get stress predictions for all.")
    
    if st.session_state.model is None:
        st.warning("No model loaded. Please upload your model files first.")
    else:
        with st.expander("Required CSV Format"):
            st.markdown("Your CSV must contain ALL these columns:")
            for f in st.session_state.original_feature_names[:10]:
                st.write(f"- `{f}`")
            if len(st.session_state.original_feature_names) > 10:
                st.write(f"- ... and {len(st.session_state.original_feature_names) - 10} more")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head())
            
            missing = [f for f in st.session_state.original_feature_names if f not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing[:5]}")
            else:
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Predicting..."):
                        predictions, probabilities = predict_stress(
                            df,
                            st.session_state.model,
                            st.session_state.scaler,
                            st.session_state.label_encoder,
                            st.session_state.original_feature_names
                        )
                        
                        if predictions is not None:
                            df['Predicted_Stress'] = predictions
                            classes = get_label_encoder_classes(st.session_state.label_encoder)
                            for i, level in enumerate(classes):
                                df[f'Probability_{level}'] = probabilities[:, i]
                            
                            st.subheader("Prediction Results")
                            st.dataframe(df)
                            
                            col1, col2, col3 = st.columns(3)
                            counts = df['Predicted_Stress'].value_counts()
                            total = len(df)
                            
                            with col1:
                                st.metric("Total Students", total)
                            with col2:
                                high_count = counts.get('High', 0)
                                st.metric("High Stress", f"{high_count} ({high_count/total*100:.1f}%)")
                            with col3:
                                low_count = counts.get('Low', 0)
                                st.metric("Low Stress", f"{low_count} ({low_count/total*100:.1f}%)")
                            
                            fig = px.pie(
                                values=counts.values,
                                names=counts.index,
                                title="Stress Level Distribution",
                                color=counts.index,
                                color_discrete_map={'Low': '#28a745', 'Moderate': '#ffc107', 'High': '#dc3545'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Store in history
                            st.session_state.prediction_history.append({
                                'timestamp': pd.Timestamp.now(),
                                'data': df,
                                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
                            })
                            
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="stress_predictions.csv",
                                mime="text/csv"
                            )

# ================================ FOOTER ===========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>XAI Stress Detection System | AdaBoost + SHAP | Powered by Streamlit</p>
    <p>FYP 2 - Explainable AI for Stress Detection</p>
</div>
""", unsafe_allow_html=True)