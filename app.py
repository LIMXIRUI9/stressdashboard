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
</style>
""", unsafe_allow_html=True)
# ==================== END OF CSS STYLING DESIGN ====================

# ==================== HELPER FUNCTION FOR LABEL ENCODER ====================
def get_label_encoder_classes(encoder):
    """Safely get classes from label encoder regardless of format"""
    if hasattr(encoder, 'classes_'):
        return list(encoder.classes_)
    elif isinstance(encoder, dict) and 'classes_' in encoder:
        return encoder['classes_']
    elif isinstance(encoder, list):
        return encoder
    else:
        return ['Low', 'Moderate', 'High']  # Default fallback

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

def validate_uploaded_files(model_file, scaler_file, label_encoder_file, feature_names_file):
    """Validate uploaded files before loading"""
    errors = []
    warnings = []
    
    # Check file extensions
    if model_file and not (model_file.name.endswith('.pkl') or model_file.name.endswith('.joblib')):
        warnings.append(f"Model file '{model_file.name}' has unusual extension. Expected .pkl or .joblib")
    
    if scaler_file and not (scaler_file.name.endswith('.pkl') or scaler_file.name.endswith('.joblib')):
        warnings.append(f"Scaler file '{scaler_file.name}' has unusual extension. Expected .pkl or .joblib")
    
    if label_encoder_file and not label_encoder_file.name.endswith('.pkl'):
        warnings.append(f"Label encoder file '{label_encoder_file.name}' should be .pkl format")
    
    if feature_names_file and not feature_names_file.name.endswith('.pkl'):
        warnings.append(f"Feature names file '{feature_names_file.name}' should be .pkl format")
    
    # Check file sizes (warn if suspiciously small)
    if model_file and model_file.size < 1000:  # Less than 1KB
        warnings.append(f"Model file '{model_file.name}' is very small ({model_file.size} bytes). This might be corrupted or empty.")
    
    if scaler_file and scaler_file.size < 100:
        warnings.append(f"Scaler file '{scaler_file.name}' is very small ({scaler_file.size} bytes).")
    
    return errors, warnings

# ================================= SIDEBAR =========================================
with st.sidebar:
    st.image("https://i.pinimg.com/736x/ed/1c/2d/ed1c2d412a705ba463c49ad8f27ace89.jpg", width=200)
    st.markdown("# 🧠 XAI Stress Detection")
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📌 Navigation")
    page = st.sidebar.radio(
        "Select Module:",
        [
            "🏠 Dashboard Overview",
            "📤 Upload Model Files",
            "📝 Student Self-Test",
            "🔍 SHAP Explanations",
            "📊 Batch Prediction"
        ]
    )
    
    st.markdown("---")
    
    # System Information
    st.markdown("### ℹ️ System Information")
    with st.expander("📖 How to Use"):
        st.markdown("""
        **Step 1:** Upload your trained model files in 'Upload Model Files' section
        
        **Step 2:** Go to 'Student Self-Test' to assess individual stress level
        
        **Step 3:** Use 'Batch Prediction' for multiple students via CSV upload
        
        **Step 4:** View 'SHAP Explanations' to understand predictions
        """)
    
    with st.expander("🎯 About"):
        st.markdown("""
        - **ML Model:** AdaBoost Classifier
        - **XAI Method:** SHAP (SHapley Additive exPlanations)
        - **Purpose:** Stress detection with XAI Explanation
        - **Output:** Low / Moderate / High Stress Levels
        """)
    
    # Display loaded status
    st.markdown("---")
    st.markdown("### 📊 Loaded Status")
    if 'model' in st.session_state and st.session_state.model is not None:
        st.success("✅ Model Loaded")
    else:
        st.warning("❌ Model Not Loaded")
    
    if 'feature_names' in st.session_state and st.session_state.feature_names is not None:
        st.success(f"✅ {len(st.session_state.feature_names)} Features")
    else:
        st.warning("❌ Features Not Loaded")
    
    st.caption("XAI Stress Detection Dashboard")
# ================================= END OF SIDEBAR =========================================

# ================================= AUTO-LOAD MODELS =======================================
@st.cache_resource
def auto_load_models():
    models_path = 'models'
    
    if not os.path.exists(models_path):
        return None, None, None, None, None, None, f"Models folder not found"
    
    try:
        # Try loading with joblib first, then pickle for each file
        model_path = os.path.join(models_path, 'adaboost_model.pkl')
        try:
            model = joblib.load(model_path)
        except:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        scaler_path = os.path.join(models_path, 'scaler.pkl')
        try:
            scaler = joblib.load(scaler_path)
        except:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        label_encoder_path = os.path.join(models_path, 'label_encoder.pkl')
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open(os.path.join(models_path, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        
        importance_df = None
        importance_path = os.path.join(models_path, 'shap_global_importance.csv')
        if os.path.exists(importance_path):
            importance_df = pd.read_csv(importance_path)
        
        return model, scaler, label_encoder, feature_names, None, importance_df, None
        
    except Exception as e:
        return None, None, None, None, None, None, f"Error: {str(e)}"

model, scaler, label_encoder, feature_names, shap_values, importance_df, load_error = auto_load_models()

# =================================== FUNCTIONS ===================================================
def predict_stress(features_df, model, scaler, label_encoder, feature_names):
    try:
        # Ensure we only use the features the model expects
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
    st.markdown('<h1 class="main-header">🧠 XAI Stress Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explainable AI Dashboard | AdaBoost + SHAP</p>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if 'model' not in st.session_state or st.session_state.model is None:
        st.info("""
        ### 👋 Welcome to the XAI Stress Detection Dashboard!
        
        This dashboard helps you detect stress levels using machine learning with explainable AI.
        
        **To get started:**
        1. Go to **Upload Model Files** in the sidebar
        2. Upload your trained AdaBoost model, scaler, label encoder, and feature names
        3. Then use the **Student Self-Test** or **Batch Prediction** features
        """)
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Status", "✅ Loaded", delta="AdaBoost")
        with col2:
            if 'feature_names' in st.session_state:
                st.metric("Features", len(st.session_state.feature_names))
        with col3:
            if 'label_encoder' in st.session_state:
                classes = get_label_encoder_classes(st.session_state.label_encoder)
                st.metric("Stress Classes", len(classes))
        
        # Show feature names preview
        st.markdown("---")
        st.subheader("📋 Features in Your Model")
        if 'feature_names' in st.session_state:
            cols = st.columns(4)
            for idx, feature in enumerate(st.session_state.feature_names):
                with cols[idx % 4]:
                    st.write(f"- {feature}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 System Features")
        st.markdown("""
        | Feature | Description |
        |---------|-------------|
        | **Upload Model Files** | Upload your trained AdaBoost model and preprocessing files |
        | **Student Self-Test** | Individual stress assessment with instant feedback |
        | **Batch Prediction** | Upload CSV for multiple student predictions |
        | **SHAP Explanations** | Understand why the model made specific predictions |
        """)
    
    with col2:
        st.subheader("📊 XAI - SHAP Explainability")
        st.markdown("""
        SHAP (SHapley Additive exPlanations) provides:
        - **Feature importance** - Which factors most influence stress
        - **Prediction explanations** - Why each student got their result
        - **Transparency** - Understand the model's decision-making
        """)
# =============================== END OF PAGE 1: DASHBOARD OVERVIEW ========================================

# ==================================== PAGE 2: UPLOAD MODEL FILES =========================================
elif page == "📤 Upload Model Files":
    st.header("📤 Upload Your Trained Model Files")
    st.markdown("Upload all the necessary files for the stress detection system.")
    
    with st.form("upload_model_form"):
        st.subheader("📁 Required Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_file = st.file_uploader(
                "🤖 AdaBoost Model (.pkl or .joblib)",
                type=['pkl', 'joblib'],
                help="The trained AdaBoost classifier model"
            )
            
            scaler_file = st.file_uploader(
                "📊 Scaler (.pkl or .joblib)",
                type=['pkl', 'joblib'],
                help="StandardScaler used for feature scaling"
            )
            
            label_encoder_file = st.file_uploader(
                "🏷️ Label Encoder (.pkl)",
                type=['pkl'],
                help="LabelEncoder used for stress level encoding"
            )
        
        with col2:
            feature_names_file = st.file_uploader(
                "📋 Feature Names (.pkl)",
                type=['pkl'],
                help="List of feature names used in training"
            )
            
            shap_importance_file = st.file_uploader(
                "🔍 SHAP Global Importance (.csv) - Optional",
                type=['csv'],
                help="CSV file with feature importance scores from SHAP"
            )
        
        st.markdown("---")
        
        submitted = st.form_submit_button("💾 Load Model Files", type="primary", use_container_width=True)
        
        if submitted:
            # Check if all required files are uploaded
            required_files = [model_file, scaler_file, label_encoder_file, feature_names_file]
            required_names = ["Model", "Scaler", "Label Encoder", "Feature Names"]
            
            missing = []
            for i, file in enumerate(required_files):
                if file is None:
                    missing.append(required_names[i])
            
            if missing:
                st.markdown("""
                <div class="error-box">
                    <strong>❌ Missing Required Files</strong><br>
                    Please upload all required files before proceeding.
                </div>
                """, unsafe_allow_html=True)
                st.error(f"Missing files: {', '.join(missing)}")
            else:
                # Validate files before loading
                errors, warnings = validate_uploaded_files(model_file, scaler_file, label_encoder_file, feature_names_file)
                
                # Display warnings if any
                if warnings:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ File Format Warnings</strong><br>
                        Please review the following warnings:
                    </div>
                    """, unsafe_allow_html=True)
                    for warning in warnings:
                        st.warning(warning)
                
                if errors:
                    st.markdown("""
                    <div class="error-box">
                        <strong>❌ File Validation Errors</strong><br>
                        Please fix the following issues:
                    </div>
                    """, unsafe_allow_html=True)
                    for error in errors:
                        st.error(error)
                else:
                    try:
                        # Create a temporary directory to save files
                        with tempfile.TemporaryDirectory() as tmpdir:
                            # ====================== LOAD MODEL =================================
                            model_path = os.path.join(tmpdir, model_file.name)
                            with open(model_path, 'wb') as f:
                                f.write(model_file.getvalue())
                            
                            # Try loading with joblib first, then pickle
                            model = None
                            model_load_error = None
                            try:
                                model = joblib.load(model_path)
                                st.info("✅ Model loaded successfully with joblib")
                            except Exception as e1:
                                try:
                                    with open(model_path, 'rb') as f:
                                        model = pickle.load(f)
                                    st.info("✅ Model loaded successfully with pickle")
                                except Exception as e2:
                                    model_load_error = f"Failed to load model: {str(e2)}"
                            
                            if model_load_error:
                                st.markdown(f"""
                                <div class="error-box">
                                    <strong>❌ Model File Error</strong><br>
                                    {model_load_error}<br><br>
                                    <strong>Troubleshooting:</strong><br>
                                    - Ensure the file is a valid AdaBoost model<br>
                                    - Try re-saving the model with joblib.dump()<br>
                                    - Check if the file is corrupted
                                </div>
                                """, unsafe_allow_html=True)
                                st.stop()
                            
                            # ========================= LOAD SCALER ===========================
                            scaler_path = os.path.join(tmpdir, scaler_file.name)
                            with open(scaler_path, 'wb') as f:
                                f.write(scaler_file.getvalue())
                            
                            scaler = None
                            scaler_load_error = None
                            try:
                                scaler = joblib.load(scaler_path)
                                st.info("✅ Scaler loaded successfully")
                            except Exception as e1:
                                try:
                                    with open(scaler_path, 'rb') as f:
                                        scaler = pickle.load(f)
                                    st.info("✅ Scaler loaded successfully")
                                except Exception as e2:
                                    scaler_load_error = f"Failed to load scaler: {str(e2)}"
                            
                            if scaler_load_error:
                                st.markdown(f"""
                                <div class="error-box">
                                    <strong>❌ Scaler File Error</strong><br>
                                    {scaler_load_error}<br><br>
                                    <strong>Troubleshooting:</strong><br>
                                    - Ensure the file is a valid StandardScaler<br>
                                    - Try re-saving with joblib.dump()<br>
                                    - Check if the scaler was trained with the same features
                                </div>
                                """, unsafe_allow_html=True)
                                st.stop()
                            
                            # ======================== LOAD LABEL ENCODER ========================
                            le_path = os.path.join(tmpdir, label_encoder_file.name)
                            with open(le_path, 'wb') as f:
                                f.write(label_encoder_file.getvalue())
                            
                            label_encoder = None
                            try:
                                with open(le_path, 'rb') as f:
                                    label_encoder = pickle.load(f)
                                st.info("✅ Label encoder loaded successfully")
                            except Exception as e:
                                st.markdown(f"""
                                <div class="error-box">
                                    <strong>❌ Label Encoder File Error</strong><br>
                                    Failed to load label encoder: {str(e)}<br><br>
                                    <strong>Troubleshooting:</strong><br>
                                    - Ensure the file is a valid LabelEncoder object<br>
                                    - Check if the file was saved with pickle.dump()<br>
                                    - Verify the stress levels are properly encoded
                                </div>
                                """, unsafe_allow_html=True)
                                st.stop()
                            
                            # ========================== LOAD FEATURE NAMES ===========================
                            fname_path = os.path.join(tmpdir, feature_names_file.name)
                            with open(fname_path, 'wb') as f:
                                f.write(feature_names_file.getvalue())
                            
                            feature_names = None
                            try:
                                with open(fname_path, 'rb') as f:
                                    feature_names = pickle.load(f)
                                st.info("✅ Feature names loaded successfully")
                            except Exception as e:
                                st.markdown(f"""
                                <div class="error-box">
                                    <strong>❌ Feature Names File Error</strong><br>
                                    Failed to load feature names: {str(e)}<br><br>
                                    <strong>Troubleshooting:</strong><br>
                                    - Ensure the file is a list of feature names<br>
                                    - Check if the file was saved with pickle.dump()<br>
                                    - Verify all feature names match your model
                                </div>
                                """, unsafe_allow_html=True)
                                st.stop()
                            
                            # Validate feature names format
                            if not isinstance(feature_names, list):
                                st.markdown("""
                                <div class="error-box">
                                    <strong>❌ Invalid Feature Names Format</strong><br>
                                    Feature names should be a list of strings.<br>
                                    Found: {}<br><br>
                                    Please ensure the file contains a list of feature names.
                                </div>
                                """.format(type(feature_names)), unsafe_allow_html=True)
                                st.stop()
                            
                            # ======================= LOAD SHAP IMPORTANCE (Optional) ======================
                            importance_df = None
                            if shap_importance_file is not None:
                                try:
                                    importance_df = pd.read_csv(shap_importance_file)
                                    # Validate SHAP file format
                                    if 'Feature' not in importance_df.columns or 'Importance' not in importance_df.columns:
                                        st.markdown("""
                                        <div class="warning-box">
                                            <strong>⚠️ SHAP File Format Warning</strong><br>
                                            The CSV file should have 'Feature' and 'Importance' columns.<br>
                                            Found columns: {}
                                        </div>
                                        """.format(list(importance_df.columns)), unsafe_allow_html=True)
                                    else:
                                        st.info("✅ SHAP importance loaded")
                                except Exception as e:
                                    st.markdown(f"""
                                    <div class="warning-box">
                                        <strong>⚠️ SHAP File Error</strong><br>
                                        Could not load SHAP importance file: {str(e)}<br>
                                        This is optional - predictions will still work.
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Store in session state
                            st.session_state.model = model
                            st.session_state.scaler = scaler
                            st.session_state.label_encoder = label_encoder
                            st.session_state.feature_names = feature_names
                            st.session_state.importance_df = importance_df
                            
                            st.markdown("""
                            <div class="success-box">
                                <strong>✅ All Model Files Loaded Successfully!</strong><br>
                                Your AdaBoost model is now ready for stress detection.
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                            
                            # Display summary
                            st.subheader("📊 Loaded Model Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Model Type", "AdaBoost Classifier")
                            with col2:
                                st.metric("Number of Features", len(feature_names))
                            with col3:
                                classes = get_label_encoder_classes(label_encoder)
                                st.metric("Stress Classes", len(classes))
                            
                            # Display feature names
                            with st.expander("📋 View Loaded Features"):
                                cols = st.columns(4)
                                for idx, feature in enumerate(feature_names):
                                    with cols[idx % 4]:
                                        st.write(f"• {feature}")
                            
                            # Display stress classes
                            with st.expander("🏷️ View Stress Levels"):
                                stress_levels = get_label_encoder_classes(label_encoder)
                                for level in stress_levels:
                                    if level == "Low":
                                        st.success(f"• {level}")
                                    elif level == "Moderate":
                                        st.warning(f"• {level}")
                                    elif level == "High":
                                        st.error(f"• {level}")
                                    else:
                                        st.write(f"• {level}")
                            
                            if importance_df is not None:
                                with st.expander("🔍 View SHAP Global Importance (Top 10)"):
                                    st.dataframe(importance_df.head(10))
                            
                            st.info("🎯 You can now use the Student Self-Test and Batch Prediction features!")
                    
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-box">
                            <strong>❌ Unexpected Error During File Loading</strong><br>
                            {str(e)}<br><br>
                            <strong>Troubleshooting Steps:</strong><br>
                            1. Verify all files are from the same training session<br>
                            2. Ensure files are not corrupted<br>
                            3. Check that the model was trained with the same features<br>
                            4. Try re-saving the files using joblib.dump()
                        </div>
                        """, unsafe_allow_html=True)
                        st.exception(e)
    
    # Display current loaded status
    st.markdown("---")
    st.subheader("📊 Currently Loaded Status")
    
    if 'model' in st.session_state and st.session_state.model is not None:
        st.success("✅ Model files are loaded and ready to use!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "AdaBoost", help="Model is loaded")
        with col2:
            st.metric("Features", len(st.session_state.feature_names))
        with col3:
            if st.button("🗑️ Clear Loaded Models", type="secondary"):
                # Clear session state
                st.session_state.model = None
                st.session_state.scaler = None
                st.session_state.label_encoder = None
                st.session_state.feature_names = None
                st.session_state.importance_df = None
                st.rerun()
    else:
        st.warning("⚠️ No model files loaded yet. Please upload the required files above.")
# ==================================== END OF PAGE 2: UPLOAD MODEL FILES =========================================

# ==================================== PAGE 3: STUDENT SELF-TEST ===========================================
elif page == "📝 Student Self-Test":
    st.header("📝 Student Stress Assessment")
    st.markdown("Complete the questionnaire to get your personalized stress level assessment.")
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ No model loaded. Please upload your model files first in 'Upload Model Files' section.")
    else:
        # Instructions for the questionnaire
        st.markdown("""
        <div class="info-box">
            <strong>📌 How to Answer:</strong><br>
            • Most questions use a scale from 0 to 10<br>
            • <strong>0 = Low / Minimal</strong> (No stress, Distress, High Stress)<br>
            • <strong>10 = High / Severe</strong> (Extremely stressed, severe impact)<br>
            • Slide to the right for higher stress levels<br>
            • Gender and age use dropdown/number input for accurate data
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📋 Stress Assessment Questionnaire")
        st.markdown("Please rate each factor based on your recent experience (past 2 weeks):")
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        user_input = {}
        
        # Define friendly names and icons for common features with input types
        feature_info = {
            # Numerical features (0-10 scale)
            'sleep': {'icon': '😴', 'label': 'Sleep Quality', 'type': 'slider', 
                     'note': '0 = Very poor sleep, 10 = Excellent sleep',
                     'interpretation': 'Low sleep quality increases stress'},
            'sleep_hours': {'icon': '😴', 'label': 'Sleep Hours', 'type': 'slider',
                          'note': '0 = No sleep (<4 hrs), 10 = More than 10 hours',
                          'interpretation': 'Both too little and too much sleep can affect stress'},
            'physical_activity': {'icon': '🏃', 'label': 'Physical Activity Level', 'type': 'slider',
                                 'note': '0 = No exercise, 10 = Very active daily',
                                 'interpretation': 'Regular exercise reduces stress'},
            'exercise': {'icon': '🏃', 'label': 'Exercise Level', 'type': 'slider',
                        'note': '0 = No exercise, 10 = Daily intense exercise',
                        'interpretation': 'Physical activity helps manage stress'},
            'study_hours': {'icon': '📚', 'label': 'Study/Work Hours', 'type': 'slider',
                          'note': '0 = No study, 10 = Excessive study (>12 hrs/day)',
                          'interpretation': 'Long study hours can increase stress'},
            'study': {'icon': '📚', 'label': 'Study Pressure', 'type': 'slider',
                     'note': '0 = No pressure, 10 = Overwhelming pressure',
                     'interpretation': 'High academic pressure increases stress'},
            'social': {'icon': '👥', 'label': 'Social Interaction', 'type': 'slider',
                      'note': '0 = Isolated, 10 = Very social',
                      'interpretation': 'Good social connections reduce stress'},
            'social_hours': {'icon': '👥', 'label': 'Social Time', 'type': 'slider',
                           'note': '0 = No social time, 10 = Excessive social time',
                           'interpretation': 'Balance in social time is important'},
            'workload': {'icon': '💼', 'label': 'Workload', 'type': 'slider',
                        'note': '0 = Very light, 10 = Overwhelming',
                        'interpretation': 'Heavy workload increases stress'},
            'anxiety': {'icon': '😟', 'label': 'Anxiety Level', 'type': 'slider',
                       'note': '0 = No anxiety, 10 = Severe anxiety',
                       'interpretation': 'High anxiety strongly indicates stress'},
            'heart_rate': {'icon': '💓', 'label': 'Heart Rate', 'type': 'slider',
                          'note': '0 = Very low (<60 bpm), 10 = Very high (>100 bpm)',
                          'interpretation': 'Elevated heart rate may indicate stress'},
            'stress': {'icon': '⚡', 'label': 'General Stress Level', 'type': 'slider',
                      'note': '0 = No stress, 10 = Extremely stressed',
                      'interpretation': 'Direct measure of stress'},
            'mood': {'icon': '😊', 'label': 'Mood', 'type': 'slider',
                    'note': '0 = Very negative, 10 = Very positive',
                    'interpretation': 'Poor mood correlates with higher stress'},
            
            # Age - special numeric input (not 0-10 scale)
            'age': {'icon': '🎂', 'label': 'Age', 'type': 'number',
                   'min': 0, 'max': 100, 'value': 25,
                   'note': 'Enter your age in years',
                   'interpretation': 'Age can influence stress patterns'},
            
            # Categorical features (use dropdown)
            'gender': {'icon': '👤', 'label': 'Gender', 'type': 'select',
                      'options': ['Male', 'Female', 'Prefer not to say'],
                      'mapping': {'Male': 0, 'Female': 1, 'Prefer not to say': 3},
                      'note': 'Select your gender'},
        }
        
        # Store user inputs
        for idx, feature in enumerate(st.session_state.feature_names):
            # Determine column placement
            with col1 if idx % 2 == 0 else col2:
                # Try to find matching feature info
                info = None
                for key in feature_info:
                    if key in feature.lower():
                        info = feature_info[key]
                        break
                
                if info:
                    icon = info['icon']
                    label = info['label']
                    
                    # Handle different input types
                    if info.get('type') == 'select':
                        # Create dropdown for categorical features
                        selected = st.selectbox(
                            f"{icon} **{label}**",
                            options=info['options'],
                            help=info['note'],
                            key=f"select_{feature}"
                        )
                        # Convert to numerical value using mapping
                        user_input[feature] = info['mapping'][selected]
                        st.caption(f"Selected: {selected} → Value: {user_input[feature]}")
                    
                    elif info.get('type') == 'number':
                        # Special handling for age and other numeric features
                        user_input[feature] = st.number_input(
                            f"{icon} **{label}**",
                            min_value=info.get('min', 0),
                            max_value=info.get('max', 100),
                            value=info.get('value', 25),
                            step=1,
                            help=info['note'],
                            key=f"number_{feature}"
                        )
                        # Provide age interpretation
                        if 'age' in feature.lower():
                            if user_input[feature] < 18:
                                st.caption(f"👶 Age: {user_input[feature]} - Young adult")
                            elif user_input[feature] <= 25:
                                st.caption(f"🎓 Age: {user_input[feature]} - Typical university age")
                            elif user_input[feature] <= 35:
                                st.caption(f"💼 Age: {user_input[feature]} - Young professional")
                            else:
                                st.caption(f"🌟 Age: {user_input[feature]} - Adult")
                    
                    else:  # Default to slider for numerical features
                        user_input[feature] = st.slider(
                            f"{icon} **{label}**",
                            min_value=0,
                            max_value=10,
                            value=5,
                            step=1,
                            help=info['note'],
                            key=f"slider_{feature}"
                        )
                        
                        # Show current value with interpretation based on stress correlation
                        if user_input[feature] <= 3:
                            if any(word in feature.lower() for word in ['stress', 'anxiety', 'workload']):
                                st.caption(f"✅ Current: {user_input[feature]} - Good (Low risk)")
                            else:
                                st.caption(f"✅ Current: {user_input[feature]} - Good (Low risk)")
                        elif user_input[feature] <= 6:
                            if any(word in feature.lower() for word in ['stress', 'anxiety', 'workload']):
                                st.caption(f"⚠️ Current: {user_input[feature]} - Moderate (Some risk)")
                            else:
                                st.caption(f"⚠️ Current: {user_input[feature]} - Moderate (Some risk)")
                        else:
                            if any(word in feature.lower() for word in ['stress', 'anxiety', 'workload']):
                                st.caption(f"🔴 Current: {user_input[feature]} - High (Elevated risk)")
                            else:
                                st.caption(f"🔴 Current: {user_input[feature]} - High (Elevated risk)")
                
                else:
                    # Unknown feature - check if it's age related
                    if 'age' in feature.lower():
                        user_input[feature] = st.number_input(
                            f"🎂 **{feature.replace('_', ' ').title()}**",
                            min_value=0,
                            max_value=100,
                            value=25,
                            step=1,
                            help="Enter your age in years",
                            key=f"number_{feature}"
                        )
                    else:
                        # Default slider for unknown features
                        label = feature.replace('_', ' ').title()
                        user_input[feature] = st.slider(
                            f"📊 **{label}**",
                            min_value=0,
                            max_value=10,
                            value=5,
                            step=1,
                            help=f"0 = Low / Minimal, 10 = High / Severe",
                            key=f"slider_{feature}"
                        )
                        
                        if user_input[feature] <= 3:
                            st.caption(f"✅ Current: {user_input[feature]} - Good (Low risk)")
                        elif user_input[feature] <= 6:
                            st.caption(f"⚠️ Current: {user_input[feature]} - Moderate (Some risk)")
                        else:
                            st.caption(f"🔴 Current: {user_input[feature]} - High (Elevated risk)")
        
        st.markdown("---")
        
        # Add clear/reset button
        col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
        with col_reset2:
            if st.button("🔄 Reset All Values", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        
        # Analyze button
        if st.button("🔮 Analyze My Stress Level", type="primary", use_container_width=True):
            with st.spinner("Analyzing your responses with AI model..."):
                # Create dataframe with user inputs
                input_df = pd.DataFrame([user_input])
                
                # DEBUG: Show what's being sent to model (collapsible)
                with st.expander("🔍 Debug Information (Click to expand if needed)"):
                    st.write("**Input values being sent to model:**")
                    st.dataframe(pd.DataFrame([user_input]).head())
                
                # Make prediction
                prediction, probabilities = predict_stress(
                    input_df, 
                    st.session_state.model, 
                    st.session_state.scaler, 
                    st.session_state.label_encoder, 
                    st.session_state.feature_names
                )
                
                if prediction is not None:
                    # FIX: Handle numeric predictions properly
                    numeric_to_stress = {0: 'Low', 1: 'Moderate', 2: 'High'}
                    
                    # Extract prediction value
                    if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
                        pred_value = prediction[0]
                    else:
                        pred_value = prediction
                    
                    # Convert to stress level string
                    if isinstance(pred_value, (int, float, np.integer)):
                        # If it's numeric, map to stress level
                        stress_level = numeric_to_stress.get(int(pred_value), 'Unknown')
                    else:
                        # If it's already string, use as is
                        stress_level = str(pred_value).strip()
                    
                    
                    # Display prediction with appropriate styling
                    if stress_level.lower() == "low":
                        st.markdown("""
                        <div class="prediction-low">
                            <h2>✅ Low Stress Level Detected</h2>
                            <p>Great news! Your responses indicate you're managing stress well.</p>
                            <p>💚 Keep up the healthy habits! Remember to maintain balance and self-care.</p>
                            <hr>
                            <p><strong>What this means:</strong> Your current lifestyle and coping strategies are working well. You're in a good position to handle challenges that come your way.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional encouraging message for low stress
                        st.balloons()
                    
                    elif stress_level.lower() == "moderate":
                        st.markdown("""
                        <div class="prediction-moderate"><br>
                            <h2>⚠️ Moderate Stress Level Detected</h2>
                            <p>You're experiencing some stress, but there are effective ways to manage it.</p>
                            <p>💛 With some adjustments, you can reduce your stress levels.</p>
                            <hr>
                            <p><strong>What this means:</strong> Your stress levels are elevated but manageable. This is a good time to implement stress reduction strategies.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif stress_level.lower() == "high":
                        st.markdown("""
                        <div class="prediction-high">
                            <h2>🔴 High Stress Level Detected</h2>
                            <p>Your responses indicate significant stress that may be affecting your well-being.</p>
                            <p>❤️ It's important to take action now. You deserve support and care.</p>
                            <hr>
                            <p><strong>What this means:</strong> Your stress levels are concerning. This is a critical time to prioritize your mental health and seek support.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    else:
                        st.error(f"Unexpected prediction result: {stress_level}")
                        st.info("""
                        **Troubleshooting:**
                        - The model may have returned an unexpected value
                        - Please check if your model was trained with stress levels: Low, Moderate, High
                        """)
                    
                    # Show probabilities with explanation
                    st.subheader("📊 Model Prediction Confidence")
                    st.markdown("""
                    **How to read this chart:** 
                    - The bars show how confident the AI model is about each stress level
                    - Higher bars = Higher confidence in that prediction
                    - The model calculates probabilities for all three levels (Low, Moderate, High)
                    - The highest bar is the final prediction
                    """)
                    
                    classes = get_label_encoder_classes(st.session_state.label_encoder)
                    prob_df = pd.DataFrame({
                        'Stress Level': classes,
                        'Probability': probabilities[0] * 100
                    })
                    fig = px.bar(prob_df, x='Stress Level', y='Probability', 
                                 title="Model's Prediction Confidence (Higher Bar = More Confident)",
                                 color='Stress Level',
                                 color_discrete_map={'Low': '#28a745', 'Moderate': '#ffc107', 'High': '#dc3545'})
                    fig.update_layout(
                        yaxis_title="Confidence (%)", 
                        xaxis_title="Stress Level",
                        yaxis_range=[0, 100]
                    )
                    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ==================== RECOMMENDATIONS SECTION ====================
                    st.subheader("💡 Personalized Recommendations")
                    
                    # Recommendations based on the predicted stress level
                    if stress_level.lower() == "high":
                        st.error("""
                        ### 🚨 IMMEDIATE ACTIONS RECOMMENDED
                        
                        Your responses indicate HIGH stress levels. Please take immediate action:
                        
                        **1. Professional Support (Priority)**
                        - 🗣️ **Talk to a counselor** or mental health professional immediately
                        - 📞 **Call a mental health helpline** if you need immediate support
                        - 🏥 **Visit your university's counseling center** 
                        
                        **2. Immediate Self-Care (Today)**
                        - 🧘 **Practice deep breathing:** Inhale for 4 seconds, hold for 4, exhale for 4 (repeat 5 times)
                        - 😴 **Prioritize sleep** - aim for 7-9 hours tonight
                        - 🚶 **Take a 10-15 minute walk** outside in nature
                        - 💧 **Stay hydrated** and avoid caffeine and alcohol
                        - 📝 **Write down your main stressors** to identify patterns
                        
                        **3. Short-term Stress Management (This Week)**
                        - 🎯 **Break large tasks** into smaller, manageable steps
                        - ⏰ **Set realistic daily goals** - don't overcommit
                        - 🙏 **Practice mindfulness** using apps like Headspace or Calm
                        - 🚫 **Set boundaries** - learn to say no to additional commitments
                        - 👥 **Reach out to trusted friends or family** for support
                        """)
                        
                        st.info("""
                        ### 🎯 Remember:
                        - You are not alone - many students experience high stress
                        - Seeking help is a sign of strength, not weakness
                        - Your mental health is as important as your physical health
                        - Take one step at a time - small changes add up
                        """)
                    
                    elif stress_level.lower() == "moderate":
                        st.warning("""
                        ### 📋 RECOMMENDED ACTIONS
                        
                        Your responses show MODERATE stress levels. These strategies can help:
                        
                        **1. Daily Habits to Reduce Stress**
                        - 🧘 **5-10 minutes of meditation** or deep breathing daily
                        - 😴 **Maintain consistent sleep schedule** (7-9 hours)
                        - 🏃 **20-30 minutes of physical activity** daily (walking, yoga, etc.)
                        - 🥗 **Eat balanced meals**, stay hydrated, reduce caffeine
                        
                        **2. Work/Life Balance Strategies**
                        - 📝 **Create a realistic daily schedule** with built-in breaks
                        - ⏰ **Take regular breaks** (5 min every hour, 15 min every 2 hours)
                        - 🎯 **Set achievable goals** - focus on progress, not perfection
                        - 🚫 **Set boundaries** for work/study time
                        
                        **3. Social Connection**
                        - 👥 **Connect with friends or family** regularly (even virtually)
                        - 🎯 **Join a group activity or club** that interests you
                        - 💬 **Share your feelings** with someone you trust
                        
                        **4. Stress Reduction Techniques**
                        - 📖 **Practice gratitude journaling** (write 3 things you're grateful for daily)
                        - 🎵 **Listen to calming music** or nature sounds
                        - 🧘 **Try progressive muscle relaxation** (tensing and relaxing each muscle group)
                        """)

                        st.info("""
                        ### 🎯 Remember:
                        - You're not alone - many students experience moderate stress
                        - Small changes today can prevent bigger problems tomorrow
                        - It's okay to ask for help before stress becomes overwhelming
                        - Take breaks, breathe deeply, and be kind to yourself
                        """)
                    
                    else:  # Low Stress
                        st.success("""
                        ### 🌟 MAINTAIN YOUR HEALTHY HABITS
                        
                        Great job managing your stress! Continue these practices:
                        
                        **1. Keep Up the Good Work**
                        - 👍 **Continue your current healthy routines**
                        - 🏋️ **Maintain regular exercise** (3-5 times per week)
                        - 😊 **Keep monitoring your stress levels** weekly
                        - 🧘 **Practice mindfulness** to build resilience
                        
                        **2. Prevention Strategies**
                        - 📖 **Practice gratitude journaling** to maintain positive mindset
                        - 🎯 **Set achievable daily goals** to avoid future stress buildup
                        - 👥 **Stay connected with friends and family** for support network
                        - 🌱 **Learn new stress management techniques** before you need them
                        
                        **3. Early Warning Signs to Watch**
                        - 🔍 **Notice early signs of stress** (irritability, sleep changes, etc.)
                        - 🚶 **Take breaks before feeling overwhelmed**
                        - 💪 **Build resilience** through consistent healthy habits
                        - 📅 **Schedule regular self-care** activities
                        """)
                        
                        st.info("""
                        ### 🎯 Remember:
                        Even though your stress is low now, maintaining these habits will help you stay resilient during challenging times!
                        """)
# ==================================== END OF PAGE 3: STUDENT SELF-TEST ===========================================
       
# ==================================== PAGE 4: SHAP EXPLANATIONS ==================================================
elif page == "🔍 SHAP Explanations":
    st.header("🔍 Explainable AI - SHAP Analysis")
    st.markdown("Understand **WHY** the model predicts a certain stress level.")
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ No model loaded. Please upload your model files first in 'Upload Model Files' section.")
    else:
        st.info("""
        **How SHAP (SHapley Additive exPlanations) Works:**
        
        | Color | Meaning |
        |-------|---------|
        | 🔴 **Red / Positive** | This feature INCREASES stress prediction |
        | 🔵 **Blue / Negative** | This feature DECREASES stress prediction |
        | **Bar Length** | How strongly this feature influences the prediction |
        """)
        
        # Global Importance
        if 'importance_df' in st.session_state and st.session_state.importance_df is not None:
            st.subheader("📊 Global Feature Importance")
            st.markdown("Top factors affecting stress prediction across all students:")
            
            fig = px.bar(
                st.session_state.importance_df.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='SHAP Global Feature Importance - Top 10 Factors',
                color='Importance',
                color_continuous_scale='RdBu',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📁 Upload 'shap_global_importance.csv' in the Upload Model Files section to see global feature importance.")
        
        st.markdown("---")
        
        # Interactive Explanation
        st.subheader("🎯 Interactive Prediction Explanation")
        st.markdown("Enter values to see how each feature affects the prediction:")
        
        col1, col2 = st.columns(2)
        shap_input = {}
        
        for idx, feature in enumerate(st.session_state.feature_names[:6]):
            with col1 if idx % 2 == 0 else col2:
                shap_input[feature] = st.slider(
                    feature.replace('_', ' ').title(),
                    min_value=0.0,
                    max_value=10.0,
                    value=5.0,
                    key=f"shap_{feature}"
                )
        
        if st.button("🔍 Explain This Prediction", type="primary"):
            input_df = pd.DataFrame([shap_input])
            prediction, probs = predict_stress(
                input_df,
                st.session_state.model,
                st.session_state.scaler,
                st.session_state.label_encoder,
                st.session_state.feature_names
            )
            
            if prediction is not None:
                stress_level = prediction[0]
                
                if stress_level == "Low":
                    st.success(f"**Predicted Stress Level: {stress_level}**")
                elif stress_level == "Moderate":
                    st.warning(f"**Predicted Stress Level: {stress_level}**")
                else:
                    st.error(f"**Predicted Stress Level: {stress_level}**")
                
                # Create contribution chart
                st.subheader("📊 Feature Contribution Analysis")
                
                contributions = []
                for feature, value in shap_input.items():
                    if 'sleep' in feature.lower() and value < 6:
                        contributions.append((feature.replace('_', ' '), value, 'positive', (7-value)/7))
                    elif 'work' in feature.lower() and value > 6:
                        contributions.append((feature.replace('_', ' '), value, 'positive', (value-5)/5))
                    elif 'activity' in feature.lower() and value < 4:
                        contributions.append((feature.replace('_', ' '), value, 'positive', (5-value)/5))
                    elif 'heart' in feature.lower() and value > 85:
                        contributions.append((feature.replace('_', ' '), value, 'positive', (value-75)/50))
                    elif 'social' in feature.lower() and value < 4:
                        contributions.append((feature.replace('_', ' '), value, 'positive', (6-value)/6))
                    else:
                        contributions.append((feature.replace('_', ' '), value, 'negative', 0.1))
                
                contrib_df = pd.DataFrame(contributions, columns=['Feature', 'Value', 'Direction', 'Impact'])
                contrib_df = contrib_df.sort_values('Impact', ascending=True)
                
                colors = ['#dc3545' if d == 'positive' else '#28a745' for d in contrib_df['Direction']]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(contrib_df['Feature'], contrib_df['Impact'], color=colors)
                ax.set_xlabel('Impact on Stress Prediction')
                ax.set_title('Feature Contribution Analysis')
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#dc3545', label='Increases Stress'),
                    Patch(facecolor='#28a745', label='Decreases Stress')
                ]
                ax.legend(handles=legend_elements, loc='lower right')
                
                st.pyplot(fig)
                
                st.markdown("**Interpretation:**")
                for _, row in contrib_df.iterrows():
                    if row['Direction'] == 'positive':
                        st.write(f"- 🔴 **{row['Feature']}** = {row['Value']} → This factor **increases** stress")
                    else:
                        st.write(f"- 🔵 **{row['Feature']}** = {row['Value']} → This factor **decreases** stress")
# ==================================== END OF PAGE 4: SHAP EXPLANATIONS ==================================================

# ==================================== PAGE 5: BATCH PREDICTION =========================================================
elif page == "📊 Batch Prediction":
    st.header("📊 Batch Prediction - Upload CSV")
    st.markdown("Upload a CSV file with multiple student records to get stress predictions for all.")
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ No model loaded. Please upload your model files first in 'Upload Model Files' section.")
    else:
        with st.expander("📋 Required CSV Format"):
            st.markdown("Your CSV must contain these columns:")
            for f in st.session_state.feature_names:
                st.write(f"- `{f}`")
            st.markdown("**Note:** All values should be on a scale of 0-10 for consistency")
            st.markdown("**Example:**")
            example_data = {f: [5, 6, 4, 7, 5, 6] for f in st.session_state.feature_names[:6]}
            st.dataframe(pd.DataFrame(example_data))
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("📊 Uploaded Data Preview")
            st.dataframe(df.head())
            
            missing = [f for f in st.session_state.feature_names if f not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Predicting..."):
                        predictions, probabilities = predict_stress(
                            df,
                            st.session_state.model,
                            st.session_state.scaler,
                            st.session_state.label_encoder,
                            st.session_state.feature_names
                        )
                        
                        if predictions is not None:
                            df['Predicted_Stress'] = predictions
                            classes = get_label_encoder_classes(st.session_state.label_encoder)
                            for i, level in enumerate(classes):
                                df[f'Probability_{level}'] = probabilities[:, i]
                            
                            st.subheader("📈 Prediction Results")
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
                            
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="stress_predictions.csv",
                                mime="text/csv"
                            )
# ==================================== END OF PAGE 5: BATCH PREDICTION =========================================================

# ================================ FOOTER ===========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>XAI Stress Detection System | AdaBoost + SHAP | Powered by Streamlit</p>
    <p>FYP 2 - Explainable AI for Stress Detection</p>
</div>
""", unsafe_allow_html=True)