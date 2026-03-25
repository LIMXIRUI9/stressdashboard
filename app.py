import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import matplotlib.pyplot as plt

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="XAI Stress Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header styling */
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
    
    /* Prediction boxes */
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
    
    /* Sidebar styling */
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Feature card */
    .feature-card {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin: 5px;
        text-align: center;
    }
    
    /* Divider */
    .custom-divider {
        margin: 20px 0;
        border-top: 2px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
    st.markdown("# 🧠 XAI Stress Detection")
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📌 Navigation")
    page = st.radio(
        "Select Module:",
        [
            "🏠 Dashboard Overview",
            "📤 Upload Model Files",
            "📝 Student Self-Test",
            "🔍 SHAP Explanations",
            "📊 Batch Prediction"
        ],
        format_func=lambda x: x.replace("📤 ", "").replace("📝 ", "").replace("🔍 ", "").replace("🏠 ", "").replace("📊 ", "")
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
        - **Purpose:** Stress detection with explainable AI
        - **Output:** Low / Moderate / High Stress Levels
        """)
    
    st.markdown("---")
    st.caption("FYP 2: XAI Stress Detection System")

# ==================== FUNCTIONS ====================
def load_uploaded_model(uploaded_file):
    """Load model from uploaded file"""
    try:
        return joblib.load(uploaded_file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_uploaded_pickle(uploaded_file):
    """Load pickle file from uploaded file"""
    try:
        return pickle.load(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def predict_stress(features_df, model, scaler, label_encoder, feature_names):
    """Make prediction using loaded model"""
    try:
        X = features_df[feature_names].copy()
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        stress_levels = label_encoder.inverse_transform(predictions)
        return stress_levels, probabilities
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# ==================== PAGE 1: DASHBOARD OVERVIEW ====================
if page == "🏠 Dashboard Overview":
    st.markdown('<h1 class="main-header">🧠 XAI Stress Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explainable AI Dashboard | AdaBoost + SHAP</p>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if 'model' not in st.session_state or st.session_state.model is None:
        st.info("""
        ### 👋 Welcome to the XAI Stress Detection System!
        
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
                st.metric("Stress Classes", len(st.session_state.label_encoder.classes_))
    
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
    
    st.markdown("---")
    
    st.subheader("📋 Quick Start Guide")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload Model**")
        st.caption("Go to Upload Model Files → Upload all required files")
    with col2:
        st.markdown("**2. Test Individual**")
        st.caption("Go to Student Self-Test → Fill survey questions → Get prediction")
    with col3:
        st.markdown("**3. View Explanation**")
        st.caption("Go to SHAP Explanations → See why the model predicted that stress level")

# ==================== PAGE 2: UPLOAD MODEL FILES ====================
elif page == "📤 Upload Model Files":
    st.header("📤 Upload Trained Model Files")
    st.markdown("Upload your trained AdaBoost model and preprocessing files from Google Colab.")
    
    st.info("""
    **Required Files:**
    - `adaboost_model.pkl` - Your trained AdaBoost model
    - `scaler.pkl` - StandardScaler used during training
    - `label_encoder.pkl` - LabelEncoder for stress levels
    - `feature_names.pkl` - List of feature names
    
    **Optional Files:**
    - `shap_values.pkl` - Precomputed SHAP values
    - `shap_global_importance.csv` - Global feature importance
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Required Files")
        
        model_file = st.file_uploader("AdaBoost Model (.pkl)", type="pkl", key="model")
        scaler_file = st.file_uploader("Scaler (.pkl)", type="pkl", key="scaler")
        encoder_file = st.file_uploader("Label Encoder (.pkl)", type="pkl", key="encoder")
        features_file = st.file_uploader("Feature Names (.pkl)", type="pkl", key="features")
        
        if model_file and scaler_file and encoder_file and features_file:
            with st.spinner("Loading model files..."):
                try:
                    st.session_state.model = joblib.load(model_file)
                    st.session_state.scaler = joblib.load(scaler_file)
                    st.session_state.label_encoder = joblib.load(encoder_file)
                    st.session_state.feature_names = pickle.load(features_file)
                    
                    st.success("✅ All required files loaded successfully!")
                    st.write(f"**Features:** {len(st.session_state.feature_names)}")
                    st.write(f"**Stress Levels:** {', '.join(st.session_state.label_encoder.classes_)}")
                except Exception as e:
                    st.error(f"Error loading files: {e}")
    
    with col2:
        st.subheader("📁 Optional Files (SHAP)")
        
        shap_values_file = st.file_uploader("SHAP Values (.pkl)", type="pkl", key="shap")
        importance_file = st.file_uploader("Global Importance (.csv)", type="csv", key="importance")
        
        if shap_values_file:
            try:
                st.session_state.shap_values = pickle.load(shap_values_file)
                st.success("✅ SHAP values loaded!")
            except Exception as e:
                st.error(f"Error loading SHAP values: {e}")
        
        if importance_file:
            try:
                st.session_state.importance_df = pd.read_csv(importance_file)
                st.success("✅ Global importance loaded!")
            except Exception as e:
                st.error(f"Error loading importance: {e}")
    
    # Display loaded files summary
    st.markdown("---")
    st.subheader("📋 Loaded Files Summary")
    
    loaded_files = []
    if 'model' in st.session_state and st.session_state.model is not None:
        loaded_files.append("✅ AdaBoost Model")
    else:
        loaded_files.append("❌ AdaBoost Model")
    
    if 'scaler' in st.session_state and st.session_state.scaler is not None:
        loaded_files.append("✅ Scaler")
    else:
        loaded_files.append("❌ Scaler")
    
    if 'label_encoder' in st.session_state and st.session_state.label_encoder is not None:
        loaded_files.append("✅ Label Encoder")
    else:
        loaded_files.append("❌ Label Encoder")
    
    if 'feature_names' in st.session_state and st.session_state.feature_names is not None:
        loaded_files.append(f"✅ Feature Names ({len(st.session_state.feature_names)} features)")
    else:
        loaded_files.append("❌ Feature Names")
    
    col1, col2 = st.columns(2)
    with col1:
        for item in loaded_files[:2]:
            st.write(item)
    with col2:
        for item in loaded_files[2:]:
            st.write(item)

# ==================== PAGE 3: STUDENT SELF-TEST ====================
elif page == "📝 Student Self-Test":
    st.header("📝 Student Stress Assessment")
    st.markdown("Complete the questionnaire to get your personalized stress level assessment.")
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ No model loaded. Please upload your model files first in 'Upload Model Files' section.")
    else:
        st.subheader("📋 Questionnaire")
        st.markdown("Please answer the following questions honestly for accurate results.")
        
        col1, col2 = st.columns(2)
        user_input = {}
        
        # Feature ranges (adjust based on your dataset)
        feature_ranges = {
            'Heart_Rate': ('💓 Heart Rate', 60, 120, 75, 'bpm'),
            'Sleep_Hours': ('😴 Sleep Hours', 0, 12, 7, 'hours'),
            'Physical_Activity': ('🏃 Physical Activity', 0, 15, 5, 'hours/week'),
            'Study_Hours': ('📚 Study Hours', 0, 12, 4, 'hours/day'),
            'Social_Interaction': ('👥 Social Interaction', 0, 20, 6, 'hours/week'),
            'Workload': ('💼 Workload Level', 1, 10, 5, 'scale 1-10')
        }
        
        # Create input fields for each feature
        for idx, feature in enumerate(st.session_state.feature_names):
            if feature in feature_ranges:
                label, min_val, max_val, default, unit = feature_ranges[feature]
                with col1 if idx % 2 == 0 else col2:
                    user_input[feature] = st.slider(
                        f"{label} ({unit})",
                        min_value=min_val,
                        max_value=max_val,
                        value=default,
                        key=feature
                    )
            else:
                with col1 if idx % 2 == 0 else col2:
                    user_input[feature] = st.number_input(
                        feature.replace('_', ' ').title(),
                        value=0.0,
                        key=feature
                    )
        
        st.markdown("---")
        
        if st.button("🔮 Analyze My Stress Level", type="primary", use_container_width=True):
            with st.spinner("Analyzing your responses..."):
                input_df = pd.DataFrame([user_input])
                prediction, probabilities = predict_stress(
                    input_df, 
                    st.session_state.model, 
                    st.session_state.scaler, 
                    st.session_state.label_encoder, 
                    st.session_state.feature_names
                )
                
                if prediction is not None:
                    stress_level = prediction[0]
                    
                    if stress_level == "Low":
                        st.markdown("""
                        <div class="prediction-low">
                            <h2>✅ Low Stress Level Detected</h2>
                            <p>Your responses indicate good stress management. Keep up the healthy habits!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif stress_level == "Moderate":
                        st.markdown("""
                        <div class="prediction-moderate">
                            <h2>⚠️ Moderate Stress Level Detected</h2>
                            <p>Some stress indicators detected. Consider stress management strategies.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="prediction-high">
                            <h2>🔴 High Stress Level Detected</h2>
                            <p>Significant stress indicators detected. We recommend seeking support.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show probabilities
                    prob_df = pd.DataFrame({
                        'Stress Level': st.session_state.label_encoder.classes_,
                        'Probability': probabilities[0] * 100
                    })
                    fig = px.bar(prob_df, x='Stress Level', y='Probability', 
                                 title="Model's Prediction Confidence",
                                 color='Stress Level',
                                 color_discrete_map={'Low': '#28a745', 'Moderate': '#ffc107', 'High': '#dc3545'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Personalized Recommendations")
                    if stress_level == "High":
                        st.warning("""
                        - 🧘 Practice deep breathing or meditation daily
                        - 😴 Ensure 7-9 hours of quality sleep
                        - 🏃 Take regular breaks and exercise
                        - 🗣️ Talk to friends, family, or counselors
                        - 📅 Prioritize tasks and set realistic goals
                        """)
                    elif stress_level == "Moderate":
                        st.info("""
                        - 📝 Take regular breaks during study sessions
                        - 🚶 Go for short walks between classes
                        - 🎵 Listen to relaxing music
                        - 📅 Plan your schedule to avoid last-minute pressure
                        - 🥗 Maintain a balanced diet
                        """)
                    else:
                        st.success("""
                        - 👍 Continue your healthy habits
                        - 🏋️ Maintain regular exercise routine
                        - 😊 Keep monitoring your stress levels
                        - 👥 Stay connected with friends and family
                        """)

# ==================== PAGE 4: SHAP EXPLANATIONS ====================
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
                
                # Add legend
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

# ==================== PAGE 5: BATCH PREDICTION ====================
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
            st.markdown("**Example:**")
            example_data = {f: [75, 7, 5, 4, 6, 5] for f in st.session_state.feature_names[:6]}
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
                            for i, level in enumerate(st.session_state.label_encoder.classes_):
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

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>XAI Stress Detection System | AdaBoost + SHAP | Powered by Streamlit</p>
    <p>FYP 2 - Explainable AI for Stress Detection</p>
</div>
""", unsafe_allow_html=True)