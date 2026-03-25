import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="XAI Stress Detection", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4CAF50; text-align: center; }
    .prediction-low { background-color: #d4edda; padding: 20px; border-radius: 10px; text-align: center; }
    .prediction-moderate { background-color: #fff3cd; padding: 20px; border-radius: 10px; text-align: center; }
    .prediction-high { background-color: #f8d7da; padding: 20px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧠 XAI Stress Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    """Load all trained models and SHAP files"""
    try:
        model = joblib.load('models/adaboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load SHAP values
        with open('models/shap_values.pkl', 'rb') as f:
            shap_values = pickle.load(f)
        
        # Load global importance
        importance_df = pd.read_csv('models/shap_global_importance.csv')
        
        return model, scaler, label_encoder, feature_names, shap_values, importance_df
    except Exception as e:
        st.error(f"Error loading: {str(e)}")
        st.info("Please ensure all files are in 'models/' folder")
        return None, None, None, None, None, None

# Load everything
model, scaler, label_encoder, feature_names, shap_values, importance_df = load_models()

# ==================== PREDICTION FUNCTION ====================
def predict_stress(features_df):
    if model is None:
        return None, None
    try:
        X = features_df[feature_names].copy()
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        stress_levels = label_encoder.inverse_transform(predictions)
        return stress_levels, probabilities
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# ==================== SIDEBAR ====================
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard Overview", "📊 Student Self-Test", "🔍 SHAP Explanations"]
)

# ==================== PAGE 1: OVERVIEW ====================
if page == "🏠 Dashboard Overview":
    st.header("📊 Dashboard Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if model is not None:
            st.success("✅ AdaBoost Model Loaded Successfully!")
            st.write(f"**Number of features:** {len(feature_names)}")
            if label_encoder is not None:
                st.write(f"**Stress levels:** {', '.join(label_encoder.classes_)}")
    
    with col2:
        if importance_df is not None:
            st.subheader("📊 Global Feature Importance (SHAP)")
            fig = px.bar(importance_df.head(10), 
                         x='Importance', y='Feature', 
                         orientation='h',
                         title='Top 10 Features by SHAP Importance',
                         color='Importance', 
                         color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Features Used in Model")
    if feature_names:
        cols = st.columns(3)
        for idx, feature in enumerate(feature_names):
            with cols[idx % 3]:
                st.write(f"- {feature}")

# ==================== PAGE 2: STUDENT SELF-TEST ====================
elif page == "📊 Student Self-Test":
    st.header("🧪 Student Self-Test")
    st.markdown("Enter student data to predict stress level")
    
    if model is not None:
        col1, col2 = st.columns(2)
        user_input = {}
        
        # Feature ranges (adjust based on your dataset)
        feature_ranges = {
            'Heart_Rate': (60, 120, 75),
            'Sleep_Hours': (0, 12, 7),
            'Physical_Activity': (0, 15, 5),
            'Study_Hours': (0, 12, 4),
            'Social_Interaction': (0, 20, 6),
            'Workload': (1, 10, 5)
        }
        
        # Create input fields
        with col1:
            for feature in feature_names[:len(feature_names)//2]:
                if feature in feature_ranges:
                    min_val, max_val, default = feature_ranges[feature]
                    user_input[feature] = st.slider(
                        feature.replace('_', ' '), 
                        min_val, max_val, default
                    )
                else:
                    user_input[feature] = st.number_input(feature, value=0.0)
        
        with col2:
            for feature in feature_names[len(feature_names)//2:]:
                if feature in feature_ranges:
                    min_val, max_val, default = feature_ranges[feature]
                    user_input[feature] = st.slider(
                        feature.replace('_', ' '), 
                        min_val, max_val, default
                    )
                else:
                    user_input[feature] = st.number_input(feature, value=0.0)
        
        if st.button("🔮 Predict Stress Level", type="primary"):
            input_df = pd.DataFrame([user_input])
            prediction, probabilities = predict_stress(input_df)
            
            if prediction is not None:
                stress_level = prediction[0]
                
                # Display prediction
                if stress_level == "Low":
                    st.markdown(f"""
                    <div class="prediction-low">
                        <h2>✅ Low Stress Level</h2>
                        <p>The student appears to be managing stress well.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif stress_level == "Moderate":
                    st.markdown(f"""
                    <div class="prediction-moderate">
                        <h2>⚠️ Moderate Stress Level</h2>
                        <p>Some stress indicators detected.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-high">
                        <h2>🔴 High Stress Level</h2>
                        <p>Significant stress indicators detected.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probabilities
                prob_df = pd.DataFrame({
                    'Stress Level': label_encoder.classes_,
                    'Probability': probabilities[0] * 100
                })
                fig = px.bar(prob_df, x='Stress Level', y='Probability', 
                             title='Prediction Confidence',
                             color='Stress Level',
                             color_discrete_map={'Low': 'green', 'Moderate': 'orange', 'High': 'red'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if stress_level == "High":
                    st.warning("""
                    - Reduce workload and take breaks
                    - Ensure adequate sleep (7-9 hours)
                    - Practice deep breathing exercises
                    - Seek support from counselors
                    """)
                elif stress_level == "Moderate":
                    st.info("""
                    - Take regular breaks during study
                    - Incorporate light exercise
                    - Practice mindfulness
                    """)
                else:
                    st.success("""
                    - Maintain good habits
                    - Continue regular exercise
                    - Keep monitoring stress levels
                    """)
    else:
        st.warning("⚠️ Model not loaded. Please ensure all files are in 'models/' folder.")

# ==================== PAGE 3: SHAP EXPLANATIONS ====================
elif page == "🔍 SHAP Explanations":
    st.header("🔍 SHAP Explanations")
    
    if importance_df is not None:
        st.subheader("Global Feature Importance")
        st.markdown("This shows which features most influence stress prediction across all students:")
        
        fig = px.bar(importance_df, 
                     x='Importance', y='Feature', 
                     orientation='h',
                     title='SHAP Global Feature Importance',
                     color='Importance', 
                     color_continuous_scale='RdBu',
                     height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📖 How to Interpret")
        st.markdown("""
        **SHAP values explain the model's predictions:**
        - **Positive SHAP value** → Feature increases stress level
        - **Negative SHAP value** → Feature decreases stress level
        - **Larger magnitude** → Greater impact on prediction
        
        **Key Insights from your model:**
        - Features with highest importance are the strongest predictors of stress
        - Higher heart rate and lower sleep typically increase stress prediction
        - Physical activity and social interaction help reduce stress
        """)
        
        st.markdown("---")
        st.subheader("🔬 SHAP Summary Plot")
        st.markdown("Distribution of SHAP values for each feature across the dataset:")
        
        # Create a sample summary visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'][:10], importance_df['Importance'][:10], color='steelblue')
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('Feature Importance (Mean SHAP Values)')
        st.pyplot(fig)
        
    else:
        st.warning("⚠️ SHAP importance file not found. Please add 'shap_global_importance.csv' to models folder.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>FYP 2: XAI Stress Detection | AdaBoost + SHAP | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)