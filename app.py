import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="XAI Stress Detection", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧠 XAI Stress Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by AdaBoost & XAI</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state for model and data
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard Overview", "📁 Upload & Train Model", "🧪 Student Self-Test", "🔍 XAI Explanations"]
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    "**How it works:**\n"
    "1. Upload your stress dataset\n"
    "2. Train AdaBoost model\n"
    "3. Students can self-test or upload data\n"
    "4. Get predictions with SHAP explanations"
)

# Function to generate sample dataset
def generate_sample_data(n_samples=500):
    np.random.seed(42)
    
    # Generate synthetic stress-related features
    heart_rate = np.random.normal(75, 15, n_samples)
    sleep_hours = np.random.normal(7, 1.5, n_samples)
    physical_activity = np.random.normal(5, 2, n_samples)
    study_hours = np.random.normal(4, 2, n_samples)
    social_interaction = np.random.normal(6, 2, n_samples)
    workload = np.random.normal(5, 2, n_samples)
    
    # Generate stress labels based on rules
    stress_score = (heart_rate > 85).astype(int) * 0.3 + \
                   (sleep_hours < 6).astype(int) * 0.3 + \
                   (physical_activity < 3).astype(int) * 0.2 + \
                   (study_hours > 6).astype(int) * 0.1 + \
                   (workload > 6).astype(int) * 0.1
    
    stress_level = np.where(stress_score > 0.5, "High", 
                           np.where(stress_score > 0.2, "Moderate", "Low"))
    
    df = pd.DataFrame({
        'Heart_Rate': heart_rate,
        'Sleep_Hours': sleep_hours,
        'Physical_Activity': physical_activity,
        'Study_Hours': study_hours,
        'Social_Interaction': social_interaction,
        'Workload': workload,
        'Stress_Level': stress_level
    })
    
    return df

# Function to train model
def train_model(df, target_col='Stress_Level'):
    try:
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Handle categorical features
        X_processed = X.copy()
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X_processed[col] = LabelEncoder().fit_transform(X_processed[col])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train AdaBoost
        model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store in session state
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.label_encoder = le
        st.session_state.feature_names = feature_cols
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.y_train = y_train
        
        return model, accuracy, le.classes_
    
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, 0, None

# Function to predict stress
def predict_stress(features_df):
    if st.session_state.model is None:
        return None, None
    
    try:
        # Process features
        X = features_df[st.session_state.feature_names].copy()
        
        # Handle categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        
        # Scale
        X_scaled = st.session_state.scaler.transform(X)
        
        # Predict
        predictions = st.session_state.model.predict(X_scaled)
        probabilities = st.session_state.model.predict_proba(X_scaled)
        
        # Decode predictions
        stress_levels = st.session_state.label_encoder.inverse_transform(predictions)
        
        return stress_levels, probabilities
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Dashboard Overview Page
if page == "📊 Dashboard Overview":
    st.header("📊 Dashboard Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Model Performance")
        if st.session_state.model is not None:
            accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
            st.metric("Model Accuracy", f"{accuracy:.2%}")
            
            # Confusion Matrix
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=st.session_state.label_encoder.classes_,
                        yticklabels=st.session_state.label_encoder.classes_, ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        else:
            st.info("No model trained yet. Go to 'Upload & Train Model' to get started.")
    
    with col2:
        st.subheader("📊 Sample Data Preview")
        if 'data' in st.session_state:
            st.dataframe(st.session_state.data.head())
        else:
            st.info("No data loaded. Upload your dataset or use sample data.")
    
    st.markdown("---")
    st.subheader("🎯 Key Features")
    
    feature_importance = st.session_state.model.feature_importances_ if st.session_state.model is not None else None
    if feature_importance is not None:
        importance_df = pd.DataFrame({
            'Feature': st.session_state.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance (AdaBoost)',
                     color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

# Upload & Train Model Page
elif page == "📁 Upload & Train Model":
    st.header("📁 Upload Dataset & Train Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if st.button("Use Sample Dataset"):
            df = generate_sample_data()
            st.session_state.data = df
            st.success("Sample dataset loaded!")
            st.dataframe(df.head())
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.success("Dataset loaded successfully!")
        st.dataframe(df.head())
        
        # Show dataset info
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
    
    with col2:
        st.subheader("Train Model")
        if 'data' in st.session_state:
            target_col = st.selectbox("Select Target Column", 
                                      st.session_state.data.columns,
                                      index=len(st.session_state.data.columns)-1 if len(st.session_state.data.columns) > 0 else 0)
            
            if st.button("🚀 Train AdaBoost Model"):
                with st.spinner("Training model..."):
                    model, accuracy, classes = train_model(st.session_state.data, target_col)
                    if model is not None:
                        st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
                        st.write(f"Stress levels: {', '.join(classes)}")
        else:
            st.info("Please upload a dataset first or use sample data.")

# Student Self-Test Page
elif page == "🧪 Student Self-Test":
    st.header("🧪 Student Self-Test")
    st.markdown("Answer the following questions to assess your stress level")
    
    if st.session_state.model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            heart_rate = st.slider("Heart Rate (bpm)", 60, 120, 80)
            sleep_hours = st.slider("Sleep Hours (per day)", 0, 12, 7)
            physical_activity = st.slider("Physical Activity (hours/week)", 0, 15, 5)
            study_hours = st.slider("Study Hours (per day)", 0, 12, 4)
        
        with col2:
            social_interaction = st.slider("Social Interaction (hours/week)", 0, 20, 6)
            workload = st.slider("Perceived Workload (1-10)", 1, 10, 5)
            stress_history = st.selectbox("Past Stress History", ["Low", "Moderate", "High"])
        
        # Create feature dataframe
        test_data = pd.DataFrame({
            'Heart_Rate': [heart_rate],
            'Sleep_Hours': [sleep_hours],
            'Physical_Activity': [physical_activity],
            'Study_Hours': [study_hours],
            'Social_Interaction': [social_interaction],
            'Workload': [workload]
        })
        
        if st.button("🔮 Predict My Stress Level", type="primary"):
            with st.spinner("Analyzing..."):
                prediction, probabilities = predict_stress(test_data)
                
                if prediction is not None:
                    st.markdown("---")
                    st.subheader("📊 Your Results")
                    
                    # Display prediction
                    stress_level = prediction[0]
                    colors = {"Low": "green", "Moderate": "orange", "High": "red"}
                    
                    st.markdown(f"""
                    <div class="prediction-box" style="background-color: {colors[stress_level]}20;">
                        <h2>Stress Level: <span style="color: {colors[stress_level]};">{stress_level}</span></h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show probabilities
                    prob_df = pd.DataFrame({
                        'Stress Level': st.session_state.label_encoder.classes_,
                        'Probability': probabilities[0] * 100
                    })
                    fig = px.bar(prob_df, x='Stress Level', y='Probability', 
                                 title='Prediction Probabilities',
                                 color='Stress Level',
                                 color_discrete_map={'Low': 'green', 'Moderate': 'orange', 'High': 'red'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    if stress_level == "High":
                        st.warning("""
                        - Consider reducing workload
                        - Practice deep breathing exercises
                        - Ensure adequate sleep (7-9 hours)
                        - Seek support from counselors or trusted individuals
                        """)
                    elif stress_level == "Moderate":
                        st.info("""
                        - Take regular breaks during study sessions
                        - Incorporate light exercise into your routine
                        - Practice mindfulness or meditation
                        - Maintain a balanced schedule
                        """)
                    else:
                        st.success("""
                        - Keep up the good habits!
                        - Maintain regular exercise and sleep schedule
                        - Continue monitoring stress levels
                        """)
    else:
        st.warning("⚠️ No model trained yet. Please go to 'Upload & Train Model' to train a model first.")
        if st.button("Train with Sample Data"):
            df = generate_sample_data()
            st.session_state.data = df
            train_model(df)

# XAI Explanations Page
elif page == "🔍 XAI Explanations":
    st.header("🔍 Explainable AI with SHAP")
    st.markdown("Understand why the model made its predictions")
    
    if st.session_state.model is not None:
        st.subheader("How SHAP Works")
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** explains predictions by showing:
        - Which features contributed most to the prediction
        - Whether each feature increased or decreased stress level
        - The magnitude of each feature's impact
        """)
        
        # Calculate SHAP values
        with st.spinner("Calculating SHAP explanations..."):
            # Create a SHAP explainer
            explainer = shap.TreeExplainer(st.session_state.model)
            
            # Get sample for explanation
            X_sample = st.session_state.X_test[:100]
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            st.subheader("Feature Impact Summary")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, 
                             feature_names=st.session_state.feature_names,
                             show=False)
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("Interactive Prediction Explanation")
            
            # Let user input values for explanation
            st.markdown("Enter values to see SHAP explanation for your prediction:")
            
            cols = st.columns(len(st.session_state.feature_names))
            user_input = {}
            
            for idx, feature in enumerate(st.session_state.feature_names):
                with cols[idx % len(cols)]:
                    user_input[feature] = st.number_input(
                        feature, 
                        value=float(st.session_state.X_test[0][idx]) if len(st.session_state.X_test) > 0 else 0.0,
                        key=f"shap_{feature}"
                    )
            
            if st.button("Explain This Prediction"):
                input_df = pd.DataFrame([user_input])
                
                # Scale input
                input_scaled = st.session_state.scaler.transform(input_df)
                
                # Get prediction
                pred = st.session_state.model.predict(input_scaled)[0]
                pred_label = st.session_state.label_encoder.inverse_transform([pred])[0]
                
                st.info(f"Predicted Stress Level: **{pred_label}**")
                
                # Get SHAP explanation for this instance
                shap_values_instance = explainer.shap_values(input_scaled)
                
                # Create force plot
                st.subheader("SHAP Force Plot")
                fig, ax = plt.subplots(figsize=(20, 3))
                shap.force_plot(explainer.expected_value, shap_values_instance[0], 
                               input_df, matplotlib=True, show=False)
                st.pyplot(fig)
                
                # Feature contribution table
                st.subheader("Feature Contributions")
                contributions = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'SHAP Value': shap_values_instance[0]
                }).sort_values('SHAP Value', key=abs, ascending=False)
                
                fig = px.bar(contributions, x='SHAP Value', y='Feature', orientation='h',
                            title='Feature Contributions to Prediction',
                            color='SHAP Value', color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No model trained yet. Please train a model first to see SHAP explanations.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>XAI Stress Detection System | Powered by AdaBoost & SHAP</p>
</div>
""", unsafe_allow_html=True)