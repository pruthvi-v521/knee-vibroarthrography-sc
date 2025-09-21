import streamlit as st
from predictor import predict_knee_condition

# App title
st.title("Knee Osteoarthritis Condition Predictor")
st.write("Enter the signal analysis values to predict knee condition:")

# Create input fields
st.subheader("Input Parameters")

rms_amplitude = st.number_input(
    "RMS Amplitude:", 
    min_value=0.5,
    max_value=1.5,
    value=1.0, 
    step=0.001,
    format="%.3f"
)

spectral_entropy = st.number_input(
    "Spectral Entropy:", 
    min_value=-4000.0,
    max_value=-1500.0,
    value=-1500.0,
    step=0.001,
    format="%.3f"
)

zero_crossing_rate = st.number_input(
    "Zero Crossing Rate:", 
    min_value=0.0, 
    max_value=0.01,
    value=0.0,
    step=0.001,
    format="%.3f"
)

mean_frequency = st.number_input(
    "Mean Frequency:", 
    min_value=30.0,
    max_value=50.0,
    value=40.0, 
    step=0.1,
    format="%.1f"
)

peak_frequency = st.number_input(
    "Peak Frequency (default):", 
    min_value=0.0, 
    max_value=100.0, 
    value=20.0, 
    step=1.0,
    format="%.1f"
)

# Predict button
if st.button("🔍 Predict Knee Condition", type="primary"):
    try:
        # Make prediction
        knee_condition, severity_level, treatment_advised = predict_knee_condition(
            rms_amplitude, spectral_entropy, zero_crossing_rate, mean_frequency, peak_frequency
        )
        
        # Display results
        st.subheader("📊 Prediction Results:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Knee Condition", knee_condition)
        
        with col2:
            st.metric("Severity Level", severity_level)
            
        with col3:
            st.metric("Treatment Advised", treatment_advised)
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error("Please check that all model files are saved correctly.")

# Add some info
st.sidebar.header("About")
st.sidebar.info("This app uses vibroarthrography data to predict knee osteoarthritis conditions using machine learning.")
