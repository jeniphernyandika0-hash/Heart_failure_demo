import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model = joblib.load('heart_model_lr.joblib')  # or your best model
scaler = joblib.load('scaler.joblib')
# Try to load a plain calibrator (saved as dict {'method':..., 'calibrator':...})
calibrator = None
try:
    calibrator = joblib.load('heart_model_calibrator.joblib')
    # expected keys: 'method' and 'calibrator'
    calib_method = calibrator.get('method')
    calib_obj = calibrator.get('calibrator')
except Exception:
    calibrator = None

# Helper: risk category
def get_risk_category(prob):
    """Return (label, color_hex) for a probability in [0,1].
    Thresholds: Low < 10%, Moderate 10-20%, High >= 20%.
    """
    try:
        p = float(prob)
    except Exception:
        return ("Unknown", "#95a5a6")
    if p < 0.10:
        return ("Low", "#2ecc71")
    elif p < 0.20:
        return ("Moderate", "#f39c12")
    else:
        return ("High", "#e74c3c")

st.set_page_config(page_title="Heart Disease Risk Predictor", layout='centered')
st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter patient details to estimate risk (model accuracy ~89%)")

# Layout: basic vitals on the left, categoricals on the right
with st.expander("Patient information", expanded=True):
    cols = st.columns([1, 1])
    with cols[0]:
        age = st.slider("Age", 20, 90, 50)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 60, 220, 120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
        max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 0.0, help="ST depression induced by exercise relative to rest")
        st.info("Oldpeak is ST depression induced by exercise relative to rest")
    with cols[1]:
        sex = st.selectbox("Sex", ["F", "M"], index=0)
        # chest pain type mapping: TA=typical angina, ATA=atypical, NAP=non-anginal, ASY=asymptomatic
        chest_pain = st.selectbox("Chest Pain Type", ["Typical (TA)", "Atypical (ATA)", "Non-anginal (NAP)", "Asymptomatic (ASY)"], index=3)
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["< 120 mg/dl", ">= 120 mg/dl"], index=0)
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality (ST)", "Left ventricular hypertrophy (LVH)"], index=0)
    exercise_angina = st.radio("Exercise induced angina", ["No", "Yes"], index=0)
    st_slope_selected = st.selectbox("ST Slope", ["Up", "Flat", "Down"], index=0)

    # Advanced inputs grouped separately
    with st.expander('Advanced / additional features (optional)', expanded=False):
        ca = st.number_input('Number of major vessels (0-3) colored by fluoroscopy (ca)', 0, 3, 0)
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed defect', 'Reversible defect'], index=0)

        st.caption('These advanced features (ca, thal) may not be present in the deployed scaler and will be ignored if not used at training time.')

if st.button("Predict Risk"):
    # Map categorical inputs to one-hot style columns expected by the scaler
    input_dict = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
    }
    # Sex
    input_dict['Sex_M'] = 1 if sex == 'M' else 0

    # Chest pain -> set the one-hot expected by scaler if present
    # Our scaler expects columns like ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA
    chest_code = chest_pain.split('(')[-1].strip(')') if '(' in chest_pain else chest_pain
    input_dict['ChestPainType_ATA'] = 1 if chest_code == 'ATA' else 0
    input_dict['ChestPainType_NAP'] = 1 if chest_code == 'NAP' else 0
    input_dict['ChestPainType_TA'] = 1 if chest_code == 'TA' else 0

    # Fasting blood sugar
    input_dict['FastingBS_1'] = 1 if fasting_bs.startswith('>=') else 0

    # Resting ECG
    input_dict['RestingECG_Normal'] = 1 if resting_ecg.startswith('Normal') else 0
    input_dict['RestingECG_ST'] = 1 if 'ST' in resting_ecg else 0

    # Exercise angina
    input_dict['ExerciseAngina_Y'] = 1 if exercise_angina == 'Yes' else 0

    # ST Slope
    input_dict['ST_Slope_Flat'] = 1 if st_slope_selected == 'Flat' else 0
    input_dict['ST_Slope_Up'] = 1 if st_slope_selected == 'Up' else 0

    # Advanced: ca, thal included if scaler expects them; otherwise ignored by reindex
    input_dict['ca'] = ca
    input_dict['thal'] = thal
    
    # Create full DataFrame with all columns from training
    input_df = pd.DataFrame([input_dict])
    # Re-order columns to match exactly what scaler/model expects
    # Try to get feature names from the fitted scaler or model (set when fitted on a DataFrame)
    feature_cols = None
    if hasattr(scaler, 'feature_names_in_'):
        feature_cols = list(getattr(scaler, 'feature_names_in_'))
    elif hasattr(model, 'feature_names_in_'):
        feature_cols = list(getattr(model, 'feature_names_in_'))

    # Fallback: minimal explicit list (adjust if your model expects more one-hot columns)
    if feature_cols is None:
        feature_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'Sex_M']

    input_df = input_df.reindex(columns=feature_cols, fill_value=0)
    
    # Scale
    input_scaled = scaler.transform(input_df)

    # Uncalibrated prediction
    prob_uncal = model.predict_proba(input_scaled)[0][1]
    pred_class = model.predict(input_scaled)[0]

    # Compute calibrated probability if calibrator is available
    prob_cal = None
    if calibrator is not None and calib_obj is not None:
        # get raw scores
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(input_scaled)
        else:
            p = np.clip(model.predict_proba(input_scaled)[:, 1], 1e-15, 1 - 1e-15)
            scores = np.log(p / (1 - p))

        if calib_method == 'sigmoid':
            prob_cal = float(calib_obj.predict_proba(scores.reshape(-1, 1))[0][1])
        else:
            # isotonic or other
            prob_cal = float(calib_obj.predict(scores)[0])

    # Let user choose whether to display calibrated probability
    use_calibrated = False
    if prob_cal is not None:
        use_calibrated = st.checkbox('Use calibrated probability (if available)', value=True)
    # Small explanatory note about calibration
    st.caption("Calibration adjusts the model's raw scores to better match observed frequencies\n" 
           "on a validation set. 'Isotonic' is flexible and can fit non-linear maps;\n" 
           "'Sigmoid' (Platt) is smoother. Use calibrated probabilities for more\n" 
           "reliable risk estimates, especially at low/high extremes.")

    display_prob = prob_cal if (use_calibrated and prob_cal is not None) else prob_uncal

    # Color-coded risk category display
    label, color = get_risk_category(display_prob)
    risk_html = f"""
    <div style='padding:14px;border-radius:8px;background:{color};color:#fff;text-align:center'>
        <strong style='font-size:18px'>Estimated risk: {display_prob:.1%} — {label}</strong>
    </div>
    """
    st.markdown(risk_html, unsafe_allow_html=True)

    # Legend
    legend_html = """
    <div style='display:flex;gap:8px;margin-top:8px'>
        <div style='background:#2ecc71;width:18px;height:18px;border-radius:4px'></div><div>Low (&lt;10%)</div>
        <div style='width:12px'></div>
        <div style='background:#f39c12;width:18px;height:18px;border-radius:4px'></div><div>Moderate (10–20%)</div>
        <div style='width:12px'></div>
        <div style='background:#e74c3c;width:18px;height:18px;border-radius:4px'></div><div>High (&gt;=20%)</div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    # Show both values when calibrator exists for transparency
    if prob_cal is not None:
        st.write(f"Uncalibrated probability: {prob_uncal:.1%}")
        st.write(f"Calibrated probability ({calib_method}): {prob_cal:.1%}")

    # Show feature importance for logistic regression models
    try:
        if hasattr(model, 'coef_'):
            coefs = model.coef_.ravel()
            import pandas as _pd
            fi = _pd.Series(data=coefs, index=feature_cols).abs().sort_values(ascending=False).head(10)
            st.subheader('Top feature importances (absolute coefficient)')
            st.bar_chart(fi)
    except Exception:
        # don't block the app if plotting fails
        pass

    if pred_class == 1:
        st.warning("Higher risk detected — consider medical consultation")
    else:
        st.info("Lower risk detected")

    with st.expander('Disclaimer / Notes', expanded=False):
        st.markdown(
            """
            **Disclaimer:** This tool provides probabilistic estimates generated by a machine learning model trained on a public dataset. It is for educational/demo purposes only and is not medical advice.
            Predictions may be inaccurate for populations different from the training data. Always consult a qualified healthcare professional for medical decisions.
            """
        )
        