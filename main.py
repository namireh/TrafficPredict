import streamlit as st
import pandas as pd
import pickle
import mapie
import warnings
warnings.filterwarnings('ignore')


st.title('Traffic Volume Predictor') 
st.write("Utilize our advanced machne learning app to predict traffic volume")
st.image('traffic_image.gif', width = 400)

with open('XGB.pickle', 'rb') as model_pickle:
    reg = pickle.load(model_pickle)

with open('MAPIE.pickle', 'rb') as MAPIE_pickle:
    regm = pickle.load(MAPIE_pickle)

default_df = pd.read_csv('Traffic_Volume.csv')
default_df = default_df.drop(columns = ["traffic_volume"])
default_df.dropna()
default_df['holiday'] = default_df['holiday'].fillna('None')
default_df['date_time'] = pd.to_datetime(default_df['date_time'], errors='coerce')
default_df['Month'] = default_df['date_time'].dt.strftime('%B')
default_df['hour'] = default_df['date_time'].dt.hour.astype(str)
default_df['Weekday'] = default_df['date_time'].dt.strftime('%A')
default_df = default_df.drop(columns=['date_time'])

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file) if uploaded_file is not None else None

#Chat GPT helped with this
if "mode" not in st.session_state:
    st.session_state.mode = None

with st.sidebar.form("User Input"):
    st.image('traffic_sidebar.jpg', width = 400)

    with st.expander("Option 1: Upload Your CSV File", expanded=False):
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        original_df = load_data(uploaded_file)
        #chat gpt helped with condensing this line
        st.markdown("<h4 style='color: yellow;'>⚠️ Make sure CSV follows this format.</h4>", unsafe_allow_html=True)
        st.dataframe(default_df)
        upload_submit = st.form_submit_button("Predict From CSV")
        if upload_submit:
            st.session_state.mode = "upload"

    with st.expander("Option 2: Fill Out Form", expanded=False):
        st.write("### Enter Traffic Data Manually")
        holiday = st.selectbox('Is it a Holiday?', options = ['None','Columbus Day' ,'Veterans Day' ,'Thanksgiving Day',
        'Christmas Day', 'New Years Day', 'Washingtons Birthday', 'Memorial Day',
        'Independence Day', 'State Fair', 'Labor Day' ,'Martin Luther King Jr Day'])
        temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=310.0, value=0.0, step=1.0, help="Temperature in degrees Celsius.")
        rain = st.number_input('Rainfall (mm)', min_value=0.0, max_value=9831.0, value=0.0, step=1.0, help="Rainfall in millimeters.")
        snow = st.number_input('Snowfall (mm)', min_value=0.0, max_value=.51, value=0.0, step=0.01, help="Snowfall in millimeters.")
        clouds = st.number_input('Cloud Cover (%)', min_value=0.0, max_value=100.0, value=0.0, step=1.0, help="Cloud cover as a percentage.")
        weather_main = st.selectbox("Weather Condition", options=['Clouds', 'Clear', 'Rain', 'Drizzle', 'Mist', 'Haze', 'Fog', 'Thunderstorm',
        'Snow', 'Squall', 'Smoke'])
        month = st.selectbox("Month", options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
        'September', 'October', 'November', 'December'])
        weekday = st.selectbox("Weekday", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        hour = st.selectbox("Hour of the Day", options=list(range(0, 24)), help="Select the hour of the day (0-23).")
        manual_submit = st.form_submit_button("Predict Manually")
        if manual_submit:
            st.session_state.mode = "manual"


if st.session_state.mode == None:
        #chat gpt helped with condensing this line
        st.markdown("<h4 style='color: blue;'>🚹 Choose a Data Entry Type.</h4>", unsafe_allow_html=True)
else:
    st.write("Using uploaded data:" if st.session_state.mode == "upload" and uploaded_file else "Using manual data:")
    alpha = st.slider('Alpha', min_value=.01, max_value=.99, value=.1, step=.01, help="Range: 0-1")

if st.session_state.mode == "upload" and uploaded_file:
    original_df['hour'] = original_df['hour'].astype(str)
    encode_dummy_df = pd.get_dummies(original_df,columns = ['holiday','weather_main', 'month', 'weekday','hour'])
    encode_dummy_df = encode_dummy_df.reindex(columns=pd.get_dummies(default_df).columns, fill_value=0)

    predictions = []
    lower_limits = []
    upper_limits = []

    for i in range(len(encode_dummy_df)):
        row = encode_dummy_df.iloc[[i]]

        prediction, intervals = regm.predict(row, alpha=alpha)
        pred_value = prediction[0].round(0)
        lower_limit = intervals[:, 0][0][0].round(2) if intervals[:, 0][0][0].round(2) > 0 else 0
        upper_limit = intervals[:, 1][0][0].round(2)

        predictions.append(f"{pred_value}")
        lower_limits.append(f"{lower_limit}")
        upper_limits.append(f"{upper_limit}")

    original_df["Predicted Traffic Volume"] = predictions
    original_df["Confidence Interval Lower"] = lower_limits
    original_df["Confidence Interval Upper"] = upper_limits

    st.write("### Predicted Traffic Volume with Confidence Intervals")
    st.dataframe(original_df)
elif st.session_state.mode == "manual":
    manual_data = pd.DataFrame({
        'holiday': [holiday],
        'temp': [temp],
        'rain_1h': [rain],
        'snow_1h': [snow],
        'clouds_all': [clouds],
        'weather_main': [weather_main],
        'month': [month],
        'weekday': [weekday],
        'hour': [str(hour)]
    })
    encode_df = pd.get_dummies(manual_data)
    encode_df = encode_df.reindex(columns=pd.get_dummies(default_df).columns, fill_value=0)

    prediction, intervals = regm.predict(encode_df, alpha=alpha)
    pred_value = prediction[0].round(0)
    lower_limit = intervals[:, 0][0][0].round(2) if intervals[:, 0][0][0].round(2) > 0 else 0
    upper_limit = intervals[:, 1][0][0].round(2)

    st.write("### Manual Prediction Results")
    st.metric(label="Predicted Traffic Volume", value=f"{pred_value:.2f}")
    st.metric(label=f"Confidence Interval {(1-alpha) * 100}%", value=f"[{lower_limit:.2f}, {upper_limit:.2f}]")


if st.session_state.mode != None:
    st.subheader("Model Insights")
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted Vs. Actual", "Coverage Plot"])

    with tab1:
        st.write("### Feature Importance")
        st.image('feature_xgb_imp.svg')
        st.caption("Relative importance of features in prediction.")
    with tab2:
        st.write("### Histogram of Residuals")
        st.image('dis_res.svg')
        st.caption("Distribution of residuals to evaluate prediction quality.")
    with tab3:
        st.write("### Plot of Predicted Vs. Actual")
        st.image('pred_v.svg')
        st.caption("Visual comparison of predicted and actual values.")
    with tab4:
        st.write("### Coverage Plot")
        st.image('PIAC.svg')
        st.caption("Range of predictions with confidence intervals.")