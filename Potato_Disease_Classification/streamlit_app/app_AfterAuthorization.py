import streamlit as st
import requests
from PIL import Image
import io
from datetime import datetime


API_BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{API_BASE_URL}/login"
PREDICTIONS_URL = f"{API_BASE_URL}/predictions"

st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="wide"
)

st.markdown(
    """
    <style>
      /* this selector targets the helper-text container on all text_input widgets */
      div[data-testid="InputInstructions"] {
        visibility: hidden;
      }
    </style>
    """,
    unsafe_allow_html=True
)

if 'token' not in st.session_state:
    st.session_state.token = None
if 'username' not in st.session_state:
    st.session_state.username = None

st.title("🥔 Potato Disease Classification Portal")
st.markdown("---")

st.sidebar.title("👤 User Authentication")

if st.session_state.token is None:
    st.sidebar.subheader("Please Log In")
    email = st.sidebar.text_input("Email", key="login_email")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    
    if st.sidebar.button("Login"):
        login_data = {'username': email, 'password': password}
        try:
            response = requests.post(LOGIN_URL, data=login_data)
            
            if response.status_code == 200:
                st.session_state.token = response.json()['access_token']
                st.session_state.username = email
                st.sidebar.success("Login successful!")
                st.rerun()
            else:
                st.sidebar.error(f"Login Failed: {response.json().get('detail', 'Invalid credentials')}")
        except requests.exceptions.RequestException:
            st.sidebar.error("Connection failed. Is the API server running?")

if st.session_state.token:
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.session_state.username = None
        st.rerun()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🔬 Make a New Prediction")
        uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_container_width=True)
            
            if st.button("Classify Image"):
                with st.spinner('Analyzing the leaf...'):
                    img_byte_arr = io.BytesIO()
                    image.convert('RGB').save(img_byte_arr, format='JPEG')
                    files = {'file': (uploaded_file.name, img_byte_arr.getvalue(), "image/jpeg")}
                    
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}

                    try:
                        response = requests.post(PREDICTIONS_URL, files=files, headers=headers)
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"**Prediction: {result['predicted_class']}**")
                            confidence_percent = result['confidence'] * 100
                            st.info(f"**Confidence:** {confidence_percent:.2f}%")
                        else:
                            st.error(f"Prediction failed: {response.json().get('detail')}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"API Error: {e}")

    with col2:
        st.header("📜 Your Prediction History")
        
        if st.button("Refresh History"):
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            try:
                response = requests.get(PREDICTIONS_URL, headers=headers)
                if response.status_code == 200:
                    history = response.json()
                    if not history:
                        st.info("You have no predictions yet.")
                    else:
                        for item in reversed(history):
                            dt_object = datetime.fromisoformat(item['timestamp'])
                            formatted_date = dt_object.strftime("%B %d, %Y at %I:%M %p")
                            with st.expander(f"Prediction from {formatted_date}"):
                                st.write(f"**Result:** {item['predicted_class']}")
                                st.write(f"**Confidence:** {item['confidence']*100:.2f}%")
                                st.write(f"**Original Filename:** {item['filename']}")
                else:
                    st.error(f"Failed to fetch history: {response.json().get('detail')}")
            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {e}")

else:
    st.info("👋 Welcome! Please log in using the sidebar to access the classifier.")