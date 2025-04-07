
import sys
import logging

def handle_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = handle_exception

import streamlit as st


st.set_page_config(
    page_title="YOLO V8 Prompt-based Object Detection App",
    layout="wide",
    page_icon = '🧠'
    # page_icon='./images/home.png'
)


st.title("YOLO Prompt-based Object Detection App")
st.caption("Custom and real-time object detection"
           )


st.markdown("""
---

### Welcome to the YOLO V8 Prompt-based Custom Object Detection App

This web app uses  YOLO V8 pre-trained model for transfer learning on 20 different  objects, supported by a prompt to detect custom classes. You can also play a live video for real-time object detection . 

- **Custom Class Selection:** Choose the classes you want to detect using the prompt for focused object detection.
- **Adjustable Thresholds:** 
  1. **Confidence Threshold:** Adjust the confidence level to control the balance between precision and recall.
  2. **Class Score Threshold:** The minimum class probability required for detection.
- **Real-Time Object Detection:** Process live video streams using the above adjustable parameters.


""")



st.markdown("""
    <style>
        .stPageLink {
            display: block;
            font-size: 18px;
            font-weight: 600;
            color: #2E7D32;
            text-decoration: none;
            padding: 10px 20px;
            border-radius: 8px;
            background-color: #E4ECF2;
            transition: all 0.3s ease;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .stPageLink:hover {
            background-color: #E4ECF2;
            color: #1B5E20;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
    </style>
""", unsafe_allow_html=True)

st.page_link("pages/1_YOLO_Vision.py", label="Vision Processing", icon="👁️")  # Search/vision-related icon
st.page_link("pages/2_YOLO_Live.py", label="Live Processing", icon="📹")    # Camera/live video-related icon



st.markdown("""
---

**Reference Repo**: [YOLO Prompt-based Customized App](https://github.com/amerob/yolo-prompt-based-detection-app)

""")


