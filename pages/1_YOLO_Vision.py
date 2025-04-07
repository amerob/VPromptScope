import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
import cv2
import tempfile
import subprocess
import os

# page configuration
st.set_page_config(
    page_title="YOLO Object Detection",
    layout="wide",
    page_icon="👁️"
)

# sidebar to control threshold
with st.sidebar:
    st.header("Threshold Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.01
    )
    class_score = st.slider(
        "Class Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.01
    )

# loading the yolo model
with st.spinner("loading the YOLO model, please wait..."):
    yolo = YOLO_Pred(
        onnx_model="./models/best_model.onnx",
        data_yaml="./models/data.yaml",
        confidence=confidence,
        class_score=class_score
    )

classes = [
    "person", "car", "chair", "bottle", "sofa", "bicycle", "horse", "boat",
    "motorbike", "cat", "tvmonitor", "cow", "sheep", "airplane", "train",
    "diningtable", "bus", "pottedplant", "bird", "dog"
]

def upload_file():
    """Handles file upload and validates the file type."""
    file = st.file_uploader(
        label="Upload an Image or Video",
        type=["png", "jpeg", "jpg", "mp4", "avi", "mov"]
    )

    if file is not None:
        size_mb = file.size / (1024 ** 2)
        file_details = {
            "filename": file.name,
            "filetype": file.type,
            "filesize": "{:,.2f} MB".format(size_mb)
        }

        if file_details["filetype"] in ("image/png", "image/jpeg"):
            st.success("Valid image file type (PNG, JPG, or JPEG)")
            return {"type": "image", "file": file, "details": file_details}
        elif file_details["filetype"] in ("video/mp4", "video/avi", "video/quicktime"):
            st.success("Valid video file type (MP4, AVI, MOV)")
            return {"type": "video", "file": file, "details": file_details}
        else:
            st.error("Invalid file type. please upload PNG, JPG, JPEG, MP4, AVI, MOV files only.")
            return None

def process_video(uploaded_file, selected_classes):

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_file.getvalue())
            input_path = tmp_input.name

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Error opening video file")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output_opencv:
            opencv_output_path = tmp_output_opencv.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(opencv_output_path, fourcc, fps, (width, height))

        # frame by frame processing
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yolo.confidence = confidence
            yolo.class_score = class_score
            pred_frame = yolo.predictions(frame_rgb, classes_to_detect=selected_classes)
            pred_frame_bgr = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR)
            out.write(pred_frame_bgr)

        cap.release()
        out.release()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output_ffmpeg:
            ffmpeg_output_path = tmp_output_ffmpeg.name

        cmd = [
            "ffmpeg",
            "-y",  # to overwr the outputfile
            "-i", opencv_output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",  # to enable streaming
            ffmpeg_output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return ffmpeg_output_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

def main():
    uploaded_file = upload_file()

    if uploaded_file:
        file_type = uploaded_file["type"]
        file_obj = uploaded_file["file"]
        details = uploaded_file["details"]

        col1, col2 = st.columns(2)

        with col1:
            st.info("uploaded File")
            if file_type == "image":
                image_obj = Image.open(file_obj)
                st.image(image_obj)
            elif file_type == "video":
                st.video(file_obj)

        with col2:
            st.subheader("file Details")
            st.json(details)
            # class selection
            selected_classes = st.multiselect("Select classes to detect:", classes, default=classes)
            button = st.button("Get detection from YOLO")

        if button:
            with st.spinner("Detecting objects, please wait..."):
                if file_type == "image":
                    image_obj = Image.open(file_obj)
                    image_array = np.array(image_obj)
                    # updating thresholds
                    yolo.confidence = confidence
                    yolo.class_score = class_score
                    pred_img = yolo.predictions(image_array, classes_to_detect=selected_classes)
                    pred_img_obj = Image.fromarray(pred_img)
                    st.subheader("Detected objects in image")
                    st.image(pred_img_obj)
                elif file_type == "video":
                    output_path = process_video(file_obj, selected_classes)
                    if output_path:
                        st.subheader("Detected objects in video")
                        st.video(output_path)

if __name__ == "__main__":
    main()
