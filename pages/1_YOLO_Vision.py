import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
import cv2
import tempfile
import subprocess
import os
import imageio_ffmpeg

st.set_page_config(
    page_title="YOLO Vision",
    layout="wide",
    page_icon="👁️"
)

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

with st.spinner("Loading the YOLO model, please wait..."):
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
            st.error("Invalid file type. Please upload PNG, JPG, JPEG, MP4, AVI, or MOV files only.")
            return None

#capture image from camera
def capture_from_camera():
    captured_file = st.camera_input("capture an image")
    if captured_file is not None:
        file_details = {
            "filename": "captured_image.png",
            "filetype": "image/png",
            "filesize": f"{len(captured_file.getvalue()) / (1024 ** 2):,.2f} MB"
        }
        st.success("image captured successfully from camera!")
        return {"type": "image", "file": captured_file, "details": file_details}
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

        # Process video frame by frame
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
        
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output_ffmpeg:
            ffmpeg_output_path = tmp_output_ffmpeg.name

        cmd = [
            "ffmpeg",
            "-y",  
            "-i", opencv_output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",  
            ffmpeg_output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return ffmpeg_output_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None


def main():
    st.title("YOLO Vision")

    # choose the source type
    source_option = st.radio(
        "Select Image/Video Source:",
        ("Upload File", "Camera Capture")
    )

    if source_option == "Upload File":
        input_data = upload_file()
    else:
        input_data = capture_from_camera()

    if input_data:
        file_type = input_data["type"]
        file_obj = input_data["file"]
        details = input_data["details"]

        col1, col2 = st.columns(2)

        with col1:
            st.info("Input Preview")
            if file_type == "image":
                image_obj = Image.open(file_obj)
                st.image(image_obj)
            elif file_type == "video":
                st.video(file_obj)

        with col2:
            st.subheader("File Details")
            st.json(details)
            # class selection
            selected_classes = st.multiselect("Select classes to detect:", classes, default=classes)
            button = st.button("Run YOLO Detection")

        if button:
            with st.spinner("Detecting objects, please wait..."):
                if file_type == "image":
                    image_obj = Image.open(file_obj)
                    image_array = np.array(image_obj)
                    # update thresholds
                    yolo.confidence = confidence
                    yolo.class_score = class_score
                    pred_img = yolo.predictions(image_array, classes_to_detect=selected_classes)
                    pred_img_obj = Image.fromarray(pred_img)
                    st.subheader("Detected Objects in Image")
                    st.image(pred_img_obj)
                elif file_type == "video":
                    output_path = process_video(file_obj, selected_classes)
                    if output_path:
                        st.subheader("Detected Objects in Video")
                        st.video(output_path)


if __name__ == "__main__":
    main()
