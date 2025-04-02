import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from yolo_predictions import YOLO_Pred

st.set_page_config(
    page_title="Real-Time YOLO Detection",
    page_icon="🎥",
    layout="centered"
)


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.model = YOLO_Pred(
            onnx_model='models/best_model.onnx',
            data_yaml='models/data.yaml'
        )
        self.confidence_threshold = 0.4  # default conf threshold

    def set_confidence(self, threshold):
        self.confidence_threshold = threshold

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        processed_img = self.model.predictions(img)

        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")



st.title("Real-time Object Detection with YOLOv8")


with st.sidebar:
    st.header("Threshold Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        help="adjust the minimum confidence level for object detection"
    )

# webRTC component
ctx = webrtc_streamer(
    key="yolo-live-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOVideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

# updating confidence threshold
if ctx.video_processor:
    ctx.video_processor.set_confidence(confidence_threshold)

