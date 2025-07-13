import streamlit as st
from streamlit_webrtc import webrtc_streamer , VideoProcessorBase
import os
import base64
import fitz  # pip install pymupdf
from PIL import Image
import io
import cv2
import mediapipe as mp


# BASIC SETUP
st.set_page_config(
    page_title="Hidetsune.T",
    page_icon=":mortar_board:",
    layout="centered"
)



st.title("Hidetsune Takahashi")

mp_face_detection = mp.solutions.face_detection


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detector = mp_face_detection.FaceDetection(
            min_detection_confidence=0.6,
            model_selection=0
        )

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                score = detection.score
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame_bgr.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                text = f"Face: {score[0]*100:.1f}%"

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, text, (x1, int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 3)

        return frame.from_ndarray(frame_bgr, format="bgr24")

# Use st.cache_data for functions that load data (images, videos)
# This prevents re-reading files from disk on every rerun, making reloads faster.
@st.cache_data
def get_encoded_paper_images():
    """Reads and base64 encodes paper images for the sidebar."""
    data_dir = os.path.join(".", "data")
    encoded_images = {}
    image_names = {
        "task1": "task1.png",
        "task3": "task3.png",
        "task4": "task4.png",
        "task10": "task10.png"
    }

    for key, name in image_names.items():
        image_path = os.path.join(data_dir, name)
        if os.path.exists(image_path):
            try:
                with open(image_path, "rb") as image_file:
                    encoded_images[key] = base64.b64encode(image_file.read()).decode()
            except Exception as e:
                st.error(f"Error reading image {name}: {e}")
                encoded_images[key] = None
        else:
            st.warning(f"Image file not found: {name} at {image_path}")
            encoded_images[key] = None
    return encoded_images

@st.cache_data
def get_lib_detection_video_bytes():
    """Reads the LIB detection video file into bytes."""
    lib_detection_video_path = os.path.join(".", "data", "lib_detection_video.mp4")
    if os.path.exists(lib_detection_video_path):
        try:
            with open(lib_detection_video_path, "rb") as video_file:
                return video_file.read()
        except Exception as e:
            st.error(f"Error reading video file: {e}")
            return None
    else:
        st.warning(f"Video file not found: {lib_detection_video_path}")
        return None

# Helper function to create clickable image HTML
def create_clickable_image_html(encoded_image, url, description_text):
    """Generates HTML for a clickable image with a description."""
    # Always include the heading for consistency, even if image fails
    html_content = f'<h3 style="font-size: 1.1em; margin-bottom: 8px;">{description_text}</h3>'
    if encoded_image:
        html_content += f'''
            <a href="{url}" target="_blank" style="display: block;">
                <img src="data:image/png;base64,{encoded_image}" style="max-width:100%; height:auto; border-radius:8px; box-shadow:2px 2px 8px rgba(0,0,0,0.2);"/>
            </a>
        '''
    else:
        html_content += "<p style='color: red; font-size:0.9em;'>Image could not be loaded.</p>"
    return f'<div style="margin-bottom: 20px;">{html_content}</div>'


with st.sidebar:
    st.title("Academic Contributions")
    st.header("📄 SemEval 2024")

    # Call the cached function to get image data
    images_data = get_encoded_paper_images()

    st.markdown("---")  # Separator
    st.markdown(create_clickable_image_html(
        images_data.get("task1"),
        "https://aclanthology.org/2024.semeval-1.2/",
        "Textual Relatedness Evaluation System"
    ), unsafe_allow_html=True)

    st.markdown("---")  # Separator
    st.markdown(create_clickable_image_html(
        images_data.get("task10"),
        "https://aclanthology.org/2024.semeval-1.58/",
        "Emotion Detection in Complex Contexts"
    ), unsafe_allow_html=True)

    st.markdown("---")  # Separator
    st.markdown(create_clickable_image_html(
        images_data.get("task4"),
        "https://aclanthology.org/2024.semeval-1.57/",
        " Multilingual Propaganda Memes Detection"
    ), unsafe_allow_html=True)

    st.markdown("---")  # Separator
    st.markdown(create_clickable_image_html(
        images_data.get("task3"),
        "https://aclanthology.org/2024.semeval-1.55/",
        "Emotion Classification & Cause Analysis"
    ), unsafe_allow_html=True)
    st.markdown("---")  # Separator


# --- CONTENT ---
split1 = st.columns((0.2, 1))
with split1[0]:
    st.image(os.path.join(".", "data", "photo_me.jpg"), width=300)
with split1[1]:
    st.markdown(
        """
        <p style='margin-bottom:2px; font-size:30px'><b>🎓 Waseda University</b></p>
        <p style='margin-bottom:2px; font-size:22px'>School of Creative Science and Engineering</p>
        <p style='margin-bottom:2px; font-size:22px'>Dept. Modern Mechanical Engineering (4th year)</p>
        <p style='margin-bottom:2px; font-size:22px'>Tokyo, Japan</p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("Contact: [takahashi78h@toki.waseda.jp](mailto:takahashi78h@toki.waseda.jp)")

# --- SECTION 1 ---
st.header("What do I do...?", divider="green")
split2 = st.columns((1, 1, 1))
with split2[0]:
    st.markdown("""
    <h3 style='margin-bottom: 0.5em; font-size: 24px; font-weight: 600;'>
    Artificial<br>Intelligence
    </h3>
    """, unsafe_allow_html=True)

    st.image(os.path.join(".", "data", "AI_image.jpg"), width=300)
    st.markdown('<a href=#target_section_AI><button style="font-size:18px;">Know more...</button></a>', unsafe_allow_html=True)
with split2[1]:
    #st.subheader("Mechanical Engineering")
    st.markdown("""
    <h3 style='margin-bottom: 0.5em; font-size: 24px; font-weight: 600;'>
    Mechanical<br>Engineering
    </h3>
    """, unsafe_allow_html=True)

    st.image(os.path.join(".", "data", "mechanical_engineering_image.jpg"), width=300)
    st.markdown('<a href=#target_section_mechanical_engineering><button style="font-size=18px;">Know more...</button></a>', unsafe_allow_html=True)

with split2[2]:
    #st.markdown('<h3 style="white-space: nowrap;">Environmental Engineering</h3>', unsafe_allow_html=True)
    st.markdown("""
    <h3 style='margin-bottom: 0.5em; font-size: 24px; font-weight: 600;'>
    Environmental<br>Engineering
    </h3>
    """, unsafe_allow_html=True)

    st.image(os.path.join(".", "data", "earth_image.jpg"), width=300)
    st.markdown('<a href="#target_section_environmental_engineering"><button style="font-size=18px;">Know more...</button></a>', unsafe_allow_html=True)


# --- SECTION 2 ---
st.header("Overview of My Research and Studies", divider="green")

# AI anchor section
st.markdown('<a name="target_section_AI"></a>', unsafe_allow_html=True)
st.markdown('<p style="font-size:30px; color:#2196f3;"><strong>⭐Research Topics in Artificial Intelligence (AI)⭐</strong></p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:24px;"><b>➡ I fine-tune AI models to apply them in natural language and computer vision tasks.</b></p>', unsafe_allow_html=True)
split3 = st.columns(2)
with split3[0]:
    st.markdown('<p style="font-size:28px;"><strong>✅Natural Language Processing (NLP)</strong></p>', unsafe_allow_html=True)
    st.markdown('''
    <ul style="font-size:24px;">
    <li>Sentiment Analysis
        <ul>
        <li>Emotion Detection</li>
        <li>Emotion Intensity Evaluation</li>
        </ul>
    </li>
    <li>Semantic Similarity Evaluation</li>
    <li>Participation in SemEval2024 & SemEval2025</li>
    </ul>
    ''', unsafe_allow_html=True
    )




with split3[1]:
    st.markdown('<p style="font-size:28px;"><b>✅Computer Vision (CV)</b></p>', unsafe_allow_html=True)
    st.markdown('''
    <ul style="font-size:24px;">
    <li>Exploration of Various Methods for CV Tasks
        <ul>
        <li>Utilization of CNN-based Models (ex: Yolo11x)</li>
        <li>Application of Transformers (ex: Florence2)</li>
        </ul>
    </li>
    <li>Image Classification</li>
    <li>Object Detection</li>
    </ul>
    ''', unsafe_allow_html=True)



st.header("**Demo: Real-time Face Detection System**")
webrtc_streamer(key="example", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False})

# Mechanical Engineering anchor section
st.markdown('<a name="target_section_mechanical_engineering"></a>', unsafe_allow_html=True)
st.markdown('<p style="font-size:30px; color:#ff9800;"><strong>⭐Areas of Study in Mechanical Engineering⭐</strong></p>', unsafe_allow_html=True)
split4 = st.columns(2, gap="small")
with split4[0]:
    st.markdown("### ✅ Areas I studied include:")
    st.markdown('''
    <ul style="font-size:24px;">
    <li>Mechanical Dynamics</li>
    <li>Mechanics of Materials</li>
    <li>Thermodynamics</li>
    <li>Fluid Dynamics</li>
    </ul>
    ''', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size:24px; hyphens: auto; -webkit-hyphens: auto; -moz-hyphens: auto; text-align: justify; line-height:1.2;">
            &nbsp;&nbsp;&nbsp;&nbsp;As a mechanical engineering student, I explored all the fundamental areas of the field.
            Both theoretical and project-based learning helped strengthen my skills and understanding.
            While I now focus on AI development, these undergraduate studies laid a solid foundation for my growth as a future engineer.
            On the right, you can preview one of my course papers, and below, you can view a technical drawing of a shaft
            that I created in a drafting class.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.image(os.path.join(".", "data", "Z9.jpg"))
with split4[1]:
    fuel_cell_report_path = os.path.join(".", "data", "fuel_cell.pdf")
    if 'pdf_document' not in st.session_state:
        # Check if PDF file exists before opening
        if os.path.exists(fuel_cell_report_path):
            st.session_state.pdf_document = fitz.open(fuel_cell_report_path)
            st.session_state.total_pages = len(st.session_state.pdf_document)
            #st.session_state.page_slider = 14
            #st.session_state.current_page = 14  # 1-indexed
        else:
            st.error(f"PDF file not found: {fuel_cell_report_path}")
            st.session_state.pdf_document = None
            st.session_state.total_pages = 0
            st.session_state.page_slider = 0
            #st.session_state.current_page = 0

    doc = st.session_state.pdf_document
    total_pages = st.session_state.total_pages

    st.markdown("### 📄 Lab Report on Fuel Cell Experiments")

    if doc: # Only display controls and PDF if document loaded successfully
        col_prev_fuel_cell, col_slider_fuel_cell, col_next_fuel_cell = st.columns([2, 5, 2])

        with col_prev_fuel_cell:
            st.write("")
            st.write("")
            if st.button("⬅️ Previous") and st.session_state.get("page_slider", 14) > 1:
                st.session_state.page_slider = st.session_state.get("page_slider", 14) - 1

        with col_next_fuel_cell:
            st.write("")
            st.write("")
            if st.button("Next ➡️") and st.session_state.get("page_slider", 14) < total_pages:
                st.session_state.page_slider = st.session_state.get("page_slider", 14) + 1

        with col_slider_fuel_cell:
            selected_page = st.slider(
                "**Select Page**",
                min_value=1,
                max_value=total_pages,
                #value=st.session_state.page_slider,
                value=st.session_state.get("page_slider", 14),
                key="page_slider"
            )

        # Display PDF
        if doc:
            page = doc[st.session_state.page_slider - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes()))
            st.image(img, caption=f"Page {st.session_state.page_slider} of {total_pages}", use_container_width=True)
        else:
            st.info("Fuel Cell Lab Report could not be loaded.")


# Environmental Engineering anchor section
st.markdown('<a name="target_section_environmental_engineering"></a>', unsafe_allow_html=True)
st.markdown('<p style="font-size:30px; color:#4caf50;"><strong>⭐Focus Areas in Environmental Engineering⭐</strong></p>', unsafe_allow_html=True)

st.markdown("### ✅ Advancing Sorting Processes of Products with Lithium-ion Batteries")
st.markdown(
        """
        <p style="font-size:24px; hyphens: auto; -webkit-hyphens: auto; -moz-hyphens: auto; text-align: justify; line-height:1.2;">
            &nbsp;&nbsp;&nbsp;&nbsp;I am currently focusing on the application of AI in the sorting of small domestic appliances (SDAs) that contain lithium-ion batteries (LiBs)—a crucial case study
            for improving the recycling of end-of-life (EOL) products. Improper handling of LiBs can lead to fires and explosions,
            as actual incidents have increasingly occurred at sorting facilities in recent years. With my background in AI,
            I aim to address this issue by developing systems that can automatically identify LiB-containing SDAs based on their appearance.
        </p>
        """,
        unsafe_allow_html=True
    )

video_bytes = get_lib_detection_video_bytes()
if video_bytes:
    st.video(video_bytes)
else:
    st.info("LIB detection video could not be loaded.")
st.write("")


# --- SECTION 3 ---
st.header("What do I aim for...?", divider="blue")
st.markdown("""
<p style="font-size:24px; hyphens: auto; -webkit-hyphens: auto; -moz-hyphens: auto; text-align: justify; line-height:1.2;">
&nbsp;&nbsp;&nbsp;&nbsp;I am currently focused on developing an AI-powered detection system
that identifies products containing lithium-ion batteries (LiBs), as a case study to address environmental challenges.
The goal of this project is to automatically determine whether a product includes a LiB based solely on its external appearance.
Although this task has long been considered too complex for direct solutions,
my prior research suggests that recent advancements in Vision-Language Models (VLMs) and Large Language Models (LLMs)
offer new opportunities to tackle such problems.</p>
""", unsafe_allow_html=True)
st.markdown("""
<p style="font-size:24px; hyphens: auto; -webkit-hyphens: auto; -moz-hyphens: auto; text-align: justify; line-height:1.2;">
&nbsp;&nbsp;&nbsp;&nbsp;Specifically, I am working toward creating an AI-based system that could be deployed in local facilities such as grocery stores, ensuring proper sorting even for those unfamiliar with recycling rules.
I also plan to implement this system through a user-friendly web application.
What makes my approach unique is the application of Transformer models with self-attention mechanisms (see the Transformer architecture below).
In particular, I propose a text-based computer vision strategy, utilizing the high captioning abilities of VLMs
in addition to direct object detection capabilities provided by Vision Transformers (ViTs).
This approach holds potential for detecting non-trained or unseen object types,
potentially surpassing the limitations of CNN-based models such as the YOLO family with lower resource.
The image on the bottom right is from my undergraduate lecture on utilizing AI for such solutions.
</p>
""", unsafe_allow_html=True)
st.markdown("""
<p style="font-size:24px; hyphens: auto; -webkit-hyphens: auto; -moz-hyphens: auto; text-align: justify; line-height:1.2;">
&nbsp;&nbsp;&nbsp;&nbsp;Through this case study, I dedicate my student life
to establishing AI-driven solutions for some of the world’s most challenging environmental problems.
I would be happy to hear any thoughts or advice you may have on my studies.<br>
(contact:<a href="mailto:takahashi78h@toki.waseda.jp">takahashi78h@toki.waseda.jp</a>)
</p>
""", unsafe_allow_html=True)

st.write("")

split5 = st.columns(2)
with split5[0]:
    st.image(os.path.join(".", "data", "transformer_architecture.png"), width=600)
with split5[1]:
    st.image(os.path.join(".", "data", "lecture.jpg"), width=600)