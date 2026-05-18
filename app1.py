import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

# IMPORT FROM UTILS
from utils.detection_utils import (
    preprocessing,
    detect_plate
)

from utils.ocr_utils import (
    preprocess_plate_for_ocr,
    extract_text_from_plate
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="ANPR System",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------
st.markdown("""
<style>

/* Main background */
[data-testid="stAppViewContainer"] {
    background-color: #0e1117 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
}

/* Text */
html, body, [class*="css"]  {
    color: #e6e6e6 !important;
}

/* Container spacing */
.block-container {
    padding-top: 2rem;
}

/* Headings */
h1, h2, h3 {
    color: #4CC9F0 !important;
}

/* Buttons */
.stButton>button {
    background-color: #2dd4bf !important;
    color: black !important;
    border-radius: 8px;
    font-weight: bold;
}

/* Download button */
.stDownloadButton>button {
    background-color: #4CC9F0 !important;
    color: black !important;
    border-radius: 8px;
    font-weight: bold;
}

/* Hyperlinks */
a {
    color: #4CC9F0 !important;
    text-decoration: none;
}

a:hover {
    transform: scale(1.05);
    transition: 0.2s;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("🚘 ANPR Guide")

# HOW TO USE
with st.sidebar.expander(" How to Use", expanded=False):

    st.write("""
    1. Upload vehicle image  
    2. Click detect button  
    3. Wait for OCR processing  
    4. View extracted text  
    5. Download cropped plate  
    """)

# SUPPORTED IMAGES
with st.sidebar.expander(" Recommended Images"):

    st.write("""
    ✔ Front vehicle images  
    ✔ Clear number plates  
    ✔ Daylight images  
    ✔ High resolution images  
    """)

# BAD IMAGES
with st.sidebar.expander(" Not Recommended"):

    st.write("""
    - Blurry vehicle images  
    - Dark/night photos  
    - Very tiny plates  
    - Extreme camera angles  
    """)

# SETTINGS
st.sidebar.title(" Settings")

show_processed = st.sidebar.checkbox(
    "Show Processed Plate",
    True
)

show_detected = st.sidebar.checkbox(
    "Show Detection Image",
    True
)

# SIDEBAR STATUS
st.sidebar.markdown("---")

st.sidebar.success("🟢 EasyOCR Loaded")
st.sidebar.success("🟢 OpenCV Loaded")

st.sidebar.markdown("---")

st.sidebar.info("""
### 💡 OCR Tips

- Clear images improve accuracy
- Avoid motion blur
- Keep plate centered
- Higher resolution gives better OCR
""")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.markdown(
    "<h2> Automatic Number Plate Recognition</h2>",
    unsafe_allow_html=True
)

st.write("""
Upload a vehicle image to detect and extract
number plate text automatically 
""")

# ---------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    " Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)

# INFO
st.info("""
###  Best Results Tips

- Use clear images
- Avoid blurry plates
- Keep vehicle close to camera
- Use proper lighting
""")

# ---------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------
if uploaded_file is not None:

    progress = st.progress(0)

    # READ IMAGE
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    col1, col2 = st.columns(2)

    # ---------------------------------------------------
    # ORIGINAL IMAGE
    # ---------------------------------------------------
    with col1:

        st.subheader(" Original Image")

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    progress.progress(20)

    # ---------------------------------------------------
    # PROCESS BUTTON
    # ---------------------------------------------------
    if st.button(" Detect Number Plate"):

        with st.spinner(
            "🔍 Detecting Number Plate..."
        ):

            # PREPROCESSING
            processed_image = preprocessing(image)

            progress.progress(40)

            # DETECTION
            result_image, cropped_plates = detect_plate(
                processed_image
            )

            progress.progress(70)

        # ---------------------------------------------------
        # DETECTED IMAGE
        # ---------------------------------------------------
        if show_detected:

            with col2:

                st.subheader(" Detection Result")

                st.image(
                    cv2.cvtColor(
                        result_image,
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )

        progress.progress(100)

        st.markdown("---")

        # ---------------------------------------------------
        # OCR RESULTS
        # ---------------------------------------------------
        if cropped_plates:

            st.success(
                f" {len(cropped_plates)} Plate(s) Detected"
            )

            os.makedirs(
                "cropped_outputs",
                exist_ok=True
            )

            for idx, plate in enumerate(cropped_plates):

                st.markdown(
                    f"## 🚘 Plate {idx+1}"
                )

                processed_plate = preprocess_plate_for_ocr(
                    plate
                )

                extracted_text = extract_text_from_plate(
                    processed_plate
                )

                c1, c2 = st.columns(2)

                # ---------------------------------------------------
                # CROPPED IMAGE
                # ---------------------------------------------------
                with c1:

                    if show_processed:

                        st.image(
                            cv2.cvtColor(
                                processed_plate,
                                cv2.COLOR_BGR2RGB
                            ),
                            caption="Processed Plate",
                            use_container_width=True
                        )

                # ---------------------------------------------------
                # OCR TEXT
                # ---------------------------------------------------
                with c2:

                    st.subheader(" Extracted Text")

                    if extracted_text:

                        st.success(
                            extracted_text
                        )

                    else:

                        st.error(
                            "No text detected"
                        )

                    st.info("""
                     If OCR is incorrect:
                    - Use clearer images
                    - Avoid blur
                    - Keep plate closer
                    """)

                # ---------------------------------------------------
                # SAVE CROPPED IMAGE
                # ---------------------------------------------------
                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                save_path = (
                    f"cropped_outputs/"
                    f"plate_{idx}_{timestamp}.jpg"
                )

                cv2.imwrite(save_path, plate)

                # DOWNLOAD BUTTON
                with open(save_path, "rb") as file:

                    st.download_button(
                        label="⬇️ Download Cropped Plate",
                        data=file,
                        file_name=f"plate_{idx}.jpg",
                        mime="image/jpeg"
                    )

                st.markdown("---")

        else:

            st.error(
                " No Number Plate Detected"
            )

            st.warning("""
            Try another image with:
            - clearer plate
            - better lighting
            - less blur
            """)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown("""
##  About Developer
""")

st.markdown("""
<div style="display:flex; gap:20px; align-items:center;">

<a href="https://github.com/" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="40"/>
</a>

<a href="https://linkedin.com/" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="40"/>
</a>

</div>

<br>

**Noman Khan**  
Computer Vision & AI Enthusiast  
Building intelligent ANPR systems 
""", unsafe_allow_html=True)