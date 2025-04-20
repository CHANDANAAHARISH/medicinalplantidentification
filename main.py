import streamlit as st
import cv2
import numpy as np
import os
from openai_vision import identify_plant

# Page config
st.set_page_config(
    page_title="Plant Identifier",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    /* Light theme styling */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #ffffff 0%, #f0f9f4 100%);
        color: #2d3748;
    }

    /* Header styling */
    .stTitle {
        color: #8B4513;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(240,249,244,0.9) 0%, rgba(209,250,229,0.7) 100%);
        box-shadow: 0 4px 15px -1px rgba(0,0,0,0.1);
        border: 1px solid #8B4513;
        text-shadow: 2px 2px 4px rgba(139, 69, 19, 0.2);
    }

    /* Upload section styling */
    .upload-section {
        background: #f0fdf4;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border: 2px dashed #10b981;
    }

    /* Results section styling */
    .stSuccess, .stInfo, .stWarning {
        padding: 1rem !important;
        border-radius: 10px !important;
        margin: 1rem 0 !important;
        font-weight: 500 !important;
        background: #f0fdf4 !important;
        border: 1px solid #10b981 !important;
    }

    /* Dataset section styling */
    .dataset-stats {
        background: #f0fdf4;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px -1px rgba(0,0,0,0.1);
        margin-top: 1rem;
        border: 1px solid #10b981;
    }

    /* Navigation styling */
    .stSelectbox {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        border: 1px solid #10b981 !important;
        color: #2d3748 !important;
    }

    .stSelectbox > div[role="button"] {
        color: #2d3748 !important;
    }

    /* Image display styling */
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 15px -1px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #10b981;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff !important;
    }

    .css-1r6slb0 {  /* Navigation container */
        background-color: #f0fdf4 !important;
        border: 1px solid #10b981 !important;
        padding: 1rem !important;
        border-radius: 10px !important;
    }

    /* Radio button styling */
    .st-cc {
        background-color: #ffffff !important;
        border-color: #10b981 !important;
    }

    .st-dd {
        background-color: #059669 !important;
    }

    .st-ce {
        color: #059669 !important;
    }

    /* Navigation text color */
    .st-c0 {
        color: #059669 !important;
    }

    .css-16idsys p {
        color: #059669 !important;
    }

    /* Hover effects */
    .st-ce:hover {
        color: #047857 !important;
        background-color: #f0fdf4 !important;
    }

    /* Text colors */
    p, h1, h2, h3, h4, h5, h6 {
        color: #2d3748;
    }

    /* Button styling */
    .stButton>button {
        background-color: #ffffff !important;
        color: #059669 !important;
        border: 1px solid #10b981 !important;
    }

    .stButton>button:hover {
        border-color: #059669 !important;
        box-shadow: 0 0 10px rgba(5, 150, 105, 0.3) !important;
    }

    /* Upload button styling */
    .css-1offfwp {
        border-color: #10b981 !important;
        color: #059669 !important;
    }

    /* File uploader styling */
    .css-1cpxqw2 {
        background-color: #ffffff !important;
        border-color: #10b981 !important;
        color: #059669 !important;
    }

    /* Category text styling */
    .css-10trblm {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }

    /* Dropdown options styling */
    .css-1d0tddh {
        color: #2d3748 !important;
    }

    .css-81oif8 {
        color: #2d3748 !important;
    }

    /* Selected option styling */
    .css-qrbaxs {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }

    /* Option hover effect */
    .css-9ycgxx:hover {
        background-color: #f0fdf4 !important;
        color: #059669 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation with icon
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 2rem; background: #f0fdf4; padding: 1.5rem; border-radius: 10px; border: 1px solid #10b981;'>
    <h3 style='color: #059669; margin: 0;'>🌿 Navigation</h3>
</div>
""", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Plant Identification", "Training Dataset"], key="main_nav")

if page == "Plant Identification":
    st.title("🌿 Medicinal Plant Identifier")

    # File upload section with instructions
    st.markdown("""
    <div class="upload-section">
        <h3 style='text-align: center; color: #059669;'>Upload Plant Image</h3>
        <p style='text-align: center; color: #4b5563;'>
            Upload a clear image of a medicinal plant for identification
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            # Read the image data
            file_bytes = uploaded_file.getvalue()

            # Display the image
            image_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
            image = cv2.imdecode(image_array, 1)
            st.image(image, channels="BGR", use_container_width=True)

            # Get prediction using OpenAI Vision
            prediction, confidence = identify_plant(file_bytes)

            # Display results with confidence score
            if prediction != "Unknown":
                st.markdown(f"""
                <div style='background: #f0fdf4; padding: 2rem; border-radius: 10px; margin: 2rem 0; border: 1px solid #10b981;'>
                    <h3 style='color: #059669; text-align: center; margin-bottom: 1rem;'>Plant Identification Results</h3>
                    <div style='display: flex; justify-content: space-between; gap: 2rem;'>
                        <div style='flex: 1; background: #dcfce7; padding: 1rem; border-radius: 8px; text-align: center;'>
                            <h4 style='color: #059669; margin: 0;'>🌿 Identified Plant</h4>
                            <p style='color: #059669; font-size: 1.5rem; margin: 0.5rem 0;'>{prediction}</p>
                        </div>
                        <div style='flex: 1; background: #dcfce7; padding: 1rem; border-radius: 8px; text-align: center;'>
                            <h4 style='color: #059669; margin: 0;'>📊 Confidence Score</h4>
                            <p style='color: #059669; font-size: 1.5rem; margin: 0.5rem 0;'>{confidence:.1f}%</p>
                        </div>
                    </div>
                    <div style='margin-top: 1rem; text-align: center;'>
                        <p style='color: #4b5563;'>
                            Upload another image or try different lighting conditions for better results
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("""
                Could not identify this plant with sufficient confidence. Please ensure:
                - The image is clear and well-lit
                - The plant is in focus and visible
                - The image shows the plant's distinctive features (leaves, stems, etc.)
                """)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            print(f"Error in image processing: {str(e)}")

elif page == "Training Dataset":
    st.title("📚 Training Dataset Management")

    # Plant category selection with styled container
    st.markdown("""
    <div style='background: #f0fdf4; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 15px -1px rgba(0,0,0,0.1); border: 1px solid #10b981;'>
        <h3 style='color: #059669; margin-bottom: 1rem;'>Select Plant Category</h3>
    </div>
    """, unsafe_allow_html=True)

    plant_category = st.selectbox(
        "",
        ["Tulsi", "Neem", "Aloe_Vera", "Mint"]
    )

    # File upload section with options
    st.markdown("""
    <div style='margin: 2rem 0; background: #f0fdf4; padding: 2rem; border-radius: 10px; border: 1px solid #10b981;'>
        <h3 style='color: #059669;'>Upload Training Images</h3>
        <p style='color: #4b5563;'>Choose how to add images to the dataset</p>
    </div>
    """, unsafe_allow_html=True)

    upload_type = st.radio("Upload Type", ["Individual Files", "Bulk Upload"])

    if upload_type == "Individual Files":
        uploaded_files = st.file_uploader(
            "",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )

        if uploaded_files:
            save_dir = f"data/training/{plant_category}"
            os.makedirs(save_dir, exist_ok=True)

            saved_count = 0
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    file_bytes = uploaded_file.read()
                    image_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
                    image = cv2.imdecode(image_array, 1)
                    if image is not None:
                        file_path = os.path.join(save_dir, uploaded_file.name)
                        cv2.imwrite(file_path, image)
                        saved_count += 1
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {i+1}/{len(uploaded_files)} images...")
                except Exception as e:
                    st.error(f"Error saving {uploaded_file.name}: {str(e)}")

            if saved_count > 0:
                st.success(f"Successfully saved {saved_count} images for {plant_category}")

    else:  # Bulk Upload
        st.markdown("""
        <div style='background: #f0fdf4; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
            <p style='color: #059669;'>💡 Upload a ZIP file containing multiple images</p>
            <p style='color: #4b5563; font-size: 0.9em;'>Supported formats: .zip containing .jpg, .jpeg, or .png files</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_zip = st.file_uploader("", type=['zip'])

        if uploaded_zip:
            import zipfile
            import io

            try:
                save_dir = f"data/training/{plant_category}"
                os.makedirs(save_dir, exist_ok=True)

                # Create a BytesIO object from the uploaded file's bytes
                zip_bytes = io.BytesIO(uploaded_zip.read())

                # Open the zip file
                with zipfile.ZipFile(zip_bytes) as zip_ref:
                    # Get list of image files from zip
                    image_files = [f for f in zip_ref.namelist()
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    if not image_files:
                        st.warning("No image files found in the ZIP archive.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        saved_count = 0

                        for i, image_file in enumerate(image_files):
                            try:
                                # Extract and read the image
                                image_data = zip_ref.read(image_file)
                                image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
                                image = cv2.imdecode(image_array, 1)

                                if image is not None:
                                    # Get the filename from the path
                                    filename = os.path.basename(image_file)
                                    save_path = os.path.join(save_dir, filename)

                                    # Save the image
                                    cv2.imwrite(save_path, image)
                                    saved_count += 1

                                # Update progress
                                progress = (i + 1) / len(image_files)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing {i+1}/{len(image_files)} images...")

                            except Exception as e:
                                st.error(f"Error processing {image_file}: {str(e)}")
                                continue

                        if saved_count > 0:
                            st.success(f"Successfully extracted and saved {saved_count} images for {plant_category}")
                        else:
                            st.warning("No images were successfully processed from the ZIP file.")

            except zipfile.BadZipFile:
                st.error("The uploaded file is not a valid ZIP archive.")
            except Exception as e:
                st.error(f"An error occurred while processing the ZIP file: {str(e)}")

    # Display current dataset statistics
    st.markdown("""
    <div class='dataset-stats'>
        <h3 style='color: #059669; margin-bottom: 1rem;'>Dataset Statistics</h3>
    </div>
    """, unsafe_allow_html=True)

    total_images = 0
    for category in ["Tulsi", "Neem", "Aloe_Vera", "Mint"]:
        path = f"data/training/{category}"
        if os.path.exists(path):
            image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            num_images = len(image_files)
            total_images += num_images
            st.markdown(f"""
            <div style='background: #ffffff; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; display: flex; justify-content: space-between; align-items: center; border: 1px solid #10b981;'>
                <span style='color: #059669;'>📁 {category}</span>
                <span style='color: #4b5563; font-weight: 500;'>{num_images} images</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background: #dcfce7; padding: 1rem; border-radius: 8px; margin-top: 1rem; text-align: center; border: 1px solid #10b981;'>
        <h4 style='color: #059669; margin: 0;'>📊 Total Dataset Size</h4>
        <p style='color: #059669; font-size: 1.5rem; margin: 0.5rem 0;'>{total_images} images</p>
    </div>
    """, unsafe_allow_html=True)