import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import sys
import requests
from io import BytesIO
from source import preprocess_input_image, batch_predict, conv_float_int, combine_image, load_trained_model, burn_area

# Load style.css
with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# Set title of the app
st.title("CS4710: Wildfire Detection AI")

# Load the model
model = load_trained_model("temp_model.h5")

# Initialize session state for image processing and upload controls
if 'images_processed' not in st.session_state:
    st.session_state['images_processed'] = []
if 'upload_again' not in st.session_state:
    st.session_state['upload_again'] = True

def process_image():
    uploaded_files = st.file_uploader("Upload a raw satellite image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    image_url = st.text_input("Or enter an image URL:")

    if uploaded_files or image_url:
        st.session_state['upload_again'] = False  # Processing done, set upload_again to False
        if uploaded_files:
            uploaded_image = Image.open(uploaded_files)
            image_name = uploaded_files.name
        else:
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    uploaded_image = Image.open(BytesIO(response.content))
                    image_name = image_url.split('/')[-1]
                else:
                    st.error("Failed to download the image. Please check the URL and try again.")
                    st.stop()
            except Exception as e:
                st.error(f"Error downloading the image: {e}")
                st.stop()

        st.image(uploaded_image, caption="Original Raw Image", use_column_width=True)

        with st.spinner("Pre-processing the image..."):
            input_image_array = np.array(uploaded_image)
            original_width, original_height, _ = input_image_array.shape
            new_image_array, row_num, col_num = preprocess_input_image(input_image_array)

        with st.spinner("Making the prediction..."):
            preds = batch_predict(new_image_array, model)
            output_pred = conv_float_int(combine_image(preds, row_num, col_num, original_width, original_height, remove_ghost=True)[:, :, 0])

        fig, ax = plt.subplots()
        ax.imshow(output_pred)
        st.write("Predicted Burn Area")
        st.pyplot(fig)

        preds_t = (preds > 0.25).astype(np.uint8)
        output_mask = conv_float_int(combine_image(preds_t, row_num, col_num, original_width, original_height, remove_ghost=False)[:, :, 0])

        forest_type = st.selectbox("Please select the type of forest:", ['Tropical Forest', 'Temperate Forest', 'Boreal Forest', 'Shrublands', 'Grasslands'])
        resolution = st.text_input("Please enter the image resolution value:", '10')

        try:
            resolution_value = float(resolution)
            area, biomass_burnt, equal_days = burn_area(output_mask=output_mask, resolution=resolution_value, forest_type=forest_type)
            st.write(f"**The total burnt area is:** {area / 1e6:.2f} km^2")
            st.write(f"**The total CO2 emitted is:** {biomass_burnt / 1e6:.2f} tons")
            if equal_days > 0:
                st.write(f"This is equivalent to: {equal_days:.2f} days of California's daily electricity power emission")
        except ValueError:
            st.error("Please enter a valid number for the image resolution.")

        fig2, ax2 = plt.subplots()
        ax2.imshow(output_mask)
        st.write("Processed Mask")
        st.pyplot(fig2)

def show_upload_prompt():
    st.write('Would you like to add more images?')
    st.write('Click YES BUTTON to upload new image or click NO to exit')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Yes'):
            st.session_state['images_processed'] = []  # Reset any stored data
            st.session_state['upload_again'] = True   # Allow new uploads
    with col2:
        if st.button('No'):
            st.success('Thank you for using our Wildfire Detection AI!')
            st.session_state['upload_again'] = False  # Stop further uploads
            time.sleep(5)
            sys.exit()

if st.session_state['upload_again']:
    process_image()
    show_upload_prompt()
else:
    show_upload_prompt()
