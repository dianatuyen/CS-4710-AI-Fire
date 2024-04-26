import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from source import preprocess_input_image, batch_predict, conv_float_int, combine_image, load_trained_model, burn_area

# Load style.css
with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# Set title of the app
st.title("CS4710: Wildfire Detection AI")

# Initialize session state variables if they don't exist
if 'images' not in st.session_state:
    st.session_state['images'] = []
if 'upload_again' not in st.session_state:
    st.session_state['upload_again'] = True

# Load the model
model = load_trained_model("temp_model.h5")

# Display previous images and their results
for img_data in st.session_state['images']:
    st.image(img_data['image'], caption="Processed Image", use_column_width=True)
    st.write(f"Total burnt area: {img_data['area']} km²")
    st.write(f"Total CO2 emitted: {img_data['co2']} tons")
    st.write(f"Equivalent days of emissions: {img_data['days']} days")

# Show uploader based on session state
if st.session_state['upload_again']:
    uploaded_file = st.file_uploader("Upload a raw satellite image", type=["png", "jpg"], key="file_uploader")
    image_url = st.text_input("Or enter an image URL:", key="url_input")

    if uploaded_file or image_url:
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)
        else:
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    uploaded_image = Image.open(BytesIO(response.content))
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
        st.pyplot(fig)

        preds_t = (preds > 0.25).astype(np.uint8)
        output_mask = conv_float_int(combine_image(preds_t, row_num, col_num, original_width, original_height, remove_ghost=False)[:, :, 0])

        forest_type = st.selectbox("Please select the type of forest:", ['Tropical Forest', 'Temperate Forest', 'Boreal Forest', 'Shrublands', 'Grasslands'], key="forest_type")
        resolution = st.slider("Please enter the image resolution value:", min_value=1, max_value=20, value=10, step=1, key="resolution")

        try:
            area, biomass_burnt, equal_days = burn_area(output_mask=output_mask, resolution=resolution, forest_type=forest_type)
            result = {'image': uploaded_image, 'area': f"{area / 1e6:.2f} km²", 'co2': f"{biomass_burnt / 1e6:.2f} tons", 'days': f"{equal_days:.2f} days"}
            st.session_state['images'].append(result)
            st.metric("Total burnt area", f"{result['area']} km²")
            st.metric("Total CO2 emitted", f"{result['co2']} tons")
            if equal_days > 0:
                st.metric("Equivalent days of California's electricity emission", f"{equal_days:.2f} days")
        except ValueError:
            st.error("Please enter a valid number for the image resolution.")


st.write('Would you like to add more images?')

col1, col2 = st.columns(2)
with col1:
    if st.button('Yes'):
        reset_state()
        st.experimental_rerun()
with col2:
    if st.button('No'):
        st.success('Thank you for using our Wildfire Detection AI!')
        st.session_state.upload_again = False  # Set this to control other parts of the app if needed
