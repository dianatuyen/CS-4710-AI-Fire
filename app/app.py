import streamlit as st
import matplotlib.pyplot as plt
from source import preprocess_input_image, batch_predict, conv_float_int, combine_image, load_trained_model, burn_area
import numpy as np
from PIL import Image

# Load style.css
with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# Set title of the app
st.title("CS4710: Wildfire Detection AI")

# Load the model
model = load_trained_model("temp_model.h5")

# File uploader widget
uploaded_file = st.file_uploader("Please upload a raw satellite image", type=["png"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Original Raw Image", use_column_width=True)
    
    with st.spinner("Pre-processing the image..."):
        input_image_array = np.array(uploaded_image)
        original_width, original_height, _ = input_image_array.shape
        new_image_array, row_num, col_num = preprocess_input_image(input_image_array)

    with st.spinner("Making the prediction..."):
        preds = batch_predict(new_image_array, model)
        output_pred = conv_float_int(combine_image(preds, row_num, col_num, original_width, original_height, remove_ghost=True)[:, :, 0])

    # Display the prediction probability
    fig, ax = plt.subplots()
    ax.imshow(output_pred)
    st.pyplot(fig)

    preds_t = (preds > 0.25).astype(np.uint8)
    output_mask = conv_float_int(combine_image(preds_t, row_num, col_num, original_width, original_height, remove_ghost=False)[:, :, 0])

    # CO2 Emission Calculator
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

    # Display the predicted mask
    fig2, ax2 = plt.subplots()
    ax2.imshow(output_mask)
    st.write("**The Predicted Mask is:**")
    st.pyplot(fig2)
