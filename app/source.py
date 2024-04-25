import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

def add_image_margin(input_image, im_width=720, im_height=480, color=0):
    # Pad the image to match the target size
    height, width, vec = input_image.shape
    new_image = np.pad(input_image, ((0, im_height-height), (0, im_width-width), (0, 0)), 'constant', constant_values=color)
    return new_image

def preprocess_input_image(input_image, im_height=480, im_width=720):
    # Normalize the pixel values
    input_image = input_image / np.max(input_image.astype('float'))
    height, width, vec = input_image.shape
    row_num = height // im_height if height % im_height == 0 else height // im_height + 1
    col_num = width // im_width if width % im_width == 0 else width // im_width + 1

    new_image_array = np.zeros((row_num * col_num, im_height, im_width, vec))

    for i in range(row_num):
        for j in range(col_num):
            temp_image = input_image[im_height*i:im_height*(i+1), im_width*j:im_width*(j+1), :]
            new_image_array[i*col_num+j] = add_image_margin(temp_image, im_width=im_width, im_height=im_height, color=0)
    return new_image_array, row_num, col_num

def combine_image(input_image, row_num, col_num, original_height, original_width, im_height=480, im_width=720, remove_ghost=True):
    num, height, width, vec = input_image.shape
    new_image = np.zeros((height * row_num, width * col_num, vec))
    for i in range(row_num):
        for j in range(col_num):
            if remove_ghost:
                # Pad the edges to reduce boundary effects
                input_image[i*col_num+j, :, :, :] = np.pad(input_image[i*col_num+j, 4:height-4, 4:width-4, :], ((4, 4), (4, 4), (0, 0)), 'edge')
            new_image[im_height*i:im_height*(i+1), im_width*j:im_width*(j+1), :] = input_image[i*col_num+j, :, :, :]  
    return new_image[:original_height, :original_width, :]

def batch_predict(input_image_array, model):
    num, height, width, vec = input_image_array.shape
    preds_array = np.zeros((num, height, width, 1))
    for ii in range(num):
        try:
            preds_array[ii] = model.predict(np.expand_dims(input_image_array[ii, :, :, :], axis=0), verbose=1)
        except Exception as e:
            print(f"Error during model prediction: {e}")
    return preds_array


def conv_float_int(image):
    if not np.any(image):  # check if the array is all zero 
        return image.astype(int)
    return ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(int)

def load_trained_model(model_location):
    try:
        loaded_model = load_model(model_location)
        return loaded_model
    except Exception as e:
        raise ValueError(f"Failed to load the model: {e}")


def burn_area(output_mask, resolution, forest_type):
    try:
        biomass_type = {
            'Tropical Forest': 28076, 'Temperate Forest': 10492, 
            'Boreal Forest': 25000, 'Shrublands': 5705, 'Grasslands': 976
        }
        area = np.count_nonzero(output_mask) * resolution**2
        biomass_burnt = area * biomass_type[forest_type] / 1e3 * 1624  # unit in g
        ca_co2_daily = 4.24e8 / 365.  # California daily CO2 emission from power generation
        equal_days = biomass_burnt / 1e6 / ca_co2_daily
        return area, biomass_burnt, equal_days
    except KeyError:
        raise ValueError("Invalid forest type selected.")
    except Exception as e:
        raise ValueError(f"An error occurred while calculating burn area: {e}")
