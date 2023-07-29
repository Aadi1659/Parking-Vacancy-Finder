import cv2
import tensorflow as tf
import numpy as np
import streamlit as st
import os
import random
import glob
import csv

# Load the model
model = tf.keras.models.load_model("my_model")

# Define constants
img_size = (200, 200)

def prepare(img):
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255, input_shape=(200,200, 3))
    img = rescale(img)
    return img

def draw_boxes(image, label):
    
    return image

def display_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    st.image(image, caption='Random Image', use_column_width=True)

def predict_parking_vacancy(image):
    # Prepare the image
    prepared_image = prepare(image)

    prepared_image = tf.image.resize(prepared_image, [200,200])

    # Predict the label
    output = model.predict(np.expand_dims(prepared_image, axis=0))
    label = 1 if output[0] > 0.5 else 0

    return label

def return_list(path):
    files = os.listdir(path)
    return files

def random_choose_file(path):
    files = os.listdir(path)
    if (len(files)==0): 
        print("No files in directory") 
        return None
    else:
        random_image = random.choice(files)
        random_file_path = os.path.join(path, random_image)
        return random_file_path

def make_prediction(image_path_list,camera):

    #the model
    model = tf.keras.models.load_model("my_model")

    for img_path in  image_path_list:

    #read the image
        image = cv2.imread(img_path)
        #read boxes
        plain_name = ['parking_dataset_plain', 'CNR-EXT-Patches-150x150', 'PATCHES']
        
        csv_path = os.path.join(os.path.dirname((os.path.abspath(__file__))), plain_name[0],
                            'CNR-EXT_FULL_IMAGE_1000x750', (camera + '.csv'))
        
        with open(csv_path, newline='') as f:
            boxes = list(csv.reader(f))
        #crop images
        image_input = np.zeros(shape=(len(boxes[1:]), img_size[0], img_size[1], 3))
        for i, one_box in enumerate(boxes[1:]):
            one_box_x = int(int(one_box[1])/2.6)
            one_box_y = int(int(one_box[2])/2.6)
            one_box_w = int(int(one_box[3])/2.5)
            one_box_h = int(int(one_box[4])/2.5)
            img_resized = cv2.resize(image[one_box_y:one_box_y+one_box_h, one_box_x:one_box_x+one_box_w], img_size,
                                    interpolation=cv2.INTER_CUBIC)
            image_input[i] = (prepare(np.expand_dims(img_resized, axis=0)))
        
        #predict the labels
        output = model.predict(image_input)

        label = np.array(list(map(lambda x: 1 if x > 0.5 else 0, output)))

        for i, one_box in enumerate(boxes[1:]):
            one_box_x = int(int(one_box[1])/2.6)
            one_box_y = int(int(one_box[2])/2.6)
            one_box_w = int(int(one_box[3])/2.5)
            one_box_h = int(int(one_box[4])/2.5)

            image = cv2.rectangle(image, (one_box_x, one_box_y), (one_box_x + one_box_w, one_box_y + one_box_h),
                                (36, int(255 * label[i]), 12), 2)
            cv2.putText(image, ('ID:' + str(one_box[0])), (one_box_x, one_box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (36, int(255 * label[i]), 12), 1)
        
        #show prediction
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        st.image(image, caption='Random Image', use_column_width=True)
        st.text(f"Total number of parking spaces: {len(label)}")
        st.text(f"Number of occupied spaces: {np.sum(label == 1)}")

def main():
    st.title("Parking Vacancy Identifier")

    weather = st.selectbox("Select Weather Type", ["OVERCAST","RAINY","SUNNY"])
    if weather == "SUNNY":
        date = st.selectbox("Select the Date",["2015-11-12","2015-11-22","2015-11-27","2015-12-10","2015-12-17","2016-01-12","2016-01-13","2016-01-15","2016-01-16","2016-01-18"])
    elif weather == "RAINY":
        date = st.selectbox("Select the Date",["2015-11-21","2015-12-22","2016-01-08","2016-01-09","2016-01-14","2016-02-12"])
    elif weather == "OVERCAST":
        date = st.selectbox("Select the Date",["2015-11-16","2015-11-20","2015-11-25","2015-11-29","2015-12-03","2015-12-18","2015-12-19",])

    camera = st.selectbox("Select the Camera Number",["camera1","camera2","camera3","camera4","camera5","camera6","camera7","camera8","camera9"])

    plain_name = ["parking_dataset_plain", 'CNR-EXT-Patches-150x150', 'PATCHES']

    img_path_search = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    plain_name[0],
    'CNR-EXT_FULL_IMAGE_1000x750',
    'FULL_IMAGE_1000x750',
    weather,
    date,
    camera,
    )
    img_path_search_list = return_list(img_path_search)

    image = st.selectbox("Select the Image", img_path_search_list)
    img_path = os.path.join(img_path_search, image)
    # img_path = random_choose_file(img_path_search)
    display_image(img_path)
    image_path_list = glob.glob(img_path)
    prediction_button = st.button("Make Predictions")
    if prediction_button:
        make_prediction(image_path_list=image_path_list,camera=camera)

    
    
if __name__ == "__main__":
    main()
