import cv2
import tensorflow as tf
import numpy as np
import os
import glob
import csv

batch_size = 32
img_size = (200,200)
num_classes = 2 #either occupied or unoccupied

def prepare(img):
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255, input_shape=(img_size[0],img_size[1],3))
    img = rescale(img)
    return img

model = tf.keras.models.load_model("my_model")

#loading a sample image
plain_name = ["parking_dataset_plain", 'CNR-EXT-Patches-150x150', 'PATCHES']
date_name = '2015-12-22'
weather = 'RAINY'
camera_name = 'camera1'
img_path_search = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..',  # Parent directory
    plain_name[0],
    'CNR-EXT_FULL_IMAGE_1000x750',
    'FULL_IMAGE_1000x750',
    weather,
    date_name,
    camera_name,
    '2015-12-22_1648.jpg'
)
img_path_list = glob.glob(img_path_search)
# print(os.listdir(img_path_search))
for img_path in  img_path_list:

    #read the image
    image = cv2.imread(img_path)
    #read boxes
    plain_name = ['parking_dataset_plain', 'CNR-EXT-Patches-150x150', 'PATCHES']
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), plain_name[0],
                        'CNR-EXT_FULL_IMAGE_1000x750', (camera_name + '.csv'))
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
    cv2.imshow('prediction', image)
    cv2.waitKey(0)

    #save preds
    save_name = os.path.join(os.path.dirname(img_path), ('pred_' + os.path.basename(img_path)))
    cv2.imwrite(save_name, image)
