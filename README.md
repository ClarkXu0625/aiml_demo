# Description
aiml group project, uploading images of tongue, palm + eye + finger nail (left and right), predicting hemoglobin level.
Please look into .ipynb files for detailed information about model training and image segmentation details.

notebooks/image_process.ipynb - contains all codes in image_process.py (In case when there's trouble when environment having trouble solving environment. Upload to Google Colab to run the code, and upload all files in "model" folder and "uploads" folder to the temporary directory in google). This file contains the whole pipline for image inference, including segmentation and prediction. The predicted hemoglobin value is at the end of this file

# required package 
flask
flask-cors 
pillow 
numpy 
pandas 
scikit-learn
os
cv2
csv
skimage
tensorflow
keras
seaborn
matplotlib
xgboost

# File description
app.py - the web on local server, calling function from image_process.py and prediction.py
image_process.py - call models stored in model folder, convert image in uploads file to a dictionary, which contains average RGB value in each region of insterest
prediction.py - predict the hemoglobin value using xgboost



# Folders
image_segmentation_training/ - contains all the code to train image segmentation, which generates .h5 models stored in model/SegModels
uploads/ - inmages of body part of one individual as inference
notebooks/ contain all jupyter notebooks that used to train different classification models
notebooks/xgboost.ipynb - contains tutorial for xgboost, take mean rgb values to predict hemoglobin value
notebooks/train1.py    - contains the process to train random forest



# Run the app
Run the app by type following command to terminal:
$ python app.py
Local server: http://127.0.0.1:8000
Select images for all 7 body parts and upload them, they will automatically show up in uploads/ folder
Then select your age and gender, then click predict