# Tomato leaves health classifier
A  image classifier transfer-learning based model web application with Tensorflow and Flask. The application allows to upload image file and determines from it whether a tomato leaf is (<b>healthy</b> or <b>unhealthy</b>).
The model used is stored in the folder models with the filename: `transfer_learning.h5`


## Getting Started
1. Clone this repository with Git Large File Storage(LFS) `git lfs clone https://github.com/mitkir/keras-flask-image-classifier`, Use Git LFS because the model H5 file is very heavy and simple git can not dowload or upload it.
2. Install all necessary dependencies `pip install -r requirements.txt`
3. Run application `python application.py`
4. Open the link shown in your terminal on your browser
5. Click the file select button and select test image for classifier.



### DEMO
![Screenshot](demo.png)