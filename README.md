# Anomaly-breast-cancer-detection

This open repository has been developed for the Master Thesis "Abnormality detection in mammographic images" during the summer semester 2022 at the Open University of Catalonia (UOC). The trained and tested models are the Faster R-CNN and YOLO algorithms. 

To train the different models, the open database MIAS has been used. Further information can be found in the following link: https://www.repository.cam.ac.uk/handle/1810/250394


This repository consists of the following files:

* `README.md`: information on the contents of the repository is exposed
* `requirements.txt`: required packages for the execution of the project are listed together with the corresponding versions
* `LICENSE`: 

And the following directories:
* `src`: main source code is saved here
    * `preprocess`: preprocessing and cleansing of the raw data can be found here
    * `train`: training of Faster R-CNN and YOLO algorithms can be found here, together with the corresponding codes of useful functions and its algorithms.
        * `algorithms`: code for the development of the model are saved here.
            * `faster_rcnn`: algorithm of the Faster R-CNN model.
            * `yolo`: it contains the algorithm of the YOLO model.
        * `utils`: complementary functions to the algorithms.
        * `notebooks`: module, which contains the training of the models in notebooks.
* `test`: test of the models with test data
