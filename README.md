# Traffic Sign Detection
## Objectives
Implemented two models for detecting the presence of traffic signs and differentiating the front and rear views of the vehicles in images or video streams.
* Built image datasets and image annotations in TensorFlow record format.
* Trained Faster R-CNN on the LISA Traffic Signs dataset to detect and recognize 47 United States traffic sign types.
* Trained SSD on the Davis King’s vehicles dataset to differentiate the front and rear views of the vehicles.
* Evaluate the accuracy and apply the trained Faster R-CNN and SSD models to input images and
video streams.

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 4.1.0
* [keras](https://keras.io/) 2.2.4
* [Tensorflow](https://www.tensorflow.org/install/) 1.13.0
  * [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.4.2
* [scikit-learn](https://scikit-learn.org/stable/) 0.20.1
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/) 1.16.2
* [PIL](https://pillow.readthedocs.io/en/stable/) 5.4.1

## Approaches
The dataset is from [The Laboratory for Intelligent and Safe Automobiles (LISA)](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html). The dataset consists of 47 different United States traffic sign types, including stop signs, pedestrian signs, and etc. The dataset is initially captured in video format but the individual frames and associated annotations are also included. There are a total of 7855 annotations on 6610 frames. Road signs vary in resolution, from 6 × 6 to 167 × 168 pixels. Furthermore, some images are captured in a low resolution 640 × 480 camera while other images are captured on a high resolution 1024 × 522 pixels. Some images are gray scale while other images are in color.

### Setup configurations
In order properly import functions inside TensorFlow Object Detection API, `setup.sh`([check here](https://github.com/meng1994412/Traffic_Sign_Detection/blob/master/setup.sh)) helps to update the `PYTHONPATH` variable. Thus, remember to source it every time, as shown below.
```
source setup.sh
```

The `lisa_config.py` ([check here](https://github.com/meng1994412/Traffic_Sign_Detection/blob/master/config/lisa_config.py)) under `config/` directory contains the configurations for the project, including path to the annotations file and training & testing sets, training/testing split ratio, and classification labels.

### Build Tensorflow record dataset
According to Tensorflow Object Detection API, we need to have a number of attributes to makes up data points for object detection. They are including: (1) the Tensorflow encoded image, (2) the height and width of the image, (3) the file encoding of the image, (4) the filename of the image, (5) a list of bounding box coordinates (normalized in range [0, 1], for the image), (6) a list of class labels for each bounding box, (7) a flag used to encode if the bounding box is "difficult" or not.

The `tfannotation.py` ([check here](https://github.com/meng1994412/Traffic_Sign_Detection/blob/master/pipeline/utils/tfannotation.py)) under `pipeline/utils/` directory build a class to encapsulate encoding an object detection data point in Tensorflow `record` format.

The `build_lisa_records.py` ([check here](https://github.com/meng1994412/Traffic_Sign_Detection/blob/master/build_lisa_records.py)) convert raw image dataset and annotation file into Tensorflow `record` format dataset with corresponding class label file, in order to train a network by using Tensorflow Object Detection API.

We could use following command to build the Tensorflow `record` datasets.
```
time python build_lisa_records.py
```

### Train & evaluate Faster R-CNN model
In order to train the Faster R-CNN model via transfer learning, we could download the pre-trained Faster R-CNN so we can fine-tune the network. And we also need to set up the Tensorflow Object Detection API configuration file for training.

All the pre-trained models can found in the [Tensorflow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). In this project, I use `faster_rcnn_resnet101_coco` model, which is the Faster R-CNN with ResNet 101 as base model pre-trained on COCO dataset.

The `faster_rcnn_resnet101_lisa.config` ([check here](https://github.com/meng1994412/Traffic_Sign_Detection/blob/master/lisa/experiments/training/faster_rcnn_resnet101_lisa.config)) under `lisa/experiments/training/` directory sets up the Tensorflow Object Detection API configuration file for training.

The `model_main.py` ([check here](https://github.com/tensorflow/models/blob/master/research/object_detection/model_main.py)) inside Tensorflow Object Detection API is used to train the model. The following command demonstrates the proper start of the training process (make sure we are current under `models/research/` directory of Tensorflow Object Detection API and source the `setup.sh` file).
```
python object_detection/model_main.py \
--pipeline_config_path=YOUR_CONFIGURATION_PATH \
--model_dir=YOUR_PRE_TRAINED_MODEL_PATH \
--num_train_steps=100000 \
--sample_1_of_n_eval_examples=1 \
--alsologtostderr
```

Figure 1 and Figure 2 below show the evaluation of the model including precision and recall of detection boxes.

<img src="https://github.com/meng1994412/Traffic_Sign_Detection/blob/master/output/DetectionBoxes_Precision.png" height="500">

Figure 1: Precision evaluation of the model.

<img src="https://github.com/meng1994412/Traffic_Sign_Detection/blob/master/output/DetectionBoxes_Recall.png" height="500">

Figure 2: Recall evaluation of the model.

The `export_inference_graph.py` ([check here](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py)) in Tensorflow Object Detection API is used to export our model for inference. The following command can be used to export the model.
```
python object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path YOUR_CONFIGURATION_PATH
--trained_checkpoint_prefix YOUR_TRAINING_MODEL_PATH/model.ckpt-100000 \
--output YOUR_PATH_TO_EXPORT_MODEL
```

### Apply Faster R-CNN model to images and videos
The `predict_image.py` ([check here](https://github.com/meng1994412/Traffic_Sign_Detection/blob/master/predict_image.py)) applys the network we trained to an input image outside the dataset it is trained on. And the `predict_video.py` ([check here](https://github.com/meng1994412/Traffic_Sign_Detection/blob/master/predict_video.py)) apply to an input video.

Figure 3 and Figure 4 show two samples for detecting the some of the traffic signs.

<img src="https://github.com/meng1994412/Traffic_Sign_Detection/blob/master/output/sample_1.png" height="400">

Figure 3: Sample #1 for detecting stop signs and pedestrian crossing sign in the image.

<img src="https://github.com/meng1994412/Traffic_Sign_Detection/blob/master/output/DetectionBoxes_Recall.png" height="600">

Figure 2: Sample #2 for detect signal ahead signs in the image
