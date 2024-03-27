# CS6476_Project
Final Project for CS 6476 - Computer Vision

## Project Update
By: Amulya Cherian, Hilary Present, and Nabil Rmiche

## Introduction
Radiologists are responsible for reading and interpreting hundreds of images per day. This intense and tedious workload can easily result in human error. In fact, inaccurate diagnostic radiology can lead to “treatment delays, poor outcomes, higher healthcare costs” for the patient (GE Healthcare, 2023). A study by the American Medical Association (AMA) found that 40.2% of the radiologists out of the 3,500 physicians in their sample have been sued in their career so far (Guardado, 2023). In addition, “errors in diagnosis” are the most common cause of “malpractice suits against radiologists,” thus pointing to the need for a transformative improvement in diagnostic radiology (Whang et al., 2013).

Incorporating machine learning and AI into radiology can help alleviate some of the pressure that radiologists experience. For instance, automating image processing and extraction of quantitative information from medical imagery can help streamline the process of interpreting imagery for radiologists. We are attempting to build a model that incorporates both image data and patient data to predict diagnosis. This kind of model can help healthcare professionals improve in both accuracy and efficiency of diagnosing patients, which can translate to better patient outcomes and lower cost of healthcare.

We thus focus our efforts on the classification task of X-Ray images via modern deep learning architectures (CNN, ViT and Diffusion Classifiers) with and without pre-training to assess how modern medicine can benefit the most from these state-of-the-art models.

## Related Works
### Vision Transformers:
- [1] A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” arXiv:2010.11929 [cs], Oct. 2020.
  - This paper by Dosovitskiy et al. on Vision Transformer (ViT) is relevant to our work because it shows an alternative to CNNs for image classification. As outlined in the paper, ViT offers a promising alternative approach from the traditional neural networks for a potentially more efficient and more accurate way to classify medical images.
    
- [2] Z. Liu et al., “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” 2021 IEEE/CVF International Conference on Computer Vision (ICCV), Oct. 2021, doi: https://doi.org/10.1109/iccv48922.2021.00986.
  - We chose this paper because it is relevant to our project in the way it discusses the challenges of adapting Transformers for visual tasks, which is crucial for the analysis of high-resolution medical images like the chest X-rays in the NIH dataset. By incorporating the techniques of shifted windows and hierarchical representation, we can potentially improve the efficiency and scalability of our own deep learning models, which will lead to far more enhanced image classification.

### YOLO Algorithm: 
- [3] N. Palanivel, Deivanai S, Lakshmi Priya G, Sindhuja B, and Shamrin Millet M, “The Art of YOLOv8 Algorithm in Cancer Diagnosis using Medical Imaging,” Nov. 2023, doi: https://doi.org/10.1109/icscan58655.2023.10395046
  - This study implements the You Only Look Once (YOLO) v8 method for early diagnosis of different types of cancer such as leukemia, skin cancer, cervical cancer, and lung cancer. This article was selected because the dataset is quite diverse and consists of different types of imaging. For instance, the dataset includes Pap smear images of cervical cancer, blood smear images of leukemia, and histopathological images of lung cancer. YOLO v8 works by dividing up each image into a matrix of cells and then predicting the class of each detected object, which should apply well for our project with the chest X-ray dataset.
    
- [4] C.-Y. Wang, I-Hau. Yeh, and H.-Y. M. Liao, “YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information,” arXiv (Cornell University), Feb. 2024, doi: https://doi.org/10.48550/arxiv.2402.13616.
  - This study by Wang et al. covers how to integrate the YOLO v9 algorithm with Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN). The combination of YOLO, PGI, and GELAN results in object detection performance that surpasses existing object detectors, hence why we are interested in this article.

### Diffusion for Classification: 
- [5] X. Han, H. Zheng, and M. Zhou, “CARD: Classification and Regression Diffusion Models,” arXiv.org, Dec. 06, 2022. https://arxiv.org/abs/2206.07275 (accessed Mar. 27, 2024).
  - This article gives an overview of different diffusion models and their applications to object detection. These applications include restoring high-resolution images from low-resolution inputs, reconstructing missing regions in an image, denoising with resampling iterations to better condition the image, semantic segmentation to label each image pixel, and more. These applications of diffusion models can be used for our project, especially with regards to the preprocessing step.
    
- [6] H. Greenspan et al., Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. Springer Nature, 2023.
  - This study proposes a diffusion-based model (DiffMIC) to assist with medical image classification. DiffMIC works by eliminating noise in order to glean semantic representation. This study tests the model on three types of medical data: ultrasound images, dermatoscopic images, and fundus images. Since the DiffMIC model can handle a variety of data, we feel that this article can be helpful in processing and analyzing chest X-ray images.

### Deep Learning for medical images: 
- [7] E. Çallı, E. Sogancioglu, B. van Ginneken, K. G. van Leeuwen, and K. Murphy, “Deep learning for chest X-ray analysis: A survey,” Medical Image Analysis, vol. 72, p. 102125, Aug. 2021, doi: https://doi.org/10.1016/j.media.2021.102125.
  - This paper applies image-level prediction with both classification and regression models, segmentation, image generation, and localization for chest X-ray datasets. This article is especially helpful for our project because it gives us guidance on what to consider when training models on chest X-ray data, especially when medical imagery varies so much across different tissue matters and different patients. 

- [8] X. Chen et al., “Recent advances and clinical applications of deep learning in medical image analysis,” Medical Image Analysis, vol. 79, p. 102444, Jul. 2022, doi: https://doi.org/10.1016/j.media.2022.102444.
  - This paper is relevant to our project since it involves the challenge of limited large-sized, well-annotated datasets in medical image analysis, which is an obstacle we were facing when we first began looking for datasets to begin our project. By reviewing recent advancements in unsupervised and semi-supervised deep learning for medical imaging, this paper provides insights that help to inform our approach for improving our models’ performance, even with some dataset constraints.

- [9] S. T. H. Kieu, A. Bade, M. H. A. Hijazi, and H. Kolivand, “A Survey of Deep Learning for Lung Disease Detection on Medical Images: State-of-the-Art, Taxonomy, Issues and Future Directions,” Journal of Imaging, vol. 6, no. 12, p. 131, Dec. 2020, doi: https://doi.org/10.3390/jimaging6120131.
  - This paper by Kieu et al. is relevant to our project because it offers an overview of deep learning methods for lung disease detection in medical images. Insights from this survey can guide our data augmentation, algorithm selection, and transfer learning approaches, which will ultimately assist in enhancing our deep learning models for pulmonary disease diagnosis.

### TorchIO:
- [10] F. Pérez-García, R. Sparks, and S. Ourselin, “TorchIO: A Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning,” Computer Methods and Programs in Biomedicine, vol. 208, p. 106236, Sep. 2021, doi: https://doi.org/10.1016/j.cmpb.2021.106236.
  - This paper by Pérez-García et al. on TorchIO is relevant to our project since it provides a comprehensive overview to efficiently handle medical images, which is necessary for our project with the NIH chest X-ray dataset. Borrowing the techniques seen in this paper – TorchIO's capabilities for preprocessing, augmentation, and patch-based sampling – will allow us to streamline our data processing pipeline and enhance the performance of our deep learning models.

## Methods
We are utilizing the Clinical Center Chest X-Ray dataset from the National Institute of Health (NIH). The dataset consists of 112,120 frontal-view chest X-ray PNG images along with the metadata for these images such as the image index, diagnostic finding labels, view position, and original dimensions. The dataset also has patient information that corresponds to each image such as patient ID, age, gender, and number of followup appointments. This dataset was chosen because it is fairly representative of the real patient population distributions and has a variety of thoracic pathologies. These X-Ray images come from 30,805 unique patients, which makes for a large sample size as well. 

Initially, we processed our dataset by resizing the images to a uniform size and converting them to PyTorch tensors, ready for model input. As we began experimenting, the dataset was originally split into training and validation sets using an 80-20 ratio. In the future, we will adjust so we have a 70-15-15 split for the training, validation, and test datasets, respectively. This change will allow us to more effectively compare the performance of the various models we plan to explore.

In addition to the initial data preparation, we conducted exploratory data analysis (EDA) to check the quality of our data and to better understand the distribution of labels. During the EDA process, we first checked for null values within the ‘Finding Labels’ column. Then, created a new column ‘Labels’ to handle instances where images have multiple labels – which were strings concatenated with '|' in the ‘Finding Label’s column. The ‘Labels’ column split the concatenated labels into lists of labels, allowing for better processing. Finally, we tallied the counts for each label to ascertain which categories had the most and the least data, which is crucial for understanding the balance of our dataset and informing potential strategies for model training and evaluation.

For the first model, we implemented a simple convolutional neural network (CNN) with two convolutional layers for feature extraction. These were followed by a subsampling max-pooling layer to reduce spatial dimensions, a dropout layer for regularization to prevent overfitting, and fully connected layers to combine features from the previous layers to make the final label predictions. We trained this CNN for 10 epochs using a binary cross-entropy loss function and Adam optimizer, with a learning rate of 0.001.

Our second method that we implemented involved deploying all our pipeline onto the PACE-ICE cluster. We were able to train Yolov8 on a very simple NIH dataset for 15 epochs with mostly default parameters.

## Results 
Our training processes for the first CNN involved tracking the loss and accuracy for both the training and validation sets. We observed that the training and validation losses decreased over time, indicating that the model was learning effectively. The final training and validation accuracies were both around 94.85%, suggesting that our model was able to generalize well to unseen data. However, the high accuracy levels and the similarity between the training and validation sets hint at potential overfitting. To mitigate this in future iterations, we plan to experiment with introducing more dropout layers, implementing early stopping, and using more complex regularization methods.

For the experiment with Yolov8 on the cluster, our goal wasn’t to get good results necessarily, we primarily wanted to make sure that our pipeline was working end-to-end without issue and also to get a rough estimate of the cost (compute and time) associated with training a state-of-the-art model on a significant dataset.
Future Direction

Going forward, we plan to build a model that can predict thoracic pathologies based on both the data from the chest x-ray images and corresponding patient data (such as age and gender). To do this, we will preprocess the images and then split the dataset into a training set, testing set, and validation set. We also plan to explore and compare models by developing them with different frameworks such as a PyTorch convolutional neural network, utilizing the Yolo V8 algorithm, and building a transformer-based model. We intend to compare these three models with the testing set to determine which framework is ideal given the type of data we have. Furthermore, the dataset presents a lot of issues (significant class imbalance and large uncertainty regarding ground truth labels) and will require a careful curation.



## Contributions

| Task                                                                       | Assigned to | Deadline |
|----------------------------------------------------------------------------|-------------|----------|
| Preprocess entire data                                                     | Everyone    | 04/03    |
| Develop & hyperparameter tune Yolo V8 model                                | Amulya      | 04/11    |
| Develop & hyperparameter tune transformer-based model                      | Nabil       | 04/11    |
| Develop & hyperparameter tune PyTorch CNN model                            | Hilary      | 04/11    |
| Evaluate each model with the validation set, compare models and select the best option, test final model on test set | Everyone | 04/13 |
| Wrap up Final Project                          | Everyone    | 04/18    |


*Github: https://github.com/hilpresent/CS6476_Project 

*Note: waiting for gatech to accept my application for githubt.gatech.edu






## References 
- GE Healthcare. (2023, June 13). Improving Accuracy in Radiology Images and Reports. GE Healthcare. https://www.gehealthcare.com/insights/article/improving-accuracy-in-radiology-images-and-reports 
- Guardado, J. R. (2023). Medical Liability Claim Frequency Among U.S. Physicians. American Medical Association: Policy Research Perspectives. 
- Whang, J. S., Baker, S. R., Patel, R., Luk, L., Castro, A. (2013, February 1). The Causes of Medical Malpractice Suits against Radiologists in the United States. Radiology, 266(2). https://doi.org/https://doi.org/10.1148/radiol.12111119 



