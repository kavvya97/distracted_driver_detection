# Distracted Driver Detection

A Deep learning project for detecting distracted drivers using the Distracted Driver Dataset.

## Introduction
Distracted driving is a major concern that significantly impacts road safety. This project aims to develop a solution for detecting distracted drivers using machine learning techniques. 

## Dataset
The Distracted Driver Dataset, available on Kaggle, serves as the foundation for our project.\
https://www.kaggle.com/competitions/state-farm-distracted-driver-detection

## Approach
- Implemented prelimnary baseline methods using resnet50 and densenet121 pretrained model
- Autoencoder approach to get the image latent space and derive the most important characterstics of the image.
- Semantic Segmentation of the images
- Ensemble methods using pretrained models like VGGnet, Resnet and densenet.

## Results
Our models have achieved promising results in detecting distracted drivers. The ensembled method, combining VGGNET, DenseNet121, and ResNet12 models, has demonstrated the highest accuracy of 98%. These results showcase the effectiveness of our solution in addressing the issue of distracted driving.

## License
This project is licensed under the MIT License.

## Acknowledgements
We would like to express our gratitude to Kaggle for providing the Distracted Driver Dataset, which has been instrumental in the development of this project.

