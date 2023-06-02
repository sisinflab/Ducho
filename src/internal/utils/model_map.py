# LEGACY FILE: IT HAS NO LONGER USE

import tensorflow as tf
import torchvision.models as models

tensorflow_models_for_extraction = {
    'ResNet50': tf.keras.applications.ResNet50,
    'VGG19': tf.keras.applications.VGG19,
    'ResNet152': tf.keras.applications.ResNet152
}

torch_models_for_extraction = {
    'AlexNet': models.alexnet,
    'VGG19': models.vgg19,
    'VGG11': models.vgg11,
    'ResNet50': models.resnet50
}

tensorflow_models_for_normalization = {
    'ResNet50': tf.keras.applications.resnet,
    'VGG19': tf.keras.applications.vgg19,
    'ResNet152': tf.keras.applications.resnet
}
