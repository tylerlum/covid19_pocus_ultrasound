from .model import (
    get_cam_model, get_model, get_mobilenet_v2_model, get_nasnet_model,
    get_dense_model
)

import cv2
from .video_model import (
    get_CNN_LSTM_model, get_3D_CNN_model, get_2plus1D_CNN_model, get_CNN_transformer_no_pos_model,
    get_CNN_transformer_model, get_2D_CNN_average_model, get_2D_then_1D_model, get_gate_shift_model, get_tea_model,
    get_CNN_GRU_model, get_CNN_LSTM_integrated_model, get_CNN_RNN_model, get_CNN_LSTM_bidirectional_model,
    get_CNN_GRU_bidirectional_model, get_CNN_RNN_bidirectional_model, get_CNN_LSTM_integrated_bidirectional_model,
    get_baseline_model, get_model_genesis_model, get_2stream_average_model,
    get_2stream_LSTM_integrated_bidirectional_model, get_2stream_transformer_model,
    get_2D_CNN_average_evidential_model, get_CNN_LSTM_integrated_bidirectional_evidential_model,
    get_CNN_transformer_evidential_model, get_2stream_LSTM_integrated_bidirectional_evidential_model,
    get_2D_3D_model, get_CNN_transformer_multihead_model,
)
import tensorflow
from tensorflow.keras.applications import VGG16, MobileNetV2, NASNetMobile, EfficientNetB0, ResNet50, ResNet50V2


MODEL_FACTORY = {
    'vgg_base': get_model,
    'vgg_cam': get_cam_model,
    'mobilenet_v2': get_mobilenet_v2_model,
    'nasnet': get_nasnet_model,
    'dense': get_dense_model
}

VIDEO_MODEL_FACTORY = {
    # No information baseline
    "baseline": get_baseline_model,

    # Simple
    "2D_CNN_average": get_2D_CNN_average_model,

    # Recurrent
    "CNN_LSTM": get_CNN_LSTM_model,
    "CNN_GRU": get_CNN_GRU_model,
    "CNN_RNN": get_CNN_RNN_model,
    "CNN_LSTM_integrated": get_CNN_LSTM_integrated_model,

    "CNN_LSTM_bidirectional": get_CNN_LSTM_bidirectional_model,
    "CNN_GRU_bidirectional": get_CNN_GRU_bidirectional_model,
    "CNN_RNN_bidirectional": get_CNN_RNN_bidirectional_model,
    "CNN_LSTM_integrated_bidirectional": get_CNN_LSTM_integrated_bidirectional_model,

    # Evidential
    "2D_CNN_average_evidential": get_2D_CNN_average_evidential_model,
    "CNN_LSTM_integrated_bidirectional_evidential": get_CNN_LSTM_integrated_bidirectional_evidential_model,
    "CNN_transformer_evidential": get_CNN_transformer_evidential_model,
    "2stream_LSTM_integrated_bidirectional_evidential": get_2stream_LSTM_integrated_bidirectional_evidential_model,

    # Convolutional
    "3D_CNN": get_3D_CNN_model,
    "2plus1D_CNN": get_2plus1D_CNN_model,
    "2D_then_1D": get_2D_then_1D_model,

    # Transformer
    "CNN_transformer": get_CNN_transformer_model,
    "CNN_transformer_multihead": get_CNN_transformer_multihead_model,
    "CNN_transformer_no_pos": get_CNN_transformer_no_pos_model,

    # Model Genesis
    "model_genesis": get_model_genesis_model,

    # Two stream optical flow
    "2stream_average": get_2stream_average_model,
    "2stream_LSTM_integrated_bidirectional": get_2stream_LSTM_integrated_bidirectional_model,
    "2stream_transformer": get_2stream_transformer_model,

    # CVPR
    "gate_shift": get_gate_shift_model,
    "tea": get_tea_model,

    # 2D to 3D CNN
    "2D_3D": get_2D_3D_model,
}

OPTICAL_FLOW_ALGORITHM_FACTORY = {
    "farneback": cv2.optflow.createOptFlow_Farneback,
    "dtvl1": cv2.optflow.createOptFlow_DualTVL1,
    "deepflow": cv2.optflow.createOptFlow_DeepFlow,
    "denserlof": cv2.optflow.createOptFlow_DenseRLOF,
    "pcaflow": cv2.optflow.createOptFlow_PCAFlow,
    "simpleflow": cv2.optflow.createOptFlow_SimpleFlow,
    "sparserlof": cv2.optflow.createOptFlow_SparseRLOF,
    "sparsetodense": cv2.optflow.createOptFlow_SparseToDense,
}

PRETRAINED_CNN_FACTORY = {
    "vgg16": VGG16,
    "efficientnet": EfficientNetB0,
    "resnet50": ResNet50,
    "resnet50_v2": ResNet50V2,
}

PRETRAINED_CNN_PREPROCESS_FACTORY = {
    "vgg16": tensorflow.keras.applications.vgg16.preprocess_input,
    # "vgg16": lambda x : x / 255.0,
    "efficientnet": tensorflow.keras.applications.efficientnet.preprocess_input,
    "resnet50": tensorflow.keras.applications.resnet.preprocess_input,
    "resnet50_v2": tensorflow.keras.applications.resnet_v2.preprocess_input,
}
