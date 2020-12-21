from .model import (
    get_cam_model, get_model, get_mobilenet_v2_model, get_nasnet_model,
    get_dense_model
)
from .video_model import get_CNN_LSTM_model, get_3D_CNN_model, get_2plus1D_CNN_model, get_CNN_transformer_model, get_2D_CNN_average_model, get_2stream_model, get_2D_then_1D_model, get_gate_shift_model, get_tea_model, get_CNN_GRU_model, get_CNN_LSTM_integrated_model
from .unet3d_genesis import unet_model_3d

MODEL_FACTORY = {
    'vgg_base': get_model,
    'vgg_cam': get_cam_model,
    'mobilenet_v2': get_mobilenet_v2_model,
    'nasnet': get_nasnet_model,
    'dense': get_dense_model
}

VIDEO_MODEL_FACTORY = {
    "2D_CNN_average": get_2D_CNN_average_model,
    "CNN_LSTM": get_CNN_LSTM_model,
    "CNN_GRU": get_CNN_GRU_model,
    "CNN_LSTM_integrated": get_CNN_LSTM_integrated_model,
    "3D_CNN": get_3D_CNN_model,
    "2plus1D_CNN": get_2plus1D_CNN_model,
    "CNN_transformer": get_CNN_transformer_model,
    "2stream": get_2stream_model,
    "2D_then_1D": get_2D_then_1D_model,
    "gate_shift": get_gate_shift_model,
    "tea": get_tea_model,
}
