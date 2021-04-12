# Save tf.keras model in .savedmodel directory format.

import tensorflow as tf


for fold in range(5):
    saved_model_dir = "multihead_model_outputs/Apr-12-2021_05-03-48/multihead_best_fold_{}".format(fold)


    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    open(f"converted_model_fold{fold}.tflite", "wb").write(tflite_model)