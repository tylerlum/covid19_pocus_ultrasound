import cv2
import numpy as np
import tensorflow as tf
from keras.layers import TimeDistributed

# Code reference:
# https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
# Converted for videos


class VideoGradCAM:
    def __init__(self, model, layerName=None):
        """
        Initialize video grad cam for specific class
        Args:
            model (tf.keras.Model): tf.keras model to inspect. Must be video input model,
                                    with a TimeDistributed layer outputing
                                    (batch_size, seq_len, height, width, embed_dim)
            layer_name (str): Targeted layer for VideoGradCAM. If no layer is
                              provided, it is automatically infered from the model
                              architecture.
        """
        self.model = model
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        # (batch_size, seq_len, height, width, embed_dim), so len(shape) == 5
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 5:
                print(f"find_target_layer: found {layer.name} {layer.output_shape}")
                return layer.name
        # otherwise, we could not find a 5D layer so the VideoGradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply VideoGradCAM.")

    def compute_heatmaps(self, video, classIdx, eps=1e-8):
        # Expect video.shape = (seq_len, height, width, channels)

        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 5D layer in the network, and (3) the output of the
        # softmax activations from the model
        cnn_layer_name = self.layerName
        gradModel = tf.keras.models.Model([self.model.inputs], [self.model.get_layer(cnn_layer_name).output, self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the video tensor to a float-32 data type, pass the
            # video through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(np.expand_dims(video, axis=0), tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, classIdx]

        # use automatic differentiation to compute the gradients
        print(f"convOutputs.shape = {convOutputs.shape}, from 2d should be (1, 7, 7, 512)")
        grads = tape.gradient(loss, convOutputs)
        print(f"grads.shape = {grads.shape}, from 2d should be (1, 7, 7, 512)")

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        print(f"guidedGrads.shape = {guidedGrads.shape}, from 2d should be (1, 7, 7, 512)")

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        print(f"AFTER convOutputs.shape = {convOutputs.shape}, from 2d should be (7, 7, 512)")
        print(f"AFTER guidedGrads.shape = {guidedGrads.shape}, from 2d should be (7, 7, 512)")
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        # guidedGrads.shape = (seq_len, height, width, embed_dim)
        # Average across height and width
        weights = tf.reduce_mean(guidedGrads, axis=(1, 2))
        print(f"weights.shape = {weights.shape}, from 2d should be (512,)")

        # weights.shape = (seq_len, embed_dim)
        # convOutputs.shape = (seq_len, height, width, embed_dim)
        # For each frame in seq_len, multiply convOutputs by weights
        # Then average across embed_dim
        cam = []
        for i in range(len(weights)):
            cam.append(tf.reduce_sum(tf.multiply(weights[i], convOutputs[i]), axis=-1))
        cam = np.array(cam)  # cam.shape = (seq_len, height, width)
        print(f"cam.shape = {cam.shape}, from 2d should be (7, 7)")

        # grab the spatial dimensions of the input video and resize
        # the output class activation map to match the input video
        # dimensions
        seq_len, height, width, channels = video.shape
        heatmaps = []
        for i in range(len(cam)):
            heatmaps.append(cv2.resize(cam[i], (width, height)))
        heatmaps = np.array(heatmaps)  # heatmaps.shape = (seq_len, width, height)
        print(f"heatmaps.shape = {heatmaps.shape}, from 2d should be (224, 224)")

        # normalize the heatmaps such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        # Can either normalize each frame individually, or across all frames
        # Currently do individually, so use the max and min of each frame, not of video
        scaled_heatmaps = []
        for i in range(len(heatmaps)):
            numer = heatmaps[i] - np.min(heatmaps[i])
            denom = (heatmaps[i].max() - heatmaps[i].min()) + eps
            scaled_heatmap = numer / denom
            scaled_heatmap = (scaled_heatmap * 255).astype("uint8")
            scaled_heatmaps.append(scaled_heatmap)
        scaled_heatmaps = np.array(scaled_heatmaps)

        # OPTIONAL: Clean output with zeroing like
        # scaled_heatmaps[np.where(scaled_heatmaps < 0.4)] = 0

        print(f"scaled_heatmaps.shape = {scaled_heatmaps.shape}, from 2d should be (224, 224)")
        # return the resulting heatmaps to the calling function
        return scaled_heatmaps

    def overlay_heatmap(self, heatmaps, video, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
        # Expect heatmaps.shape = (seq_len, height, width)
        # Expect video.shape = (seq_len, height, width, channels)

        overlays = []
        new_heatmaps = []
        for i in range(len(heatmaps)):
            heatmap = heatmaps[i]
            image = video[i]

            # apply the supplied color map to the heatmap and then
            # overlay the heatmap on the input image
            heatmap = cv2.applyColorMap(heatmap, colormap)
            heatmap = heatmap.astype(image.dtype)

            overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
            overlays.append(overlay)
            new_heatmaps.append(heatmap)

        # return a 2-tuple of the color mapped heatmap and overlaid video
        return (np.array(new_heatmaps), np.array(overlays))
