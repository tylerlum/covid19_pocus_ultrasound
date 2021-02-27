from pocovidnet.transformer import TransformerBlock
from keras.models import Model
import numpy as np


class AttentionExplanation:
    def __init__(self, model, layerName=None):
        self.model = model
        self.layerName = layerName

        # if the layer name is None, attempt to automatically find the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the first transformer block layer in the network
        for layer in self.model.layers:
            if isinstance(layer, TransformerBlock):
                print(f"find_target_layer: found {layer.name} {layer.output_shape}")
                return layer.name
        raise ValueError("Could not find TransformerBlock. Cannot apply AttentionExplanation")

    def compute_attention_map(self, video):
        # Expect video.shape = (seq_len, height, width, channels)

        # Get model that outputs both attention weights and predictions
        transformer_output, attn_weights_output = self.model.get_layer(self.layerName).output
        attn_model = Model(self.model.input, [attn_weights_output, self.model.output])
        (attn_matrix_by_head, predictions) = attn_model(np.expand_dims(video, axis=0))
        # attn_matrix_by_head.shape = (batch_size, num_heads, seq_len, seq_len)

        # batch_size dimension is just 1, so remove that.
        attn_matrix_by_head = attn_matrix_by_head[0]
        # attn_matrix_by_head.shape = (num_heads, seq_len, seq_len)

        # Average attentions across heads
        attn_matrix = np.mean(attn_matrix_by_head, axis=0)
        # attn_matrix_by_head.shape = (seq_len, seq_len)

        # Average attentions across frames to show most relevant frames to output
        attn_weights = np.mean(attn_matrix, axis=0)
        # attn_weights.shape = (seq_len)

        # Reshape attention so it can be plotted nicely
        attn_weights = np.reshape(attn_weights, (-1, len(attn_weights)))
        # attn_weights.shape = (1, seq_len)

        return attn_weights
