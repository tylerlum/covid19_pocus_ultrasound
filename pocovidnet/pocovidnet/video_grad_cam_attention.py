from pocovidnet.attention_explanation import AttentionExplanation
from pocovidnet.video_grad_cam import VideoGradCAM
import numpy as np


class VideoGradCAMAttention:
    def __init__(self, model):
        self.model = model
        self.video_grad_cam = VideoGradCAM(model)
        self.attention_explanation = AttentionExplanation(model)

    def compute_attention_maps(self, video):
        # Expect video.shape = (seq_len, height, width, channels)
        heatmaps = self.video_grad_cam.compute_heatmaps(video)
        attn_weights = self.attention_explanation.compute_attention_weights(video)

        # Combine
        scaled_heatmaps = []
        for i in range(len(heatmaps)):
            scaled_heatmap = heatmaps[i] * attn_weights[0][i] * attn_weights.size  # Scale by weights, then scale up to same mean
            scaled_heatmap = np.clip(scaled_heatmap, 0, 255)  # Avoid out of bounds
            # scaled_heatmap[np.where(scaled_heatmap < 100)] = 0  # Filter out low activations
            scaled_heatmaps.append(scaled_heatmap)
        scaled_heatmaps = np.array(scaled_heatmaps)

        (scaled_heatmaps, overlays) = self.video_grad_cam.overlay_heatmaps(scaled_heatmaps, video, alpha=0.5)
        return scaled_heatmaps, overlays
