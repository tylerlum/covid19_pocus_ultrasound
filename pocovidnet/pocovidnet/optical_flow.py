import numpy as np
import cv2
from tqdm import tqdm
from pocovidnet import OPTICAL_FLOW_ALGORITHM_FACTORY


def get_optical_flow_data(data_3d, optical_flow_type):
    def flow_to_img(raw_flow, bound):
        '''
        this function scale the input pixels to 0-255 with bi-bound

        :param raw_flow: input raw pixel value (not in 0-255)
        :param bound: upper and lower bound (-bound, bound)
        :return: pixel value scale from 0 to 255
        '''
        flow = raw_flow
        flow[flow > bound] = bound
        flow[flow < -bound] = -bound
        flow += bound
        flow *= (255 / float(2 * bound))
        return flow

    # Optical flow of video clips
    flow_type = optical_flow_type.lower()
    optical_flow_interface = OPTICAL_FLOW_ALGORITHM_FACTORY[flow_type]()
    optical_flows = []

    # Video clips
    print("Optical flow calculations:")
    for num, images in enumerate(tqdm(data_3d)):
        optical_flow_frames = []

        # Frames
        for i in range(len(images)):
            # Get 2 frames for calculation
            is_already_grey = (len(images[i].shape) == 2 or images[i].shape[2] == 1)
            curr_grey = images[i]
            prev_grey = images[i - 1] if i > 0 else np.zeros_like(curr_grey)  # Handle first image case
            if not is_already_grey:
                prev_grey = cv2.cvtColor(prev_grey, cv2.COLOR_BGR2GRAY)
                curr_grey = cv2.cvtColor(curr_grey, cv2.COLOR_BGR2GRAY)

            # Calculate and clean optical flow
            flow = optical_flow_interface.calc(prev_grey, curr_grey, None)
            bound = 15
            flow_x = flow_to_img(flow[..., 0], bound)
            flow_y = flow_to_img(flow[..., 1], bound)
            flow_xy = (np.abs(flow_x) + np.abs(flow_y)) / 2

            # Stack flow in different directions
            stacked_flow = np.array([flow_x, flow_y, flow_xy])
            stacked_flow = np.transpose(stacked_flow, [1, 2, 0])  # Channels last

            optical_flow_frames.append(stacked_flow)

        optical_flows.append(optical_flow_frames)

    optical_flow_data = np.asarray(optical_flows)
    return optical_flow_data
