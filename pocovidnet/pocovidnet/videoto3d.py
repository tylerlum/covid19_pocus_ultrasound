import numpy as np
import cv2
import pickle
import os
import math
from tqdm import tqdm


class Videoto3D:

    def __init__(self, vid_path, width=224, height=224, depth=5, framerate=5, grayscale=False, optical_flow_type=None):
        self.vid_path = vid_path
        self.width = width
        self.height = height
        self.depth = depth
        self.framerate = framerate
        # self.max_vid = {"cov": 10, "pne": 10, "reg": 10}
        self.max_vid = {"cov": 100, "pne": 100, "reg": 100}
        self.grayscale = grayscale
        self.optical_flow_type = optical_flow_type

    def save_data(self, data_3d, labels_3d, files_3d, save_path):
        print("SAVE DATA", data_3d.shape, np.max(data_3d))
        with open(save_path, "wb") as outfile:
            pickle.dump((data_3d, labels_3d, files_3d), outfile)

    def video3d(self, vid_files, labels, save=None):
        # Iterate to fill data
        data_3d, labels_3d, files_3d = [], [], []
        for vid, label in zip(vid_files, labels):
            cap = cv2.VideoCapture(os.path.join(self.vid_path, vid))
            video_framerate = cap.get(cv2.CAP_PROP_FPS)
            video_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            show_every = math.ceil(video_framerate / self.framerate)  # ceil to avoid 0
            frames_available = video_num_frames / show_every
            video_clips_available = frames_available // self.depth
            print(f"{vid}, {video_framerate} FPS, {video_num_frames} frames, show every {show_every} frames, available frames: {frames_available}, available video clips: {video_clips_available}")

            video_clips_counter = 0
            current_data = []

            while cap.isOpened():
                frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
                ret, frame = cap.read()
                if not ret:
                    break

                image = frame if not self.grayscale else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (self.width, self.height))
                from tensorflow.keras.applications.efficientnet import preprocess_input
                # image = preprocess_input(image)

                # Grab image every X frames
                if frame_id % show_every == 0:
                    current_data.append(image)

                # Store video clip
                if len(current_data) == self.depth:
                    data_3d.append(current_data)
                    labels_3d.append(label)
                    files_3d.append(vid)
                    current_data = []
                    video_clips_counter += 1

                # Got max number of clips for this video
                if video_clips_counter >= self.max_vid[label]:
                    print(f"already {video_clips_counter} clips taken from this video")
                    break

            cap.release()

        if self.optical_flow_type is not None:
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
            flow_type = self.optical_flow_type.lower()
            if flow_type == "farneback":
                optical_flow_interface = cv2.optflow.createOptFlow_Farneback()
            elif flow_type == "dtvl1":
                optical_flow_interface = cv2.optflow.createOptFlow_DualTVL1()
            elif flow_type == "deepflow":
                optical_flow_interface = cv2.optflow.createOptFlow_DeepFlow()
            elif flow_type == "denserlof":
                optical_flow_interface = cv2.optflow.createOptFlow_DenseRLOF()
            elif flow_type == "pcaflow":
                optical_flow_interface = cv2.optflow.createOptFlow_PCAFlow()
            elif flow_type == "simpleflow":
                optical_flow_interface = cv2.optflow.createOptFlow_SimpleFlow()
            elif flow_type == "sparserlof":
                optical_flow_interface = cv2.optflow.createOptFlow_SparseRLOF()
            elif flow_type == "sparsetodense":
                optical_flow_interface = cv2.optflow.createOptFlow_SparseToDense()
            else:
                raise ValueError(f"Invalid flow_type = {flow_type}")
            optical_flows = []
            for num, images in enumerate(tqdm(data_3d)):
                optical_flow_frames = []
                for i in range(len(images)):
                    is_already_grey = (len(images[i].shape) == 2 or images[i].shape[2] == 1)
                    curr_grey = images[i]
                    prev_grey = images[i - 1] if i > 0 else np.zeros_like(curr_grey)
                    if not is_already_grey:
                        prev_grey = cv2.cvtColor(prev_grey, cv2.COLOR_BGR2GRAY)
                        curr_grey = cv2.cvtColor(curr_grey, cv2.COLOR_BGR2GRAY)

                    flow = optical_flow_interface.calc(prev_grey, curr_grey, None)

                    bound = 15
                    flow_x = flow_to_img(flow[..., 0], bound)
                    flow_y = flow_to_img(flow[..., 1], bound)
                    flow_xy = (np.abs(flow_x) + np.abs(flow_y)) / 2

                    stacked_flow = np.array([flow_x, flow_y, flow_xy])
                    stacked_flow = np.transpose(stacked_flow, [1, 2, 0])  # Channels last

                    optical_flow_frames.append(stacked_flow)

                optical_flows.append(optical_flow_frames)

            optical_flow_data = np.asarray(optical_flows)

        # If grayscale, still want a channel size of 1
        data = np.asarray(data_3d) if not self.grayscale else np.expand_dims(np.asarray(data_3d), 4)

        # Bring together
        if self.optical_flow_type is not None:
            data = np.concatenate([data, optical_flow_data], axis=4)

        data = data / 255.0

        # Save
        if save is not None:
            self.save_data(data, labels_3d, files_3d, save)
        return data, labels_3d, files_3d
