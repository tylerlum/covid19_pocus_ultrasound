import numpy as np
import cv2
import pickle
import os
import math


class Videoto3D:

    def __init__(self, vid_path, width=224, height=224, depth=5, framerate=5, grayscale=False):
        self.vid_path = vid_path
        self.width = width
        self.height = height
        self.depth = depth
        self.framerate = framerate
        # self.max_vid = {"cov": 10, "pne": 10, "reg": 10}
        self.max_vid = {"cov": 100, "pne": 100, "reg": 100}
        self.grayscale = grayscale

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

        def ToImg(raw_flow, bound):
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
        dtvl1 = cv2.optflow.createOptFlow_DualTVL1()
        optical_flows = []
        for num, images in enumerate(data_3d):
            print(f"Working on image {num} of {len(data_3d)}")
            optical_flow_frames = []
            for i in range(1, len(images)):
                is_already_grey = (len(images[i].shape) == 2 or images[i].shape[2] == 1)
                prev_grey = images[i - 1]
                curr_grey = images[i]
                if not is_already_grey:
                    prev_grey = cv2.cvtColor(prev_grey, cv2.COLOR_BGR2GRAY)
                    curr_grey = cv2.cvtColor(curr_grey, cv2.COLOR_BGR2GRAY)

                # flow_type = "dtvl1"
                flow_type = "farneback"
                if flow_type.lower() == "dtvl1":
                    flow = dtvl1.calc(prev_grey, curr_grey, None)
                elif flow_type.lower() == "farneback":
                    flow = cv2.calcOpticalFlowFarneback(prev_grey, curr_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                else:
                    # Default dtvl1
                    flow = dtvl1.calc(prev_grey, curr_grey, None)

                bound = 15
                flow_x = ToImg(flow[..., 0], bound)
                flow_y = ToImg(flow[..., 1], bound)
                flow_xy = (np.abs(flow_x) + np.abs(flow_y)) / 2

                stacked_flow = np.array([flow_x, flow_y, flow_xy])
                stacked_flow = np.transpose(stacked_flow, [1, 2, 0])  # Channels last
                cv2.imwrite(f"flow_x_{num}_{i}.jpg", flow_x)
                cv2.imwrite(f"flow_y_{num}_{i}.jpg", flow_y)
                cv2.imwrite(f"flow_xy_{num}_{i}.jpg", flow_xy)

                optical_flow_frames.append(stacked_flow)

            optical_flows.append(optical_flow_frames)

        print(f"len(optical_flows) = {len(optical_flows)}")
        print(f"len(optical_flow_frames) = {len(optical_flow_frames)}")
        optical_flow_data = np.asarray(optical_flows)
        print(f"optical_flow_data.shape = {optical_flow_data.shape}")
        print(f"np.mean(optical_flow_data) = {np.mean(optical_flow_data)}")

        # If grayscale, still want a channel size of 1
        data = np.asarray(data_3d) if not self.grayscale else np.expand_dims(np.asarray(data_3d), 4)
        print(f"data.shape = {data.shape}")
        print(f"np.mean(data) = {np.mean(data)}")

        # Normalize
        optical_flow_data = optical_flow_data / 255
        data = data / 255

        # Save
        if save is not None:
            self.save_data(data, labels_3d, files_3d, save)
        return data, labels_3d, files_3d
