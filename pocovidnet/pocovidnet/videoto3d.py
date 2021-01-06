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
        # iterate to fill data
        data_3d, labels_3d, files_3d = [], [], []
        for train_vid, train_label in zip(vid_files, labels):
            cap = cv2.VideoCapture(os.path.join(self.vid_path, train_vid))
            video_framerate = cap.get(cv2.CAP_PROP_FPS)
            video_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            show_every = math.ceil(video_framerate / self.framerate)  # ceil to avoid 0
            frames_available = video_num_frames / show_every
            video_clips_available = frames_available // self.depth
            print(f"{train_vid}, {video_framerate} FPS, {video_num_frames} frames, show every {show_every} frames, available frames: {frames_available}, available video clips: {video_clips_available}")

            video_clips_counter = 0
            current_data = []

            while cap.isOpened():
                frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
                ret, frame = cap.read()
                if (ret != True):
                    break

                image = frame if not self.grayscale else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (self.width, self.height))

                # Grab image every X frames
                if frame_id % show_every == 0:
                    current_data.append(image)

                # Store video clip
                if len(current_data) == self.depth:
                    data_3d.append(current_data)
                    labels_3d.append(train_label)
                    files_3d.append(train_vid)
                    current_data = []
                    video_clips_counter += 1

                # Got max number of clips for this video
                if video_clips_counter >= self.max_vid[train_label]:
                    print(f"already {video_clips_counter} clips taken from this video")
                    break

            cap.release()

        # If grayscale, still want a channel size of 1
        data = np.asarray(data_3d) if not self.grayscale else np.expand_dims(np.asarray(data_3d), 4)
        data = data / 255
        if save is not None:
            self.save_data(data, labels_3d, files_3d, save)
        return data, labels_3d, files_3d
