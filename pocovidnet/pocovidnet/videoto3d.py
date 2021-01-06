import numpy as np
import cv2
import pickle
import os


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
        for train_vid, train_lab in zip(vid_files, labels):
            cap = cv2.VideoCapture(os.path.join(self.vid_path, train_vid))
            fr = cap.get(5)
            show_every = round(fr / self.framerate)
            print(
                train_vid, fr, cap.get(7), "available frames:",
                cap.get(7) / show_every
            )
            frames_available = cap.get(7) / show_every
            end_is_close = frames_available % self.depth >= 4
            number_selected = int(
                end_is_close
            ) + frames_available // self.depth
            print(number_selected, cap.get(7), "show every", show_every)
            vid_part_count = 0
            current_data = []
            # for frame_id in range(int(cap.get(7))):
            while cap.isOpened():
                frame_id = cap.get(1)
                ret, frame = cap.read()
                if (ret != True):
                    break
                # plt.imshow(image)
                # plt.show()
                image = frame if not self.grayscale else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (self.width, self.height))

                if frame_id % show_every == 0 or (
                    end_is_close and frame_id == int(cap.get(7) - 1)
                ):
                    current_data.append(image)
                if len(current_data) == self.depth:
                    data_3d.append(current_data)
                    labels_3d.append(train_lab)
                    files_3d.append(train_vid)
                    current_data = []
                    vid_part_count += 1
                if vid_part_count >= self.max_vid[train_lab]:
                    print("already 5 parts of this video")
                    break
            cap.release()
        # If grayscale, still want a channel size of 1
        data = np.asarray(data_3d) if not self.grayscale else np.expand_dims(np.asarray(data_3d), 4)
        data = data / 255
        if save is not None:
            self.save_data(data, labels_3d, files_3d, save)
        return data, labels_3d, files_3d
