from pocovidnet.read_mat import loadmat
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_lung_dataset", type=str, required=True, help='path to lung dataset')
args = parser.parse_args()
print(f"Looking at {args.path_to_lung_dataset}")

all_mat_files = []

# Patients
for patient_dir in tqdm(os.listdir(args.path_to_lung_dataset)):
    path_to_patient_dir = os.path.join(args.path_to_lung_dataset, patient_dir)

    # Mat files OR directories to mat files
    for mat_or_dir in os.listdir(path_to_patient_dir):
        path_to_mat_or_dir = os.path.join(path_to_patient_dir, mat_or_dir)

        # Mat file
        if not os.path.isdir(path_to_mat_or_dir):
            all_mat_files.append(path_to_mat_or_dir)

        # Directory to mat files
        else:
            for mat_file in os.listdir(path_to_mat_or_dir):
                all_mat_files.append(os.path.join(path_to_mat_or_dir, mat_file))

print(f"Got {len(all_mat_files)} mat files")
print("Loading mat files in now")
b_lines_list = []
stop_frame_list = []
start_frame_list = []
subpleural_consolidations_list = []
pleural_irregularities_list = []
a_lines_list = []
lobar_consolidations_list = []
pleural_effusions_list = []
no_lung_sliding_list = []
width_list = []
height_list = []
frames_list = []
view_list = []
frame_time_list = []

a_line_b_line_tuple_list = []
len_b_line_tuple_list = []
full_len_b_line_tuple_list = []
for mat_file in tqdm(all_mat_files):
    mat = loadmat(mat_file)

    # Get labels
    b_lines = mat['labels']['B-lines']
    stop_frame = mat['labels']['stop_frame']
    start_frame = mat['labels']['start_frame']
    subpleural_consolidations = mat['labels']['Sub-pleural consolidations']
    pleural_irregularities = mat['labels']['Pleural irregularities']
    a_lines = mat['labels']['A-lines']
    lobar_consolidations = mat['labels']['Lobar consolidations']
    pleural_effusions = mat['labels']['Pleural effussions']
    no_lung_sliding = mat['labels']['No lung sliding']

    b_lines_list.append(b_lines)
    stop_frame_list.append(stop_frame)
    start_frame_list.append(start_frame)
    subpleural_consolidations_list.append(subpleural_consolidations)
    pleural_irregularities_list.append(pleural_irregularities)
    a_lines_list.append(a_lines)
    lobar_consolidations_list.append(lobar_consolidations)
    pleural_effusions_list.append(pleural_effusions)
    no_lung_sliding_list.append(no_lung_sliding)
    a_line_b_line_tuple_list.append((a_lines, b_lines))
    len_b_line_tuple_list.append((stop_frame-start_frame, b_lines))

    # Check data
    cine = mat['cleaned']
    height_list.append(cine.shape[0])
    width_list.append(cine.shape[1])
    frames_list.append(cine.shape[2])
    full_len_b_line_tuple_list.append((cine.shape[2], b_lines))

    # Check view
    view = mat['labels']['view']
    view_list.append(view)

    # Check framerate
    if 'FrameTime' in mat['dicom_info'].keys():
        frame_time_list.append(mat['dicom_info']['FrameTime'])
    else:
        frame_time_list.append(-100)

print(f"b_lines_list = {np.unique(b_lines_list, return_counts=True)}")
print(f"stop_frame_list = {np.unique(stop_frame_list, return_counts=True)}")
print(f"start_frame_list = {np.unique(start_frame_list, return_counts=True)}")
print(f"subpleural_consolidations_list = {np.unique(subpleural_consolidations_list, return_counts=True)}")
print(f"pleural_irregularities_list = {np.unique(pleural_irregularities_list, return_counts=True)}")
print(f"a_lines_list = {np.unique(a_lines_list, return_counts=True)}")
print(f"lobar_consolidations_list = {np.unique(lobar_consolidations_list, return_counts=True)}")
print(f"pleural_effusions_list = {np.unique(pleural_effusions_list, return_counts=True)}")
print(f"no_lung_sliding_list = {np.unique(no_lung_sliding_list, return_counts=True)}")
print(f"width_list = {np.unique(width_list, return_counts=True)}")
print(f"height_list = {np.unique(height_list, return_counts=True)}")
print(f"frames_list = {np.unique(frames_list, return_counts=True)}")
print(f"view_list = {np.unique(view_list, return_counts=True)}")

data_source = args.path_to_lung_dataset.split(os.sep)[-1]

plt.figure()
plt.hist(b_lines_list)
plt.title("b lines severity histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} b_lines_list.png")

plt.figure()
plt.hist(stop_frame_list)
plt.title("stop frame histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} stop_frame_list.png")

plt.figure()
plt.hist(start_frame_list)
plt.title("start frame histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} start_frame_list.png")

plt.figure()
plt.hist(subpleural_consolidations_list)
plt.title("subpleural consolidations histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} subpleural_consolidations_list.png")

plt.figure()
plt.hist(pleural_irregularities_list)
plt.title("pleural irregularities histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} pleural_irregularities_list.png")

plt.figure()
plt.hist(a_lines_list)
plt.title("a lines histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} a_lines_list.png")

plt.figure()
plt.hist(lobar_consolidations_list)
plt.title("lobar consolidations histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} lobar_consolidations_list.png")

plt.figure()
plt.hist(pleural_effusions_list)
plt.title("pleural effusions histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} pleural_effusions_list.png")

plt.figure()
plt.hist(no_lung_sliding_list)
plt.title("no lung sliding histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} no_lung_sliding_list.png")

plt.figure()
plt.hist(width_list)
plt.title("frame width histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} width_list.png")

plt.figure()
plt.hist(height_list)
plt.title("frame height histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} height_list.png")

plt.figure()
plt.hist(frames_list)
plt.title("total number of frames histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} frames_list.png")

plt.figure()
plt.hist(view_list)
plt.title("view histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} view_list.png")

plt.figure()
plt.hist(frame_time_list)
plt.title("time between frames (ms) histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} frame_time_list.png")

b_lines_with_a_lines = np.array([b for a, b in a_line_b_line_tuple_list if a == 1])
b_lines_without_a_lines = np.array([b for a, b in a_line_b_line_tuple_list if a == 0])
print(f"len(b_lines_without_a_lines) = {len(b_lines_without_a_lines)}")
print(f"len(b_lines_with_a_lines) = {len(b_lines_with_a_lines)}")
print(f"len(a_line_b_line_tuple_list) = {len(a_line_b_line_tuple_list)}")
plt.figure()
plt.hist(b_lines_without_a_lines)
plt.title("b lines severity (without a lines) histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} b_lines_without_a_lines.png")

plt.figure()
plt.hist(b_lines_with_a_lines)
plt.title("b lines severity (with a lines) histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} b_lines_with_a_lines.png")

plt.figure()
stop_minus_start_list = np.array(stop_frame_list) - np.array(start_frame_list)
plt.hist(stop_minus_start_list)
plt.title("used number of frames (stop - start) histogram")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} stop_minus_start_list.png")

len_with_b_lines = np.array([len_ for len_, b in len_b_line_tuple_list if b != 0])
len_without_b_lines = np.array([len_ for len_, b in len_b_line_tuple_list if b == 0])
print(f"len(len_with_b_lines) = {len(len_with_b_lines)}")
print(f"len(len_without_b_lines) = {len(len_without_b_lines)}")

plt.figure()
plt.hist(len_without_b_lines)
plt.title("Len without b lines")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} len_without_b_lines.png")

plt.figure()
plt.hist(len_with_b_lines)
plt.title("Len with b lines")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} len_with_b_lines.png")

full_len_with_b_lines = np.array([len_ for len_, b in full_len_b_line_tuple_list if b != 0])
full_len_without_b_lines = np.array([len_ for len_, b in full_len_b_line_tuple_list if b == 0])
print(f"len(full_len_with_b_lines) = {len(full_len_with_b_lines)}")
print(f"len(full_len_without_b_lines) = {len(full_len_without_b_lines)}")

plt.figure()
plt.hist(full_len_without_b_lines)
plt.title("full_Len without b lines")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} full_len_without_b_lines.png")

plt.figure()
plt.hist(full_len_with_b_lines)
plt.title("full_Len with b lines")
plt.ylabel("Frequency")
plt.savefig(f"{data_source} full_len_with_b_lines.png")
