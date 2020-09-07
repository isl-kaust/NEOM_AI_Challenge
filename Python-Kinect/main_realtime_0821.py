
import ctypes
import _ctypes
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread


def body_joints(joints, joint0):
    joint0State = joints[joint0].TrackingState;
    positions = np.zeros(3)
    # both joints are not tracked
    if joint0State == PyKinectV2.TrackingState_NotTracked:
        return positions
    # both joints are not *really* tracked
    if joint0State == PyKinectV2.TrackingState_Inferred:
        return positions
    # ok, at least one is good
    positions[0] = joints[joint0].Position.x
    positions[1] = joints[joint0].Position.y
    positions[2] = joints[joint0].Position.z
    return positions

# Kinect runtime object, we want only color and body frames
_kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
image_width = _kinect.color_frame_desc.Width
image_height = _kinect.color_frame_desc.Height
print(image_width)

## import AI model


import AI_model
from AI_model import *
import ray


state_path = "state_dict_binary-wudu-T20-allP-overlap1.pth"
num_class = 2
max_hop = 1
dilation = 1
num_node = 25
dropout = 0
edge, center = get_edge()
hop_dis = get_hop_distance(num_node, edge, max_hop=max_hop)
A = get_adjacency(hop_dis, center, num_node, max_hop, dilation)
A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
model = Model(in_channels=3, num_class=num_class, A=A, edge_importance_weighting=True, dropout=dropout)
model.load_state_dict(torch.load(state_path, map_location=torch.device('cpu')))

features_test_npy = np.load("sample_data.npy")
print(features_test_npy.shape)

datapoint = features_test_npy
ray.shutdown()
ray.init(num_cpus=3, num_gpus=0)

@ray.remote
def parallelModel(pred_i):
    model_output = model(model_input[:, :, pred_i:pred_i + 20, :, :])
    state_output = F.log_softmax(model_output, dim=1).max(1)[1]
    #     probs = softmax(model_output) # Can be used to monitor the probabilities
    return state_output.item()

softmax = nn.Softmax(dim=1)
next_n_frames = 3
pred = Predictor(n=next_n_frames - 1)  # Loading motion prediction model
zeros = np.zeros((1, 3, next_n_frames - 1, 25, 1))
# datapoint = np.random.rand(1, 10)
# datapoint[0, 0:9] = datapoint[0, 1:10]
# datapoint[0, 10] = 0



## Run main code


tt = time.time()
fps_pre = 0
while True:
    # update RGB frame
    if _kinect.has_new_color_frame():   # check new color frame from kinect
        frame = _kinect.get_last_color_frame()
        frame0 = np.reshape(frame, (1080, 1920, 4))
        img_color = frame0[:, :, 0:3]
        # imgplot = plt.imshow(img_color)
        scale_ratio = 0.6  # ratio of original size
        width = int(img_color.shape[1] * scale_ratio )
        height = int(img_color.shape[0] * scale_ratio )
        dim = (width, height)
        # resize image
        img_color_resized = cv2.resize(img_color, dim, interpolation=cv2.INTER_AREA)
        # imgplot = plt.imshow(img_color_resized)

    # update body joints data
    if _kinect.has_new_body_frame():    # check new body frame from kinect
        _bodies = _kinect.get_last_body_frame()
        if _bodies is not None:
            # points_str = []
            for i in range(0, _kinect.max_body_count):
                body = _bodies.bodies[i]
                if not body.is_tracked:
                    continue
                joints = body.joints

                # joints location on RGB image
                joint_points = _kinect.body_joints_to_color_space(joints)
                for joint_i in range(0,25):
                    if 0< joint_points[joint_i].x < image_width and 0 < joint_points[joint_i].y < image_height:
                        pointx = joint_points[joint_i].x*scale_ratio
                        pointy = joint_points[joint_i].y*scale_ratio
                        # Center coordinates
                        center_coordinates = (int(pointx), int(pointy))
                        radius = 5
                        color = (255, 0, 0)
                        thickness = 2
                        image2 = cv2.circle(img_color_resized, center_coordinates, radius, color, thickness)

                # 3d joints to feed the AI model
                joints_3d = np.zeros((3, 25))
                for joint_i in range(0, 25):
                    joints_3d[:, joint_i] = body_joints(joints, joint_i)
                joints_3d = np.reshape(joints_3d, (1, 3, 1, 25, 1))
                # joints_3d.shape
                # imgplot = plt.imshow(img_color_resized)
                # joint_head = body_joints(joints, 3)
                # print(joint_head)

    datapoint[:, :, 1:20, :, :] = datapoint[:, :, 0:19, :, :]
    datapoint[:, :, 0:1, :, :] = joints_3d

    # datapoint = np.random.rand(1, 3, 20, 25, 1)
    # tt = time.time()
    with torch.no_grad():
        # datapoint.shape = (1 batch size, 3 dims, 20 frames, 25 joints, 1 objects)
        model_input = pred.transform(np.append(datapoint, zeros, axis=2))
        # Shape = (1 batch size, 3 dims, 22 frames, 25 joints, 1 objects)
        model_input = torch.FloatTensor(pre_normalization(model_input))
        # state_output_ray = [parallelModel.remote(pred_i) for pred_i in range(next_n_frames)]
        state_output_ray = [parallelModel.remote(pred_i) for pred_i in range(3)]
        state_output = ray.get(state_output_ray)  # output example: [1, 1, 0]
    # print("AI model:" + str(time.time() - tt))

    # print(state_output)
    # print(time.time() - tt)
    # str(state_output)
    # show fps
    interval = time.time() - tt
    fps = 1/interval
    tt = time.time()
    if np.abs(fps_pre - fps) > 1:
        fps_pre = fps
    # print(fps)
    img_color_resized = cv2.putText(img_color_resized, "fps: " + "{0:.1f}".format(fps_pre), (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (209, 80, 0, 255), 3)
    img_color_resized = cv2.putText(img_color_resized, "state: " + str(state_output), (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (209, 80, 0, 255), 3)

    # imgplot = plt.imshow(img_color_resized)

    cv2.imshow('frame', img_color_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


##

