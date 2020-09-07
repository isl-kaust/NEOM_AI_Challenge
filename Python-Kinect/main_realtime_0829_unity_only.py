# import libraries
import sys
import cv2
# import matplotlib.pyplot as plt
import PyKinectV2
import PyKinectRuntime
import threading
import zmq
import numpy as np
import time
if sys.hexversion >= 0x03000000:
    pass
else:
    pass


## Initialize Kinect
def body_joints(joints, joint_ind):     # obtain 25 joints in 3D
    joint_state = joints[joint_ind].TrackingState
    positions = np.zeros(3)
    # joint is not tracked
    if joint_state == PyKinectV2.TrackingState_NotTracked:
        return positions
    # joint is not *really* tracked
    if joint_state == PyKinectV2.TrackingState_Inferred:
        return positions
    positions[0] = joints[joint_ind].Position.x
    positions[1] = joints[joint_ind].Position.y
    positions[2] = joints[joint_ind].Position.z
    return positions


# Kinect runtime object, we want only color and body frames
_kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
image_width = _kinect.color_frame_desc.Width
image_height = _kinect.color_frame_desc.Height
print(image_width)
print("Kinect Initialized")

## Initialize AI model [Abdulwahab's code]
# from AI_model import *
#
# state_path1 = "T20_next4_nframe1.pth"
# state_path2 = "T20_next4_nframe2.pth"
# state_path3 = "T20_next4_nframe3.pth"
# state_path4 = "T20_next4_nframe4.pth"
#
# num_class = 2
# max_hop = 1
# dilation = 1
# num_node = 25
# dropout = 0
# torch.cuda.is_available()

# edge, center = get_edge()
# hop_dis = get_hop_distance(num_node, edge, max_hop=max_hop)
# A = get_adjacency(hop_dis, center, num_node, max_hop, dilation)
# A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
# model1 = Model(in_channels=3, num_class=num_class, A=A, edge_importance_weighting=True, dropout=dropout)
# model2 = Model(in_channels=3, num_class=num_class, A=A, edge_importance_weighting=True, dropout=dropout)
# model3 = Model(in_channels=3, num_class=num_class, A=A, edge_importance_weighting=True, dropout=dropout)
# model4 = Model(in_channels=3, num_class=num_class, A=A, edge_importance_weighting=True, dropout=dropout)



# model1.load_state_dict(torch.load(state_path1, map_location=torch.device('cpu')))
# model1.load_state_dict(torch.load(state_path1))
# model2.load_state_dict(torch.load(state_path2))
# model3.load_state_dict(torch.load(state_path3))
# model4.load_state_dict(torch.load(state_path4))

# cuda = torch.cuda.is_available()
# if cuda:
#     A = A.cuda()
#     model1 = model1.cuda()
#     model2 = model2.cuda()
#     model3 = model3.cuda()
#     model4 = model4.cuda()


# features_test_npy = np.load("sample_data.npy")
# print(features_test_npy.shape)
#
# datapoint = features_test_npy
#
# softmax = nn.Softmax(dim=1)
# next_n_frames = 3
# pred = Predictor(n=next_n_frames - 1)  # Loading motion prediction model
# zeros = np.zeros((1, 3, next_n_frames - 1, 25, 1))
print("AI Model Loaded...")

## Define Threads

threadLock = threading.Lock()


# Kinect thread
def kinect_update():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    b = np.zeros((25, 3))
    count = 0
    duty = 30
    tt = time.time()
    loc = np.random.rand(25, 3) * 1
    tap_state = "on"
    image_update = True
    global _kinect, datapoint, img_color_resized, done, stop_flag, state_output, fps
    while True:
        message = socket.recv()     # waiting for message from unity

        if _kinect.has_new_color_frame():   # check new color frame from kinect
            frame = _kinect.get_last_color_frame()
            frame0 = np.reshape(frame, (1080, 1920, 4))
            img_color = frame0[:, :, 0:3]   # ignore the 4-th channel (depth)
            # imgplot = plt.imshow(img_color)
            scale_ratio = 0.3  # ratio of original size
            width = int(img_color.shape[1] * scale_ratio )
            height = int(img_color.shape[0] * scale_ratio )
            dim = (width, height)
            # resize image for visualization
            img_color_resized = cv2.resize(img_color, dim, interpolation=cv2.INTER_AREA)
            # imgplot = plt.imshow(img_color_resized)


        if _kinect.has_new_body_frame():    # check new body frame from kinect
            _bodies = _kinect.get_last_body_frame()
            if _bodies is not None:
                for i in range(0, _kinect.max_body_count):  # how about multi-person?
                    body = _bodies.bodies[i]
                    if not body.is_tracked:
                        continue
                    joints = body.joints
                    # print("tracked person index: " + str(i))

                    # joints location on RGB image
                    joint_points = _kinect.body_joints_to_color_space(joints)
                    for joint_i in range(0, 25):
                        if 0 < joint_points[joint_i].x < image_width and 0 < joint_points[joint_i].y < image_height:
                            pointx = joint_points[joint_i].x*scale_ratio
                            pointy = joint_points[joint_i].y*scale_ratio
                            # Center coordinates
                            center_coordinates = (int(pointx), int(pointy))
                            b[joint_i, 0] = pointx/200
                            b[joint_i, 1] = pointy/200
                            radius = 3
                            color = (0, 255, 0)
                            thickness = 3
                            img_color_resized = cv2.circle(img_color_resized, center_coordinates, radius, color, thickness)
                    # 3d joints to feed the AI model
                    image_update = True
                    joints_3d = np.zeros((3, 25))
                    for joint_i in range(0, 25):
                        joint_temp = body_joints(joints, joint_i)
                        if np.sum(joint_temp) != 0:
                            joints_3d[:, joint_i] = joint_temp
                    joints_3d = np.reshape(joints_3d, (1, 3, 1, 25, 1))
                    # print(joints_3d)
                    # threadLock.acquire()    # protect datapoint, update joints
                    # datapoint[:, :, 1:20, :, :] = datapoint[:, :, 0:19, :, :]
                    # datapoint[:, :, 0:1, :, :] = joints_3d
                    # threadLock.release()

        # send joints data and tap state to unity
        # if state_output[0] == 0:
        #     tap_on = "on"
        # else:
        #     tap_on = "off"
        tap_on = "on"
        b = b - b[0:1, :]
        str_b = np.array2string(b, formatter={'float_kind': lambda x: "%.2f" % x}, separator=',',
                                suppress_small=True)

        str_b = str_b + "," + tap_on
        str_b = str_b.replace('\n ', '')
        str_b = str_b.replace(' ', '')
        str_b = str_b.replace('[', '')
        str_b = str_b.replace(']', '')
        # print(str_b)
        socket.send(str_b.encode())

        if image_update:
            # img_color_resized = cv2.putText(img_color_resized, "fps: " + "{0:.1f}".format(fps), (20, 30),
            #                                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (209, 80, 0, 255), 3)
            # img_color_resized = cv2.putText(img_color_resized, "state: " + str(state_output) + " | Tap: " + tap_on, (20, 80),
            #                                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (209, 80, 0, 255), 3)
            cv2.imshow('frame', img_color_resized)
            image_update = False
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag = 1
                break
    cv2.destroyAllWindows()


def model_update():
    global state_output, datapoint, pred, zeros, img_color_resized, fps_pre, done, \
        model_input, state_output_ray, fps, stop_flag
    # cc = 1
    fps_pre = 0
    tt = time.time()
    # model1.eval()
    # model2.eval()
    # model3.eval()
    # model4.eval()
    # state_output_temp = np.zeros((4))

    while True:
        # with torch.no_grad():
        #     # datapoint.shape = (1 batch size, 3 dims, 20 frames, 25 joints, 1 objects)
        #     model_input1 = pred.transform(
        #         np.append(datapoint, zeros, axis=2))  # Shape = (1 batch size, 3 dims, 22 frames, 25 joints, 1 objects)
        #     model_input2 = torch.FloatTensor(pre_normalization(model_input1))
        #     if cuda:
        #         model_input2 = model_input2.cuda()
        #     model_output = model1(model_input2[:, :, 0:0 + 20, :, :])
        #     state_output_temp[0] = F.log_softmax(model_output, dim=1).max(1)[1]
        #     model_output = model2(model_input2[:, :, 1:1 + 20, :, :])
        #     state_output_temp[1] = F.log_softmax(model_output, dim=1).max(1)[1]
        #     model_output = model3(model_input2[:, :, 2:2 + 20, :, :])
        #     state_output_temp[2] = F.log_softmax(model_output, dim=1).max(1)[1]
        #     model_output = model4(model_input2[:, :, 3:3 + 20, :, :])
        #     state_output_temp[3] = F.log_softmax(model_output, dim=1).max(1)[1]
        #     val = state_output_temp[1]
            # if val == 1:
            #     ser.write(b'z');
            #     print(b'z')
            # else:
            #     ser.write(b'a');
            #     print(b'a')

        time.sleep(0.03)
        # cc = cc + 1
        threadLock.acquire()    # protect state_output, fps
        # state_output = state_output_temp
        interval = time.time() - tt
        fps = 1 / interval
        tt = time.time()
        if np.abs(fps_pre - fps) > 1:
            fps_pre = fps
        # kinect_ready_flag = False
        threadLock.release()
        # print("I am running..." + str(cc) + "| FPS = " + str(fps))
        if stop_flag:
            break


## Main
img_color_resized = np.zeros((648, 1152, 3), dtype="uint8") # image for opencv visualization
state_output = [0, 0, 0, 0]        # predictions
stop_flag = False               # stop when exit opencv
fps = 0.0

# initialize thread
kinect_thread = threading.Thread(target=kinect_update)
model_thread = threading.Thread(target=model_update)
# communication_thread = threading.Thread(target=communication_update)

kinect_thread.start()
model_thread.start()
# communication_thread.start()

kinect_thread.join()
model_thread.join()
# communication_thread.start()
##

