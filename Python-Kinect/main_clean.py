import sys
import time
import numpy as np
import cv2

import PyKinectV2
import PyKinectRuntime

if sys.hexversion >= 0x03000000:
    pass
else:
    pass


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

##
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
                # joints_3d = np.zeros(25, 3)
                # joint_head = body_joints(joints, 3)

                # imgplot = plt.imshow(img_color_resized)
                # joint_head = body_joints(joints, 3)
                # print(joint_head)

    # show fps
    interval = time.time() - tt
    fps = 1/interval
    tt = time.time()
    if np.abs(fps_pre - fps) > 3:
        fps_pre = fps
    text_position = (50, 50)
    img_color_resized = cv2.putText(img_color_resized, "fps: " + "{0:.1f}".format(fps_pre), text_position,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
    # imgplot = plt.imshow(img_color_resized)

    cv2.imshow('frame', img_color_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()



##

