#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq
import numpy as np
import time
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
count = 0
duty = 30

tt = time.time()
loc = np.random.rand(25, 3) * 1
tap_state = "on"
while True:

    #  Wait for next request from client
    message = socket.recv()
    # print(message)
    # time_a = time.time()
    # time_a = str(time_a)
    # socket.send(time_a.encode())
    #  In the real world usage, you just need to replace time.sleep() with
    #  whatever work you want python to do.
    time.sleep(0.03)
    if count == duty:
        count = 0
        if tap_state == "on":
            tap_state = "off"
        else:
            tap_state = "on"

        print(tap_state + " | " + str(time.time()-tt))
        tt = time.time()
    else:
        count += 1


    # count = 66
    a = np.random.rand(3)*.1
    a[0] = np.cos(count/duty*2*np.pi)*0.1
    a[1] = np.sin(count/duty*2*np.pi)*0.1
    a[2] = 0.5
    b = loc + a
    str_b = np.array2string(b, formatter={'float_kind':lambda x: "%.2f" % x}, separator=',', suppress_small=True)
    str_b = str_b + "," + tap_state

    str_b = str_b.replace('\n ', '')
    str_b = str_b.replace('[', '')
    str_b = str_b.replace(']', '')
    # print(str_b)
    socket.send(str_b.encode())

    #  Send reply back to client
    #  In the real world usage, after you finish your work, send your output here
    # socket.send(b"world_123123123")


