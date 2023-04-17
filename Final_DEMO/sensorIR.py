"""
Created on Friday Apr 07 00:15:00 2023

@author: Gabriela Hilario Acuapan & Luis Alberto Pineda Gómez
File: sensorIR.py
Comments: Official primense openni2 and nite2 python bindings.

            Streams infra-red camera

ref:
    https://github.com/elmonkey/Python_OpenNI2/blob/master/samples/ex6_ird_stream.py
"""

from Preprocessing import img_prepro # Preprocessing image module

from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np
import cv2 as cv

openni2.initialize(r'C:\Users\gabri\OneDrive\Documentos\RYMA-Maestria-Tesis\win64\win64\bin')     # can also accept the path of the OpenNI redistribution
if (openni2.is_initialized()):
    print ("openNI2 initialized")
else:
    print ("openNI2 not initialized")

dev = openni2.Device.open_any()
print (dev.get_sensor_info(openni2.SENSOR_IR)) # sensor information
depth_stream = dev.create_depth_stream()
ir_stream = dev.create_ir_stream()

## Set stream speed and resolution
w = 640
h = 480
fps = 30

## Set the video properties
#print 'Get b4 video mode', depth_stream.get_video_mode()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=w, resolutionY=h, fps=fps))
#print 'Get after video mode', depth_stream.get_video_mode()

# ----------------------------------------Start the stream -----------------------------------------------------
depth_stream.start()
ir_stream.start() 
print ("Started!") 

def get_ir():
    """
    Returns numpy ndarrays representing raw and ranged infra-red(IR) images.
    Outputs:
        ir    := raw IR, 1L ndarray, dtype=uint16, min=0, max=2**12-1
        ir4d  := IR for display, 3L ndarray, dtype=uint8, min=0, max=255
    """
    ir_frame = ir_stream.read_frame()
    ir_frame_data = ir_stream.read_frame().get_buffer_as_uint16()
    ir4d = np.ndarray((ir_frame.height, ir_frame.width),dtype=np.uint16, buffer = ir_frame_data).astype(np.float32)
    ir4d = np.uint8((ir4d/ir4d.max()) * 255)
    ir4d = cv.cvtColor(ir4d,cv.COLOR_GRAY2RGB)
    return ir_frame, ir4d

## main loop
frame_idx = 0
done = False
while not done:

    key = cv.waitKey(1)
    key = cv.waitKey(1) & 255

    if (key&255) == 27:
        print ("\t key detected!")
        done = True

    elif chr(key) =='s': #screen capture
        print(key)
        done = True
    # Infrared method

    _, ir4d = get_ir()
    cv.imshow("IR image", ir4d)
    frame_idx+=1
# end while

cv.imwrite('Input_image.png', ir4d)
print ("\tSaving frame")

## Release resources and terminate
cv.destroyAllWindows()
depth_stream.stop()
openni2.unload()
print ("Terminated!") 
# --------- Finish the stream -------------------------------------------------------

# ------------------- Preprocessing input image -------------------------------------
#  s = save image
#  any other = try again!
img_prepro()