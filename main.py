print("Minecraft auto fisher\n")

from multiprocessing import Process, Queue
import cv2
import numpy as np
import time
import cv2
import mss
import numpy
import imutils
from skimage import data, color, io, exposure
from skimage.feature import match_template
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('TKagg')
from IPython.display import clear_output
from PIL import Image
import mss
import pyautogui
import mss.tools

pyautogui.FAILSAFE = False


# Display the manually selected template of the object to be tracked
# template = mpimg.imread("template.jpeg")

# resize the frame and convert it to grayscale (while still retaining 3 channels)
template = color.rgb2gray(io.imread('template.jpeg'))
template = exposure.rescale_intensity(template)

template_height = template.shape[0]
template_width = template.shape[1]
print(f'template_shape:')
print(f'height:\t{template_height}')
print(f'width:\t{template_width}')

plt.title("Manually selected flöjt template")
plt.imshow(template)

plt.show()


def grab(queue):
    # type: (Queue) -> None

    rect = {"top": 350, "left": 1080 + 200, "width": 700, "height": 700}

    with mss.mss() as sct:
        while True:

            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(rect))
            queue.put(img)
            # time.sleep(0.2)
            time.sleep(0.25)

    # Tell the other worker to stop
    queue.put(None)


def process(queue):
    # type: (Queue) -> None

    frame_count = 0
    # output = "screenshots/file_{}.png"
    # to_png = mss.tools.to_png

    print(f'frame_count: {frame_count}')

    flöjt_positions = []

    while True:
        last_time = time.time()
        img = queue.get()

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ksize
        ksize = (11, 11)
  
        # Using cv2.blur() method 
        image = cv2.blur(gray_image, ksize) 
        # image = gray_image

        brightness = 125
        contrast = 250
        img = np.int16(image)
        img = img * (contrast/127+1) - contrast + brightness
        img = np.clip(img, 0, 255)
        image = np.uint8(img)

        # image = img
        result = match_template(image, template)
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]
        
        # Store the ball position
        flöjt_positions.append((x + (template_width / 2), y + (template_height / 2)))


        #print(flöjt_positions[-1])

        # Do we have napp på kroken?
        # If the current flöjt position differs grately from the last position
        # then we probably have napp
        if (len(flöjt_positions) > 20):

            current_position = flöjt_positions[-1]
            last_position = flöjt_positions[-2]
           
            # If some new position is way out there, then we take the previous value and move on
            if current_position[1] - last_position[1] > 30:
                continue
                #current_position[1] = last_position[1]

            x_temp = current_position[0] - last_position[0]
            y_temp = current_position[1] - last_position[1]

            x_temp_2 = flöjt_positions[-2][0] - flöjt_positions[-2][0]

            y_temp_2 = flöjt_positions[-2][1] - flöjt_positions[-3][1]
            y_temp_3 = flöjt_positions[-3][1] - flöjt_positions[-4][1]
            
            # print(f'x_temp: {x_temp}')
            # print(f'y_temp: {y_temp} y_temp_2: {y_temp_2} y_temp_3: {y_temp_3}')

            if (abs(y_temp) > 25):
                if (abs(y_temp_2) > 20 or abs(y_temp_3) > 20):
                    print(f'WE HAVE NAPP!')
                    pyautogui.rightClick(0, 0)
                    time.sleep(2)
                    pyautogui.rightClick(0, 0)

                    flöjt_positions = []

            if (len(flöjt_positions) > 2):
                image = cv2.rectangle(image, 
                    (int(flöjt_positions[-1][0] - template_width / 2), 
                    (int(flöjt_positions[-1][1] - template_height / 2))
                ),  (int(flöjt_positions[-1][0] + template_width / 2), 
                    (int(flöjt_positions[-1][1] + template_height / 2))), 
                    (255, 0, 255),
                5)

        cv2.imshow('minecraft_ai_fisher', image)


        # print("time to process the frame: {}".format(1 / (time.time() - last_time)))


        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        # plt.show(img)
        if img is None:
            break
            

        # to_png(img.rgb, img.size, output=output.format(number))
        frame_count += 1


if __name__ == "__main__":
    # The screenshots queue
    queue = Queue()  # type: Queue

    # 2 processes: one for grabing and one for saving PNG files
    Process(target=grab, args=(queue,)).start()
    Process(target=process, args=(queue,)).start()