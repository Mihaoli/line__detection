#calibrate.py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider, Button

cam = cv.VideoCapture('test1.mp4')
cam.set(3, 1920)
cam.set(4, 1080)
# reading the input using the camera
result, image = cam.read()
# If image will detected without any error, 
# show result
if result:
    # showing result, it take frame name and image 
    #cv.imshow("test_road", image)
    # saving image in local storage
    cv.imwrite("test_road.jpg", image)
    cv.waitKey(0)
    #cv.destroyWindow("test_road")
# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please! try again")
#================= Plot draw

bl = 200 #bottom left point
tl = 400 #top left point
tr = 600 #top right point
br = 1080 #bottom right point

blo = None #bottom left point
tlo = None #top left point
tro = None #top right point
bro = None #bottom right point

# Create the figure and the line that we will manipulate

img = "test_road.jpg"
im = plt.imread(img)
fig, ax = plt.subplots()
fig.tight_layout()
im = ax.imshow(im, extent=[0, 1920, 0, 1080])
pts = np.array([[200,0], [400,405], [600,405], [1080,0]]) #
p = Polygon(pts, closed=True, alpha=0.4, fc='green', ec="black")
ax = plt.gca()
ax.add_patch(p)
# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.4)
# Make a point slider to edit coordinates.
bleft = fig.add_axes([0.20, 0.27, 0.63, 0.05])
bleft_slider = Slider(
    ax=bleft,
    label='bottom-left',
    valmin=1,
    valmax=1920,
    valinit=200,
)
tleft = fig.add_axes([0.20, 0.20, 0.63, 0.05])
tleft_slider = Slider(
    ax=tleft,
    label="top-left",
    valmin=1,
    valmax=1920,
    valinit=342
)
tright = fig.add_axes([0.20, 0.13, 0.63, 0.05])
tright_slider = Slider(
    ax=tright,
    label="top-right",
    valmin=1,
    valmax=1920,
    valinit=600,
)
bright = plt.axes([0.20, 0.06, 0.63, 0.05])
bright_slider = Slider(
    ax=bright,
    label='bottom-right',
    valmin=1,
    valmax=1920,
    valinit=1080,
    closedmax=True
)
# The function to be called anytime a slider's value changes

def update(val):
    # Update the coordinates of the vertices
    pts[0][0] = bleft_slider.val
    pts[1][0] = tleft_slider.val
    pts[2][0] = tright_slider.val
    pts[3][0] = bright_slider.val
    # Update the Polygon object with the new coordinates
    p.set_xy(pts)
    # Update the limits of the x-axis
    ax.set_xlim(bleft_slider.val, bright_slider.val)
    fig.canvas.draw_idle()

# The function saves the polygon coordiantes after clicking save button
def save(val):
  x2=np.array([pts[0][0],pts[1][0],pts[2][0],pts[3][0]])
  y2=np.array([0,340,340,0])
  ax.plot(x2,y2,color="green", marker="o")
  global blo
  global tlo
  global tro
  global bro
  blo=pts[0][0]
  tlo=pts[1][0] 
  tro=pts[2][0]
  bro=pts[3][0]
  #print(blo,tlo,tro,bro)
  plt.close()
#save button
axes = plt.axes([0.84, 0.000001, 0.1, 0.075])
bsave = Button(axes, 'Save',color='green')
bsave.on_clicked(save)
# register the update function with each slider
bleft_slider.on_changed(update)
bright_slider.on_changed(update)
tleft_slider.on_changed(update)
tright_slider.on_changed(update)
plt.show()