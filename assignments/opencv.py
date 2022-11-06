import cv2
import os
import matplotlib.pyplot as plt

# read the image
imgPath = os.path.join(os.path.dirname(__file__), '..', 'assets', 'paradise.png')
original = cv2.imread(imgPath)
# declare the size of the img
size = original.shape
print(size)

columns = 3 # columns on the x_axis
rows = 2 # knife cuts on the y_axis
dividers = (int(size[1]/columns), int(size[0]/rows)) # size of each cut (dx,dy)

for i in range(0, rows):
    for j in range(0, columns):
        dx = dividers[0]
        dy = dividers[1]
        hx0 = dx * j
        hx1 = dx * (j + 1)
        hy0 = dy * i
        hy1 = dy * (i + 1)
        print(hx0, hx1, hy0, hy1)
        img = original[hy0:hy1, hx0:hx1]
        cv2.imwrite(f"assets/paradise_{i}_{j}.png", img)
        # resize the image to be as large as the original image
        # img = cv2.resize(img, (size[1], size[0]))
        # save the image to a new file
        # cv2.imwrite(f"assets/paradise_{i}_{j}_resized.png", img)


fig, ax = plt.subplots(nrows=rows, ncols=columns)
for i in range(0, rows):
    for j in range(0, columns):
        img = plt.imread(f"assets/paradise_{i}_{j}.png")
        ax[i,j].imshow(img)
        ax[i,j].axis('off')
plt.tight_layout()
plt.show()
