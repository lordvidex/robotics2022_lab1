import cv2
import numpy as np
import matplotlib.pyplot as plt
import median_gauss_blur as blur

noise = 20 # 0-100
img = cv2.imread('assets/paradise.png')
def on_slide():
    print('hello')
    


window_name = 'Original with Noise'

# frequency filters
def mean_filter(img, kernel_size):
    return cv2.blur(img, (kernel_size, kernel_size))

def mag_spectrum(img):    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))

if __name__ == '__main__':
    while True:
        salted = blur.add_salt_pepper_noise(img, float(noise)/100.0)
        cv2.imshow(window_name, salted)
        cv2.createTrackbar('Noise Slider', window_name, noise, 100, on_slide)
        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('+'):
            noise = min(100, noise + 10)
        elif k == ord('-'):
            noise = max(0, noise - 10)
        plt.imshow(mag_spectrum(cv2.imread('assets/paradise.png', 0)), cmap='gray')
        plt.show()
        cv2.waitKey(1)
    cv2.destroyAllWindows()