'''
Created on 11-Oct-2018

@author: Pramey
'''
import numpy as np 
from scipy.signal import convolve2d as conv2
import cv2
import scipy
from scipy import signal
from scipy import ndimage
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.measure import compare_ssim as  ssim

blurred_img1 = cv2.imread('blurred_img1.jpg')
blurred_img2 = cv2.imread('blurred_img2.jpg')
blurred_img3 = cv2.imread('blurred_img3.jpg')
blurred_img4 = cv2.imread('blurred_img4.jpg')

original_img = cv2.imread('original_img.jpg')

kernel1 = cv2.imread('kernel1.png',0)
kernel2 = cv2.imread('kernel2.png',0)
kernel3 = cv2.imread('kernel3.png',0)
kernel4 = cv2.imread('kernel4.png',0)

cropped_kernel1 = kernel1[25:55, 20:50]
cropped_kernel2 = kernel2[15:45, 20:50]
cropped_kernel3 = kernel3[15:45, 20:50]
cropped_kernel4 = kernel4[20:50, 25:55]

padded_array1 = np.zeros((1600, 1600, 3))
padded_array1[0:800, 0:800, 0:3] = blurred_img1
padded_array2 = np.zeros((1600, 1600, 3))
padded_array2[0:800, 0:800, 0:3] = blurred_img2
padded_array3 = np.zeros((1600, 1600, 3))
padded_array3[0:800, 0:800, 0:3] = blurred_img3
padded_array4 = np.zeros((1600, 1600, 3))
padded_array4[0:800, 0:800, 0:3] = blurred_img4

r1, g1, b1 = cv2.split(padded_array1)
r2, g2, b2 = cv2.split(padded_array2)
r3, g3, b3 = cv2.split(padded_array3)
r4, g4, b4 = cv2.split(padded_array4)

def psnr(img1, img2):
    img1 = cv2.normalize(img1, None, alpha = 0 , beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2 = cv2.normalize(img2, None, alpha = 0 , beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img1Y, img1Cr, img1Cb = cv2.split(cv2.cvtColor(img1.astype(np.float32), cv2.COLOR_BGR2YCrCb))
    img2Y, img2Cr, img2Cb = cv2.split(cv2.cvtColor(img2.astype(np.float32), cv2.COLOR_BGR2YCrCb))
    mse = np.mean((img1Y-img2Y)**2)
    
    max_val = np.max(img1Y)
    if mse == 0:
        return 100
    else: 
        return 10*np.log10((max_val)**2/mse)

def ssim_i(img1, img2):
    img1 = cv2.normalize(img1, None, alpha = 0 , beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2 = cv2.normalize(img2, None, alpha = 0 , beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    s = ssim(img1, img2, multichannel = True)
    return np.mean(s)

def filter_Butterworth_LP(d,n,b):

    D = np.zeros((1600,1600))
    H = np.zeros((1600,1600))
    x = np.linspace(0, 1600, 1600)
    y = np.linspace(0, 1600, 1600)

    D = np.sqrt(x**2 + y**2)
    
    if b == 1:
        H = 1/(1 + (D/d)**(2*n))
    else:
        H = np.exp(-(D/np.sqrt(2)*d)**2)

    return H


def filter(r, g, b, kernel, type, init_val):
    r_fft = np.fft.fft2(r)
    g_fft = np.fft.fft2(g)
    b_fft = np.fft.fft2(b)
    
    kernel_fft = np.fft.fft2(kernel, (1600, 1600))
    p = 0
    
    if type == 0:
        p = 1
    elif type == 1:
        #K = np.median((np.abs(kernel_fft))**2)
        c = init_val
        p = (np.abs(kernel_fft))**2/((np.abs(kernel_fft))**2 + c)
    elif type == 2:
        p = filter_Butterworth_LP(150, 10, 1)
    elif type == 3:
        gamma =  init_val
        Pxy = np.array([[0,-1,0], [-1, 4,-1],[0,-1,0]])
        Puv_fft = np.fft.fft2(Pxy,(1600,1600))
        Huv_conj = np.conjugate(kernel_fft)
        p = (kernel_fft*Huv_conj)/((np.abs(kernel_fft))**2 + gamma*(np.abs(Puv_fft))**2) 
            
    deblurred_r = 255.0*np.fft.ifft2((r_fft/kernel_fft)*p).real
    deblurred_g = 255.0*np.fft.ifft2((g_fft/kernel_fft)*p).real
    deblurred_b = 255.0*np.fft.ifft2((b_fft/kernel_fft)*p).real
    
    clipped_deblurred_r = 2*deblurred_r.real/np.max(np.abs(deblurred_r.real))
    clipped_deblurred_g = 2*deblurred_g.real/np.max(np.abs(deblurred_g.real))
    clipped_deblurred_b = 2*deblurred_b.real/np.max(np.abs(deblurred_b.real))
    
    deblurred_img = np.clip(cv2.merge((clipped_deblurred_r, clipped_deblurred_g, clipped_deblurred_b)),0, 255)
    cropped_deblurred_img = deblurred_img[0:800,0:800, 0:3]
    print("SSIM:", np.mean(ssim_i(original_img, cropped_deblurred_img)))
    print("PSNR:" , psnr(original_img, cropped_deblurred_img))
    return cropped_deblurred_img
    
  
def interactive_value():
    
    D_min = 100
    D_max = 10000
    D_init = 1500
    fig = plt.figure()
    plt.axis("off")
    recovered_image = filter(b4, g4, r4, cropped_kernel4, 3, D_init )
    recovered_image_plot = plt.imshow(cv2.cvtColor(recovered_image.astype(np.float32), cv2.COLOR_BGR2RGB))
    
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
    value_slider = Slider(slider_ax, 'value', D_min, D_max, valinit=D_init)


    def update(value):
        recovered_image = filter(b4, g4, r4, cropped_kernel4, 3, value )
        recovered_image_plot.set_data(recovered_image)
        fig.canvas.draw_idle()

    value_slider.on_changed(update)

    plt.show()

interactive_value()  