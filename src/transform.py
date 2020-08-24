import numpy as np
from skimage import io
import os.path as osp
import math

def load_image(file_name):
    """
    Load image from disk
    :param file_name:
    :return: image: numpy.ndarray
    """
    if not osp.exists(file_name):
        print('{} not exist'.format(file_name))
        return
    image = io.imread(file_name)
    return np.array(image)

def save_image(image, file_name):
    """
    Save image to disk
    :param image: numpy.ndarray
    :param file_name:
    :return:
    """
    io.imsave(file_name,image)


def cs4243_resize(image, new_width, new_height):
    """
    5 points
    Implement the algorithm of nearest neighbor interpolation for image resize,
    Please round down the value to its nearest interger, 
    and take care of the order of image dimension.
    :param image: ndarray
    :param new_width: int
    :param new_height: int
    :return: new_image: numpy.ndarray
    """
    image_len_two = False
    new_image = np.zeros((new_height, new_width, 3), dtype = 'uint8')
    if len(image.shape) == 2:
        image_len_two = True
        new_image = np.zeros((new_height, new_width), dtype = 'uint8')
        
    ###Your code here####
    old_height = image.shape[0]
    old_width = image.shape[1]

    width_ratio = old_width / new_width
    height_ratio = old_height / new_height
    
    for x in range(0, new_height):  
        for y in range(0, new_width):
            src_x = int(x * height_ratio)
            src_y = int(y * width_ratio)
            if (image_len_two):
                new_image[x][y]= image[src_x][src_y]
            else: 
                new_image[x][y]= image[src_x][src_y][0:3]
    ###
    return new_image

def cs4243_rgb2grey(image):
    """
    5 points
    Implement the rgb2grey function, use the
    weights for different channel: (R,G,B)=(0.299, 0.587, 0.114)
    Please scale the value to [0,1] by dividing 255
    :param image: numpy.ndarray
    :return: grey_image: numpy.ndarray
    """
    RED = 0.299
    GREEN = 0.587
    BLUE = 0.114
    
    if len(image.shape) != 3:
        print('Image should have 3 channels')
        return
    
    ###Your code here####
    height = image.shape[0]
    width = image.shape[1]
    new_image = np.zeros((height, width))
    
    for x in range(0, height):
        for y in range(0, width):
            new_image[x][y] = (image[x][y][0]*RED)/255 + (image[x][y][1]*GREEN)/255 + (image[x][y][2]*BLUE)/255
             
    ###
    return new_image

def cs4243_histnorm(image, grey_level=256):
    """
    5 points 
    Stretch the intensity value to [0, 255]
    :param image : ndarray
    :param grey_level
    :return res_image: hist-normed image
    Tips: use linear normalization here https://en.wikipedia.org/wiki/Normalization_(image_processing)
    """
    res_image = image.copy()

    ##your code here ###
    low = image.min()
    high = image.max()
    res_image = ((res_image - low) / (high - low) * (grey_level-1))
    ####
    
    return res_image



def cs4243_histequ(image, grey_level=256):
    """
    10 points
    Apply histogram equalization to enhance the image.
    the cumulative histogram will aso be returned and used in the subsequent histogram normalization function.
    :param image: numpy.ndarray(float64)
    :return: ori_hist: histogram of original image
    :return: cum_hist: cumulated hist of original image, pls normalize it with image size.
    :return: res_image: image after being applied histogram equalization.
    :return: uni_hist: histogram of the enhanced image.
    Tips: use numpy buildin funcs to ease your work on image statistics
    """
    ###your code here####
    ori_hist = np.bincount(image.flatten(), minlength=grey_level)
    cum_hist = np.zeros(len(ori_hist), dtype='float64')
    cumulated_value = 0
    image_size = image.shape[0] * image.shape[1]
    for i in range(len(cum_hist)):  
        cumulated_value = cumulated_value + ori_hist[i]      
        cum_hist[i] = cumulated_value / image_size 
    uniform_hist = np.zeros(len(ori_hist), dtype='float64')
    for j in range(len(uniform_hist)):
        uniform_hist[j] = math.floor(cum_hist[j] * (grey_level - 1))
    ###

    # Set the intensity of the pixel in the raw image to its corresponding new intensity 
    height, width = image.shape
    res_image = np.zeros(image.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            res_image[i,j] = uniform_hist[image[i,j]]
    
    uni_hist = np.bincount(res_image.flatten(), minlength=grey_level)
    return ori_hist, cum_hist, res_image, uni_hist

def cs4243_histmatch(ori_image, refer_image):
    """
    10 points
    Map value according to the difference between cumulative histogram.
    Note that the cum_hists of the two images can be very different. It is possible
    that a given value cum_hist[i] != cum_hist[j] for all j in [0,255]. In this case, please
    map to the closest value instead. if there are multiple intensities meet the requirement,
    choose the smallest one.
    :param ori_image #image to be processed
    :param refer_image #image of target gray histogram 
    :return: ori_hist: histogram of original image
    :return: ref_hist: histogram of reference image
    :return: res_image: image after being applied histogram matching.
    :return: res_hist: histogram of the enhanced image.
    Tips: use cs4243_histequ to help you
    """
    ###your code here####
    # (1) Form histogram of original image
    ori_hist = np.bincount(ori_image.flatten(), minlength=256)
    cum_hist = np.zeros(len(ori_hist), dtype='float64')
    cumulated_value=0
    image_size=ori_image.shape[0]*ori_image.shape[1]
    
    for i in range(len(cum_hist)):
        cumulated_value=cumulated_value+ori_hist[i]
        cum_hist[i]=cumulated_value/image_size
    # (cum_hist)
    # Index -> intensity
    # Value -> Cumulative freq
    
    # (2) Form histogram of template image
    refer_hist = np.bincount(refer_image.flatten(), minlength=256)
    refer_cum_hist = np.zeros(len(refer_hist), dtype='float64')
    ref_cumulated_value=0
    refer_image_size=refer_image.shape[0]*refer_image.shape[1]
    
    print(refer_hist[255])
    
    for i in range(len(refer_cum_hist)):
        ref_cumulated_value=ref_cumulated_value+refer_hist[i]
        refer_cum_hist[i]=ref_cumulated_value/refer_image_size
    # (refer_cum_hist)
    # Index -> intensity
    # Value -> Cumulative freq
    
    # (3) 
    match_hist=np.zeros(len(cum_hist), dtype='float64')
    for intensity, frequency in enumerate(cum_hist):
        # note that index of refer_cum_hist=intensity val
        # we want to find the intensity/index that has the closest matching cum freq with 
        # cum_hist
        # https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        # In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned.
        new_intensity = (np.abs(refer_cum_hist - frequency)).argmin()
        match_hist[intensity]=new_intensity
    # (match_hist)
    # index -> Old intensity
    # value -> New intensity
    
    ####
    
    # Set the intensity of the pixel in the raw image to its corresponding new intensity      
    height, width = ori_image.shape
    res_image = np.zeros(ori_image.shape, dtype='uint8')  # Note the type of elements
    
    for i in range(height):
        for j in range(width):
            # ori_image[i,j] contains the Old intensity value
            res_image[i,j] = match_hist[ori_image[i,j]]

    
    res_hist = np.bincount(res_image.flatten(), minlength=256)
    ori_hist = np.bincount(ori_image.flatten(), minlength=256)
    ref_hist = np.bincount(refer_image.flatten(), minlength=256)
    
    return ori_hist, ref_hist, res_image, res_hist

# def cs4243_histmatch(ori_image, refer_image):
#     """
#     10 points
#     Make ori_image have the similar intensity distribution as refer_image
#     :param ori_image #image to be processed
#     :param refer_image #image of target gray histogram 
#     :return: ori_hist: histogram of original image
#     :return: ref_hist: histogram of reference image
#     :return: res_image: image after being applied histogram normalization.
#     :return: res_hist: histogram of the enhanced image.
#     Tips: use cs4243_histequ to help you
#     """
    
#     ##your code here ###
    
#     ##
#     # Set the intensity of the pixel in the raw image to its corresponding new intensity      
#     height, width = ori_image.shape
#     res_image = np.zeros(ori_image.shape, dtype='uint8')  # Note the type of elements
#     for i in range(height):
#         for j in range(width):
#             res_image[i,j] = map_value[ori_image[i,j]]
    
#     res_hist = np.bincount(res_image.flatten(), minlength=256)
    
#     return ori_hist, ref_hist, res_image, res_hist

def cs4243_rotate180(kernel):
    """
    Rotate the matrix by 180. 
    Can utilize build-in Funcs in numpy to ease your work
    :param kernel:
    :return:
    """
    kernel = np.flip(np.flip(kernel, 0),1)
    return kernel

def cs4243_gaussian_kernel(ksize, sigma):
    """
    5 points
    Implement the simplified Gaussian kernel below:
    k(x,y)=exp(((x-x_mean)^2+(y-y_mean)^2)/(-2sigma^2))
    Make Gaussian kernel be central symmentry by moving the 
    origin point of the coordinate system from the top-left
    to the center. Please round down the mean value. In this assignment,
    we define the center point (cp) of even-size kernel to be the same as that of the nearest
    (larger) odd size kernel, e.g., cp(4) to be same with cp(5).
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    """
    kernel = np.zeros((ksize, ksize))
    ###Your code here####

    ###

    return kernel / kernel.sum()

def cs4243_filter(image, kernel):
    """
    10 points
    Implement the convolution operation in a naive 4 nested for-loops,
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return:
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))
    
    ###Your code here####
    
    # Pad the image with 0s 
    height_padding_length = int(Hk / 2)
    width_padding_length = int(Wk / 2)
    padded_image = pad_zeros(image, height_padding_length, width_padding_length)
    
    # Traverse through every pixel in the padded_image
    for i in range(0 + height_padding_length, Hi + height_padding_length):  
        for j in range(0 + width_padding_length, Wi + width_padding_length):
            temp = 0 
            for u in range(-height_padding_length, height_padding_length + 1):
                for v in range(-width_padding_length, width_padding_length + 1):
                    # Convolution formula is Pi-u, j-v
                    temp += padded_image[i - u][j - v] * kernel[u + height_padding_length][v + width_padding_length]
            # Remember that the dimensions of filtered_image is smaller than padded_image hence need to 
            # adjust index to account for padded 0s positions in padded_image
            filtered_image[i - height_padding_length][j- width_padding_length] = temp
    ###

    return filtered_image

def pad_zeros(image, pad_height, pad_width):
    """
    Pad the image with zero pixels, e.g., given matrix [[1]] with pad_height=1 and pad_width=2, obtains:
    [[0 0 0 0 0]
    [0 0 1 0 0]
    [0 0 0 0 0]]
    :param image: numpy.ndarray
    :param pad_height: int
    :param pad_width: int
    :return padded_image: numpy.ndarray
    """
    height, width = image.shape
    new_height, new_width = height+pad_height*2, width+pad_width*2
    padded_image = np.zeros((new_height, new_width))
    padded_image[pad_height:new_height-pad_height, pad_width:new_width-pad_width] = image
    return padded_image

def cs4243_filter_fast(image, kernel):
    """
    10 points
    Implement a fast version of filtering algorithm.
    take advantage of matrix operation in python to replace the 
    inner 2-nested for loops in filter function.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    Tips: You may find the functions pad_zeros() and cs4243_rotate180() useful
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####

    ###

    return filtered_image

def cs4243_filter_faster(image, kernel):
    """
    10 points
    Implement a faster version of filtering algorithm.
    Pre-extract all the regions of kernel size,
    and obtain a matrix of shape (Hi*Wi, Hk*Wk),also reshape the flipped
    kernel to be of shape (Hk*Wk, 1), then do matrix multiplication, and rehshape back
    to get the final output image.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    Tips: You may find the functions pad_zeros() and cs4243_rotate180() useful
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####
    
    ###

    return filtered_image

def cs4243_downsample(image, ratio):
    """
    Downsample the image to its 1/(ratio^2),which means downsample the width to 1/ratio, and the height 1/ratio.
    for example:
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = downsample(A, 2)
    B=[[1, 3], [7, 9]]
    :param image:numpy.ndarray
    :param ratio:int
    :return:
    """
    width, height = image.shape[1], image.shape[0]
    return image[0:height:ratio, 0:width:ratio]

def cs4243_upsample(image, ratio):
    """
    upsample the image to its 2^ratio, 
    :param image: image to be upsampled
    :param kernel: use same kernel to get approximate value for additional pixels
    :param ratio: which means upsample the width to ratio*width, and height to ratio*height
    :return res_image: upsampled image
    """
    width, height = image.shape[1], image.shape[0]
    new_width, new_height = width*ratio, height*ratio
    res_image = np.zeros((new_height, new_width))
    res_image[0:new_height:ratio, 0:new_width:ratio] = image
    return res_image


def cs4243_gauss_pyramid(image, n=4):
    """
    10 points
    build a Gaussian Pyramid of level n
    :param image: original grey scaled image
    :param n: level of pyramid
    :return pyramid: list, with list[0] corresponding to original image.
	:e.g., img0->blur&downsample->img1->blur&downsample->img2	
    Tips: you may need to call cs4243_gaussian_kernel() and cs4243_filter_faster()
	The kernel for blur is given, do not change it.
    """
    kernel = cs4243_gaussian_kernel(7, 1)
    pyramid = []
    ## your code here####
    
    ##
    return pyramid

def cs4243_lap_pyramid(gauss_pyramid):
    """
    10 points
    build a Laplacian Pyramid from the corresponding Gaussian Pyramid
    :param gauss_pyramid: list, results of cs4243_gauss_pyramid
    :return lap_pyramid: list, with list[0] corresponding to image at level n-1 in Gaussian Pyramid.
	Tips: The kernel for blurring during upsampling is given, you need to scale its value following the standard pipeline in laplacian pyramid.
    """
    #use same Gaussian kernel 

    kernel = cs4243_gaussian_kernel(7, 1)
    n = len(gauss_pyramid)
    lap_pyramid = [gauss_pyramid[n-1]] # the top layer is same as Gaussian Pyramid
    ## your code here####
    
    ##
    
    return lap_pyramid
    
def cs4243_Lap_blend(A, B, mask):
    """
    10 points
    blend image with Laplacian pyramid
    :param A: image on the left
    :param B: image on the right
    :param mask: mask [0, 1]
    :return blended_image: same size as input image
    Tips: use cs4243_gauss_pyramid() & cs4243_lap_pyramid() to help you
    """
    kernel = cs4243_gaussian_kernel(7, 1)
    blended_image = None
    ## your code here####
    
    ##
    
    return blended_image