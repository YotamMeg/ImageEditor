#################################################################
# FILE : cartoonify.py
# WRITER : yotam megged , yotam267 , 319134912
# EXERCISE : intro2cs ex6 2022C
# DESCRIPTION: A program that makes a cartooned copy of a given picture
# WEB PAGES I USED: www.how2staySane.com
#################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
from ex6_helper import *
from typing import Optional
import math
import sys


def get_image_scales(image):
    """
    calculates the height (number of rows) and width (number of columns) of an image
    :param image: an image
    :return: a tuple containing the height and width of an image
    """
    height = len(image)
    width = len(image[0])
    return height, width


def separate_channels(image: ColoredImage) -> List[List[List[int]]]:
    """
        find a better way using a third function
        :param image:
        :return:
        """
    height, width = get_image_scales(image)
    color_length = len(image[0][0])
    final_list = []
    for i in range(color_length):
        row_list = []
        for j in range(height):
            color_list = []
            for k in range(width):
                color_list.append(image[j][k][i])
            row_list.append(color_list)
        final_list.append(row_list)
    return final_list


def combine_channels(channels: List[List[List[int]]]) -> ColoredImage:
    """
    receives a list of channels and combines them into a colored image
    :param channels: a list of channels
    :return: a colored image
    """
    row_len = len(channels[0])
    col_len = len(channels[0][0])
    colored_image = []
    for i in range(row_len):
        colored_row = []
        for j in range(col_len):
            colored_col = []
            for k in range(len(channels)):
                colored_col.append(channels[k][i][j])
            colored_row.append(colored_col)
        colored_image.append(colored_row)
    return colored_image


def grey_scale_pixel(rgb):
    """
    gets 3 color parameters and returns the updated grey scale color
    :param rgb: the color values - a list with the parameters of red, green and blue values
    :return:a grey scale color
    """
    single_value = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
    return round(single_value)


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    """
        turns a colored image to a grey scale image
        :param colored_image: an image with 3 channels
        :return: a grey scale image
        """
    grey_scale_image = []
    for row in colored_image:
        row_list = []
        for col in row:
            new_pixel = grey_scale_pixel(col)
            row_list.append(new_pixel)
        grey_scale_image.append(row_list)
    return grey_scale_image


def blur_kernel(size: int) -> Kernel:
    """
        creates a kernel of size # size, with each cell containing the same value-therefore
        the kernel can be used for blurring an image
        :param size: a number
        :return: a kernel with size rows and columns, and the same value in each cell
        """
    cell = 1 / size ** 2
    updated_kernel = []
    for i in range(size):
        col_list = []
        for j in range(size):
            col_list.append(cell)
        updated_kernel.append(col_list)
    return updated_kernel


def update_cell_kernel(patch, kernel):
    """
    gets a patch - a block of an image with same number of rows and columns and a kernel
    with the same size. multiplies every cell of the patch with the matching cell from the
    kernel and returns the sum
    :param patch: a list with n lists, each containing n values
    :param kernel: a kernel with the same size as the patch
    :return: the sum of the multiplication of the kernel's and the patch's cells
    """
    updated_patch = 0
    for i in range(len(patch)): # for row
        updated_row = 0
        for j in range(len(patch[i])): # for col
            # print(patch[i][j])
            # print(kernel[i][j])
            updated_value = patch[i][j] * kernel[i][j] # updates every value of the cell (each value of color)
            updated_row += updated_value
        updated_patch += updated_row
    if updated_patch > 255:
        return 255
    elif updated_patch < 0:
        return 0
    return round(updated_patch)


def create_patch(image, location, kernel): # location = [row,col]
    """
    creates a patch from the image with n rows and columns, with n being the length of the kernel.
    adds the looked at pixel to the patch where the index of the patch falls out of the image
    :param image: an image
    :param location: a list of with two elements, the first being the rows index of the image and the second
    being the columns index
    :param kernel: a kernel
    :return: a patch from the image, in its center is the pixel matching the given location
    """
    patch = []
    for i in range(location[0] - len(kernel)//2, location[0] + len(kernel)//2 + 1):
        patch_row = []
        for j in range(location[1] - len(kernel)//2, location[1] + len(kernel)//2 + 1):
            if len(image) > i >= 0 and len(image[i]) > j >= 0:
                # if the current cell of the patch is inside the image
                patch_row.append(image[i][j])
            else:
                # else we add to the patch the pixel matching the given location
                patch_row.append(image[location[0]][location[1]])
        patch.append(patch_row)
    return patch


def apply_kernel(image: SingleChannelImage, kernel: Kernel) -> SingleChannelImage:
    """
    gets an image, and a blur kernel, and runs every pixel of the image through the kernel to make a blurred image
    :param image: an image
    :param kernel: a blur kernel
    :return: a blurred image
    """
    blurred_image = []
    for i in range(len(image)):
        blurred_row = []
        for j in range(len(image[i])):
            location = [i, j]
            new_cell = 0
            patch = create_patch(image, location, kernel)
            new_cell += update_cell_kernel(patch, kernel)
            blurred_row.append(new_cell)
        blurred_image.append(blurred_row)
    return blurred_image


def falls_between_pixels(image, y, x):
    """
    checks if indexes of a pixel in the resized image falls exactly between two pixels
    in the original image, and calculates the bilinear interpolation in this case
    :param image: the image
    :param y: the coordinates of the rows of the image
    :param x: the coordinates of the columns of the image
    :return:
    """
    if int(x) == x:
        # if the pixel falls on one of the columns of the image
        x = int(x)
        a = math.floor(y)
        b = math.ceil(y)
        pixel_value = image[a][x] * (1-(y % 1)) + image[b][x] * (y % 1)
    if int(y) == y:
        # if it falls on one of the rows of the image
        y = int(y)
        a = math.floor(x)
        b = math.ceil(x)
        pixel_value = image[y][a] * (1 - (x % 1)) + image[y][b] * (x % 1)
    return pixel_value


def bilinear_interpolation(image: SingleChannelImage, y: float, x: float) -> int:
    """
    gets an image and location of a pixel that "falls" somewhere within the image, and returns the
    pixel's value according to the surrounding pixels
    :param image: an image
    :param y: the location in the row of the image
    :param x: the location in the column of the image
    :return: the value of the pixel
    """
    delta_x = x % 1
    delta_y = y % 1
    if int(x) == x or int(y) == y:
        sum = round(falls_between_pixels(image, y, x))
    else:
        a = math.floor(y), math.floor(x)
        b = math.ceil(y), math.floor(x)
        c = math.floor(y), math.ceil(x)
        d = math.ceil(y), math.ceil(x)

        s = image[a[0]][a[1]]
        n = image[b[0]][b[1]]
        z = image[c[0]][c[1]]
        k = image[d[0]][d[1]]
        sum = round(s * (1-delta_x) * (1-delta_y) + n * (1-delta_x) * delta_y + z * delta_x * (1-delta_y) + k * delta_x * delta_y)
        # using known formula
    return sum


def is_location_corner(height, width, location):
    """
    gets the height, width of an image and a location - [height index, width index], and returns true
    if the location is a corner
    :param height: number of rows of an image
    :param width: number of columns of an image
    :param location: a location of a specific pixel
    :return: true if the location is of a corner, false otherwise
    """
    condition_1 = location[0] == 0 and location[1] == 0
    condition_2 = location[0] == height - 1 and location[1] == width - 1
    condition_3 = location[0] == 0 and location[1] == width - 1
    condition_4 = location[0] == height - 1 and location[1] == 0
    if condition_1 or condition_2 or condition_3 or condition_4:
        return True
    return False


def resize(image: SingleChannelImage, new_height: int, new_width: int) -> SingleChannelImage:
    """
    receives an image, a height parameter and a width parameter and creates a new image with the given sizes (height
    and width), that matches the original image using the bilinear interpolation function
    :param image:
    :param new_height: number of rows of the new image
    :param new_width: number of columns of the new image
    :return: the new image
    """
    x_height, x_width = get_image_scales(image)
    height_ratio = (x_height - 1) / (new_height - 1)  # formula for pixel location calculation
    width_ratio = (x_width - 1) / (new_width - 1)
    new_image = []
    for i in range(new_height):
        new_row = []
        for j in range(new_width):
            location = [i, j]
            if is_location_corner(new_height, new_width, location):
                new_row.append(image[int(i * height_ratio)][int(j * width_ratio)])
                # add the matching corner of the original image
            else:
                bilinear_x = width_ratio * j
                bilinear_y = height_ratio * i
                # location of where the pixel of the new image "falls" in the original image
                new_row.append(bilinear_interpolation(image, bilinear_y, bilinear_x))
        new_image.append(new_row)
    return new_image


def image_in_size(image, max_size):
    """
    receives an image and a max size, and checks if both the number of rows and columns is equal
    or smaller than the given size
    :param image: an image
    :param max_size: a number
    :return: true if the image is equal or shorter than the given size, false otherwise
    """
    height, width = get_image_scales(image)
    return height <= max_size and width <= max_size


def wrong_proportions(image, max_size):
    """
    checks if it is not possible to scale down the image to the size
    and keep the original proportions
    :param image: an image
    :param max_size: a maximum size - a number
    :return: true if it is impossible to scale down the image to the required size and keep the original proportions -
    the smaller size would have to be 0 or less
    """
    height, width = get_image_scales(image)
    bigger_side = max(height, width)
    smaller_side = min(height, width)
    return bigger_side / smaller_side > max_size


def find_new_scales(image, max_size):
    """
    receives an image and a maximum size and calculates the new height and width that matches the maximum size
    and keep the original proportions
    :param image: an image
    :param max_size: a maximum size - a number
    :return: a tuple containing the new height at the [0] location and the new width at the [1] location
    """
    height, width = get_image_scales(image)
    bigger_side, smaller_side = max(height, width), min(height, width)
    new_bigger_size = max_size
    new_smaller_size = round(smaller_side * new_bigger_size / bigger_side)
    if height >= width:
        return new_bigger_size, new_smaller_size
    return new_smaller_size, new_bigger_size


def scale_down_colored_image(image: ColoredImage, max_size: int) -> Optional[ColoredImage]:
    """
    receives an image and a maximum size, and returns an updated image, with the same proportions as the
    original one, but not bigger than the maximum size (both height and width)
    :param image: an image
    :param max_size: a maximum size - a number
    :return: an updated image
    """
    if image_in_size(image, max_size):
        # no need to resize
        return None
    elif wrong_proportions(image, max_size):
        height, width = get_image_scales(image)
        if height > max_size:
            new_height = max(max_size, 2)
            new_width = max(round(width * new_height / height), 2)
        else:
            new_width = max(max_size, 2)
            new_height = max(round(height * new_width / width), 2)
            # the image should have at least 2 rows and columns else the bilinear interpolation
            # formula won't work
    else:
        # get the new height and width
        new_height, new_width = find_new_scales(image, max_size)
    channel_list = separate_channels(image)
    scaled_down_images = []
    for i in range(len(channel_list)):
        new_channel = resize(channel_list[i], new_height, new_width)
        scaled_down_images.append(new_channel)
    scaled_down_img = combine_channels(scaled_down_images)
    return scaled_down_img


def rotate_image_left(image):
    """
    receives an image and rotates is 90 degrees to the left, works for both colored and grey scale images
    :param image: a given image
    :return: the same image rotated
    """
    height, width = get_image_scales(image)
    new_image = []
    for i in range(width-1, -1, -1):
        new_row = []
        for j in range(height):
            new_row.append(image[j][i])
        new_image.append(new_row)
    return new_image


def rotate_image_right(image):
    """
    receives an image and rotates is 90 degrees to the right, works for both colored and grey scale images
    :param image: a given image
    :return: the same image rotated
    """
    height, width = get_image_scales(image)
    new_image = []
    for i in range(width):
        new_row = []
        for j in range(height-1, -1, -1):
            new_row.append(image[j][i])
        new_image.append(new_row)
    return new_image


def rotate_90(image: Image, direction: str) -> Image:
    """
    rotates an image to the given direction
    :param image: an image
    :param direction: "R" to rotate right, "L" to rotate left
    :return: a rotated image
    """
    if direction == "R":
        new_image = rotate_image_right(image)
    else:
        new_image = rotate_image_left(image)
    return new_image


def get_average(block):
    """
    receives a block of an image and calculates the average value of its cells
    :param block: a block of an image
    :return:
    """
    size = len(block)
    sum = 0
    for i in range(size):
        for j in range(size):
            sum += block[i][j]
    return sum / size ** 2


def get_edges(image: SingleChannelImage, blur_size: int, block_size: int, c: int) -> SingleChannelImage:
    """
    receives a grey scale image, and 3 numbers, and returns the edges of that image
    :param image: a grey scale image (single dimensional)
    :param blur_size: an odd number required to blur the image
    :param block_size: an odd number required to create a block to calculate the threshold
    :param c: a number required to make sure we find only edges and not pixels that are just relatively different
    from their neighbors
    :return: a black and white image, with the black pixels being the edges of the original image
    """
    height, width = get_image_scales(image)
    block = blur_kernel(block_size)  # creates a block
    kernel = blur_kernel(blur_size)
    blured_image = apply_kernel(image, kernel)
    edged_image = []
    for i in range(height):
        edged_row = []
        for j in range(width):
            location = [i, j]
            patch = create_patch(blured_image, location, block)
            average_of_block = get_average(patch)
            if blured_image[i][j] > average_of_block - c:
                edged_row.append(255)
            else:
                edged_row.append(0)
        edged_image.append(edged_row)
    return edged_image


def quantize(image: SingleChannelImage, N: int) -> SingleChannelImage:
    """
    reeceives a single dimensional image and returns an image with N (number) of shades
    :param image: an image
    :param N: a number
    :return: the updated image with N shades
    """
    height, width = get_image_scales(image)
    quantized_image = []
    for i in range(height):
        quantized_row = []
        for j in range(width):
            quantized_pixel = round(math.floor(image[i][j] * N / 256) * 255 / (N - 1))
            quantized_row.append(quantized_pixel)
        quantized_image.append(quantized_row)
    return quantized_image


def quantize_colored_image(image: ColoredImage, N: int) -> ColoredImage:
    """
    receives a colored image and returns an updated image with N shades for each color channel
    :param image: a colored image
    :param N: number of shades
    :return: the quantized image
    """
    channels = separate_channels(image)
    list_of_channels = []
    for i in range(len(channels)):
        quantized_channel = quantize(channels[i], N)
        list_of_channels.append(quantized_channel)
    quantized_image = combine_channels(list_of_channels)
    return quantized_image


def image_is_colored(image):
    """
    checks if an image is colored or single dimensional
    :param image: an image
    :return: true if colored, false if not
    """
    return type(image[0][0]) == list


def mask_single_channel(channel1, channel2, mask):
    """
    combines two channels with the same number of rows and columns
    (single dimensional images) together using the formula given in the instructions
    :param channel1: the first image
    :param channel2: the second image
    :param mask: a given mask- a single dimensional image of the same size, with pixel vales of 0 - 1
    :return: a combined image
    """
    height, width = get_image_scales(channel1)
    new_channel = []
    for i in range(height):
        new_row = []
        for j in range(width):
            new_row.append(round(channel1[i][j] * mask[i][j] + channel2[i][j] * (1 - mask[i][j])))
        new_channel.append(new_row)
    return new_channel


def add_mask(image1: Image, image2: Image, mask: List[List[float]]) -> Image:
    """
    combine two images, whether they are colored or not
    :param image1: the first image
    :param image2: the second image
    :param mask: a given mask
    :return: the combined image
    """
    if image_is_colored(image1):
        channels1 = separate_channels(image1)
        channels2 = separate_channels(image2)
        masked_channels = []
        for i in range(len(channels1)):
            masked_channels.append(mask_single_channel(channels1[i], channels2[i], mask))
        new_image = combine_channels(masked_channels)
        return new_image
    else:
        # single dimensional images
        return mask_single_channel(image1, image2, mask)


def black_white_to_mask(black_white_image):
    """
    creates a mask from a black and white image
    :param black_white_image: a black and white image
    :return: a mask
    """
    height, width = get_image_scales(black_white_image)
    mask = []
    for i in range(height):
        mask_row = []
        for j in range(width):
            mask_row.append(black_white_image[i][j] / 255)
        mask.append(mask_row)
    return mask


def cartoonify(image: ColoredImage, blur_size: int, th_block_size: int,
               th_c: int, quant_num_shades: int) -> ColoredImage:
    """
    receives an image and 5 numbers and returns a cartoon of that image, by combining its edges with its quantized
    version
    :param image: a given image
    :param blur_size: a number required to blur the image
    :param th_block_size: a number required to make a block to calculate the threshold
    :param th_c: a number required to make sure we find the edges and not relatively dark or bright pixels
    :param quant_num_shades: number of shades of each color channel for the quantized image
    :return: the cartooned image
    """
    if image_is_colored(image):
        grey_scale_image = RGB2grayscale(image)
        quantized_image = quantize_colored_image(image, quant_num_shades)
        edged_image = get_edges(grey_scale_image, blur_size, th_block_size, th_c)
        mask = black_white_to_mask(edged_image)
        # we use the mask from the edged image because it will "take" all the pixels that are not the
        # image borders from the first image, and all the pixel that are the borders from the second
        # image, and that will give us the cartooned image
        channels = separate_channels(quantized_image)
        cartooned_channels = []
        for i in range(len(channels)):
            cartooned_channels.append(add_mask(channels[i], edged_image, mask))
        cartooned_image = combine_channels(cartooned_channels)
    else:
        quantized_image = quantize(image, quant_num_shades)
        edged_image = get_edges(image, blur_size, th_block_size, th_c)
        mask = black_white_to_mask(edged_image)
        cartooned_image = add_mask(quantized_image, edged_image, mask)
    return cartooned_image


def main():
    """
    using the helper file, loads an image, rezises it to the given maax size, makes a cartooned copy
    of it, saves it to the typed path and shows it
    :return: none
    """
    if len(sys.argv) != 8:
        sys.exit("Invalid numbers of parameters!")
    image = load_image(sys.argv[1])
    resized_image = scale_down_colored_image(image, int(sys.argv[3]))
    cartooned_image = cartoonify(resized_image, int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
    save_image(cartooned_image, sys.argv[2])
    show_image(cartooned_image)


if __name__ == '__main__':
    main()



