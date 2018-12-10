#!/usr/bin/env/ python
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate as turn


class ImageGenerator(object):

    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """

        # The ImageGenerator instance has to store the following information:
        # x, y, num_of_samples, height, width, number of pixels translated, degree of rotation, is_horizontal_flip,
        # is_vertical_flip, is_add_noise. By default, set boolean values to False.

        #import and copy the exact input
        self.x = x.copy()
        self.y = y.copy()
        
        #dimensions
        self.N, self.height,self.width, self.channels = x.shape
                
        #transform parameters
        self.pixels_translated = None
        self.degree_rotation = None
        self.is_horizontal_flip = False
        self.is_vertical_flip= False
        self.is_add_noise = False

        # One way to use augmented data is to store them after transformation (and then combine all of them to form new data set).
        # Following variables (along with create_aug_data() function) is one kind of implementation.
        # You can either figure out how to use them or find out your own ways to create the augmented dataset.
        # if you have your own idea of creating augmented dataset, just feel free to comment any codes you don't need
        
        self.translated = None
        self.rotated = None
        self.flipped = None
        self.added = None
        self.x_aug = self.x.copy()
        self.y_aug = self.y.copy()
        self.N_aug = self.N

    def create_aug_data(self):
        # If you want to use function create_aug_data() to generate new dataset, you can perform the following operations in each
        # transformation function:
        #
        # 1.store the transformed data with their labels in a tuple called self.translated, self.rotated, self.flipped, etc. 
        # 2.increase self.N_aug by the number of transformed data,
        # 3.you should also return the transformed data in order to show them in task4 notebook
        # 
        
        '''
        Combine all the data to form a augmented dataset 
        '''
        if self.translated:
            self.x_aug = np.vstack((self.x_aug,self.translated[0]))
            self.y_aug = np.hstack((self.y_aug,self.translated[1]))
        if self.rotated:
            self.x_aug = np.vstack((self.x_aug,self.rotated[0]))
            self.y_aug = np.hstack((self.y_aug,self.rotated[1]))
        if self.flipped:
            self.x_aug = np.vstack((self.x_aug,self.flipped[0]))
            self.y_aug = np.hstack((self.y_aug,self.flipped[1]))
        if self.added:
            self.x_aug = np.vstack((self.x_aug,self.added[0]))
            self.y_aug = np.hstack((self.y_aug,self.added[1]))
        
        #increase N
        self.N_aug = self.x_aug.shape[0]
        
    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data infinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """
        
        # 1. The generator should return batches endlessly.
        # 2. Make sure the shuffle only happens after each sample has been visited once. Otherwise some samples might
        # not be output.
        
        #input is for image selection
        x = self.x_aug
        y = self.y_aug
        N = self.N_aug
        
        batch_count = 0
        max_batch = N//batch_size#   calculate the total number of batches possible (if the rest is not sufficient to make up a batch, ignore)
        
        while True:
            if(batch_count < max_batch):
                #plt.imshow(x[batch_count])
                yield [x[batch_count * batch_size : (batch_count + 1) * batch_size], y[batch_count * batch_size : (batch_count + 1) * batch_size]]
                batch_count += 1

            else:
                if shuffle == True:
                    p = np.random.permutation(N) #create random array of indexes
                    x = x[p]
                    y = y[p]
                
                batch_count = 0 # reset batch count        

    def show(self, images):
        """
        Plot the top 16 images (index 0~15) for visualization.
        :param images: images to be shown
        """
        
        #select top 16
        preview = images[:16]

        #plot 4 by 4
        r = 4
        f, axarr = plt.subplots(r, r, figsize=(32,32))

        for i in range(r):
            for j in range(r):
                img = preview[r*i+j]
                axarr[i][j].imshow(img)
                
    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return translated: translated dataset
        """
       
        self.translated = np.roll(self.x, shift = shift_height, axis = 1) #vertical roll
        self.translated = [np.roll(self.translated, shift = shift_width, axis = 2), self.y] #horizontal roll
        self.pixels_translated = [shift_height, shift_width]
        return self.translated
    
    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.
        :return rotated: rotated dataset
        - https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
        """
        self.rotated = turn(self.x, angle, axes=(2,1), reshape=False, order=3, mode='constant', cval=0.0, prefilter=True)
        self.degree_rotation = angle
        self.rotated = [self.rotated, self.y]

        return self.rotated
        
        
    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        :return flipped: flipped dataset
        """
        if mode == 'h':
            self.flipped = np.flip(self.x, axis = 2)
            self.is_horizontal_flip = not self.is_horizontal_flip
        elif mode == 'v':
            self.flipped = np.flip(self.x, axis = 1)
            self.is_vertical_flip = not self.is_vertical_flip
        elif mode == 'hv':
            self.flipped = np.flip(self.x, axis = 1)
            self.flipped = np.flip(self.flipped, axis = 2)
            self.is_vertical_flip = not self.is_vertical_flip
            self.is_horizontal_flip = not self.is_horizontal_flip
        else:
            print("not a correct parameter")
            
        self.flipped = [self.flipped, self.y]
        return self.flipped

    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        :return added: dataset with noise added
        """
        #select random portion of images
        num_rand = int(portion * self.N)
        rand_ind = np.random.choice(self.N, size = num_rand, replace = False)
        
        #add noise to those images without going over 0 or 1 for RGB
        temp = self.x[rand_ind].astype(float)
        noise = np.random.uniform(low = -amplitude, high=amplitude, size=temp.shape)
        
        #limit size of images to be between 0 and 255
        temp = np.clip(temp + noise, 0, 255)/255
        
        self.is_add_noise =  True
        
        self.added = [temp, self.y[rand_ind]]
        return self.added
        
