import math
import random
import numpy as np
import pandas as pd

# numpy random
from numpy.random import uniform
from numpy.random import random_integers

# sci-image  
from skimage.io import imread
from skimage.util import random_noise
from sklearn.utils import shuffle as pd_shuffle

# sci-image transform
from skimage.transform import AffineTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp


def MultiChannelImageDataGenerator( groups,
                                    batch_size,
                                    stats=None,
                                    shuffle=True,
                                    horizontal_flip=False,
                                    vertical_flip=False,
                                    rotation_range=0,
                                    shear_range=0,
                                    scale_range=1,
                                    transform_range=0,
                                    filling_mode='edge',
                                    speckle=None,
                                    crop=False,
                                    crop_size=None ):

    """
    generate batches of augmented multichannel imagery catalogued in data frame list
    """
    
    # shuffle data frames
    if shuffle:
        for group in groups:
            group = pd_shuffle(group)
    else:
        # verify single dataframe without shuffle
        assert ( len( groups ) == 1 )

    # create index
    current_idx = 0
    while True:
        
        # initialise batch lists
        batch_X = []
        batch_Y = []
        
        # create empty
        sample = pd.DataFrame( columns=groups[ 0 ].columns )
        if shuffle:

            # construct random stratified sample from input dataframes
            for idx in range( batch_size ):
            
                # randomly select grouping and random pick single record
                fid = random.randrange( 0, len( groups ) )
                sample = sample.append ( groups[ fid ].sample() )

        else:

            # get sample by slicing dataframe at incremental index 
            sample = groups[ 0 ][ current_idx : current_idx + batch_size ]
            current_idx += batch_size

            # reset at end
            if current_idx >= len ( groups[ 0 ] ):
                current_idx = 0

            # check sample size equals batch size
            if batch_size > len ( sample ):
                sample = sample.append ( groups [ 0 ] [ 0 : ( batch_size - len ( sample ) ) ] )
                current_idx = ( batch_size - len ( sample ) )

        # iterate through sample rows
        for idx, row in sample.iterrows():

            # read image
            image = np.array( imread( row[ 'pathname'] ), dtype=float )

            # apply normalisation / standardisation
            if stats is not None:
                image = standardiseImage( image, stats )

            # optionally apply random flip
            if horizontal_flip or vertical_flip:
                image = applyFlip(  image,
                                    horizontal_flip=horizontal_flip,
                                    vertical_flip=vertical_flip )
        
            # optionally apply random affine transformation
            image = applyAffineTransform(   image, 
                                            rotation_range=rotation_range, 
                                            shear_range=shear_range,
                                            scale_range=scale_range,
                                            transform_range=transform_range, 
                                            warp_mode=filling_mode )

            # optionally apply random speckle noise
            if speckle is not None:
                image = applySpeckleNoise( image, speckle )

            # optionally apply random crop
            if crop:
                if crop_size is None:
                    crop_size = image.shape[0:2]
                image = applyCentreCrop( image, crop_size )

            # add image and one hot result to batch 
            batch_X += [ image ]
            batch_Y += [ row[ 'target'] ]

        # convert lists to np.array
        X = np.array(batch_X)
        Y = np.array(batch_Y)

        # pump out batch
        yield ( X, Y )

    return


def standardiseImage(   image, 
                        stats ):

    """
    compute z-score image 
    """

    # compute z score across each channel
    for idx, row in stats.iterrows():
        image[..., idx] -= row[ 'mean' ]

        if 'stdev' in stats.columns:
            image[..., idx] /= row[ 'stdev' ]

    return image


def applyFlip(  image, 
                horizontal_flip=False, 
                vertical_flip=False ):

    """
    apply random flip to N-channel image
    """

    # randomly flip image up/down
    if horizontal_flip:
        if random.choice([True, False]):
            image = np.flipud(image)
    
    # randomly flip image left/right
    if vertical_flip:
        if random.choice([True, False]):
            image = np.fliplr(image)

    return image


def applyAffineTransform(   image, 
                            rotation_range=0, 
                            shear_range=0, 
                            scale_range=1, 
                            transform_range=0,
                            warp_mode='edge' ):

    """
    apply optional random affine transformation to N-channel image
    """

    # generate transformation parameters
    image_shape = image.shape

    rotation_angle = uniform(low=-abs(rotation_range), high=abs(rotation_range) )
    shear_angle = uniform(low=-abs(shear_range), high=abs(shear_range))
    scale_value = uniform(low=abs(1 / scale_range), high=abs(scale_range))
    translation_values = (random_integers(-abs(transform_range), abs(transform_range)), random_integers(-abs(transform_range), abs(transform_range)))

    # initialise transformations
    transform_toorigin = SimilarityTransform(   scale=(1, 1), 
                                                rotation=0, 
                                                translation=(-image_shape[0], -image_shape[1]))
    
    transform_revert = SimilarityTransform( scale=(1, 1), 
                                            rotation=0, 
                                            translation=(image_shape[0], image_shape[1]))

    # generate affine transformation
    transform = AffineTransform(    scale=(scale_value, scale_value), 
                                    rotation=np.deg2rad(rotation_angle),
                                    shear=np.deg2rad(shear_angle), 
                                    translation=translation_values)

    # apply affine transform
    image = warp(   image, 
                    ((transform_toorigin) + transform) + transform_revert, 
                    mode=warp_mode, 
                    preserve_range=True )

    return image


def applySpeckleNoise( image, var ):

    """
    add optional random speckle noise
    """

    # normalise image
    image_max = np.max(np.abs(image), axis=(0, 1))
    image /= image_max

    # add speckle noise and rescale
    image = random_noise(image, mode='speckle', var=var)
    image *= image_max

    return image


def applyCentreCrop(    image, 
                        target_size ):

    """
    apply predefined centre crop to image
    """

    # check bounds
    x_crop = min(image.shape[0], target_size[0])
    y_crop = min(image.shape[1], target_size[1])
    midpoint = [math.ceil(image.shape[0] / 2), math.ceil(image.shape[1] / 2)]

    # apply crop
    crop_image = image[int(midpoint[0] - math.ceil(x_crop / 2)):int(midpoint[0] + math.floor(x_crop / 2)),
                    int(midpoint[1] - math.ceil(y_crop / 2)):int(midpoint[1] + math.floor(y_crop / 2)),
                    :]
    
    assert crop_image.shape[0:2] == target_size
    return crop_image
