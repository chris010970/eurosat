import os
import glob
import math
import random
import pandas as pd
import numpy as np

# one hot encoder
from skimage.io import imread
from keras.utils import to_categorical


class Eurosat:

    def __init__( self ):

        """
        constructor
        """

        # dataset vital statistics
        self._height = self._width = 64
        self._channels = 13

        # zip class names into dict
        labels = [  "AnnualCrop", 
                    "Forest", 
                    "HerbaceousVegetation", 
                    "Highway", 
                    "Industrial", 
                    "Pasture", 
                    "PermanentCrop", 
                    "Residential",
                    "River", 
                    "SeaLake" ]

        self._classes = dict(zip( labels,range (len(labels) ) ) )
        return


    def getSubsetFiles( self, data_path ):

        """
        create train and test csv files
        """

        def getSubset( files, start, end ):

            """
            repackage subset of files into dataframe
            """

            # create master dataframe
            df = pd.DataFrame( { 'pathname' : [], 'class' : [], 'id' : [] } )
            for idx, ( k, v ) in enumerate ( self._classes.items() ):

                # append child frame comprising pathnames, class name and id
                f = files[ idx ]
                df = df.append( pd.DataFrame( {  'pathname' : f[ start : end ], 
                                                'class' : [ k ] * len( f[ start : end ] ), 
                                                'id' : [ v ] * len( f[ start : end ] ) } ) )

            return df

        # for each class
        files = []; max_samples = 0
        for k in self._classes.keys():
            
            # get files from each class sub-directory
            path = os.path.join( data_path, k ) 
            files.append(  glob.glob( os.path.join( path, '*.tif' ) ) )

            # get maximum samples in class
            max_samples = max( max_samples, len( files[ -1 ] ) )

        # equalise sample sizes by upsampling
        for f in files:

            while len( f ) != max_samples:            
                f.append( random.choice( f ) ) 

            # shuffle lists
            random.shuffle( f )
    
        # compile train records
        train = getSubset( files, 0, int( max_samples * 0.8 ) )
        train.to_csv( os.path.join( data_path, 'train.csv' ), index=False )

        # compile test records
        test = getSubset( files, int( max_samples * 0.8 ), max_samples )
        test.to_csv( os.path.join( data_path, 'test.csv' ), index=False )

        return

    
    def getNormalisationStats( self, data_path ):

        """
        create train and test csv files
        """

        # initialise stats
        sum_x = np.zeros( self._channels ); sum_x2 = np.zeros( self._channels )
        count = np.zeros( self._channels )

        for k in self._classes.keys():
            
            # separately process each class sub-directory
            path = os.path.join( data_path, k ) 
            files = glob.glob( os.path.join( path, '*.tif' ) )
            for f in files:

                # load image
                image = np.array( imread( f ), dtype=float )
                for channel in range( self._channels ):

                    # flatten channel data
                    data = np.reshape(image[:,:,channel], -1)
                    count[ channel ] += data.shape[ 0 ]

                    # update sum and sum of squares 
                    sum_x[ channel ] += np.sum( data )
                    sum_x2[ channel ] += np.sum( data**2 )


        # for each channel
        stats = []
        for channel in range( self._channels ):

            # compute mean and stdev from summations
            mean = sum_x[ channel ] / count [ channel ]
            stdev = math.sqrt ( sum_x2[ channel ] / count [ channel ] - mean**2 )

            # append channel stats to list
            stats.append ( [ channel, mean, stdev ] )

        # convert list to dataframe and save to csv
        df = pd.DataFrame( stats, columns =['channel', 'mean', 'stdev'], dtype = float ) 
        df.to_csv( os.path.join( data_path, 'stats.csv' ), index=False )

        return


    def updateDataFrame( self, df ):

        """
        update data frame for training / prediction
        """

        # add OHE target column to subset data frames
        df[ 'target' ] = tuple ( to_categorical(  df[ 'id' ], 
                                    num_classes=len( self._classes ) ) )

        return df


# create files
# obj = Eurosat()
# obj.getNormalisationStats( 'C:\\Users\\Chris.Williams\\Documents\\GitHub\\eurosat\\data' )
# obj.getSubsetFiles( 'C:\\Users\\Chris.Williams\\Documents\\GitHub\\eurosat\\data' )
