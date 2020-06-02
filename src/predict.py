import os
import time
import argparse
import numpy as np
import pandas as pd

# metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# graphics
import seaborn as sn
import matplotlib.pyplot as plt

# local imports
from eurosat import Eurosat
from model import loadFromFile
from generator import MultiChannelImageDataGenerator


class Predict:

    def __init__( self, args ):

        """
        constructor
        """

        # initialise members            
        self._model, self._architecture = loadFromFile( args.model_path )
        self._eurosat = Eurosat()
        return


    def process( self, args ):

        """
        main path of execution
        """

        # get stats dataframe
        stats = pd.read_csv( os.path.join( args.data_path, 'stats.csv' ) )
        args.batch_size = 1

        # get train and test dataframes
        df = {  'train' : pd.read_csv( os.path.join( args.data_path, 'train.csv' ) ),
                'test' : pd.read_csv( os.path.join( args.data_path, 'test.csv' ) ) }
        
        # add OHE target column to subset data frames
        for subset in [ 'train', 'test']:
            df[ subset ] = self._eurosat.updateDataFrame( df[ subset ] )

        # generate actual vs prediction
        for subset in [ 'train', 'test' ]:
        
            actual = np.asarray( df[ subset ][ 'id' ].tolist(), dtype=int )
            predict = self.getPrediction( df[ subset ], stats, args )

            # get confusion matrix
            cm = self.getConfusionMatrix(   actual,
                                            predict,
                                            self._eurosat._classes.keys() )

            # plot confusion matrix
            self.plotConfusionMatrix( cm, subset )

        return


    def getPrediction( self, df, stats, args ):
                
        """
        generate prediction for images referenced in data frame
        """

        # create generator
        generator = MultiChannelImageDataGenerator( [ df ],
                                                    args.batch_size,
                                                    stats=stats,
                                                    shuffle=False )

        # initiate prediction
        steps = len( df ) // args.batch_size
        y_pred = self._model.predict_generator( generator, steps=steps )

        # return index of maximum softmax value
        return np.argmax( y_pred, axis=1 )


    def getConfusionMatrix( self, actual, predict, labels ):

        """
        compute confusion matrix for prediction
        """

        # compute normalised confusion matrix 
        cm = confusion_matrix( actual, predict )
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # parse normalised confusion matrix into dataframe
        return pd.DataFrame( cm, index=labels, columns=labels )


    def plotConfusionMatrix( self, cm, subset ):

        """
        plot train and test confusion matrix
        """

        # create figure
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

        # plot heatmap - adjust font and label size
        sn.set(font_scale=0.8) 
        sn.heatmap(cm, annot=True, annot_kws={"size": 12}, fmt='.2f', ax=axes )

        axes.set_title( 'Normalised Confusion Matrix: {}'.format( subset ) )
        plt.show()

        return


def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='eurosat train')
    parser.add_argument('data_path', action='store')
    parser.add_argument('model_path', action='store')

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()
    
    # create and execute training instance
    obj = Predict( args )
    obj.process( args )

    return


# execute main
if __name__ == '__main__':
    main()
