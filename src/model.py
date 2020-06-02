import os

# keras layers
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization

# keras models
from keras.models import Model
from keras.models import model_from_json

# keras apps
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50


def getVgg16( in_shape, layers ):

    """
    construct vgg16 architecture with randomly initialised weights
    """

    # create VGG16 architecture with random weights 
    model = VGG16(  weights=None,
                    include_top=False, 
                    input_shape=in_shape)

    # add flatten layer
    x = Flatten()( model.layers[-1].output )
    for layer in layers[ 'fc' ]:
        
        # add fc + optional dropout layers
        x = Dense( layer[ 'units' ], activation=layer[ 'activation' ], kernel_initializer='he_uniform' )(x)
        if 'dropout' in layer:
            x = Dropout( layer[ 'dropout' ] )( x )
                        
    # add output layers
    for layer in layers[ 'out' ]:
        x = Dense( layer[ 'units' ], activation=layer[ 'activation' ] )(x)

    return Model( inputs=model.inputs, outputs=x )


def getResNet50( in_shape, layers ):

    """
    construct resnet50 architecture with randomly initialised weights
    """

    # create resnet50 architecture with random weights 
    model = ResNet50(   weights=None,
                        include_top=False, 
                        input_shape=in_shape )

    # add flatten layer
    x = Flatten()( model.layers[-1].output )
    for layer in layers[ 'fc' ]:
        
        # add fc + optional dropout layers
        x = Dense( layer[ 'units' ], activation=layer[ 'activation' ], kernel_initializer='he_uniform' )(x)
        if 'dropout' in layer:
            x = Dropout( layer[ 'dropout' ] )( x )
                        
    # add output layers
    for layer in layers[ 'out' ]:
        x = Dense( layer[ 'units' ], activation=layer[ 'activation' ] )(x)

    return Model( inputs=model.inputs, output=x )


def loadFromFile( path ):

    """
    load model architecture and weights from file
    """
    
    # check load path exists
    model = name = None
    with open( os.path.join( path, 'model.json' ), 'r' ) as json_file:

        # read model json file
        model_json = json_file.read()
        model = model_from_json( model_json )

        # read weights into new model
        model.load_weights( os.path.join( path, 'model.h5' ) )
        print('Loaded model from disk: {}'. format ( path ) )

        # record model type
        with open( os.path.join( path, 'model.txt' ), "r" ) as txt_file:
            name = txt_file.read()

    return model, name


def saveToFile( model, path, name ):

    """
    save model architecture and weights to file
    """

    # create save path if not exists
    if not os.path.exists( path ):
        os.makedirs( path )

    # check save path exists
    if os.path.exists( path ):

        # serialize model to JSON
        model_json = model.to_json()
        with open( os.path.join( path, 'model.json' ), "w" ) as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights( os.path.join( path, 'model.h5' ) )
        print('Saved model to disk: {}'. format ( path ) )

        # record model type
        with open( os.path.join( path, 'model.txt' ), "w" ) as txt_file:
            txt_file.write(name)

    return
