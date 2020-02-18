'''
Pipeline:
    BUILD_DATA: if True, opens camera to take new data as an input
        if you are making a new dataset, then you will need to preprocess from
        scratch so boolean of preprocessing is same as boolean of inputting new
        dataset

    compileData: preprocesses raw data of images into an array:
        [image[i], label[i]]

    model.buildModel(): build a convolutional neural network, prints summary if
        summary=True. returns convmodel.

    model.train(): defines optimizer, loads data from .npy, converts to tensor,
        splits into train and test sets, fits model,  saves model checkpoints
'''
import sys, getopt
import get_data
import netmodel
import preprocess
from live_predict import prediction

#BUILD_STATE = True
#TRAIN_MODEL = True
#LIVE_PREDICT = True


def get_args(argv):
    
    try:
        opts, args = getopt.getopt(argv,"hb:i:t:", ["build=","images=","train="])
    except getopt.GetoptError:
        print('python runall.py -b <BUILD_STATE> -i <IMAGES_PER_LABEL> -t <TRAIN_MODEL>')
        print("<BUILD_STATE> should be Boolean (0 or 1). \n <IMAGES_PER_LABEL> should be int(). \n <TRAIN_MODEL> should be Boolean (1 or 0)")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h' or opt == '--help':
            print('python runall.py -b <BUILD_STATE> -i <IMAGES_PER_LABEL> -t <TRAIN_MODEL>')
            sys.exit()
        elif opt in ("-b", "--build"):
            BUILD_STATE = int(arg)
        elif opt in ("-i", "--images"):
            IMAGES_PER_LABEL = int(arg)
        elif opt in ("-t", "--train"):
            TRAIN_MODEL = int(arg)
    return BUILD_STATE, IMAGES_PER_LABEL, TRAIN_MODEL

def main(BUILD_STATE, IMAGES_PER_LABEL, TRAIN_MODEL):
    labels = [0, 1]
    get_data.main(BUILD_STATE, IMAGES_PER_LABEL, labels)
    datapath = 'data'
    dataset, plot_set = preprocess.compileData(data_path=datapath,
                                         make_new=BUILD_STATE)
    if BUILD_STATE:
        preprocess.example_plot(plot_set)

    convmodel = netmodel.buildModel(print_summary=False)
    if TRAIN_MODEL:
        netmodel.train(convmodel)
    


if __name__ == "__main__":
    BUILD_STATE, IMAGES_PER_LABEL, TRAIN_MODEL, LIVE_PREDICT = get_args(sys.argv[1:])
    main(BUILD_STATE, IMAGES_PER_LABEL, TRAIN_MODEL)

