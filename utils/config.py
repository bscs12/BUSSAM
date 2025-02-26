# This file is used to configure the training parameters for each task

class Config_AMUBUS:
    data_path = 'datasets/'
    data_subpath = 'datasets/AMUBUS/'
    output_path = 'outputs/'
    load_path = '/media/data2/lcl_e/gl/BUSSAM/outputs/BUSSAM_1118110744/checkpoints/BUSSAM_11181413_36_86.65829590482832.pth'  # checkpoint used for testing

    workers = 1  # number of data loading workers (default: 8)
    epochs = 100  # number of total epochs to run (default: 400)
    batch_size = 8  # batch size (default: 4)
    learning_rate = 1e-4  # initial learning rate (default: 0.001)
    momentum = 0.9  # momentum
    classes = 2  # the number of classes (background + foreground)
    img_size = 256  # the input size of model
    train_split = 'AMUBUS_train'  # the file name of training set
    val_split = 'AMUBUS_val'  # the file name of validating set
    test_split = 'AMUBUS_test'  # the file name of testing set
    crop = None  # the cropped image size
    eval_freq = 1  # the frequency of evaluate the model
    save_freq = 2000  # the frequency of saving the model
    device = 'cuda'  # training device, cpu or cuda
    cuda = 'on'  # switch on/off cuda option (default: off)
    gray = 'yes'  # the type of input image
    img_channel = 1  # the channel of input image
    eval_mode = 'mask_slice'  # the mode when evaluate the model, slice level or patient level
    pre_trained = False  # load pre-trained model during training
    mode = 'train'  # train or test
    visual = False  # save segmentation maps during validation
    modelname = 'BUSSAM'  # type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, BUSSAM
    
    
class Config_BUSI:
    data_path = 'datasets/'
    data_subpath = 'datasets/BUSI/'
    output_path = 'outputs/'
    load_path = '/media/data2/lcl_e/gl/BUSSAM/outputs/BUSSAM_0112202015/checkpoints/BUSSAM_01122031_3_78.48604293121788.pth'  # checkpoint used for testing

    workers = 1  # number of data loading workers (default: 8)
    epochs = 100  # number of total epochs to run (default: 400)
    batch_size = 8  # batch size (default: 4)
    learning_rate = 1e-4  # initial learning rate (default: 0.001)
    momentum = 0.9  # momentum
    classes = 2  # the number of classes (background + foreground)
    img_size = 256  # the input size of model
    train_split = 'BUSI_train'  # the file name of training set
    val_split = 'BUSI_val'  # the file name of testing set
    test_split = 'BUSI_test'  # the file name of testing set
    crop = None  # the cropped image size
    eval_freq = 1  # the frequency of evaluate the model
    save_freq = 2000  # the frequency of saving the model
    device = 'cuda'  # training device, cpu or cuda
    cuda = 'on'  # switch on/off cuda option (default: off)
    gray = 'yes'  # the type of input image
    img_channel = 1  # the channel of input image
    eval_mode = 'mask_slice'  # the mode when evaluate the model, slice level or patient level
    pre_trained = False  # load pre-trained model during training
    mode = 'train'  # train or test
    visual = False  # save segmentation maps during validation
    modelname = 'BUSSAM'  # type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, BUSSAM


# ========== get config ==========

def get_config(task=None):
    if task == 'AMUBUS':
        return Config_AMUBUS()
    elif task == 'BUSI':
        return Config_BUSI()
    else:
        assert 'We do not have the related dataset, please choose another task.'
