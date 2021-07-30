# All helper functions are written in this file.

def IO():
    """
        This is wraper function around MNE input/oupt modules
    """
    from mne import io

    return io


def eeg_reader(data_path, labels, orders, epoch_length=1e4):
    """
    This is custome writen function to read eegdata from .mat file which saved in h5 format.
    It has some preassumptions about data structure (3 channel data)
    It also takes care of signal synchronicity with labels

    INPUTS
    data_path: full path to location of data (.mat file)
    labels: labels corresponding to data
    orders: channel order
    epoch_length: length of epoch for each label
    z_score: zscore function need to be applied to signal or not
    """

    # import packages
    import numpy as np
    import h5py
    from scipy.stats import zscore

    # reading files from h5 source
    F = h5py.File(data_path)
    list(F.keys())
    CH1 = F[list(F.keys())[0]]['values']
    CH2 = F[list(F.keys())[1]]['values']
    CH3 = F[list(F.keys())[2]]['values']
    print('CH1', CH1.shape, 'CH2', CH2.shape, 'CH3', CH3.shape)

    # concatinating 3 channels
    signals = np.concatenate((CH1, CH2, CH3), axis=0)
    print(f'output signal size is: {signals.shape}')

    # making numpy array of signal and transposing it
    signals = np.array(signals.T)

    # re ordering channels
    signals = signals[:, orders]

    # adjusting signal length to label length
    nr_epochs = int(np.floor(signals.shape[0] / epoch_length))

    # warning if signal length is not equal to epoch_length * nr_epochs
    if signals.shape[0] != (nr_epochs * epoch_length):
        print('WARNING: signal length is not equal to number of epochs * epoch length')
        print('WARNING: signal will be cutted from end.')

    # cut-out signal ending which has no label
    signals = signals[0:int(nr_epochs * epoch_length), :]

    # selecting labels based on number of epochs
    labels = labels[0:nr_epochs]
    print(
        f'number of epochs is: {nr_epochs} and adjusted signals length is: {signals.shape[0]}')

    # getting number of classes
    nr_classes = len(np.unique(labels))
    print(f'number of unique classes are: {nr_classes}')

    # getting number of features (channels)
    nr_features = signals.shape[1]

    return signals, labels, epoch_length, nr_classes


def make_aux_data(data, epoch_length, labels):
    """
    This function reads 2d input data (time * features) and change it to
    3d strucure epochs * epoch_length * features

    data: A 2D matrix with shape [timesteps, feature]
    epoch_length: length of epoch
    labels: label value for all epochs
    """

    # import necessary packages
    import numpy as np

    # getting data dimension
    t, f = data.shape  # time * features

    # change epoch length to int
    epoch_length = int(epoch_length)

    # optimal number of epochs
    dim_0 = np.min(
        [np.int(np.floor(data.shape[0] / epoch_length)), len(labels)])
    print(f'optimal number of epochs are: {dim_0}')

    # initializing aux_data
    data_aux = np.zeros((dim_0, epoch_length, f), dtype=np.float16)
    print('auxilary data initial size', data_aux.shape)

    for i in range(dim_0):
        data_aux[i, :] = data[i * epoch_length:i * epoch_length + epoch_length]

    print('auxilary data final size', data_aux.shape,
          '  labels final size', labels.shape)
    if data_aux.shape[0] != len(labels):
        raise NameError('label length is different than batch nr in data')

    return data_aux, dim_0, t, f


def check_installed_packages():
    """
    using this function we are checking if user computer has all required packages
    """
    pass
