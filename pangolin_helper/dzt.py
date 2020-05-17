import siina
import numpy as np
from scipy.fftpack import fft,ifft


def transfer_time_to_freq(data, samplefrequency, freq_from, freq_to):
    """ Transfer time series data to frequency data.

    Parameters:
    data: DZT data N*M, where N is the number of channels and M is the number of times.
    samplefrequency: sample frequency of DZT data. For example, 250.(Hz).
    freq_from: Lower bound of frequency band width. For example, 0.
    freq_to: Upper bound of frequency band width. For example, 50.

    Returns:
    --------
    freq_data: N*(freq_to-freq_from) matrix, where N is the number of channels.
    """
    freq_data = []
    for chaneldata in data:
        fft_y=fft(chaneldata)
        T = 1/samplefrequency
        N = chaneldata.size
        freq_data_row = np.abs(fft_y)[:N // 2] * 1 / N
        freq_data.append(freq_data_row[freq_from:freq_to])

    return np.array(freq_data)


def read_dzt_file(filepath):
    """ Read DZT file from a given path.

    Parameter:
    filepath: str
    Path to a dzt-file

    Returns
    -------
    header : dictionary
        First header, length of 1024 bytes, unpacked.
        Other headers are found as a list of bytes under 'other_headers'.
    data : list of numpy arrays
        Each channel in Fortran (column oriented) format.
        In case of failing to reshape, returns one numpy array in a list.
        Error message is found in the header-dict.
    """
    meas = siina.Radar()
    meas.read_file(filepath)

    return meas.header, meas.data


def read_directory(filepaths):
    """ Read all the files in the given pathes.

    Parameter:
    filepaths: a list of paths, where each path contains a category of files.
        For ex, [richwater_path, broken_path]

    Returns:
    --------
    X_train: the training data set.
    y_train: the label of training data set.
    """
    X_train = []
    Y_train = []
    category = 0

    for filepath in filepaths:
        files = glob.glob(filepath + "/*.DZT")
        for file in files:
            try:
                header, data = read_dzt_file(file)
                samplefrequency = 1.0/header['samples_per_second']
                if data.shape[0]!=1024:
                    continue
                freqdata = transfer_time_to_freq(data, samplefrequency, 0, 25)
                X_train.append(freqdata)
                Y_train.append(category)
            except:
                print("error file:", file)
        category = category + 1

    print("X_train length is: " + str(len(X_train)))
    X_train = np.asarray(X_train)
    Y_train = tf.keras.utils.to_categorical(y=Y_train, num_classes=len(filepaths))

    return X_train, Y_train
