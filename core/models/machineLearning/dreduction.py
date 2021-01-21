import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_principalcomponent(input, n_components=1, **kwargs):
    im_shape = input.shape
    im_lines = None
    for i in range(im_shape[0]):
        im_line = input[i,:,:].flatten()
        if im_lines is None:
            im_lines = im_line
        else:
            im_lines = np.vstack((im_lines,im_line))
    im_lines = im_lines.transpose((1,0))

    min_max_scaler = MinMaxScaler()
    im_lines = min_max_scaler.fit_transform(im_lines)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    im_DReduction = pca.fit_transform(im_lines)
    img_pac = None
    for i in range(im_DReduction.shape[1]):
        im_pac = im_DReduction[:,i].reshape((im_shape[1],im_shape[2]))[np.newaxis,:, :]
        if img_pac is None:
            img_pac = im_pac
        else:
            img_pac = np.concatenate((img_pac,im_pac), axis=0)

    return img_pac