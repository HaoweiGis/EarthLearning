import numpy as np

def get_kmeans(input, n_clusters=10, **kwargs): 
    im_shape = input.shape
    im_lines = None
    for i in range(im_shape[0]):
        im_line = input[i,:,:].flatten()
        if im_lines is None:
            im_lines = im_line
        else:
            im_lines = np.vstack((im_lines,im_line))
    im_lines = im_lines.transpose((1,0))

    from sklearn.cluster import KMeans
    model=KMeans(n_clusters=n_clusters)

    model.fit(im_lines)
    return model