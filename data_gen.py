import os
import torch
import torch_geometric
from skimage.segmentation import slic
import numpy as np
from scipy import ndimage, stats
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
import torch.nn.functional as F
import skimage.io
import pickle

def zscore_normalize(x):
        x = x.astype(np.float32)

        std = np.std(x)
        mv = np.mean(x)
        if std <= 1.0:
            # normalize (but dont divide by zero)
            x = (x - mv)
        else:
            # z-score normalize
            x = (x - mv) / std
        return x

def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    m = counts.argmax()
    return values[m]

def get_features(x):  # This function can be edited to use any features you want. Edit the num_features variable if this function is changed.

    quants = np.quantile(x, [0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9])
    mean = np.array([np.mean(x)])
    std = np.array([np.std(x)])

    features = np.concatenate((mean, std, quants)) 
    
    return features
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate graphs for input images.')    
    parser.add_argument('-i', '--image-dir', type=str, required=True,
                        help='Filepath to the folder/directory where the input images exist')
    
    parser.add_argument('-m', '--mask-dir', type=str, required=True,
                        help='Filepath to the folder/directory where the input masks exist')

    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='Path to the file where the output graph will be saved')
                        
    args = parser.parse_args()
    image_filepath = args.image_dir
    mask_filepath = args.mask_dir
    output_filename = args.output_file
                        
    #Hyperparameters
    num_features = 11 #This is equal to the number of features outputted by the get_features function. 
    num_superpixels = 40000
    num_neighbours = 10
    
    _, _, files = list(os.walk(image_filepath))[0]
    
    data_list = []
            
    for file in files:
        
        image = skimage.io.imread(os.path.join(image_filepath, file))
        mask = skimage.io.imread(os.path.join(mask_filepath, file))
        mask = mask//255
        image = zscore_normalize(image)
    
        superpixels = slic(image.astype('double')/np.max(image), n_segments = 40000, compactness = 0.1,sigma = 1, min_size_factor = 0.1)
        num_nodes = np.amax(superpixels)+1
    
        
    
        sp_centroids = np.array(ndimage.center_of_mass(np.ones(superpixels.shape),superpixels,range(0,num_nodes)), dtype = 'float32')
        euc_distances = cdist(sp_centroids,sp_centroids)
        closest_neighbours = np.argsort(euc_distances,axis=1)[:,:num_neighbours+1]
        edge_array = np.array([np.reshape(closest_neighbours, (-1)),
                                   np.repeat(range(num_nodes), num_neighbours+1)])
        edge_index = torch.tensor(edge_array, dtype=torch.long)

        
        closest_neighbours = np.sort(euc_distances,axis=1)[:,:num_neighbours+1]
        edge_array = np.reshape(closest_neighbours, (-1))
        edge_array = np.exp(-edge_array)
        edge_weights = torch.tensor(edge_array, dtype=torch.float32)
        
        edge_index, edge_weights = torch_geometric.utils.to_undirected(edge_index, edge_weights)

    
        x = ndimage.labeled_comprehension(image,labels=superpixels,func=get_features,index=range(0,num_nodes),out_dtype='object',default=-1.0)
        x = np.stack(x)
        
    
    
        y = ndimage.labeled_comprehension(mask,labels=superpixels,func=mode,index=range(0,num_nodes),out_dtype='int32',default=-1.0)
        data = Data(x=torch.from_numpy(x).float(), y=torch.from_numpy(y), edge_index=edge_index, edge_weights = edge_weights)
    
        data_list.append(data)
                
    f = open(output_filename, 'wb')
    pickle.dump(data_list,f )
    f.close()
