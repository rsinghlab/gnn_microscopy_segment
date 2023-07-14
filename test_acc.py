import os, sys
import random
import torch
import torch_geometric
import numpy as np
import pickle
from scipy import ndimage, stats
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import Sequential,GCNConv, SAGEConv, BatchNorm
from torch.nn import Linear, ReLU,CrossEntropyLoss
import time
import GPUtil
import sklearn.metrics
import skimage.io
from skimage.segmentation import slic
from gnn_model import GCN


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

def get_features(x): # This function can be edited to use any features you want. Edit the num_features variable if this function is changed.
    quants = np.quantile(x, [0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9])
    mean = np.array([np.mean(x)])
    std = np.array([np.std(x)])

    features = np.concatenate((mean, std, quants))
    
    return features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calculate accuracy of model on test images.')    
    parser.add_argument('-i', '--image-dir', type=str, required=True,
                        help='Filepath to the folder/directory where the input images exist')
    
    parser.add_argument('-m', '--mask-dir', type=str, required=True,
                        help='Filepath to the folder/directory where the input masks exist')

    parser.add_argument('-o', '--model-dir', type=str, required=True,
                        help='Filepath to the folder/directory where the model exists and where the output will be stored')
                        
    parser.add_argument('--num-classes', type=int, default=2, help='The number of classes the model predicts')
                        
    args = parser.parse_args()
    image_filepath = args.image_dir
    mask_filepath = args.mask_dir
    model_filepath = args.model-dir
    n_classes = args.num_classes
    
    _, _, files = list(os.walk(image_filepath))[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Hyperparameters (ideally are the same as those used in the trained model)
    num_superpixels = 40000
    num_features = 11
    num_neighbours = 10
        
    acc = 0
    f1 = 0
    jac = 0
    cm = np.zeros((n_classes,n_classes))

    model = GCN(num_features, n_classes)
    model = model.to(device)

    model.load_state_dict(torch.load(os.path.join(model_filepath, "model.pt")))
    model.eval()


    output_file = open(os.path.join(model_filepath, "test_stats"), "w")
    sys.stdout = output_file
    
    
        
    for file in files:
        image = skimage.io.imread(os.path.join(image_filepath, file))
        mask = skimage.io.imread(os.path.join(mask_filepath, file))
        mask = mask//255
        image = zscore_normalize(image)
        
        
        superpixels = slic(image.astype('double')/np.max(image), n_segments = 40000, compactness = 0.1,sigma = 0, min_size_factor = 0.1)
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
        
        model.eval()
        out = model(data.to(device))
        out = model(data.to(device))
        y_pred = np.argmax(out.cpu().detach().numpy(), axis = 1)
        

        check = y_pred[superpixels].astype('int')
        
        y = mask.flatten()
        x = check.flatten()
        
        
        #acc += (np.sum(mask == check)/np.size(mask))

        acc+= sklearn.metrics.accuracy_score(y, x)

        f1+= sklearn.metrics.f1_score(y, x, average = 'macro')
 
        jac+= sklearn.metrics.jaccard_score(y, x, average = 'micro')

        cm += sklearn.metrics.confusion_matrix(y, x)

    print ("Accuracy -", acc/len(files))
    print ("F1 -", f1/len(files))
    print ("Jaccard -", jac/len(files))
    print("CM -", cm)
    
    output_file.close()
