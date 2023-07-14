import torch
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GraphConv
from torch.nn import Linear, ReLU,CrossEntropyLoss

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes ):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.n_hidden = 32
        
        self.conv_first = GraphConv(num_node_features, self.n_hidden)

        self.conv1 = GraphConv(self.n_hidden, self.n_hidden)
        #self.conv2 = GraphConv(self.n_hidden, self.n_hidden)
        #self.conv3 = GraphConv(self.n_hidden, self.n_hidden)
        #self.conv4 = GraphConv(self.n_hidden, self.n_hidden)
        #self.conv5 = GraphConv(self.n_hidden, self.n_hidden)
        #self.conv6 = GraphConv(self.n_hidden, self.n_hidden)
        #self.conv7 = GraphConv(self.n_hidden, self.n_hidden)
        #self.conv8 = GraphConv(self.n_hidden, self.n_hidden)
        #self.conv9 = GraphConv(self.n_hidden, self.n_hidden)

        self.conv_last = GraphConv(self.n_hidden, num_classes)

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_weights

        x = self.conv_first(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        
        x = self.conv1(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        '''x = self.conv2(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv4(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv5(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv6(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv7(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv8(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv9(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)'''
        
        x = self.conv_last(x, edge_index, edge_weights)

        return x
