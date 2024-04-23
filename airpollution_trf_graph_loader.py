import json
import os
import urllib
import numpy as np
from torch_geometric_temporal.signal import StaticHeteroGraphTemporalSignal

class AirpollutionDatasetLoader(object):

    def __init__(self, city):
        self.city= city
        self._read_file_data()

    def _read_file_data(self):
        
        if self.city=='bilbao':
            with open(os.path.join('data', 'graph_structure', 'bb_graph.json')) as f:
                self._dataset = json.load(f)
        elif self.city=='madrid':
            with open(os.path.join('data', 'graph_structure', 'md_graph.json')) as f:
                self._dataset = json.load(f)
        else:
            raise ValueError()
        self.n_total_snapshots= self._dataset['n_snapshots']
        
    def _get_edges(self):
        self._edges={}
        for s, values in self._dataset["edges"].items():
            edge_name= tuple(s.split('_'))
            self._edges[edge_name]= np.array(values).T

    def _get_edge_weights(self):
        self._edge_weights={}
        for s, values in self._dataset["edges_attr"].items():
            edge_name= tuple(s.split('_'))
            self._edge_weights[edge_name]= np.array(values)

    def _get_targets_and_features(self):
        self.features=[]
        self.targets=[]
        self._feature_dim={}
                
        for snapshot_index in range(0,self.n_total_snapshots-self._T):
            feat_snapshot= self._dataset['snapshots'][snapshot_index]
            feat_snapshot_dict={}
            for node, values in feat_snapshot.items():
                values = np.array(values)
                            
                if node != 'trf':
                    values= values.reshape(-1,len(values))
                
                feat_snapshot_dict[node]=values
                #print(node, values, len(values))
                #print(values.shape)
                if node not in self._feature_dim:
                    self._feature_dim[node]=values.shape[1]
                
            self.features.append(feat_snapshot_dict)
            
            target_snapshot= self._dataset['snapshots'][snapshot_index+self._T]
            target_snapshot_dict={}
            for node, values in target_snapshot.items():
                values = np.array(values)
                            
                if node != 'trf':
                    values= values.reshape(-1,len(values))
     
                target_snapshot_dict[node]=values
            self.targets.append(target_snapshot_dict)
                
        
    def get_dataset(self, T= 1) -> StaticHeteroGraphTemporalSignal:
        """Returning the Spanish Airpollution data iterator.

        Args types:
            * **T** *(int)* - The time horizon.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Spanish Air Pollution dataset.
        """
        self._T = T
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticHeteroGraphTemporalSignal(self._edges, self._edge_weights, self.features, self.targets)
        return dataset
    
    def get_feature_dim(self):
        return self._feature_dim
    
    def get_n_total_snapshots(self):
        return self.n_total_snapshots
    