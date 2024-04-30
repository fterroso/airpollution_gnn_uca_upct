import json
import os
import urllib
import numpy as np
from torch_geometric_temporal.signal import StaticHeteroGraphTemporalSignal

class AirpollutionDatasetLoader(object):

    def __init__(self, city, include_trf=True, synth=False):
        self.city= city
        self._include_trf= include_trf
        self._synth= synth
        self._read_file_data()
        

    def _read_file_data(self):
        synth_str=""
        if self._synth:
            synth_str= "_synth"

        city_str=""
        if self.city=='bilbao':
            city_str='bb'
        elif self.city=='madrid':
            city_str='md'
        else:
            raise ValueError()
        with open(os.path.join('data', 'graph_structure', f'{city_str}_graph{synth_str}.json')) as f:
                    self._dataset = json.load(f)
           
        self.n_total_snapshots= self._dataset['n_snapshots']
        print(self.n_total_snapshots)
        
    def _get_edges(self):
        self._edges={}
        for s, values in self._dataset["edges"].items():
            edge_name= tuple(s.split('_'))
            if self._include_trf or ('trf' not in edge_name):
                self._edges[edge_name]= np.array(values).T

    def _get_edge_weights(self):
        self._edge_weights={}
        for s, values in self._dataset["edges_attr"].items():
            edge_name= tuple(s.split('_'))
            if self._include_trf or ('trf' not in edge_name):
                self._edge_weights[edge_name]= np.array(values)

    def _get_targets_and_features(self):
        self.features=[]
        self.targets=[]
        self._feature_dim={}
                
        for snapshot_index in range(0,self.n_total_snapshots-self._T):
            feat_snapshot= self._dataset['snapshots'][snapshot_index]
            feat_snapshot_dict={}
            for node, values in feat_snapshot.items():
                if self._include_trf or ('trf' != node):
                    values = np.array(values)
                                
                    if node != 'trf':
                        values= values.reshape(-1,len(values))
                    
                    feat_snapshot_dict[node]=values
                    if node not in self._feature_dim:
                        self._feature_dim[node]=values.shape[1]
                
            self.features.append(feat_snapshot_dict)
            
            target_snapshot= self._dataset['snapshots'][snapshot_index+self._T]
            target_snapshot_dict={}
            for node, values in target_snapshot.items():
                if self._include_trf or ('trf' != node):
                    values = np.array(values)
                                
                    if node != 'trf':
                        values= values.reshape(-1,len(values))
         
                    target_snapshot_dict[node]=values
            self.targets.append(target_snapshot_dict)

    def _get_columns_names(self):
        self._column_names= {}
        for k,v in self._dataset['columns_ids'].items():
            if self._include_trf or ('trf' != k):
                sorted_v = sorted(v.items(), key=lambda x: x[1])
                ordered_columns = [item[0] for item in sorted_v]
                self._column_names[k]= ordered_columns
        
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
        self._get_columns_names()
        dataset = StaticHeteroGraphTemporalSignal(self._edges, self._edge_weights, self.features, self.targets)
        return dataset
    
    def get_feature_dim(self):
        return self._feature_dim
    
    def get_n_total_snapshots(self):
        return self.n_total_snapshots

    def get_column_names(self, node_type):
        return self._column_names[node_type]
        