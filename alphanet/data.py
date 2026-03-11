import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import joblib
from torch_geometric.data import Data, DataLoader, InMemoryDataset, download_url, extract_zip

def get_pic_datasets(root, name, config):
   
    if config.train_dataset is not None and config.valid_dataset is not None and config.test_dataset is not None:
      
        train_dataset = CustomPickleDataset(name=config.train_dataset, root=root, config = config)
        valid_dataset = CustomPickleDataset(name=config.valid_dataset, root=root, config = config)
        test_dataset = CustomPickleDataset(name=config.test_dataset, root=root, config = config)

        if config.train_size is not None:
            train_indices = train_dataset.get_idx_split(data_size=len(train_dataset.data.y), train_size=config.train_size, seed=config.seed)
            train_dataset = train_dataset[train_indices]
        if config.valid_size is not None:
            valid_indices = valid_dataset.get_idx_split(data_size=len(valid_dataset.data.y), valid_size=config.valid_size, seed=config.seed)
            valid_dataset = valid_dataset[valid_indices]
        if config.test_size is not None:
            test_indices = test_dataset.get_idx_split(data_size=len(test_dataset.data.y), test_size=config.test_size, seed=config.seed)
            test_dataset = test_dataset[test_indices]
    else:
      
        dataset = CustomPickleDataset(name=name, root=root, config=config)

 
        split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=config.train_size, valid_size=config.valid_size, test_size=config.test_size, seed=config.seed)
        train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    return train_dataset, valid_dataset, test_dataset


class CustomPickleDataset(InMemoryDataset):
    r"""
    Where the attributes of the output data indicates:
    * :obj:`z`: The atom type.
    * :obj:`pos`: The 3D position for atoms.
    * :obj:`y`: The property (energy) for the graph (molecule).
    * :obj:`force`: The 3D force for atoms.
    * :obj:`cell`: The size of the periodic box.
    * :obj:`natoms`: The number of atoms.
    * :obj:`stress`: The stress of the system.
    * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs
    """
    def __init__(self, root='dataset/', name='custom', transform=None, pre_transform=None, pre_filter=None, config = None):
        self.name = name
        self.folder = osp.join(root, self.name)
        self.precision = torch.float32 if config.dtype == "32" else torch.float64
        self.config = config
        
        super(CustomPickleDataset, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.pkl') or f.endswith('.pickle')]

    @property
    def processed_file_names(self):
        return f'{self.name}_pyg.pt'

    def process(self):
        data_list = []
        for raw_file_name in self.raw_file_names:
      
            data_path = osp.join(self.raw_dir, raw_file_name)
            if not osp.exists(data_path):
                raise FileNotFoundError(f"{data_path} does not exist. Please ensure the raw data file is placed correctly.")
            with open(data_path, 'rb') as f:
                data_dict = joblib.load(f)
            E = data_dict['E']
            F = data_dict['F']
            R = data_dict['R']
            z = data_dict['z']
            cell = data_dict['cell']
            natoms = data_dict['natoms']
       
            # Check if stress exists in the data
            has_stress = 'stress' in data_dict
            if has_stress:
                S = data_dict['stress']

            for i in tqdm(range(len(E))):
                R_i = torch.tensor(R[i], dtype=self.precision)
                z_i = torch.tensor(z[i], dtype=torch.int64)
                E_i = torch.tensor([E[i]], dtype=self.precision)  
                F_i = torch.tensor(F[i], dtype=self.precision)
                cell_i = torch.tensor(cell[i], dtype=self.precision)
                natoms_i = torch.tensor([natoms[i]], dtype=torch.int64)
                
                # Create data object with or without stress
                data_dict = {
                    'pos': R_i,
                    'z': z_i,
                    'y': E_i,
                    'force': F_i,
                    'cell': cell_i,
                    'natoms': natoms_i
                }
                
                if has_stress:
                    S_i = torch.tensor(S[i], dtype=self.precision)
                    data_dict['stress'] = S_i
                #if self.config.compute_forces:
                 # data_dict['pos'].requires_grad = True
                data = Data(**data_dict)
                
                data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
      
        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    #def get_idx_split(self, data_size, train_size=None, valid_size=None, seed=None):
    def get_idx_split(self, data_size, train_size=None, valid_size=None, test_size=None, seed=None):
      ids = shuffle(list(range(data_size)))
      if train_size is not None and valid_size is None:
          train_idx = ids[:train_size]
          
          return train_idx
      if valid_size is not None and train_size is None:
          valid_idx = ids[:valid_size]
          return valid_idx
  
      if train_size is not None and valid_size is not None:
          train_idx, valid_idx = train_test_split(ids, train_size=train_size, test_size=valid_size, random_state=seed)
          test_idx = [idx for idx in ids if idx not in train_idx and idx not in valid_idx]
          split_dict = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
          return split_dict

      return {'train': [], 'valid': [], 'test': []}


    

