import torch.utils.data as data
import h5py

class HDF5Dataset(data.Dataset):                
    """docstring for HDF5Dataset"""         
    def __init__(self,                                                          
                 h5file,                                                           
                 label_df,                                                                              
                 transform,            
                 colname=("index", "encoded", "filename")):

        super(HDF5Dataset, self).__init__()                                     
        self._h5file = h5file                                            
        self._label_df = label_df                               
        self._dataset = None                                     
        self._transform = transform                                        
        self._colname = colname         
                                                          
    def __getitem__(self, index):

        if self._dataset is None:
            self._dataset = h5py.File(self._h5file, "r")

        h5_index, target, fname = self._label_df.iloc[index].reindex(self._colname).values

        data = self._dataset[str(h5_index)][()]

        if self._transform:
            data = self._transform(data)

        return data, target, fname



    def __len__(self):
        return len(self._label_df)


class SubsetSequentialSampler(data.Sampler):
    r"""Samples elements sequentially from a given list of indices, always in the same order.

    Arguments:
        indices (sequence): s sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
