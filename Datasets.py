from torch.utils.data import Dataset
import torch
import numpy as np

# Define the custom class for accessing our dataset
class DatasetEnsemble(Dataset):
    def __init__(self, stft_file_list,raw_file_list, labels, transform=None, verbose = False):
        self.stft_file_list = stft_file_list
        self.raw_file_list = raw_file_list
        self.labels=labels
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # returns a training sample and its label
        stft_file_path = self.stft_file_list[idx]
        raw_file_path = self.raw_file_list[idx]
        label = torch.tensor(self.labels[idx])
        stft_vector = torch.tensor(np.load(stft_file_path)) #load from file
        raw_vector=torch.tensor(np.load(raw_file_path)) 
                
        return [stft_vector,raw_vector], label


# Define the custom class for accessing our dataset
class DatasetRGB(Dataset):
    def __init__(self, file_list, labels, transform=None, verbose = False):
        self.file_list = file_list
        self.labels=labels
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # returns a training sample and its label
        file_path = self.file_list[idx]
        label = torch.tensor(self.labels[idx])
        stft_vector = torch.tensor(np.load(file_path)) #load from file
        
        # Normalize your data here
        if self.transform:
            if(self.verbose==True):
                print("TRANSFORM: applying transform to tensor shape:",stft_vector.shape,"content:",stft_vector)
            stft_vector = self.transform(torch.unsqueeze(stft_vector, dim=0)) #unsqueeze needed for the torchvision normalize method
            if(self.verbose==True):
                print("TRANSFORM: after transform shape:",stft_vector.shape,"content:",stft_vector)
            stft_vector = torch.squeeze(stft_vector, dim=0)
            if(self.verbose==True):
                print("TRANSFORM: after squeeze shape:",stft_vector.shape,"content:",stft_vector)
                
        #do ResNet18 normalization:
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        #copy the channel 3 times (need to unsqueeze to create a new dimension first)
        #print("DATASET*  sample shape is:",stft_vector.shape,"content:",stft_vector)
        stft_vector = stft_vector.unsqueeze(0).repeat(3,1,1)
        stft_vector = stft_vector.to(torch.float32) #float32 needed for ResNet18 model (downcast from float64)
        #print("DATASET* sample shape after repeat is:",stft_vector.shape,"content:",stft_vector)
        #print("stft_vector dtype:",stft_vector.dtype)

        
        return stft_vector, label


class DatasetRaw(Dataset):
    def __init__(self, file_list, labels, transform=None, verbose=False):
        self.file_list = file_list
        self.labels=labels
        self.transform = transform
        self.verbose=verbose

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = torch.tensor(self.labels[idx])
        raw_vector = np.load(file_path).astype(np.int16) # Ensure int16 data type
        if(self.verbose==True):
            print("raw vector shape:",raw_vector.shape)
        raw_vector = torch.tensor(raw_vector)
        
        # Normalize your data here
        if self.transform:
            
            #convert to float64 tensor
            raw_vector = raw_vector.double()
            if(self.verbose==True):
                print("TRANSFORM: applying transform to tensor shape:",raw_vector.shape,"content:",raw_vector)
            raw_vector = torch.unsqueeze(raw_vector, dim=0)
            #print("TRANSFORM: after first unsqueeze:",raw_vector.shape,"content:",raw_vector)
            raw_vector = torch.unsqueeze(raw_vector, dim=0) #unsqueeze two times (needed for torchvision normalize method)
            #print("TRANSFORM: after second unsqueeze:",raw_vector.shape,"content:",raw_vector)
            raw_vector = self.transform(raw_vector) #normalize the sample
            if(self.verbose==True):
                print("TRANSFORM: after transform shape:",raw_vector.shape,"content:",raw_vector)
            raw_vector = torch.squeeze(raw_vector, dim=0)
            raw_vector = torch.squeeze(raw_vector, dim=0)
            if(self.verbose==True):
                print("TRANSFORM: after double squeeze shape:",raw_vector.shape,"content:",raw_vector)
        
        return raw_vector, label
    
# Define the custom class for accessing our dataset
class DatasetSTFT(Dataset):
    def __init__(self, file_list, labels, transform=None, verbose = False):
        self.file_list = file_list
        self.labels=labels
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # returns a training sample and its label
        file_path = self.file_list[idx]
        label = torch.tensor(self.labels[idx])
        stft_vector = torch.tensor(np.load(file_path)) #load from file
        
        # Normalize your data here
        if self.transform:
            if(self.verbose==True):
                print("TRANSFORM: applying transform to tensor shape:",stft_vector.shape,"content:",stft_vector)
            stft_vector = self.transform(torch.unsqueeze(stft_vector, dim=0)) #unsqueeze needed for the torchvision normalize method
            if(self.verbose==True):
                print("TRANSFORM: after transform shape:",stft_vector.shape,"content:",stft_vector)
            stft_vector = torch.squeeze(stft_vector, dim=0)
            if(self.verbose==True):
                print("TRANSFORM: after squeeze shape:",stft_vector.shape,"content:",stft_vector)

        
        return stft_vector, label