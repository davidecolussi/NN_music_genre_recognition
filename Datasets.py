from torch.utils.data import Dataset
import torch
import numpy as np
import os
import utils 
from tkinter import Tcl # file sorting by name

def create_single_dataset(folder_path, tracks_dataframe, genre_dictionary):    
    labels = []
   
    _, file_list = get_sorted_file_paths(folder_path)
    
    for i,file in enumerate(file_list):
        #print("considering file:",file, "({}/{})".format(i,len(file_list)))
        track_id_clip_id = file.split('.')[0]
        track_id = track_id_clip_id.split('_')[0]
        #print("track id with clip: {}, track id: {}".format(track_id_clip_id, track_id))
        genre = tracks_dataframe.loc[int(track_id)]
        #print("genre from dataframe: ", genre)
        label = genre_dictionary[genre]
        #print("label from dictionary:",label)
        labels.append(label)
    print("labels length: {}".format(len(labels)))
    return labels
    

#create the train,validation and test vectors using the files in the train/validation/test folders
def create_dataset_splitted(folder_path):
    train_folder = os.path.join(folder_path,'train') # concatenate train folder to path
    validation_folder = os.path.join(folder_path,'validation') # concatenate train folder to path
    test_folder = os.path.join(folder_path,'test') # concatenate train folder to path
    
    print("train_folder:",train_folder)
    print("validation_folder:",validation_folder)
    print("test_folder:",test_folder,"\n")
    
    AUDIO_DIR = os.environ.get('AUDIO_DIR')
    print("audio directory: ",AUDIO_DIR)
    print("Loading tracks.csv...")
    tracks = utils.load('data/fma_metadata/tracks.csv')
    
    #get only the small subset of the dataset
    small = tracks[tracks['set', 'subset'] <= 'small']
    print("small dataset shape:",small.shape)    

    small_training = small.loc[small[('set', 'split')] == 'training']['track']
    small_validation = small.loc[small[('set', 'split')] == 'validation']['track']
    small_test = small.loc[small[('set', 'split')] == 'test']['track']

    print("Track.csv: {} training samples, {} validation samples, {} test samples\n".format(len(small_training), len(small_validation), len(small_test)))

    small_training_top_genres = small_training['genre_top']
    small_validation_top_genres = small_validation['genre_top']
    small_test_top_genres = small_test['genre_top']
    
    #create dictionary of genre classes:
    unique_genres = small_training_top_genres.unique()
    unique_genres = np.array(unique_genres)
    print("there are {} unique genres".format(len(unique_genres)))
    genre_dictionary = {}
    for i,genre in enumerate(unique_genres):
        genre_dictionary[genre] = i
    print("Dictionary of genres created:",genre_dictionary)
    
    
    Y_train = create_single_dataset(train_folder, small_training_top_genres, genre_dictionary)
    Y_validation = create_single_dataset(validation_folder, small_validation_top_genres, genre_dictionary)
    Y_test = create_single_dataset(test_folder, small_test_top_genres, genre_dictionary)
    
    return Y_train, Y_validation, Y_test
 
def get_sorted_file_paths(folder_path):
    file_list = os.listdir(folder_path)
    #sort the dataset files in alphabetical order (important to associate correct labels created using track_id in track.csv)
    file_list = Tcl().call('lsort', '-dict', file_list) # sort file by name: 2_0,2_1, ... 2_9,3_0, ... 400_0,400_1, ...
    file_paths = [os.path.join(folder_path, file_name) for file_name in file_list] #join filename with folder path
    #print("There are {} in the folder: {}".format(len(file_list),file_list))
    return file_paths, file_list



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