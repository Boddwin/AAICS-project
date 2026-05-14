from sklearn import mixture
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import os


def remove_previous_models(models_folder, ubm_folder):
    #delete previous models (.gmm files first, ubm second)
    files = os.listdir(models_folder)
    for file in files:
        file_path = os.path.join(models_folder, file)
        os.remove(file_path)
    ubmpath = os.path.join(ubm_folder, "ubm.gmm")
    if os.path.isfile(ubmpath):
        os.remove(ubmpath)

def create_templated(audio_files, ubm_folder, models):
    #Step 1: Read all folders in 'audio files'. Each folder contains audio samples for a person that we want to build a model for
        
    dir_list = os.listdir(audio_files) 
    ubm_train = np.asarray(())
    for folder in dir_list:  
        folder_path = os.path.join(audio_files, folder)
        if os.path.isdir(folder_path):
            #Step 2: get all the audio files and build a model
            file_list = os.listdir(folder_path)
            features = np.asarray(()) #empty feature set
            for file in file_list: 
                # Skip non-audio files (e.g., .DS_Store)
                if not file.endswith('.wav'):
                    continue
                
                #Step 3: open the audio files and extract the features. Note: all files in the same folder are assumed to belong to the same person
                file_path = os.path.join(folder_path, file)
                sr,audio = read(file_path)
                vector   = extract_features(audio,sr)

                if features.size == 0:
                    features = vector
                else:
                    features = np.vstack((features, vector))
                if ubm_train.size == 0:
                    ubm_train = vector
                else:
                    ubm_train = np.vstack((ubm_train, vector))		
        else:
            # Skip non-audio files
            if not folder.endswith('.wav'):
                continue
                
            sr,audio = read(folder_path)
            vector   = extract_features(audio,sr)
            features = vector
            if ubm_train.size == 0:
                ubm_train = vector
            else:
                ubm_train = np.vstack((ubm_train, vector))		
        #Step 4: once all files processed for a person, build a model and save to file.
        gmm = mixture.GaussianMixture(n_components = 16,  covariance_type='diag')
        gmm.fit(features)
        
        picklefile = folder+".gmm"
        pickle.dump(gmm,open(os.path.join(models, picklefile),'wb'))
        if os.path.isdir(folder_path):
            print ('Modelling completed for speaker:',picklefile," with data point = ",features.shape, " from ",len(file_list)," audio files") 
        else:
            print ('Modelling completed for speaker:',picklefile," with data point = ",features.shape, " from 1 audio file") 

    #Step 5: generate and output UBM
    ubm = GaussianMixture(n_components = 16, covariance_type='diag')
    ubm.fit(ubm_train)
    pickle.dump(ubm,open(os.path.join(ubm_folder, "ubm.gmm"),'wb'))
    print("FINISHED")




#paths
audio_files_folder = os.path.join("..", "data", "1-to-1", "train")


models_folder = "speaker_models"
ubm_folder = "ubm_model"

#functions to call in order
remove_previous_models(models_folder, ubm_folder)
create_templated(audio_files_folder, ubm_folder, models_folder)
