import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time
from sklearn import preprocessing as p 
from statistics import mean 

def process_audio_match_templates_with_labels(models_folder, audio_files_folder, ubm_file, threshold):
    # Step 1: load in speaker models (.gmm files)
    gmm_files = [os.path.join(models_folder,fname) for fname in
                  os.listdir(models_folder) if fname.endswith('.gmm')]

    if not gmm_files:
        print("ERROR: No GMM files found in", models_folder)
        return None

    ubm = pickle.load(open(ubm_file,'rb'))

    #Load the Gaussian Mixture Models
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers  = [os.path.splitext(os.path.basename(fname))[0] for fname in gmm_files]

    correct = 0
    incorrect = 0
    notmatched = 0
    total_processed = 0

    # Step 2: Read all folders in 'test'. Each folder contains audio samples for a person that we want to test
    for it in os.scandir(audio_files_folder):
        if it.is_dir(): # Assumed directory so process all files inside
            print ("Reading data for "+it.name)
            # Step 3: get all the audio files and build a model
            file_list = os.listdir(it.path)
            features = np.asarray(()) # Empty feature set
            audio_count = 0
            for file in file_list:
                # Skip non-audio files (like .DS_Store)
                if not file.endswith('.wav'):
                    continue
                file_path = os.path.join(it.path, file)
                try:
                    # Step 3.1: open the audio files and extract the features. Note: all files in the same folder are assumed to belong to the same person
                    sr, audio = read(file_path)
                    vector   = extract_features(audio, sr)

                    if features.size == 0:
                        features = vector
                    else:
                        features = np.vstack((features, vector))
                    audio_count += 1
                except Exception as e:
                    print(f"  WARNING: Failed to process {file}: {e}")
                    continue

            if features.size == 0:
                print(f"  WARNING: No valid audio features extracted for {it.name}")
                continue

        else: # Assumed single files to match
            # Skip non-audio files
            if not it.name.endswith('.wav'):
                continue
            features = np.asarray(()) # Empty feature set
            try:
                sr, audio = read(it.path)
                vector = extract_features(audio, sr)
                features = vector
                audio_count = 1
            except Exception as e:
                print(f"  WARNING: Failed to process {it.name}: {e}")
                continue

        # Step 5: once all files processed for a person, now it is time to do the comparison
        gmm_s = np.zeros(len(models))
        ubm_s = np.zeros(len(models))
        sc = np.zeros(len(models)) # Store scores here
        for i in range(len(models)):
            gmm = models[i]
            gmm_s[i] = gmm.score(features)
            ubm_s[i] = ubm.score(features)
            sc[i] = np.array(gmm_s[i] - ubm_s[i])

        # Step 5: identify which is the best. decision boundary 0 (<0 reject, >1 accept)
        best_idx = np.argmax(sc)
        total_processed += 1

        if sc[best_idx] > 0 and gmm_s[best_idx] > threshold:
            # If else to handle correct/incorrect matches
            if it.name == speakers[best_idx]:
                correct += 1
                print(f"  ✓ CORRECT: {it.name} matched with {speakers[best_idx]} (GMM={gmm_s[best_idx]:.2f}, UBM={ubm_s[best_idx]:.2f}, score={sc[best_idx]:.2f})")
            else:
                incorrect += 1
                print(f"  ✗ INCORRECT: {it.name} matched with {speakers[best_idx]} (GMM={gmm_s[best_idx]:.2f}, UBM={ubm_s[best_idx]:.2f}, score={sc[best_idx]:.2f})")
        else:
            notmatched += 1
            best_match = speakers[best_idx]
            print(f"  - NO MATCH: {it.name} (best candidate: {best_match}, score={sc[best_idx]:.2f})")

    # Calculate False Non-Match Rate (FNMR)
    total = correct + incorrect + notmatched
    fnmr = notmatched / total if total > 0 else 0.0
    fmr = incorrect / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total Processed:       {total_processed}")
    print(f"Correct Matches:       {correct}")
    print(f"Incorrect Matches:     {incorrect}")
    print(f"No Match Found:        {notmatched}")
    print(f"Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"FNMR (No Match Rate):  {fnmr:.4f}")
    print(f"FMR (False Match Rate): {fmr:.4f}")
    print("="*60)

    return {
        'total': total_processed,
        'correct': correct,
        'incorrect': incorrect,
        'notmatched': notmatched,
        'accuracy': accuracy,
        'fnmr': fnmr,
        'fmr': fmr
    }

