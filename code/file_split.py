from pydub import AudioSegment
import os

def split_and_save_audio(input_file, output_folder):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Duration of each segment in milliseconds (1 minute = 60,000 milliseconds)
    segment_duration = 10000 

    # Get the total duration of the audio in milliseconds
    total_duration = len(audio)

    # Calculate the number of segments
    num_segments = total_duration // segment_duration
    #print('The audio files are split into' +total_duration +' smaller audios of 60 seconds')

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Split the audio into segments
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration

        # Extract the segment
        segment = audio[start_time:end_time]

        # Save the segment to the output folder
        output_file = os.path.join(output_folder, f"segment_{i + 1}.wav")
        segment.export(output_file, format="wav")

def process_folder(input_folder, output_parent_folder):
    # Iterate through files and subfolders in the input folder
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Check if the file is an audio file (you can add more file extensions if needed)
            if file.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a', '.mp4')):
                print("processing file: ", file)
            #if file.lower().endswith(('.wav')):
                # Create the corresponding output subfolder
                output_subfolder = os.path.join(output_parent_folder, os.path.relpath(root, input_folder))
                output_subfolder = os.path.normpath(output_subfolder)

                # Process the audio file and save segments in the corresponding subfolder
                input_file = os.path.join(root, file)
                print("input file: ", input_file)
                print("output location: ", output_subfolder)
                split_and_save_audio(input_file, output_subfolder)


# Replace 'input_parent_folder' with the path to the parent folder containing subfolders with audio files
input_parent_folder = os.path.join("..", "data", "large_mixed", "minutes2")
# Replace 'output_parent_folder' with the desired parent output folder
output_parent_folder = os.path.join("..", "data", "large_mixed", "testing")
process_folder(input_parent_folder, output_parent_folder)
