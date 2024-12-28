from pydub import AudioSegment
import os

def combine_audio_files(input_folder, output_file):
    # Get all audio files from the folder
    audio_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp3', '.wav', '.m4a'))]
    
    # Sort files to ensure consistent ordering
    audio_files.sort()
    
    # Initialize combined audio
    combined = AudioSegment.empty()
    
    # Combine all audio files
    for audio_file in audio_files:
        file_path = os.path.join(input_folder, audio_file)
        try:
            # Load the audio file
            audio = AudioSegment.from_file(file_path)
            # Append to combined audio
            combined += audio
            print(f"Added: {audio_file}")
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    # Export the combined audio
    combined.export(output_file, format=output_file.split('.')[-1])
    print(f"\nSuccessfully created combined audio file: {output_file}")

if __name__ == "__main__":
    input_folder = "audio_calls"  # Your input folder
    output_file = "combined_audio.mp3"  # Your desired output file
    
    combine_audio_files(input_folder, output_file)