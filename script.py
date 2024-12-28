import os

def combine_text_files(input_folder, output_file):
    # Get all text files from the folder
    text_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    # Sort files to ensure consistent ordering
    text_files.sort()
    
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Iterate over each text file
        for text_file in text_files:
            file_path = os.path.join(input_folder, text_file)
            try:
                # Open each text file in read mode
                with open(file_path, 'r', encoding='utf-8') as infile:
                    # Read the content and write it to the output file
                    content = infile.read()
                    outfile.write(content + "\n")
                    print(f"Added: {text_file}")
            except Exception as e:
                print(f"Error processing {text_file}: {str(e)}")
    
    print(f"\nSuccessfully created combined text file: {output_file}")

if __name__ == "__main__":
    input_folder = "audio_calls"  # Your input folder
    output_file = "combined_transcript.txt"  # Your desired output file
    
    combine_text_files(input_folder, output_file)