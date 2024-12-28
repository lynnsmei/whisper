import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
from tqdm import tqdm

def setup_whisper():
    """Setup the Whisper model with optimal configurations"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "openai/whisper-large-v3-turbo"
    
    # Load the model with optimal settings
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Create pipeline with chunked processing for longer audio
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    return pipe

def transcribe_audio(pipe, audio_path):
    """Transcribe a single audio file"""
    try:
        print(f"Transcribing: {audio_path}")
        
        # Configure generation parameters
        generate_kwargs = {
            "language": "ru",
            "task": "transcribe",
            "return_timestamps": True,
            "num_beams": 1,
            "no_speech_threshold": 0.6,
        }
        
        # Transcribe
        result = pipe(
            audio_path,
            generate_kwargs=generate_kwargs
        )
        
        # Save transcription
        output_file = os.path.splitext(audio_path)[0] + "_transcript.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
            
            # Save timestamps if available
            if "chunks" in result:
                f.write("\n\nTimestamps:\n")
                for chunk in result["chunks"]:
                    f.write(f"\n[{chunk['timestamp']}] {chunk['text']}")
        
        print(f"Transcription saved to: {output_file}")
        return result["text"]
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None

def batch_transcribe(input_folder):
    """Transcribe all audio files in a folder"""
    # Setup Whisper
    pipe = setup_whisper()
    
    # Get all audio files
    audio_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp3', '.wav', '.m4a'))]
    
    if not audio_files:
        print("No audio files found!")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each file
    for audio_file in tqdm(audio_files, desc="Processing files"):
        file_path = os.path.join(input_folder, audio_file)
        print(f"\nProcessing: {audio_file}")
        transcribe_audio(pipe, file_path)

def combine_transcripts(input_folder):
    """Combine all transcript files into one"""
    print("Combining all transcript files...")
    transcript_files = [f for f in os.listdir(input_folder) if f.endswith('_transcript.txt')]
    
    with open("combined_transcript.txt", "w", encoding="utf-8") as outfile:
        for transcript_file in sorted(transcript_files):
            file_path = os.path.join(input_folder, transcript_file)
            outfile.write(f"\n=== {transcript_file} ===\n\n")
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + "\n")

if __name__ == "__main__":
    input_folder = "audio_calls"
    
    print("Starting Russian speech-to-text transcription...")
    batch_transcribe(input_folder)
    combine_transcripts(input_folder)