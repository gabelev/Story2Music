from model import Story2MusicTransformer
import torch
import argparse
from miditok import REMI, TokenizerConfig
from tqdm import tqdm
from pathlib import Path
from model import Story2MusicTransformer
from dataset import StoryMidiDataset
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

def ensure_output_dir():
    """
    Check if output directory exists, create it if it doesn't.
    """
    output_dir = Path("generated_midi")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print("Created generated_midi directory")
    return output_dir

def read_story_from_file(file_path):
    """
    Read story from a text file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate MIDI music from a story')
    parser.add_argument('--story', type=str, help='Direct story text input')
    parser.add_argument('--story_file', type=str, help='Path to text file containing the story')
    parser.add_argument('--output_name', type=str, default='generated_song', help='Name for the output MIDI file (without extension)')
    args = parser.parse_args()

    # Validate input arguments
    if not args.story and not args.story_file:
        parser.error("Either --story or --story_file must be provided")
    if args.story and args.story_file:
        parser.error("Cannot provide both --story and --story_file")

    # Get story text
    if args.story:
        story = args.story
    else:
        story = read_story_from_file(args.story_file)

    # Ensure output directory exists
    output_dir = ensure_output_dir()
    
    # Load model
    model = Story2MusicTransformer("bert-base-uncased", midi_vocab_size=30000)
    model.load_state_dict(torch.load("saved_models/custom_transformer.pth"))
    model.eval()

    tokenizer_params = {
        "pitch_range": (21, 108),  # MIDI range for piano keys
        "beat_res": {(0, 4): 8, (4, 12): 4},
        "num_velocities": 32,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"]
    }

    config = TokenizerConfig(**tokenizer_params)
    midi_tokenizer = REMI(config)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer(story, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate
    generated = model.generate(input_ids, attention_mask, 4, 3, max_len=50)
    
    # Convert tokens to MIDI file
    midi_tokens = generated.squeeze().tolist()
    midi_obj = midi_tokenizer.decode(midi_tokens)
    
    # Save MIDI file
    output_path = output_dir / f"{args.output_name}.mid"
    midi_obj.dump(output_path)
    print(f"Generated MIDI file saved to: {output_path}")

if __name__ == "__main__":
    main()
