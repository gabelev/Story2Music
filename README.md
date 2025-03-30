First start by cloning this repository.

In order to construct the midi dataset, download the EMOPIA dataset from here:
```
wget https://zenodo.org/records/5090631/files/EMOPIA_1.0.zip
```
Then unzip the file:

run ``` unzip EMOPIA_1.0.zip ``` in the terminal.
now the midi files should be under a folder called EMOPIA_1.0.

The ```story_midi_matched.csv``` dataframe contains stories with their corresponding midi_id based on emotions. The dataset object reads the midi_ids for each story and reads the midi file with that ID and tokenizes it and stores it in the dataset object.

Create a python environment and install the dependencies:

```
pyenv virtualenv 3.13.0 story2music
pyenv activate
pip install -r requirements.txt
```

Once you are set up and the data is downloaded, to train the model, run:

``` python train.py --model_name "bert-base-uncased" ```

To play a midifile from command line:

```
python play_midi.py <path/to/midi/file.mid>
```

For example:
```
python play_midi.py EMOPIA_1.0/midis/Q1_0vLPYiPN7qY_0.mid
```

To generate MIDI music from a story, you can use generate.py in two ways:

1. Using direct story text:
```
python generate.py --story "Your story text here" --output_name "my_song"
```

2. Using a text file containing the story:
```
python generate.py --story_file "path/to/your/story.txt" --output_name "my_song"
```

The generated MIDI file will be saved in the `generated_midi` directory with the specified output name (defaults to "generated_song.mid" if no name is provided).

You can then play the generated MIDI file using play_midi.py:
```
python play_midi.py generated_midi/my_song.mid
```