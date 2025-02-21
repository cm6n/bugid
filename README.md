# BUGID

This project aims to identify insects by sound. 

Inspired by [Merlin Bird ID](https://merlin.allaboutbirds.org/) which allows you to identify birds by several methods including recording their calls. From what I can find, there is no equivalent Sound ID for insects. Is it inherently harder or simply has not gotten enough attention? This is an experiment for my own amusement and education to see how good (or not) a classifier I could throw together quickly. It is far less sophisticated and reliable than the Merlin Bird sound identification feature in the Merlin app, but can be made better! See *Potential Improvements* below.

## Usage

Place recordings into the following directory structure:
```
bugsounds/
    Gryllus armatus/
        recording1.wav
        recording2.wav
        ...
    Gryllus assimilis/
        recording3.wav
        ...
```
The top-level directory name and recording file names can be anything you want. The names of the directories with the recordings will be treated as species names.

Train a model by pointing to a directory of recordings.
```
python .\cli.py train ..\..\bugsounds\ --model-output ../../mymodel.joblib
```

Make a Prediction given a recording (wav, mp3, etc):
```
python .\cli.py predict 'bugsounds_test\black_horned_tree_cricket_warm.mp3' --model ../../mymodel.joblib
```

## Training Data

Finding good training data is 90% of the battle. The bare bones scraper `bugsounddownloader.py` pointed at a website with recordings like https://orthsoc.org/sina/cricklist.htm downloads 1GB of audio files (all crickets!).

## Potential Improvements

Some initial (very informal) experiments with datasets like the cricket sound archive above, show poor accuracy. That's not surprising given a small set of recordings per specifies. Merlin, in contrast, doesn't offer Sound ID for bird species with fewer than 100 high quality recordings. They also have a network of volunteers helping to submit, review, and annotate recordings. Check out the [project overview](https://merlin.allaboutbirds.org/merlin-sound-id-project-overview/) and learn how you can help!

But enough about birds, let's get back to bugs! 

1. More data

   We're gonna need a ~~bigger boat~~ bigger training set! 
   
   Finding the data is the hard part. On the plus side, we could go a lot bigger and still train locally even on a typical laptop. The training over 1 GB of audio went quite quickly - props to sklearn, librosa, and numpy for cranking through that on my laptop in less time than it took to make coffee.

2. Features matter

   Well, some of them do, anyway. Before blindly throughing more data at the problem, I probably should give a little more thought to _which_ features from the audio might help distinguish species.

3. Geographic info

   Honestly, this should be number 1! Not using geographic info to narrow down the search is likely making this much harder, if not impossible. There may be specicies with identical calls but distinct habitat ranges that we could easily distinguish if we had location information. Which brings me back to... finding good training data is 90% of the battle.

   Returning for a second look at https://orthsoc.org/sina reveals there is geographic information available for that dataset in a MS Access 2000 database along with the audio. So at least for _that_ site, I suppose I didn't need the scraper after all though frankly it was more fun than using Access. Now, if I can find range info for more than just crickets and cicadas, I might start getting somewhere.
