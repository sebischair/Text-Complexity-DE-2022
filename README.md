# Text-Complexity-DE-2022
Code used for the Text Complexity DE Challenge 2022 (https://qulab.github.io/text_complexity_challlenge/).

## TUM sebis DE Text Complexity Challenge 2022
Here, you will find all the files needed to reproduce our approach followed the for challenge. In particular, `complex_train.py` is used to train a Gaussian Process Model, as well as run a given model on the dev and test sets. Likewise, `roberta_train.py` is used to fine-tune a RoBERTa model. Note that the resulting outputs of these two files must be averaged together to follow our approach exactly.

Another note: `result.zip` must be unzipped before running `complex_train.py`.
