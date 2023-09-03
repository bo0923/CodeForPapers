# Code for paper :

## Data Preprocess
  The code for converting the FOL into graph is in utils.py for both FOLIO and Mizar40, and an additional pre-process for converting the Mizar40 into a normal FOL is located in `tptp_preprocess.py`.

## Train
  run `train.py` respectively in the FOLIO and DeepMath folder to train model and the validation/test are carried out simultaneously.

  The configure can be setted in `config.py` or you can set parameters directly in the command line.
## Prompt
  code for test on GPT3.5 is `prompot.py` and you need replace your own key.

