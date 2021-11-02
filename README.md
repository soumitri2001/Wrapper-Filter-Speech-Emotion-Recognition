# Wrapper-Filter-Speech-Emotion-Recognition
Based on our paper **"A Hybrid Wrapper-Filter Deep Feature Selection Framework for Speech Emotion Recognition"** under review in _Multimedia Tools and Applications_, Springer.

## Overall Workflow
<img src="./full_method.png" style="margin: 5px;">

# Requirements
To install the required dependencies run the following in command prompt:
`pip install -r requirements.txt`

# Running the codes:
Required directory structure: (**"data"** directory contains class-wise **spectrograms** of the raw audio files in original dataset).

```

+-- data
|   +-- .
|   +-- train
|   +-- val
+-- PasiLuukka.py
+-- WOA_FS.py
+-- __init__.py
+-- audio2spectrogram.py
+-- main.py
+-- model.py

```
Then, run the code using the command prompt as follows:

`python main.py --data_dir "./data"`

**Available arguments:**
- `--num_epochs`: number of training epochs. Default = 100
- `--learning_rate`: learning rate for training. Default = 0.0005
- `--batch_size`: batch size for training. Default = 4
- `--optimizer`: optimizer for training: SGD / Adam. Default = "SGD"
