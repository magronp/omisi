# Online Spectrogram inversion for audio source separation (OMISI)

Here is the code related to the article entitled [Online Spectrogram Inversion for Low-Latency Audio Source Separation](https://arxiv.org/abs/1911.03128).
Audio examples of recovered source signals are available on the [companion website](https://magronp.github.io/demos/spl20_omisi.html).

# Setup

The needed packages are listed in `requirements.txt`.
For convenience, you can create a virtual environment and automatically install them as follows:

    python3 −m venv env
    source env/bin/activate
    pip3 install -r requirements.txt
    
The paper uses the [Danish HINT dataset](https://www.ncbi.nlm.nih.gov/pubmed/21319937), which shall be placed in the `data/HINT` folder.
If you use this dataset, you will end up with the proper directory structure and file names.
If you want to use a different dataset, then you can edit the corresponding function (``source/audio_handler.py``) accordingly.

# Reproducing the results from the paper

This code allows to reproduce the experiment in the Oracle scenario (Table 1) conducted in the paper.
To run the benchmark as done in the paper, simply run the ``main.py`` script.


### Reference

<details><summary>If you use any of this code for your research, please cite our paper:</summary>
  
```latex
@article{Magron2020omisi,
  Title                    = {Online Spectrogram Inversion for Low-Latency Audio Source Separation},
  Author                   = {Paul Magron AND Tuomas Virtanen},
  Journal                  = {IEEE Signal Processing Letters},
  Year                     = {2020},
  Pages                    = {306--310},
  Volume                   = {27}
}
```
