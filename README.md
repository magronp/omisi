# Online MISI GitHub repository

Here, you will find the code related to the online multiple input spectrogram inversion (oMISI) algorithm for source separation.

If you use any of the things existing in this repository, please cite the [corresponding paper](https://arxiv.org/abs/1802.03156). 

You can also find an online demo with sound examples related to this work on the [companion website](https://magronp.github.io/demos/spl19_omisi.html).

This code allows to reproduce the experiment in the Oracle scenario conducted in the paper. To do so, you will need the [Danish HINT dataset](https://www.ncbi.nlm.nih.gov/pubmed/21319937) and to place its content in the `data/HINT` folder.

If you use this dataset, you will end up with the proper directory structure and file names. If you want to use a different dataset, then you can edit the audio handler function accordingly.

To run the benchmark as done in the paper, simply run the "main.py" script.
