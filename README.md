## sednn_pytorch_ignite

Improved version of monaural speech seperation using deep neural networks [sednn](https://github.com/cHemingway/sednn) for dissertation
In quite a rough state, don't expect this to work without some fiddling, performance is currently quite bad.
See [README_Original.md](README_Original.md) for original licensing and contributors.

* Dependencies using [Anaconda](https://anaconda.org/)
* Cleaner main training loop using [pytorch ignite](https://pytorch.org/ignite/)
* Multiple speakers supported added for speech seperation as well as enhancement
* Speedups by changing from pickle data format to [numpy .npz](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html)
* Automation of workflow and dependencies using [pydoit](https://pydoit.org)
* [Weights And Biases](https://www.wandb.com) dashboard support
* Benchmarking against SEGAN 
* STOI benchmarking, as well as PESQ
* MRCG features for input, using [MRCG_Python](https://github.com/cHemingway/MRCG_python)
* Cleaner output using logging module, progress bars through [tqdm](https://github.com/tqdm/tqdm)
