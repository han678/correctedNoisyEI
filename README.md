## Corrected Noisy Expected Improvement function
This is the code for "A Corrected Expected Improvement Acquisition Function Under Noisy Observations".

#### key dependencies 
(excluding commonly used packages such as scipy, numpy, torch etc.)
   * botorch (https://github.com/pytorch/botorch)
        ```bash
        pip install botorch
        ```
   * chainer (https://github.com/chainer/chainer)
       ```bash
        pip install chainer
        ```
#### toy example 
The following example compares our proposed acquisition function with expected improvement under noisy observations on a simple synthetic function.
```bash
python toy_example.py
```
![figure](https://github.com/han678/correctedNoisyEI/blob/d5acac5e4dedbc128b2a3dab42c9216e888ebc3c/toy_result/TestGaussian_1d_plots.png)
#### synthetic function 
```bash
python benchmark.py --output_dir OUTPUT_DIR --acq {acq_name}
```
```acq``` can be 'q_NEI', 'NEI', 'PI', 'UCB', 'EI_C', 'PI_C' or 'EI'.
#### model compression
##### Prepare dataset
* ImageNet (ILSVRC2012)
The dataset can be found on the official website if you are affiliated with a research organization. It is also available on Academic torrents.
Download the ILSVRC2012_img_train.tar and then extract those images under the folder './data/ILSVRC2012'
* MNIST (https://github.com/datapythonista/mnist)

#### run compression task
python compress_task.py --output_dir OUTPUT_DIR --acq {acq_name} --model {model_name}
```
```model``` can be 'Resnet50', 'VGG16' or 'FC3'.
