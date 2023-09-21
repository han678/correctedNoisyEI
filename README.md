## Corrected Noisy Expected Improvement function


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
```bash
python toy_example.py
```
![figure](https://github.com/han678/correctedNoisyEI/blob/d5acac5e4dedbc128b2a3dab42c9216e888ebc3c/toy_result/TestGaussian_1d_plots.png)
#### synthetic function 
```bash
python benchmark.py
```
#### model compression
