# VAGAN Code

Code for the **Visual Feature Attribution using Wasserstein GANs** method. The paper can be
found [here](https://arxiv.org/pdf/1711.08998.pdf). 

Author:
- Christian F. Baumgartner ([email](mailto:baumgartner@vision.ee.ethz.ch))

## How to get the code to run:

 * Step 1: Set up a virtual environment with Python 3 (I used 3.4.3), and tensorflow 
 (I used version 1.2). There is a requirements.txt file so `pip install -r requirements.txt`
  should work.

Potential caveat: We had to use the dev version of skimage to allow for 3D image 
resampling of the ADNI data. Previously, we had problems to compile it with anything 
else but Python 3.4.3. However, if you don't need this function 
(which you don't on 2D data) you can simply use the normal `pip install scikit-image`.


 * Step 2: Open `config/system.py` and change the settings to match your system.

 * Step 3: Run the method by executing `python vagan_train`, which should automatically build the synthetic data, run the WGAN training, 
 and write the progress to tensorboard (`tensorboard --logdir=logdir/gan --port=8008`).
 It takes around 1h hour or 1000 steps to train the method on synthetic data. 

 * Step 4: Get some results on the test set by running: `python vagan_test_loop.py logdir/gan/synthetic_experiment` (runs without GPU)

You can also train a classifier (`classifier_train.py`) and then generate all the 
types of saliency maps discussed in the paper using the `classifier_test_saliencies.py` 
script.

## Tailoring the code to your needs:

All you need to do is do implement a new data and data_loader script along the lines 
of `synthetic_data.py` and `synthetic_data_loader.py`. The gist is that it needs to 
generate an .hdf5 file with a field 'features' where all your images go, and a field 
'labels' where the class label goes (e.g. 0=no shadow, 1=shadow, or 0=acne, 1=no acne). You will also need to add
an `elif`statement for your new data in `vagan_train.py` around line 20 and that's it. For keeping 
things organised you may wish to create a new experiment for this in `vagan/experiments.`

## How is the code structured:

 * All scripts that can be run directly using `python <script>.py` are in the base 
 directory. 
 * All other code pertaining to VA-GAN is in the `vagan` folder. All code pertaining to the 
   classifier and the conventional saliency maps is in the folder `classifier`
 * The main code for VA-GAN can be found in `vagan/model_vagan.py`
 * All code pertaining to the different data sets is in the `data` folder
 * `tfwrapper` contains shortcuts for building layers etc. 

## Tricks for training and tuning

 * Using the `tanh` function after the generator output can make training more stable. This requires
   that the input data is scalled into the range of -1, to 1. 
 * The Wasserstein distance, i.e. negative critic loss, can be used for determining 
   how well the network is doing. Values closer to 0 usually produce better results. However, 
   the Wasserstein distance tends to be so noisy that chosing the final model on the best
   value does not work well. 
 * The best way to select a model is to look at the generated validation images in 
   tensorboard. Models get backed-up every hour during training so you should be able 
   to always find a point where the model was good. 
 * VA-GAN does overfit when trained too long or with insufficient data. For instance the synthetic experiment with only 
   1000 data points does not produce good results. If your data set is not big enough you
   may wish to play around with dropout and weight decay for the critic and map generator.
 * The default network for the synthetic experiments (`normalnet2D`) is modelled after the 3D experiment, so
   it has an unusualy layer structure. Other architectures (e.g. >1 layers per max pooling step)
   may produce better results. 
   
   
## Bugs
 
Please report bugs to [baumgartner@vision.ee.ethz.ch](mailto:baumgartner@vision.ee.ethz.ch). The code has just been freshly refactored and 
there are likely to be a couple of minor bugs. 
 
 