## topic D

### Envorinment

* python 3.7
* pytorch
* numpy
* pytorch-ignite

* All can be found in requirements.txt

### Task
Study whether bi-directional learning is helpful to overcome the problem of getting stuck in the local
optimum in phoneme recognition task.

* Auto-encoder on MFCC and Fbank feature.
* Phoneme classifier.
* Phoneme classifier with ae initialization.


### Results

#### Auto-encoder with Fbank
[ INFO : 2019-12-23 17:30:07,210 ] - Loss on Eval: 0.03

#### Auto-encoder with MFCC
[ INFO : 2019-12-23 14:40:26,005 ] - Loss on Eval: 0.03

#### Auto-encoder with MFCC and Batchnorm
[ INFO : 2019-12-23 21:21:34,626 ] - Loss on Eval: 0.03



#### DNN Classifier with Fbank(Random initialization)
[ INFO : 2019-12-23 17:24:48,137 ] - Acc on Eval: 61.40%


#### DNN Classifier with MFCC(Random initialization)
[ INFO : 2019-12-23 14:15:51,906 ] - Acc on Eval: 67.90%


#### DNN Classifier with MFCC(Random initialization and batchnorm)
[ INFO : 2019-12-23 20:28:56,031 ] - Acc on Eval: 68.05%

#### DNN Classifier with Fbank(AE initialization. Fix encoder parameter)
[ INFO : 2019-12-23 18:34:05,010 ] - Acc on Eval: 60.04%

#### DNN Classifier with MFCC(AE initialization. Fix encoder parameter)
[ INFO : 2019-12-23 16:07:44,660 ] - Acc on Eval: 67.01%

#### DNN Classifier with MFCC(Batchnorm, AE initialization. Fix encoder parameter)
[ INFO : 2019-12-23 21:44:54,356 ] - Acc on Eval: 67.22%


#### DNN Classifier with Fbank(AE initialization. Update all parameters)
[ INFO : 2019-12-23 18:00:53,524 ] - Acc on Eval: 61.05%

#### DNN Classifier with MFCC(AE initialization. Update all parameters)
[ INFO : 2019-12-23 16:21:13,392 ] - Acc on Eval: 67.20%


#### DNN Classifier with MFCC(Batchnorm, AE initialization. Update all parameters)
[ INFO : 2019-12-23 21:44:28,046 ] - Acc on Eval: 68.25%
