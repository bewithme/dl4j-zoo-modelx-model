# dl4j-zoo-modelx-model

This repo is extension of dl4j zoo.model 


Available models are:

InceptionV4

InceptionResNetV2

CDCGan (conditional deep convolutional generative adversarial networks)  Based on https://github.com/wmeddie/dl4j-gans



You can find model training entrance class under package org.freeware.dl4j.modelx.train. for example , if you want to train CDCGan by yourself you can run org.freeware.dl4j.modelx.train.gan.CDCGanTrainer class. you will see the following window. you also can watch the training process by access the url

http://localhost:9000/train/overview



The following snapshot is generator output smaples snapshot when training iteration is more than 7000 

![image](https://github.com/bewithme/dl4j-zoo-modelx-model/blob/master/snapshot/1619078093540.jpg)


SGAN(Semi-supervised learning GAN) this GAN can get above 0.9 F1 Score only use  1% of MNIST as training data set .


The following snapshot is the evaluate output of superviseDiscriminator  when the training iteration is more than 6000 

![image](https://github.com/bewithme/dl4j-zoo-modelx-model/blob/master/snapshot/sgan.jpg)


Any question pls contact me by wechatId:italybaby



