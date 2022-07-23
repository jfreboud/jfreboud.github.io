---
layout: post
title:  "Classification Task"
category: network
date:  2022-07-23
excerpt: >-
  Let us build our first deep learning model to classify images. 
---

## Introduction

In the [previous articles]({% post_url 2022-06-08-normalization %}), we have explored the neural structure of 
several 2D $ layers $. 
  
In this article, we will use them in order to address a **classification task**. Before that, let us take 
a simple **dataset** to practice.

## CIFAR Dataset

Let us take the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. It consists in 60 000 tiny images 
of 32x32 RGB pixels. From the [second dimension article]({% post_url 2022-01-22-second-dimension %}), 
we recall it means 32x32 grids of ($ red $, $ green $, $ blue $) numbers between 0 and 255.

Using the same remark as in the [second dimension article]({% post_url 2022-01-22-second-dimension %}), 
the images of **CIFAR-10** are not stored as "RGB friendly" format but as "neural friendly" format. 
This means that instead of considering the images as 32x32 grids of pixels, we prefer considering them as 3x32x32 
grids of numbers.

So we are able to download 60 000 tiny images from the internet. What to do with them ?

The **CIFAR-10** classifies these 60 000 images under 10 different **labels**: airplane, automobile, bird, 
cat, deer, dog, frog, horse, ship, truck (hence the -10 in **CIFAR-10**).

Let us define the problem we want to solve in this article: 
predicting whether an image represents a ship or a dog.

&nbsp;&nbsp;&nbsp; **ship** &nbsp;&nbsp;&nbsp; ![Classification](/_assets/images/network/Classification1.png) 
&nbsp;&nbsp;&nbsp; **dog** &nbsp;&nbsp;&nbsp; ![Classification](/_assets/images/network/Classification2.png) 

## Classification Task

Our goal is to train a $ model $ so that it can recognize what is represented in each image. Here we want 
to recognize a ship or a dog.

Before talking about the $ model $, let us get back to the 2 phases we introduced in the 
[first article]({% post_url 2021-08-05-general-concepts %}): **learning** and **inferring**. 

Let us assume we have trained 
one $ model $ on some images of **CIFAR-10**. We would like to know how well the $ model $ has been trained. 
In order to do so, we will use the **inferring** phase on some images of **CIFAR-10** to evaluate whether the $ model $ 
makes the right predictions or not. But during this **inferring** phase, we do not want to show the same images 
used during the **training** phase. We want to evaluate the performance of our $ model $ on "untouched" images, 
so that it reflects "new predictions" the $ model $ would give us. But in order to be able to note the $ model $ 
as a student at school, we have to know in advance what the correct answers are.

So the idea is pretty simple: just split the **CIFAR-10** dataset into 2 sets. We are able to train the $ model $ 
on the first set of images. Let us call it the **training set**. We keep the second set "untouched" during the 
**training** phase in order to evaluate our $ model $ during the **inferring** phase. Let us call 
this second set the **validation** set.

Once more, the **CIFAR-10** is already built in that way: out of the 60 000 images, 50 000 images are reserved for 
the **training set** and 10 000 are reserved for the **validation set**.

## Pre Processing

As we already mentioned, the numbers of our different image grids go from 0 to 255. But from a 
[previous article]({% post_url 2021-12-12-linear-function %}), we would like to use $ ReLU $ $ activation $ function 
in order to mimic the decision making of the "Biological Neuron". As the threshold for this decision appears at 0, 
we would like our signal to be centered around 0. 

For this reason we will modify our **data input** so that the numbers are in the $ [-1; 1] $ interval. 
Here is a simple operation to do so: 

![Classification](/_assets/images/network/Classification3.png)

More precisely the channels $$ ch^{0, 0} $$, $$ ch^{0, 1} $$ and $$ ch^{0, 2} $$ now contain numbers in the 
interval $$ [-0.5, 0.5] $$.

## Conclusion
