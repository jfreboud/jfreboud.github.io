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
  
In this article, we will use them in order to address a **classification task**. 

## Classification Task

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

Our goal is to train a $ model $ so that it can recognize what is represented in each image. 
But before talking about the $ model $, let us get back to the 2 phases we introduced in the 
[first article]({% post_url 2021-08-05-general-concepts %}): **learning** and **inferring**. 

Let us assume we have trained 
one $ model $ on some images of **CIFAR-10**. We would like to know how well the $ model $ has been trained. 
In order to do so, we will use the **inferring** phase on some images of **CIFAR-10** to note if the $ model $ 
makes the right predictions or not. But during this **inferring** phase, we do not want to show the same images 
used during the **training**. We want to evaluate the performance of our $ model $ on untouched images, 
so that it reflects "new predictions" the $ model $ would give us. But in order to able to note the $ model $ 
as a student at school, we have to know in advance what the correct answers are.

So the idea is pretty simple: just split the **CIFAR-10** dataset into 2 sets. We are able to train the $ model $ 
on the first set of images. Let us call it the **training set**. We keep the second set untouched during the 
**training** phase but we will use it during the **inferring** phase to evaluate our $ model $. Let us call 
this second set the **validation** set.

Once more, the **CIFAR-10** is already built in that way: out of the 60 000 images, 50 000 images are reserved for 
the **training set** and 10 000 are reserved for the **validation set**.

Now, let us define the problem we want to solve: predicting whether an image represents a ship or a dog:  

**ship** ![Classification](/_assets/images/network/Classification1.png) 
**dog** ![Classification](/_assets/images/network/Classification2.png) 

## Conclusion
