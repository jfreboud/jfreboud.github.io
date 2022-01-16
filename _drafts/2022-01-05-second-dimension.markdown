---
layout: post
title:  "The Second Dimension"
category: network
date:   2022-01-05
excerpt: >-
  In this article, we open the second dimension of our trip to Deep-Learning.
---

## Introduction

In the [previous article]({% post_url 2021-12-12-linear-function %}), we saw that $ Linear $ **network** are 
already good at finding correlations among structured **data**. 

In this article we will explore a field where the **data** is not structured : what solution could we find in such 
a case ? 

## Computer Vision

The typical field where **data** is not structured is **Computer Vision**. In this field, we would like  
machines to understand **images** and **video** the way humans do. 

The main problem is that humans are already used 
to talk about **images** as **representations** (introduced in the [second article]({% post_url 2021-08-06-inside-the-model %}), 
we talked about them more in the [previous article]({% post_url 2021-12-12-linear-function %})). 

For example when I see an **image** of another human, I tend to identify his/her main characteristics: 
the hair color, the eyes, the hands... And in fact each of these characteristics are some sort of **representations**. 
In my mind, I can play with these **representations**, comparing them for example: this human is taller than this one. 
In a way, I have a **representation** for the entire world and any **image** I will see will only refer to the 
**representations** I know. If I happen to interact with other senses, such as taste, I just have to enrich my 
visual **representation** knowing the taste and view refer to the same one. 

But for a machine, we must build this whole from scratch... 

## The Visual Information Flow

The first difference we should highlight between **Human Vision** and **Computer Vision** is the **information flow** 
(see the [second article]({% post_url 2021-08-06-inside-the-model %})).

In the human case the visual **information flow** is physical. It starts with light on our retina before being 
translated into **neurons** activations / non activations in our brain. 

In the machine case, we will only work with numerics. In a way it is easy: we just have to do 
some computations on those numerics to understand what happens. 
The bad news is that there will be so many computations that it will be difficult to fully grasp 
what **representations** are built. 
Hence, it is important to try and understand the different 
transformation of our **information flow** across the different **layers** of our **model**: this is our goal for 
the next articles.

Before building Deep-Learning **models**, let us start by converting our **images** into numerics!

## An Image

An image is a grid of a certain size. An image of size $ (height, width) $ will contain $ height * width $ pixels. 
Each pixel is a mix of the three primary colors $ (red, green, blue) $. 

Let us imagine a very small image of size $ (9, 9) $ and zoom in so that we observe a house ! 

![Image](/_assets/images/network/Image1.png)

You may have guessed that each circle in this image actually represents a pixel of the image. 
As we mentioned in the [previous paragraph](#the-visual-information-flow), our brain is triggered by the light 
on different colors. But the same image can also be seen as the different numbers of the pixels. 
In the below image, I have hidden the center of the image to show some pixels values $ (red, green, blue) $.

![Image](/_assets/images/network/Image2.png)

Note that if the entire image was only numbers, you would have no way understanding the image as a house. 
This is what from scratch means: we have to build a system that allows the computer to "understand" those numerics.

We are now going to make the first operation to build our numeric **representations**: instead of considering 
the grid of the previous numeric pixels, we are going to consider the three **channels** $ red $, $ green $ and $ blue $. 
In fact, it merely consists in splitting our grid into 3 grids in which each grid only contains the numbers of 
the **channel** considered. 

Instead of having a grid of $ height * width $ pixels, we now have 3 grids of $ height * width $ numbers.

Here is the $ red $ (first) **channel**: 

![Image](/_assets/images/network/Image3.png)

Here is the $ green $ (second) **channel**: 

![Image](/_assets/images/network/Image4.png)

And here is the $ blue $ (third) **channel**:

![Image](/_assets/images/network/Image5.png)

Tada, we are now in the presence of 3 numeric **representations** ! 
We even know the cheat in order to go back to a real image our of these 3 different **channels**.

![Image](/_assets/images/network/Image6.png)
