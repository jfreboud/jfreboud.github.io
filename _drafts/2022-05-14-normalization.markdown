---
layout: post
title:  "Normalization Layer"
category: layer
date:   2022-05-14
excerpt: >-
  Normalization layer allows us to stabilize learning. 
---

## Introduction

In the [batch learning article]({% post_url 2021-08-24-batch-learning %}), we saw a method that helps stabilizing 
the **learning  process**. , modifying the way we iterate through the **gradient descent** algorithm. 
It is a global method that modifies the behavior of the **gradient descent** algorithm itself.

In this article we will discuss of a much more localised method: applying **normalization** on the $ layer $ scope.

## The Formula

The idea is that instead of considering one global methodology that "normalizes" **gradients**, we will include 
a specific operation in some $ layers $ of our Deep learning $ model $ in order to "stabilize" the output **neurons** 
ot these specific chosen $ layers $.

First of all, how should we "stabilize" the output **neurons** ?

There is no mystery, in order to normalize we have to know about the "norm". 
In our case the "norm" will be some $ mean $, noted $ \mu $, of the output **neurons** of one particular $ layer $.

$$ 
\mu = \frac{1}{\textbf{nb elements}} . \sum_{elem=0}^{\textbf{nb elements} - 1} o_{elem}
$$

We also want to know about the typical difference we might observe between one output **neuron** of our 
considered $ layer $ and the $ mean $ above. This is called the **standard deviation**, noted $ \sigma $: 

$$ 
\sigma = \frac{1}{\textbf{nb elements}} . \sqrt{\sum_{elem=0}^{\textbf{nb elements} - 1} (o_{elem} - \mu)^2}
$$

## The Shapes of Normalization

We have the two principal elements needed to normalize the output **neurons** of one $ layer $. But there is 
still an important problem to solve...

In the [linear function]({% post_url 2022-12-12-linear-function %}) and 
the [second dimension]({% post_url 2022-01-22-second-dimension %}) articles, we respectively saw how the 
$ Linear $ and $ Convolution $ $ layers $ build **representations**.

For the $ Linear $ $ layer $, the "meaning" is hold by each of the output **neurons** themselves. 
For the $ Convolution $ $ layer $, the "meaning" is hold by the different **channels**, composed of output **neurons** 
organized on a grid.

During the **normalization** process we will have to preserve these "meaning" intact, avoiding to mix output 
**neurons** that do not **represent** "the same thing".
