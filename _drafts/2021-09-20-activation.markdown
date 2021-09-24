---
layout: post
title:  "The Activation Layer"
date:   2021-09-20 14:00:00 +0200
excerpt: >-
  The activation layer is not a learning layer per. It will help us better understand how the linear layer is 
  building representations on the data input. 
---

## Introduction

In the [previous layer]({% post_url 2021-09-19-linear %}), we mainly worked on a new approach, centered on the 
layer, that helps us computing the different elements of the $ Linear $ $ layer $ **backward pass**.

In this article we will speak about the **representations**, introduced in the **forward pass** paragraph of 
[this article]({% post_url 2021-08-06-inside-the-model %}).

## The Activation neural structure

Let us talk about the following setup: 

- $ L^k $ is a $ Linear $ $ layer $ of 2 **neurons**
- $ L^{k+1} $ is an $ Activation $ $ layer $

By definition, the $ Activation $ $ layer $ preserves the structure of the previous $ layer $. 
$ L^k $ has 2 output **neurons**, $ L^{k+1} $ will have 2 output **neurons** too.

As the $ Activation $ $ layer $ is not a **learning** layer, it does not declare any **weights**. 
We could wonder what the $ Activation $ $ layer $ actually does...

![Activation](/_assets/images/layers/Activation1.png)

Generally speaking, the $ Activation $ $ layer $ consists in evaluating an $ activation $ function 
on every input **neurons**. The choice of the $ activation $ function is up to the developer. Several reasons may 
justify the use of an $ activation $ function. 

1. It allows to transform value ranges. With the $ logistic $ $ activation $ function, we are able to transform 
input values in the range of $ [-\infty; \infty] $ to output values in the range of $ [0; 1] $.

    $$ 
    Logistic(x) = \frac{1}{1 + e^{-x}}
    $$

2. Add a non linearity in the $ model $. $ Heaviside $ example: 

    $$ 
    Heaviside(x) = \left\{\begin{align}
                            0, & \text{ if $x<0$}\\
                            1, & \text{ otherwise}
                          \end{align}
                   \right.
    $$
    
    Note that adding a non linearity after $ layers $ such as the $ Linear $ $ layer $ increases their expressiveness. 
    Let us recap that the final goal is to build a $ model $ that can "understand" data. 
    If $ model $ contains only $ Linear $ $ layers $, the global "understanding" of $ model $ will also be linear. 
    Adding a non linearity increases the spectre of functions that $ model $ is equivalent to. 

3. Inspired from the activation potential in biology. $ ReLU $ example:

    $$ 
    ReLU(x) = \left\{\begin{align}
                       0, & \text{ if $x<0$}\\
                       x, & \text{ otherwise}
                     \end{align}
              \right.
    $$
    
    We will talk about it in a later paragraph.
    
## Forward pass

Just use the $ activation $ function on each input **neurons** in order to produce the output **neurons**.

## Backward pass


