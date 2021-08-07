---
layout: post
title:  "Inside the Model"
date:   2021-08-06 15:00:00 +0200
---

## Introduction

In the [previous article]({% post_url 2021-08-05-general-concepts %}), we mentioned the prominent part the 
Deep-Learning **model** plays in the **learning**. 

In this article we will explore this **model** further. 

## The Layers

Without any further ado, here is what a typical deep-learning model looks like: 

![Layers](/_assets/images/model/Layers.png)

The two main components are the clouds and the arrows. For now we do not know what is in the clouds, just that we 
will call them **layers** (**layer L1**, **layer L2**, ...).
But we can clearly see what the arrows imply. 

## The forward pass 

The arrows imply a **sequential order**. If we recall the paragraph "Run a model" in the 
[previous article]({% post_url 2021-08-05-general-concepts %}), we saw that if we consider the **model** 
mathematical function depending on **X** we can call **model(X)**.

Now, we have a more precise form for this **model** function, as a sequential order of **inner functions** (the clouds), 
each depending of its previous function. 

### Example


