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
will call them **layers** (**L1**, **L2**, ...).
But we can clearly see what the arrows imply. 

## The forward pass 

The arrows imply a **sequential order**. If we recall the paragraph "Run a model" in the 
[previous article]({% post_url 2021-08-05-general-concepts %}), we saw that if we consider the **model** 
mathematical function depending on **X** we can call **model(X)**.

Now, we have a more precise form for this **model** function, as a sequential order of **inner functions** (the clouds), 
each depending of its previous function. 

### Example

In the [previous article]({% post_url 2021-08-05-general-concepts %}), we did not consider the inside 
of the **model** function. We only saw its dependency on a certain variable we called **X** and that 
we could produce **model(X)** thanks to some **data** of a **dataset**. If we summarize the situation, 
we had:  

![L2-1](/_assets/images/model/L2-1.png)

Now we will assume **model** is only composed of 3 layers 
(**L1**, **L2**, **L3**): 

![L2-1](/_assets/images/model/L2-2.png)

We can assemble the two previous schema in one: 

![L2-3](/_assets/images/model/L2-3.png)

Now we need some **data**, let us use the same **dataset** as in the "Example" paragraph 
in the [previous article]({% post_url 2021-08-05-general-concepts %}). 
We consider `x = (100 broccoli, 2000 Tagada strawberries, 100 workout hours)` and 
want to produce the result of **model(x)** following the [forward pass](#the-forward-pass):

1. first we have to call **L1** on **x** as **L1** is our first layer => L1 produces a new **output**, 
let **o1** be it
2. then we call **L2** on the result of **L1** which is **o1** => L2 produces a new **output**, 
let **o2** be it
3. finally we call **L2** on the result of **L2** which is **o2** => L3 produces a new **output**, 
let **o3** be it

Implicitly what happened was: 
1. **L1** is a mathematical function depending on **X1** and we produced its result on **x** as **x** 
is the global **input**
2. **L2** is a mathematical function depending on **X2** and we produced its result on **o1** as **o1** 
is the output of **L1** and thus the input of **L2** 
3. **L3** is a mathematical function depending on **X3** and we produced its result on **o2** as **o2** 
is the output of **L2** and thus the input of **L3** 

Finally it appears that `model(x) = o3`. 
