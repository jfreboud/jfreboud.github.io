---
layout: post
title:  "Inside the Model"
date:   2021-08-06 15:00:00 +0200
---

## Introduction

In the [previous article]({% post_url 2021-08-05-general-concepts %}), we mentioned the prominent part the 
deep-learning **model** plays in the **learning**. 

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

Finally it appears that: 
```
model(x) = o3
```

## The Input Layer

The **input layer** is the first layer of the **model**. It is special in that it does not need the 
output of its previous layer in the **forward pass** order to produce an **output**.

When the developer wants to run the **model** on some **x** value, the developer will in fact give this 
**x** value directly to the **input layer**. In that way it does not even need to produce a value, 
as it already **owns** this value.

![L2-4](/_assets/images/model/L2-4.png)

## The Output Layer 

The **output layer** is the last layer of the **model**. It is special in that its value is not used 
by any other layers. 

Its value is in fact the final **output** of the **model** and this this value that we want to compare 
to the **data output** expectation we saw in the paragraph "Learning, inferring" 
of the [previous article]({% post_url 2021-08-05-general-concepts %}). 

![L2-5](/_assets/images/model/L2-5.png)

## The Layer in general

As we saw in the [example](#example), each **layer** is in fact a mathematical function that depends on 
some variable. We saw that **L2** depends on **X2** and produces **o2** and that **X2** is in fact the 
**output** of **L1** and that **o2** is the **input** of **L3**.

When we run a **model** on some **data** we want to produce the **output** of its **output layer** 
(see [the output layer](#the-output-layer)). 

Using the [example](#example) **model**, produce a result on **x** is about producing the 
result of **L3** on the **output** of **L2**. The **output** of **L2** is itself the result of 
**L2** computed on the **output** of **L1**. The **output** of **L1** has been given by the 
developer (see [the input layer](#the-input-layer)).

We can put it mathematically as: 
```
model(x) = L3 o L2 o L1 (x)
```
or 
```
model(x) = L3(L2(L1(x)))
```

## Store internal results

Let us try to compute **model(x)** with the same **model** as in [this example](#example). 
When reading naively the list of calls in the [last chapter](#the-layer-in-general), we need to: 
1. compute the output of **L3** which depends on the **output** of **L2** but we have no clue about 
the **output** of **L2** yet, so we need to...
2. compute the output of **L2** which depends on the **output** of **L1** but we have no clue about 
the **output** of **L1** ? ...
3. in fact yes, we know about the output of **L1**, it has been given by the developer, so now we can`
4. use the **output** of **L1** to compute the output of **L2**
5. use the **output** of **L2** to compute the output of **L3** !

This way of solving the problem about computing **L3** is recursive, but it is clearly not effective.
The right way is in fact just to follow [the forward pass](#the-forward-pass): 

1. developer gives **x** to **L1**, **L1** **stores** **o1** 
(`o1` == `x` as we saw in [the input layer](#the-input-layer))
2. **L2** computes **o2** thanks to **o1**, **L2** **stores** **o2**
3. **L3** computes **o3** thanks to **o3**, **L3** **stores** **o3**

We can now redraw the previous schema knowing that we **store** each **output** in memory:

![L2-6](/_assets/images/model/L2-6.png)

## Conclusion

In this article, we saw that the global form of a deep-learning **model** is in fact an ordered graph 
of **layers** as in the following schema: 

![L2-7](/_assets/images/model/L2-7.png)

We must now discuss how to choose the different **layers**. But before that, we will talk about the 
[learning process]({% post_url 2021-08-09-learning-process %}) 
