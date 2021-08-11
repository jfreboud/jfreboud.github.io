---
layout: post
title:  "Inside the Model"
date:   2021-08-06 15:00:00 +0200
excerpt: >-
  II/ In this article, we explore the structure of a deep-learning model.
---

## Introduction

In the [previous article]({% post_url 2021-08-05-general-concepts %}), we mentioned the prominent part the 
deep-learning $ model $ plays in the **learning**. 

In this article we will explore the $ model $ structure further. 

## The Layers

Without any further ado, here is what a typical deep-learning $ model $ looks like: 

![Layers](/_assets/images/model/Layers.png)

The two main components are the clouds and the arrows. For now we do not know what is in the clouds, just that we 
will call them $ layers $ ($ L1 $, $ L2 $, ...).
But we can clearly see what the arrows imply. 

## The forward pass 

The arrows imply a **sequential order**. If we recall the paragraph "Run a model" in the 
[previous article]({% post_url 2021-08-05-general-concepts %}), we saw that if we consider the $ model $ 
mathematical function depending on $ X $ we can call $ model(X) $.

Now, we have a more precise structure for this $ model $ function, as a sequential order of $ inner functions $ (the clouds), 
each depending of its previous function. 

### Example

#### <span style="text-decoration:underline"> Data </span>

Same **data** as in the [previous article]({% post_url 2021-08-05-general-concepts %}).

| data input | data output (expectation) |
| ---------------- | ----- |
| (100 broccoli, 2000 Tagada strawberries, 100 workout hours) | (bad shape) |
|(200 broccoli,  0 Tagada strawberries, 0 workout hours) | (good shape) |
| (0 broccoli, 2000 Tagada strawberries, 3 000 workout hours) | (good shape) |

#### <span style="text-decoration:underline"> Model </span> 

We assume here we have a $ model $ containing only 3 $ layers $: 

![L2-3](/_assets/images/model/Layer-1.png)

Let us use: 

$$
\begin{align}
    L1(X^1)  &= X^1 \text{,} & \text{ with } X^1 = (X^1_1, X^1_2, X^1_3) \\
    L2(X^2)  &= \frac{1}{200} X^2_1 - \frac{8 800}{11 600 000}  X^2_2 + 
        \frac{1}{5 800} X^2_3 \text{,} & \text{ with } X^2 = (X^2_1, X^2_2, X^2_3) \\
    L3(X^3)  &= X^3 \text{ if } X^3 > 0 \text{, else } 0 \\ \\
    model(X) &= L3(L2(L1(X))) \text{,} & \text{ with } X = (X_1, X_2, X_3) 
\end{align}
$$

We can verify that:
- $ X $ is 3 dimensional: $ X_1 $ is the variable for broccoli, $ X_2 $ is the variable for Tagada strawberries, 
$ X_3 $ is the variable for workout hours
- $ model(X) $ is 1 dimensional

We have built a $ model $ that is composed of 3 layers ($ L1 $, $ L2 $, $ L3 $): 

#### <span style="text-decoration:underline"> Run the model </span>

Instead of using $ model(X) $ directly as in the [previous article]({% post_url 2021-08-05-general-concepts %}).
We now have to apply [the forward pass](#the-forward-pass), storing every intermediate results.

1. first we have to call $ L1 $ on $ x $ as $ L1 $ is our first layer => L1 produces a new **output**, 
let $ o1 $ be it
2. then we call $ L2 $ on the result of $ L1 $ which is $ o1 $ => L2 produces a new **output**, 
let $ o2 $ be it
3. finally we call $ L3 $ on the result of $ L2 $ which is $ o2 $ => L3 produces a new **output**, 
let $ o3 $ be it

Finally it appears that: 
$$ 
model(x) = o3 
$$

Here are the different results on the **data**:

| $ x $              | $ L1(x) \text{ or } o1 $  |
| :----------------: | :-----------------------: |
| (100, 2000, 100)   | (100, 2000, 100)          |
| (200,  0, 0)       | (200,  0, 0)              |
| (0, 2000, 3 000)   | (0, 2000, 3 000)          |

| $ o1 $             | $ L2(o1) \text{ or } o2 $ |
| :----------------: | :-----------------------: |
| (100, 2000, 100)   | (-1)                      |
| (200,  0, 0)       | (1)                       |
| (0, 2000, 3 000)   | (-1)                      |

| $ o2 $ | $ L3(o2) \text{ or } o3 $ |
| :----: | :-----------------------: |
| (-1)   | (0)                       |
| (1)    | (1)                       |
| (-1)   | (0)                       |

Finally we can summarize these results:

| $ x $ | expected result | $ model(x) \text{ or } o3 $ | correct ? |
| :----------------: | :-----: | :----: | :---: |
| (100, 2000, 100) | (<span style="color:red">bad shape</span>)    | (0) => (<span style="color:red">bad shape</span>) | ![wrong](/_assets/images/general/right.png) |
| (200,  0, 0)     | (<span style="color:green">good shape</span>) | (1) => (<span style="color:green">good shape</span>)    | ![wrong](/_assets/images/general/right.png) |
| (0, 2000, 3 000) | (<span style="color:green">good shape</span>) | (0) => (<span style="color:red">bad shape</span>) | ![right](/_assets/images/general/wrong.png) |

We can observe that although we have changed the structure of $ model $ compared to the
[previous article]({% post_url 2021-08-05-general-concepts %}), we still get exactly the same results. 
Which is in fact normal: we did not introduce any **weights** yet :smiling_imp:

## The Input Layer

The **input layer** is the first layer of the $ model $. It is special in that it does not need the 
output of its previous layer in the **forward pass** order to produce an **output**.

When the developer wants to run the $ model $ on some $ x $ value, the developer will in fact give this 
$ x $ value directly to the **input layer**. In that way it does not even need to produce a value, 
as it already **owns** this value.

Note as in [the example](#example), the **input layer** was just outputting its input without any modification.

![L2-4](/_assets/images/model/Layer-2.png)

## The Output Layer 

The **output layer** is the last layer of the $ model $. It is special in that its value is not used 
by any other layers. 

Its value is in fact the final **output** of the $ model $ and this is the value that we want to compare 
to the **data output** expectation. 

![L2-5](/_assets/images/model/Layer-3.png)

## The Layer in general

As we saw in the [example](#example), each $ layer $ is in fact a mathematical function that depends on 
some variable. We saw that $ L2 $ depends on $ X^2 $ to produce $ o2 $, that $ X^2 $ is in fact the 
**output** of $ L1 $ and that $ o2 $ is the **input** of $ L3 $.

When we run a $ model $ on some **data** we want to produce the **output** of its **output layer** 
(see [the output layer](#the-output-layer)). 

Using the [example](#example) $ model $, produce a result on $ x $ is about producing the 
result of $ L3 $ on the **output** of $ L2 $. The **output** of $ L2 $ is itself the result of 
$ L2 $ computed on the **output** of $ L1 $. The **output** of $ L1 $ has been given by the 
developer (see [the input layer](#the-input-layer)).

We can put it mathematically as: 

$$ 
\begin{align}
    model(x) &= L3 \text{ } o \text{ } L2 \text{ } o \text{ } L1 (x) \\
             & \text{or} \\
    model(x) &= L3(L2(L1(x))) \text{ as in the example} 
\end{align}
$$

## Conclusion

In this article, we saw that the global structure of a deep-learning $ model $ is in fact an ordered graph 
of $ layers $ as in the following schema: 

![L2-7](/_assets/images/model/Layer-4.png)

We must now discuss how to choose the different $ layers $. But before that, we will talk about the 
[learning process]({% post_url 2021-08-09-learning-process %}) 
