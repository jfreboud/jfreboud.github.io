---
layout: post
title:  "The Learning Process Part 1"
date:   2021-08-09 15:00:00 +0200
excerpt: >-
  3-1/ Time to work on the model's weights which are the core of the learning process.
---

## Introduction

In the [previous article]({% post_url 2021-08-06-inside-the-model %}), we explored the 
 global structure of the deep-learning $ model $: an ordered graph of $ layers $.

In this article we will explore the **model's weights**, responsible of the **learning process**. 

## The model's weights

In order to **learn** something, there must be a part of the $ model $ that can be trained. 

In the [first article]({% post_url 2021-08-05-general-concepts %}) and in the 
[previous article]({% post_url 2021-08-06-inside-the-model %}), none of the $ model $ we used had **weights**.
So what are these ?

### Introducing $ W $ 

In fact, the **weights** are just a new **variable** in the $ model $. 
For now each $ layer $ we have built depended on one variable $ X $. 
This $ X $ was used to pass a signal from the **first layer** through the **output layer** 
(see the [previous article]({% post_url 2021-08-06-inside-the-model %})).

What we are about to do now is to introduce a new variable $ W $ that is in a way **artificial** and will 
modify the behavior of the $ layer $ that uses it. 

Note that not all $ layers $ have to use these **weights**. But each $ layer L^i $ that declares 
**weights** will have two dependencies: $ X^i $ and $ W^i $. The $ layers $ $ L^k $ that do not declare 
any **weights** will only have one variable dependency: $ X^k $.

### The learning process 

We saw that this $ W $ is artificial. In what way ?

It is artificial in that we will modify it during 
the **learning** phase we talked about in the paragraph "Learning, inferring" of the 
[first article]({% post_url 2021-08-05-general-concepts %}).

During the **inferring** phase, $ W $ will stay the same because we know the $ model $ has already learnt.

This difference between $ X $ and $ W $ will have many impacts.  

## The backward pass

The first impact the difference between $ X $ and $ W $ has is the need to follow the rules of 
a **backward pass** during the **learning** phase.

In a way, this is the exact opposite as the **forward pass** we talked about in 
the [previous article]({% post_url 2021-08-06-inside-the-model %}):

![Layers](/_assets/images/backward/Layers.png)

The red arrows imply a **reversed sequential order** of calling the different $ layers $ in order to 
prepare for **updating** the **weights**.

During the **learning** phase, we will have to: 

1. follow the **forward pass** from the **input layer** to the **output layer** 
2. follow the **backward pass** from the **output layer** to the **input layer**

## The Loss function

In the [first article]({% post_url 2021-08-05-general-concepts %}), we saw that during the 
**learning phase**, we had to teach the $ model $ which one of its results was $ true $ and 
which one of its results was $ false $. 

The way to do it is via the **Loss** function which is defined by the developer. 
This function takes place in the **output layer** as we saw that it is the layer where we 
compare the $ model $ results to the expectations (see the paragraph "The Output Layer" in the 
[previous article]({% post_url 2021-08-06-inside-the-model %})).

We can see the **Loss function** as a function of two variables: $ X $ (as every $ layer $) and 
$ Y^{truth} $ (see the [first article]({% post_url 2021-08-05-general-concepts %})).

![Layer-1](/_assets/images/backward/Layer-1.png)

## The derivatives

This is what is at the **core** of updating the **weights**. 
We can say this is the nemesis of the **information flux** that goes from the **input layer** to 
the **output layer**. We could call this **reversed signal** 
(see the [backward pass](#the-backward-pass)): the **learning flux**. 

If the **input layer** is the origin of the **information flux**, the **output layer** 
and more specifically the $ Loss function $  is the origin of the **learning flux**. 

### Example

#### <span style="text-decoration:underline"> Data </span>

Same **data** as in the [first article]({% post_url 2021-08-05-general-concepts %}).

| data input | data output (expectation) |
| ---------------- | ----- |
| (100 broccoli, 2000 Tagada strawberries, 100 workout hours) | (bad shape) |
|(200 broccoli,  0 Tagada strawberries, 0 workout hours) | (good shape) |
| (0 broccoli, 2000 Tagada strawberries, 3 000 workout hours) | (good shape) |

#### <span style="text-decoration:underline"> Model </span> 

We assume here we have a $ model $ containing only 3 $ layers $: 

![Layer-1](/_assets/images/backward/Layer-1.png)

Let us use: 

$$
\begin{align}
    L1(X^1)  &= X^1 \text{,} & \text{ with } X^1 = (X^1_1, X^1_2, X^1_3) \\
    L2(X^2)  &= \frac{1}{200} X^2_1 - \frac{3 000}{11 600 000}  X^2_2 + \frac{1}{5 800} X^2_3 
        \text{,} & \text{ with } X^2 = (X^2_1, X^2_2, X^2_3) \\
    L3(X^3)  &= X^3 \\ 
    Loss(X^3, Y^{truth})  &= \frac{1}{2} (L3(X^3) - Y^{truth})^2 \\ 
                          &= \frac{1}{2} (X^3 - Y^{truth})^2 \\ \\
    model(X) &= L3(L2(L1(X))) \text{,} & \text{ with } X = (X_1, X_2, X_3) 
\end{align}
$$

We can verify that:
- $ X $ is 3 dimensional: $ X_1 $ is the variable for broccoli, $ X_2 $ is the variable for Tagada strawberries, 
$ X_3 $ is the variable for workout hours
- $ model(X) $ is 1 dimensional
- $ Loss $ is a loss function that depends on $ X^3 $ and $ Y^{truth} $.

We have built a $ model $ that is composed of 3 layers ($ L1 $, $ L2 $, $ L3 $).

#### <span style="text-decoration:underline"> Run the forward pass </span>

First of all let us apply the **forward pass**:

| $ x $              | $ o1 = L1(x) $   |
| :----------------: | :--------------: |
| (100, 2000, 100)   | (100, 2000, 100) |
| (200,  0, 0)       | (200,  0, 0)     |
| (0, 2000, 3 000)   | (0, 2000, 3 000) |

| $ o1 $             | $ o2 = L2(o1) $ |
| :----------------: | :-------------: |
| (100, 2000, 100)   | (0)             |
| (200,  0, 0)       | (1)             |
| (0, 2000, 3 000)   | (0)             |

| $ o2 $ | $ o3 = L3(o2) $ |
| :----: | :-------------: |
| (0)    | (0)             |
| (1)    | (1)             |
| (0)    | (0)             |

| $ o3 = model(x) $ | $ y^{truth} $ = expected result | $ loss = Loss(o2, y^{truth}) $ | correct ? |
| :----: | :-----: | :-----: | :---: |
| (0) | (0) | (<span style="color:green">0</span>) | ![wrong](/_assets/images/general/right.png) |
| (1) | (1) | (<span style="color:green">0</span>) | ![wrong](/_assets/images/general/right.png) |
| (0) | (1) | (<span style="color:red">0.5</span>) | ![right](/_assets/images/general/wrong.png) |

We can observe that the value of $ loss(o2, y^{truth}) $ <span style="color:green"> is 0 </span> when there is <span style="color:green"> no error </span> comparing $ o3 $ 
with $ y^{truth} $ and <span style="color:red"> is greater than 0 </span> when there is <span style="color:red"> an error </span>.
