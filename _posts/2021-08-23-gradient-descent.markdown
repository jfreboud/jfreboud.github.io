---
layout: post
title:  "The Gradient Descent Algorithm"
date:   2021-08-23 20:00:00 +0200
excerpt: >-
  7/ Toward an iterative algorithm to build a more robust learning.
---

## Introduction

In the [previous article]({% post_url 2021-08-22-recapitulation %}), we run our first **learning phase** on 
a given **dataset**.

In this article we will check the validity of our work. 

## Example: check the results

We have already run the **learning phase** on the whole **dataset**. Let us run a new **forward pass** 
in order to check the new results produced by our $ model $. 

### <span style="text-decoration:underline"> Data </span>

Same **data** as in the [first article]({% post_url 2021-08-05-general-concepts %}).

| data input | data output (expectation) |
| ---------------- | ----- |
| (100 broccoli, 2000 Tagada strawberries, 100 workout hours) | (bad shape)  |
| (200 broccoli,  0 Tagada strawberries, 0 workout hours)     | (good shape) |
| (0 broccoli, 2000 Tagada strawberries, 3 000 workout hours) | (good shape) |

### <span style="text-decoration:underline"> Model </span> 

Same $ model $ as in the [weights article]({% post_url 2021-08-19-weights %}).

$$
\begin{align}
    L1(X^1)  &= X^1 & \text{ with } X^1 = (X^1_1, X^1_2, X^1_3) \\
    L2(X^2, W^2) &= W^2 . X^2          & \text{ with } X^2 = (X^2_1, X^2_2, X^2_3) \\
                 &                     & \text{ and } W^2 = (W^2_1, W^2_2, W^2_3) \\
                 &= W^2_1 . X^2_1 + W^2_2 . X^2_2 + W^2_3 . X^2_3 \\
    L3(X^3)  &= X^3 \text{ if } X^3 \geq 0 \text{ else } 0 \\ \\
    model(X) &= L3(L2(L1(X))) & \text{ with } X = (X_1, X_2, X_3) \\ 
    Loss(X^4, Y^{truth})  &= \frac{1}{2} (X^4 - Y^{truth})^2 
\end{align}
$$

We introduced the following **weights** values for $ L2 $:

$$ 
w^2 = (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})
$$

And we finally updated them to the new values:

$$ 
w^2 = (\frac{1}{200}, 2000 - \frac{3 000}{11 600 000}, 3000 + \frac{1}{5 800})
$$

### <span style="text-decoration:underline"> Run a new forward pass </span>

As we modified the values for the $ L2 $ **weights**, the $ model $ is globally different.
We have to run a new **forward pass** to see the new values our $ model $ produces on the **data input**.

| $ x $              | $ o1 = L1(x) $   | $ o2 = L2(o1) $ | $ o3 = L3(o2) $ |
| :----------------: | :--------------: | :-------------: | :-------------: |
| (100, 2000, 100)   | (100, 2000, 100) | (4 300 000)     | (4 300 000)     |
| (200,  0, 0)       | (200,  0, 0)     | (1)             | (1)             |
| (0, 2000, 3 000)   | (0, 2000, 3 000) | (13 000 000)    | (13 000 000)    |

Wow, it seems some values have landed on Pluto. This is due to the **learning rate** value we used 
in the [previous article]({% post_url 2021-08-19-weights %}). We chose: $ \alpha = 1 $ which 
seems too high for the learning.

Yet, we may try to fix these extreme values with a **rule**.
As our expectations are: (bad shape) => 0 and (good shape) => 1, we can admit that: 
- values > 1 will be transformed to 1.

Now we have:

| $ x $              | $ o3 = model(x) $ | 
| :----------------: | :---------------: | 
| (100, 2000, 100)   | (1)               | 
| (200,  0, 0)       | (1)               | 
| (0, 2000, 3 000)   | (1)               | 

and: 

| $ o3 = model(x) $ | $ y^{truth} $ = expected result | $ loss = Loss(o3, y^{truth}) $ | correct ? |
| :----: | :-----: | :-----: | :---: |
| (1) | (0) | (<span style="color:red">0.5</span>) | ![wrong](/_assets/images/general/wrong.png) |
| (1) | (1) | (<span style="color:green">0</span>) | ![right](/_assets/images/general/right.png) |
| (1) | (1) | (<span style="color:green">0</span>)   | ![right](/_assets/images/general/right.png) |

It appears that $ model $ produces a good value for the last **data input**: the $ model $ has fixed 
its previous failure on this. But the $ model $ now fails for the first **data input**, which was 
not the case previously...

## The Gradient Descent Algorithm

In the [previous article]({% post_url 2021-08-22-recapitulation %}), we run the **learning phase** on the 
whole **dataset** with a big value for the **learning rate**. 

As the previous [paragraph](#example-check-the-results) shows, 
our **learning phase** appears to have fixed the result the $ model $ 
produces on the last **data input** but it has broken the result produced on the first **data input**. 

This suggests we should try a smaller **learning rate**. 
Following this idea, we should have many "small understading steps" in order to converge to a robust 
understanding of the **dataset**.

This is exactly what we will do in the gradient descent algorithm, just modifying the number of times we run 
every steps of the **learning phase**, as follows: 

1. pick one **data input** in the **dataset**
2. run the **forward pass** for the $ model $ on this **data input**
3. use the $ Loss $ function to compute the error between the result produced by the $ model $ and 
the expectation given by the **data output**
4. run the **backward pass** to compute:
    - the **learning flux**
    - the $ derivative $ of the $ Loss $ function according to $ W $
    
5. update the **weights** of $ model $

As before, we do it for all **data input** in the **dataset**, we call it an **epoch**.
Now we use a small **learning rate** and we have to iterate many times during several **epochs**. 
This means that we learn several times on the same **data input**.

![Tangent](/_assets/images/backward/tangent.png)

The naming of "The Gradient Descent" comes from the **weights** update formula: 

$$
\hat{W} = W - \alpha * \frac{\partial Loss}{\partial W}
$$

As we already saw, the whole **learning process** is linked to the $ derivative $ of the $ Loss $ function:
$ \frac{\partial Loss}{\partial W} $ called gradient. It is a "gradient descent" because the $ derivative $ 
of the $ Loss $ function is the direction of the tangent evaluated on the **data input** for the $ Loss $ function.
Following this direction allows to minimize the $ Loss $ function step by step.

## Example: back to the learning phase from scratch

Let us run the **learning phase** with a very small **learning rate** $ \alpha = 10^{-7} $.
Back from the beginning, the values for our **weights** were:

$$ 
w^2 = (\frac{1}{200}, \frac{3 000}{11 600 000}, \frac{1}{5 800})
$$

The $ model $ has to learn on each **data input** of our **dataset**. 
Thus we will run the **learning phase** on our 3 **data input**: this will be one **epoch** of the 
**gradient descent** algorithm.

We already know that the $ model $ does not learn anything from the first two **data input**.
We will only compute the **learning phase** on the last **data input**.

### <span style="text-decoration:underline"> Run the learning phase on the 3rd data input </span>

1: **data input**: $ x = (0, 2000, 3 000) $

2: run the **forward pass**: 

$$
\begin{align}
    o3 &= model(x) \\ 
       &= model((0, 2000, 3 000)) \\
       &= (0)
\end{align}
$$

3: compute $ loss $ 

$$ 
\begin{align}
loss &= Loss(o3, y^{truth}) \\
     &= (0.5) 
\end{align}
$$

4: run the **backward pass**:

$$ 
\begin{align}
\delta 4 &= o3 - y^{truth} \\
         &= (0) - (1) \\
         &= (-1)
\end{align}
$$

$$ 
\begin{align}
\delta 3 &= \delta 4 \text{ if } o2 \geq 0 \text{ else } 0 \\ 
         &= (-1) \text{ if } (0) \geq 0 \text{ else } 0 \\
         &= (-1)
\end{align}
$$

$$ 
\begin{align}
\delta 2 &= \delta 3 . w2 \text{ with } w^2 = (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
         &= (-1) * (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
         &= -(\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})
\end{align}
$$

$$ 
\begin{align}
\frac{\partial Loss}{\partial W^2}(o1) &= \delta 3 * o1 \\
                                       &= (-1) * (0, 2000, 3 000) \\
                                       &= -(0, 2000, 3 000)
\end{align}
$$

$$
\begin{align}
\delta 1 &= \delta 2 \\
         &= -(\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})
\end{align}
$$
    
5: update the **weights** of $ model $

$$
\begin{align}
    \hat{w^2} &= w^2 - \alpha * \frac{\partial Loss}{\partial W^2}(o1) \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) - 10^{-7} * (-(0, 2000, 3 000)) \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) + (0, 0.0002, 0.0003) \\
              &= (\frac{1}{200}, 0.0002 - \frac{3 000}{11 600 000}, 0.0003 + \frac{1}{5 800})
\end{align}
$$

Let us keep in mind the new values we computed for $ w^2 $: 

$$
\boxed{w^2 = (\frac{1}{200}, 0.0002 - \frac{3 000}{11 600 000}, 0.0003 + \frac{1}{5 800})}
$$

### <span style="text-decoration:underline"> Run a new forward pass </span>

As we modified the values for the $ L2 $ **weights**, the $ model $ is globally different.
We have to run a new **forward pass** to see the new values our $ model $ produces on the **data input**.

| $ x $              | $ o1 = L1(x) $   | $ o2 = L2(o1) $ | $ o3 = L3(o2) $ |
| :----------------: | :--------------: | :-------------: | :-------------: |
| (100, 2000, 100)   | (100, 2000, 100) | (0.43)          | (0.43)          |
| (200,  0, 0)       | (200,  0, 0)     | (1)             | (1)             |
| (0, 2000, 3 000)   | (0, 2000, 3 000) | (1.3)           | (1.3)           |

Let us use these new **rules**: 
- values < 0.5 will be transformed to 0, 
- values $ \geq $ 0.5 will be transformed to 1

Now we have:

| $ x $              | $ o3 = model(x) $ | 
| :----------------: | :---------------: | 
| (100, 2000, 100)   | (0)               | 
| (200,  0, 0)       | (1)               | 
| (0, 2000, 3 000)   | (1)               | 

and: 

| $ o3 = model(x) $ | $ y^{truth} $ = expected result | $ loss = Loss(o3, y^{truth}) $ | correct ? |
| :----: | :-----: | :-----: | :---: |
| (0) | (0) | (<span style="color:green">0</span>) | ![wrong](/_assets/images/general/right.png) |
| (1) | (1) | (<span style="color:green">0</span>) | ![right](/_assets/images/general/right.png) |
| (1) | (1) | (<span style="color:green">0</span>)   | ![right](/_assets/images/general/right.png) |

With this small **learning rate**, our $ model $ produces the right results for each **data input** !
On this simple example, there is no point in continuing the **learning process** by iterating over 
new **epochs** as we already got the results we expected. 

Yet, it will be necessary to iterate over far more **epochs** on real use cases.

## Conclusion

In this article we showed the importance of an iterative process in order to build a robust learning step 
by step. In deep-learning this algorithm is called "the gradient descent", due to the formula used to update 
the **weights**. 

The iteration consists in updating the **weights** of $ model $ for every **data input** in the **dataset**. 
We call it an **epoch**. Then we do it for several **epochs** until we are satisfied with the results.
