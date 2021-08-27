---
layout: post
title:  "Batch Learning"
date:   2021-08-24 10:00:00 +0200
excerpt: >-
  8/ Another idea to build a more robust learning: learn on multiple **data input** at once.
---

## Introduction

In the [previous article]({% post_url 2021-08-23-gradient-descent %}), we run one **epoch** of the 
**gradient descent** algorithm.

In this article we will add one more concept to stabilize this algorithm. 

## What is a batch ?

A **batch** corresponds to multiple elements of **data input** taken at once.
The main goal is to modify the way our **weights** are updated so that each update is more robust.

We typically want that our **weights** are updated according to the average direction of 
$ \frac{\partial Loss}{\partial W} $ on the **batch** of **data input**. We can still use the same 
formula to update the **weights**:

$$
\hat{W} = W - \alpha * \frac{\partial Loss}{\partial W}
$$

We are able to modify our **gradient descent** algorithm so that it is applied on a **batch** of 
**data input**:

1. pick a **batch** of **data input** in the **dataset** (just means pick several **data input**)
2. run the **forward pass** for the $ model $ on each element of the **batch**
3. use the $ Loss $ function to compute the error between the result produced by the $ model $ and 
the expectation given by the **data output**, this for each (**data input**, **data output**) of the 
different elements of the **batch**
4. run the **backward pass** to compute:
    - the **learning flux** for each elements of the **batch**
    - the $ derivative $ of the $ Loss $ function according to $ W $ for each elements of the **batch**
    
5. update the **weights** of $ model $

## The new forward pass

There is no new **forward pass**, we just have to continue running the **forward pass** as before, 
just keeping in mind that the different elements in the **batch** will be grouped together for **learning**.

For $ i $ in $ 1, 2 ... n $, indices of the batch elements, we compute $ \boxed{model(x^i)} $.

## The new Loss function

From its introduction in the article [Loss function](% post_url 2021-08-09-loss-function %), the $ Loss $ function 
has been used to systematically compare the results produced by the $ model $ with expectations.

Now, we want to compare the results and the expectations on multiple elements of a **batch**. 
Each element of the **batch** being (**data input**, **data output**).

One simple idea is to compute the average $ loss $ on these elements.

Example with the $ Loss $ function used in the previous articles where we had picked an $ x $ from the 
**data input** and $ y^{truth} $ from the **data output** and computed: 

$$
loss = Loss(model(x), y^{truth}) 
$$

Now we consider a **batch** so we have several elements (let us say $ n $ elements): 

- $ x^1 $, $ x^2 $, ... $ x^n $
- $ y^{truth, 1} $, $ y^{truth, 2} $, ... $ y^{truth, n} $

We can compute: 

$$
\begin{align}
    loss^1 &= Loss(model(x^1), y^{truth, 1}) \\ 
    loss^2 &= Loss(model(x^2), y^{truth, 2}) \\ 
           & \text{...} \\
    loss^n &= Loss(model(x^n), y^{truth, n})
\end{align}
$$

We introduce the average $ Loss^{avg} $ as the average of the errors on the **batch**:

$$ 
\boxed{Loss^{avg} = \frac{1}{n} . (Loss(model(X^1), Y^{truth, 1}) + Loss(model(X^2), Y^{truth, 2}) + \text{...} + 
Loss(model(X^n), Y^{truth, n}))}
$$ 

and we apply it on real values:

$$
loss^{avg} = \frac{1}{n} . (loss^1 + loss^2 + \text{...} + loss^n)
$$ 

What is interesting to note is that for any $ X^i $ we have:

$$
\frac{\partial Loss^{avg}}{\partial X^i} = \frac{1}{n} . \frac{\partial Loss}{\partial X^i}
$$

And we recall that $ \frac{\partial Loss}{\partial X^i} $ is used to compute the **learning flux**.

This means that the new $ Loss^{avg} $ has exactly the same impact on **learning** as 
$ Loss^{learning} $, with: 
 
$$ 
\boxed{Loss^{learning}(X, Y^{truth}) = \frac{1}{n} . Loss(X, Y^{truth})}
$$ 

In fact $ Loss^{avg} $ is just a global indicator that shows the average error at the end of the **forward pass**. 
But what is really propagated during the **learning phase** will be $ Loss^{learning} $.

## The new backward pass

There is no new **backward pass**, we just have to continue running the **backward pass** as before, 
just keeping in mind that the different elements in the **batch** will be grouped together for **learning**.

For $ i $ in $ 1, 2 ... n $, indices of the batch elements, we compute: 

$$
\frac{\partial Loss^{avg}}{\partial X^i} 
$$

and 

$$
\frac{\partial Loss^{avg}}{\partial W^i} 
$$

Thanks to the [previous paragraph](#the-new-loss-function), we know it brings down to compute:

$$
\boxed{\frac{\partial Loss^{learning}}{\partial X^i}}
$$

and 

$$
\boxed{\frac{\partial Loss^{learning}}{\partial W^i}}
$$

## The status so far

Let us summarize the status so far.
We are trying to **learn** on a **batch** of (**data input**, **data output**).
We have to apply the **learning phase**, which is nearly the same as before.

Let us concentrate on one $ L^k $ $ layer $ that declares $ W^k $ weights.
Suppose that our **batch** has $ n $ elements: 

- During the **forward pass** we have computed multiple $ o^k $ results, let us say: 
$ o^{k, 1} \text{, } o^{k, 2} \text{ ... } o^{k, n} $.
- During the **backward pass** we computed multiple $ \delta k $ for the **learning flux**: 
$ \delta k^{1} \text{, } \delta k^{2} \text{ ... } \delta k^{n} $.
- During the **backward pass** we also computed multiple $ \frac{\partial Loss^{learning}}{\partial W^k} $: 
$ \frac{\partial Loss^{learning}}{\partial W^{k, 1}} \text{, } 
  \frac{\partial Loss^{learning}}{\partial W^{k, 2}} \text{ ... } 
  \frac{\partial Loss^{learning}}{\partial W^{k, n}} $.
  
What is common for all these steps is that they are fully independent inside the **batch**.
This means that $ o^{k, 1} $, $ o^{k, 2} $ ... $ o^{k, n} $ are fully independent.
$ \delta k^{1} $, $ \delta k^{2} $ ... $ \delta k^{n} $ are fully independent.
$ \frac{\partial Loss^{learning}}{\partial W^{k, 1}} $, 
$ \frac{\partial Loss^{learning}}{\partial W^{k, 2}} $ ...
$ \frac{\partial Loss^{learning}}{\partial W^{k, n}} $ are fully independent.

The fact that they are fully independent is really interesting in terms of computing: 
we can fully parallelize their computation inside the current step (forward or backward).

Let us talk about our final goal which is to update the **weights** according to the average direction of 
$ \frac{\partial Loss}{\partial W} $.

For now, this average is clearly out of reach. 
Indeed, every **batch** element in the **forward pass** is isolated from the others.
For the **backward pass**, it is also the case. 
There is just one modification in the **backward pass**: the $ \frac{1}{n} $ coefficient in the 
$ Loss^{learning} $. But clearly this is not sufficient to say we are about to compute the average direction of 
$ \frac{\partial Loss}{\partial W} $.

The last part where things can get right is the **weights** update...

## Update the weights: the new rule

Let us recall the update formula for the **weights**: 

$$
\boxed{\hat{W} = W - \alpha * \frac{\partial Loss}{\partial W}}
$$

Our goal is to compute: 

$$
\frac{\partial Loss}{\partial W}
$$

We need to think about the [backward pass](% post_url 2021-08-13-backward-pass %) once more, 
to understand how $ W $ impacts the final $ Loss $: $ Loss^{avg} $.

Let us consider the $ L^k $ $ layer $ example we introduced in the [precedent paragraph](#the-status-so-far).
We try to compute: 

$$ 
\frac{\partial Loss^{avg}}{\partial W^k}
$$

Until now, the values for $ w^k $ have not been updated, they are still the same...
This means that every results obtained through the **forward pass** were not that independent: $ o^{k, 1} $ was computed 
with $ w^k $ as a value for the $ W^k $. But $ o^{k, 2} $ was computed with the same $ w^k $ value for 
$ W^k $ ... and $ o^{k, n} $ was computed with the same $ w^k $ value for $ W^k $ as well.

Thus $ W^k $ impacts $ L^k(X^{k, 1}, W^k) $, $ L^k(X^{k, 2}, W^k) $ ... and $ L^k(X^{k, n}, W^k) $. 
And by definition: $ L^k(X^{k, 1}, W^k) $ impacts $ Loss^{avg} $, 
$ L^k(X^{k, 2}, W^k) $ impacts $ Loss^{avg} $ ... and 
$ L^k(X^{k, n}, W^k) $ impacts $ Loss^{avg} $. 

So we must compute: 

$$ 
\begin{align}
\frac{\partial Loss^{avg}}{\partial W^k} &= \frac{\partial Loss^{avg}}{\partial W^{k, 1}} + 
\frac{\partial Loss^{avg}}{\partial W^{k, 2}} + \text{...} + \frac{\partial Loss^{avg}}{\partial W^{k, n}} \\
                                         &= \frac{\partial Loss^{learning}}{\partial W^{k, 1}} + 
\frac{\partial Loss^{learning}}{\partial W^{k, 2}} + \text{...} + \frac{\partial Loss^{learning}}{\partial W^{k, n}} \\
                                         &= \frac{1}{n} . (\frac{\partial Loss}{\partial W^{k, 1}} + 
\frac{\partial Loss}{\partial W^{k, 2}} + \text{...} + \frac{\partial Loss}{\partial W^{k, n}}) 
\end{align}
$$

We finally obtained what we were looking for: 

$ \frac{\partial Loss^{avg}}{\partial W^k} $ is the average direction of $ \frac{\partial Loss}{\partial W^{k}} $.

Though, we will keep in mind that: 

$$ 
\boxed{\frac{\partial Loss^{avg}}{\partial W^k} = \frac{\partial Loss^{learning}}{\partial W^{k, 1}} + 
\frac{\partial Loss^{learning}}{\partial W^{k, 2}} + \text{...} + \frac{\partial Loss^{learning}}{\partial W^{k, n}}}
$$
