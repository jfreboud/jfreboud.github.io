---
layout: post
title:  "The Loss function"
date:   2021-08-09 15:00:00 +0200
excerpt: >-
  3/ We complete the deep-learning model with the loss function: this is the first step toward the learning process.
---

## Introduction

In the [previous article]({% post_url 2021-08-06-inside-the-model %}), we explored the 
 generic structure of the deep-learning $ model $: an ordered graph of $ layers $.

In this article we will talk about the $ Loss function $ which is the starting point of the **learning process**. 

## The learning process 

In the paragraph "Learning, inferring" of the [first article]({% post_url 2021-08-05-general-concepts %}), 
we talked about the **learning** phase and the **inferring** phase. 
It is no surprise that the **learning** process in deep-learning happens during the **learning** phase. 
We will thus concentrate on it.

During this phase we run the **forward** pass (see the [previous article]({% post_url 2021-08-06-inside-the-model %})).
Then we are able to get the final results of our $ model $ thanks to its **output layer**. 

In the different "Example" paragraphs, we had: 
- $ o3 $: the **output layer** result when we run the $ model $ on the **data input**. 
- the expected result which is the **data output**.

We compared $ o3 $ to the expected result and had two situations: $ o3 = expectation $ or $ o3 \neq expectation $.

We are now looking for a systematic way of telling the $ model $: this result is right, that result is wrong.  

## The Loss function

The systematic way of telling the $ model $ what is right or wrong is the $ Loss function $.

This $ Loss $ function is defined by the developer. 
It is a function of two variables: $ X $ (as every $ layer $) and 
$ Y^{truth} $ (see the [first article]({% post_url 2021-08-05-general-concepts %})).

Its $ X $ variable will receive the result of the **output layer** whereas $ Y^{truth} $ will receive 
the expectation given by the **data output**. Hence, the $ Loss $ function will be able to systematically compare 
them both.

This also implies the $ Loss $ function will be called after the **output layer**: 

![Layer-1](/_assets/images/backward/Layer-1.png)

## The derivatives of the Loss function

We have to find to what extent each variable in the $ model $ is responsible for the errors that are 
highlighted by the $ Loss $ function.
 
What will give this shaming indicator is: 

$$
\boxed{\frac{\partial Loss(X, Y^{truth})}{\partial X}} 
$$

Back to the paragraph "Run a model" in the [first article]({% post_url 2021-08-05-general-concepts %}), 
we introduced the notation $ X $ for an "abstract" variable and $ x $ for a real value taken by the variable.
This will prove to be useful now. 

Indeed, $ \frac{\partial Loss(X, Y^{truth})}{\partial X} $ is itself a function and produces the same kind of results 
as $ Loss(X, Y^{truth}) $. Let $ \hat{X} $ and $ \hat{Y} $ be the new dependency variables. 
We will apply this new function on $ \hat{x} $ and $ \hat{y} $, 
the values for the variables $ \hat{X} $ and $ \hat{Y} $.

Considering that $ \hat{y} = y^{truth} $, our formula becomes:

$$
\boxed{\frac{\partial Loss(X, Y^{truth})}{\partial X}(\hat{x}, y^{truth})}
$$

This formula could be read as: we want to know to what extent the variable $ X $ has caused an error in the 
$ Loss $ function when the $ Loss $ function was evaluated on $ \hat{x} $ and $ y^{truth} $.

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
    Loss(X^4, Y^{truth})  &= \frac{1}{2} (X^4 - Y^{truth})^2 \\ \\
    model(X) &= L3(L2(L1(X))) \text{,} & \text{ with } X = (X_1, X_2, X_3) 
\end{align}
$$

We can verify that:
- $ X $ is 3 dimensional: $ X_1 $ is the variable for broccoli, $ X_2 $ is the variable for Tagada strawberries, 
$ X_3 $ is the variable for workout hours
- $ model(X) $ is 1 dimensional
- $ Loss $ is a loss function that depends on $ X^4 $ and $ Y^{truth} $.

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

| $ o3 = model(x) $ | $ y^{truth} $ = expected result | $ loss = Loss(o3, y^{truth}) $ | correct ? |
| :----: | :-----: | :-----: | :---: |
| (0) | (0) | (<span style="color:green">0</span>) | ![wrong](/_assets/images/general/right.png) |
| (1) | (1) | (<span style="color:green">0</span>) | ![wrong](/_assets/images/general/right.png) |
| (0) | (1) | (<span style="color:red">0.5</span>) | ![right](/_assets/images/general/wrong.png) |

We can observe that the value of $ loss(o3, y^{truth}) $ <span style="color:green"> is 0 </span> when there is <span style="color:green"> no error </span> comparing $ o3 $ 
with $ y^{truth} $ and <span style="color:red"> is greater than 0 </span> when there is <span style="color:red"> an error </span>.
The $ loss $ is an indicator of the error of the results produced by the $ model $ function.

#### <span style="text-decoration:underline"> Going further </span>

Let us try to compute the derivatives of our $ Loss $ function according to $ X $ in our $ model $: 

$$
\frac{\partial Loss(X, Y^{truth})}{\partial X}(\hat{x}, y^{truth})
$$

Do not forget our $ model $ is composed of several $ layers $. 
Each of these $ layers $ is responsible for the error highlighted by the $ Loss $ function.
This means we have to compute: 

$$
\begin{align}
    \delta 1 &= \frac{\partial Loss(X^1, Y^{truth})}{\partial X^1}(x, y^{truth}) \\ 
    \delta 2 &= \frac{\partial Loss(X^2, Y^{truth})}{\partial X^2}(o1, y^{truth}) \\
    \delta 3 &= \frac{\partial Loss(X^3, Y^{truth})}{\partial X^3}(o2, y^{truth}) \\
    \delta 4 &= \frac{\partial Loss(X^4, Y^{truth})}{\partial X^4}(o3, y^{truth}) \\
\end{align}
$$

These different quantities won't be as easy to compute. 
The easiest is $ \delta 4 $ because we have: 

$$ 
Loss(X^4, Y^{truth}) = \frac{1}{2} (X^4 - Y^{truth})^2
$$

Thus: 

$$ 
\begin{align}
    \delta 4 &= \frac{\partial Loss(X^4, Y^{truth})}{\partial X^4}(o3, y^{truth}) \\
             &= \frac{\partial (\frac{1}{2} (X^4 - Y^{truth})^2)}{\partial X^4}(o3, y^{truth}) \\
\end{align}
$$

$$ 
\boxed{\delta 4 = o3 - y^{truth}} \\
$$

What is the problem with the other quantities ?
Let us look at $ \delta 3 = \frac{\partial Loss(X^3, Y^{truth})}{\partial X^3}(o2, y^{truth}) $.

We need to find to what extent the variable $ X^3 $ has caused an error in the $ Loss $ function 
when the $ Loss function $ was evaluated on $ o2 $ and $ y^{truth} $. And we know that: 

$$
\begin{align}
    L3(X^3)  &= X^3 \\ 
    Loss(X^4, Y^{truth})  &= \frac{1}{2} (X^4 - Y^{truth})^2 \\  
\end{align}
$$

The question that may arise is: 
$ X^3 $ is defined in $ L3 $, not in the $ Loss $ function, 
so how could it be responsible for the error highlighted by the $ Loss $ function ?

This is due to the structure in $ layers $ of our $ model $. Changing the value that $ X^3 $ takes impacts 
the $ layers $ that use $ X^3 $ directly ($ L3 $) or indirectly ($ L4 $, $ L5 $, ..., $ Loss $) <sup>[1](#remark)</sup>

This is the beauty of deep-learning: from a single $ loss $ result, being able to find the different culprits and to 
what extent they are responsible for the error through a **chain** of $ layers $. 

## Conclusion

In this article, we saw the importance of the $ Loss $ function is order to set what results of $ model $ are right 
or wrong. 

We also saw the formula that allows to distribute the penalties to the faulty variables: 
 
$$
\boxed{\frac{\partial Loss(X, Y^{truth})}{\partial X}(\hat{x}, y^{truth})}
$$

We were able to compute this quantity for the final variable but not for inner variables yet. 
We will have to discuss how to compute this quantity for the other variables and how to use it for real because 
for the time being we did not use it anywhere :smiling_imp:

<br>

<a id="remark" class="anchor" href="#header-title">1:</a>

In fact this means the $ Loss $ function depends on more variables than just $ X4 $ and $ Y^{truth} $. 
Talking about $ Loss(X^1, X^2, X^3, X^4, Y^{truth}) $ would be more precise. 
This also applies for every $ layer $: 
$ L3(X^1, X^2, X^3) $, $ L2(X^1, X^2) $ and $ L1(X^1) $. 

For our comfort, we will keep the notation with direct variables: $ Loss(X^4, Y^{truth}) $, $ L3(X^3) $, $ L2(X^2) $ 
and $ L1(X^1) $.
