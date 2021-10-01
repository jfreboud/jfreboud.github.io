---
layout: post
title:  "The Activation Layer"
date:   2021-09-20
excerpt: >-
  The activation layer is not a learning layer per. It will help us better understand how the linear layer is 
  building representations on the data input. 
---

## Introduction

In the [previous article]({% post_url 2021-09-19-linear %}), we mainly worked on a new approach, centered on the 
layer, that helps us computing the different elements of the $ Linear $ $ layer $ **backward pass**.

In this article we will first of all talk about the $ Activation $ $ layer $. 

Then we will use this new $ layer $ 
in order to better understand the **representations** built by the $ Linear $ $ layer $. For now, we just 
introduced these **representations** in the **forward pass** paragraph of 
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

As we saw in the [weights article]({% post_url 2021-08-19-weights %}), the goal of the **backward pass** is to compute 

$$ 
\boxed{\delta w = \frac{\partial Loss}{\partial W}(x, y^{truth})}
$$

for each **weight** of every $ layer $. This is the direction to follow in the **weights**' **update** in order 
to minimize the $ Loss $ function.

We also have to use the **backward pass** in order to back propagate

$$
\boxed{\delta = \frac{\partial Loss}{\partial X}(x, y^{truth})}
$$

which is the **learning flow**: the essential part to compute the direction in the first formula. 

Yet, in our case, the $ Activation $ $ layer $ has no **weights** at all. This means we will only have to 
back propagate the **learning flow**.

## Backward pass for the learning flow 

We assume we have the same setup as in the [first paragraph](#the-activation-neural-structure): 

- $ L^k $ is a $ Linear $ $ layer $ of 2 **neurons**
- $ L^{k+1} $ is an $ Activation $ $ layer $

Let us also assume our $ activation $ function is the $ ReLU $ one.

We are currently focusing on the $ L^{k+1} $ $ layer $, trying to compute:

$$ 
\delta^{k+1} = \frac{\partial Loss}{\partial X^{k+1}}(o^k)
$$

We use the same approach as in the [previous article]({% post_url 2021-09-19-linear %}). 

The principal idea is to go back to the very structure of $ L^{k+1} $ in order to find the impacts of $ X^{k+1} $ 
on the $ Loss $ function, knowing that the "future" 
**learning flow** has already been computed (by definition of the **backward pass**). 

The structure for the $ L^{k+1} $ $ layer $ is: 
- 2 output **neurons** 
- 2 input **neurons**. 

$ \delta^{k+2, 1} $ and $ \delta^{k+2, 2} $ are the "future" **learning flow**: we admit they have already been 
computed.
We must back propagate the **learning flow** to $ \delta^{k+1, 1} $ and $ \delta^{k+1, 2} $.

![Linear](/_assets/images/layers/Activation2.png)

### Computing $ \delta^{k+1, 1} $ 

$$ 
\delta^{k+1, 1} = \frac{\partial Loss}{\partial X^{k+1, 1}}(o^k_1)
$$

The interesting variable is $ X^{k+1, 1} $. In the different diagrams it corresponds to $ o^{k}_1 $, 
its value during the current **backward pass**. There is just one output of $ L^{k+1} $ that uses $ X^{k+1, 1} $: 
$ L^{k+1, 1} $.

We are now able to build the **paths** of impacts from $ X^{k+1, 1} $ to the $ Loss $ function. 

![Linear](/_assets/images/layers/Activation3.png)

- $ X^{k+1, 1} $ impacts $ L^{k+1, 1} $ which impacts the $ Loss $ function 

We have only 1 impact, using the **chain rule**, we obtain: 

$$ 
\delta^{k+1, 1} = \delta^{k+2, 1} . \frac{\partial L^{k+1, 1}}{X^{k+1, 1}}(o^k_1)
$$

We just have to compute: 

$$ 
\begin{align}
\frac{\partial L^{k+1, 1}}{\partial X^{k+1, 1}} &= \frac{\partial (ReLU(X^{k+1, 1}))}{\partial X^{k+1, 1}} \\
                                                &= \frac{\partial (0 \text{ if } X^{k+1, 1} < 0 \text{ else } X^{k+1, 1})}{\partial X^{k+1, 1}} \\
                                                &= 0 \text{ if } X^{k+1, 1} < 0 \text{ else 1 }
\end{align}
$$

Then we evaluate this function on the values that have produced the final $ loss $:

$$ 
\frac{\partial L^{k+1, 1}}{X^{k+1, 1}}(o^k_1) = 0 \text{ if } o^k_1 < 0 \text{ else 1 }
$$

We finally use this result in the first formula:

$$ 
\boxed{\delta^{k+1, 1} = \delta^{k+2, 1} \text{ if } o^k_1 \geq 0}
$$

### Computing $ \delta^{k+1, 1} $ 

Same as in the previous paragraph. 
The result is: 

$$ 
\boxed{\delta^{k+1, 2} = \delta^{k+2, 2} \text{ if } o^k_2 \geq 0}
$$

## Example

We have already used an $ Activation $ $ layer $ in the "Example" of the previous articles. Let us have a look 
at the $ model $ we used in the [weights article]({% post_url 2021-08-19-weights %}): 

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

$ L3 $ is indeed an $ Activation $ $ layer $. Its **neural structure** is simple: it has just 1 output **neuron**.

![Linear](/_assets/images/layers/Activation4.png)

### <span style="text-decoration:underline"> Forward pass for L3 </span>

By definition, we just have to use the $ L3 $ explicit formula: 

$$ 
L3(X^3) = X^3 \text{ if } X^3 \geq 0 \text{ else } 0 
$$ 

### <span style="text-decoration:underline"> Backward pass for L3 </span>

In the [backward pass for the learning flow](#backward-pass-for-the-learning-flow), we found:

$$
\delta^{k+1, 1} = \delta^{k+2, 1} \text{ if } o^k_1 \geq 0
$$

$$ 
\delta^{k+1, 2} = \delta^{k+2, 2} \text{ if } o^k_2 \geq 0
$$

We adjust these formula to the current **neural structure**: 

$$
\boxed{\delta^{3} = \delta^{4} \text{ if } o^2 \geq 0}
$$ 

which is what we already computed in the 
[backward pass article]({% post_url 2021-08-13-backward-pass %}).
