---
layout: post
title:  "Linear Network"
date:   2021-10-06
excerpt: >-
  We are finally able to build a solid understanding of the learning flow on the simple linear network.
---

## Introduction

Here we are, having seen every $ layer $ that composes the $ model $ introduced in the "Example" of the previous
articles. We are now ready to use the different formula we found for the **learning flow** in order to observe how
they work in practice.

## Example

Here is the **neural structure** synthesis with the **forward pass** for the whole $ model $ we saw in the
[weights article]({% post_url 2021-08-19-weights %}).

![Activation](/_assets/images/network/Linear1.png)

And below is the **neural structure** with the **backward pass**:

![Activation](/_assets/images/network/Linear2.png) 

We are now going to follow the order of the **backward pass** in order to clarify the **learning flow** once more.
The order of this **backward pass** being $ Loss $ -> $ L3 $ -> $ L2 $ -> $ L1 $, let us begin with the $ Loss $.

## Loss interpretation

Let us recap the formula of the $ Loss $ function we have used in the 
[loss function article]({% post_url 2021-08-09-loss-function %}):

$$ 
Loss(X^4, Y^{truth}) = \frac{1}{2} (X^4 - Y^{truth})^2
$$

In the same article we saw how this $ Loss $ serves a "systematic way of telling the $ model $ what is right or wrong".
And from the [backward pass article]({% post_url 2021-08-13-backward-pass %}) we introduced the **learning flow** and 
we computed it for this $ Loss $ function:
 
$$ 
\delta^4 = o^3 - y^{truth}
$$

What it is interesting to note is how "pure" this formula is. The **learning flow** for the $ Loss $ 
function just compares the actual output of $ model $ with the expected output $ y^{truth} $. 
But if we look closer at the formula for the $ Loss $ function, we may see how "artificial" it is: it has 
been chosen so that its $ derivative $ gives a good looking $ \delta^4 $. 

It appears that the whole $ Loss $ function is "just a global indicator". 
What really is propagated during the **learning phase** is the **learning flow**. 
Thanks to the simple formula for $ \delta^4 $ it is really easy to understand what happens during the **learning phase**. 
We have 3 cases to see: 

- when $ model $ produces $ o^3 = y^{truth} $
- when $ model $ produces $ o^3 < y^{truth} $
- when $ model $ produces $ o^3 > y^{truth} $ 

<hr style="width: 65%; margin: auto;">

<h3 id="nothing_to_learn" style="text-align:center; margin-top: 2%;"> $ o^3 = y^{truth} $ </h3>

The perfect situation: the $ model $ already produces the expected output, nothing to learn! The result is:

$$
\delta^4 = o^3 - y^{truth}
$$

$$
\boxed{\delta^4 = 0}
$$

<hr style="width: 65%; margin: auto;">

<h3 id="update_weights" style="text-align:center; margin-top: 2%;"> $ o^3 < y^{truth} $ </h3>

The $ model $ produces a lower than expected output. The result is:

$$
\delta^4 = o^3 - y^{truth}
$$

$$
\boxed{\delta^4 < 0}
$$

Thanks to the [weights article]({% post_url 2021-08-19-weights %}),
we know what elements of the $ model $ we should modify in order to fix the error: the $ model $'s **weights**.
In our current $ model $ we only have **weights** in our $ L2 $ $ layer $. Thus we have to update them with
the formula we already saw:

$$
\hat{w^2} = w^2 - \alpha . \delta w^2
$$

In order to compute $ \delta w^2 $, we first have to back propagate the **learning flow**.

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ o^3 > y^{truth} $ </h3>

The $ model $ produces a greater than expected output. It is the same case as in the previous paragraph but with
the opposite impact!
We will see in a later paragraph how this opposite impact translates into the **weights** update.

The result is:

$$
\delta^4 = o^3 - y^{truth}
$$

$$
\boxed{\delta^4 > 0}
$$

## L3 interpretation

$ L3 $ is a $ ReLU $ $ activation $ $ layer $ with 1 output **neuron**.
In the [previous article]({% post_url 2021-10-06-activation %}),
we found:

$$
\delta^{3} = \delta^{4} \text{ if } o^2 \geq 0 \text{ else 0 }
$$

Let us cover the 3 cases coming from the [previous paragraph](#loss-interpretation):

- when $ model $ produces $ o^3 = y^{truth} $ => $ \delta^4 = 0 $
- when $ model $ produces $ o^3 < y^{truth} $ => $ \delta^4 < 0 $
- when $ model $ produces $ o^3 > y^{truth} $ => $ \delta^4 > 0 $

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^4 = 0 $ </h3>

As we saw in this [paragraph](#nothing_to_learn), we have nothing to learn in this situation.
Without any surprise we find:

$$
\delta^{3} = \delta^{4} \text{ if } o^2 \geq 0 \text{ else 0 }
$$

$$
\boxed{\delta^3 = 0}
$$

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^4 < 0 $ </h3>

In this situation, we must beware of the "$ \text{ if } o^2 \geq 0 \text{ else 0 } $" member in the $ \delta^3 $ formula.

<h4> $ o^2 \geq 0$ </h4>

$$
\delta^{3} = \delta^{4} \text{ if } o^2 \geq 0 \text{ else 0 }
$$

$$
\boxed{\delta^3 < 0}
$$

<h4 id="bad_situation"> $ o^2 < 0$ </h4>

$$
\delta^{3} = \delta^{4} \text{ if } o^2 \geq 0 \text{ else 0 }
$$

$$
\boxed{\delta^3 = 0}
$$

We are in a bad situation here: we are blocking the **learning flow** while there is something to learn ($ \delta^4 < 0 $)!
We should definitely avoid this situation and think where the problem comes from in the first place.

Looking back at the $ L3 $ formula:

$$
L3(X^3) = X^3 \text{ if } X^3 \geq 0 \text{ else } 0
$$

Immediately we find that the culprit is the member: "$ \text{ if } X^3 \geq 0 \text{ else } 0 $".
We should think why we introduced it in the [activation article]({% post_url 2021-10-06-activation %}).
There were 3 reasons to use an $ activation $ function:

1. transform value ranges
2. add a non linearity in the $ model $ to increase its expressiveness
3. mimic the activation potential in biology

The $ ReLU $ activation main interests are the 2 and 3. But it is really the 3 that causes our bad situation.
We will talk about that later.
In fact we could still preserve the 2 using another $ activation $ function like the $ leaky $ $ ReLU $:

$$
leaky \text{ } ReLU(x) = \left\{\begin{align}
                           x, & \text{ if $x \geq 0$}\\
                           0.01x, & \text{ otherwise}
                               \end{align}
                        \right.
$$

With such an $ activation $ function, we would have computed:

$$
\delta^{3} = \left\{\begin{align}
                \delta^4, & \text{ if $o^2 \geq 0$}\\
                 0.01. \delta^4, & \text{ otherwise}
                    \end{align}
             \right.
$$

Our result would have been:

$$
\delta^{3} < 0
$$

and not

$$
\delta^{3} = 0
$$

while preserving the non linearity.

<p style="color: red;">
Rather than changing our $ activation $ function, we will make the assumption we do not fall into this
bad situation. Thus we will not discuss the case where $ o^2 < 0 $ and assume we always have $ o^2 \geq 0 $.
</p>

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^4 > 0 $ </h3>

<h4> $ o^2 \geq 0$ </h4>

$$
\delta^{3} = \delta^{4} \text{ if } o^2 \geq 0 \text{ else 0 }
$$

$$
\boxed{\delta^3 > 0}
$$

<h4> $ o^2 < 0$ </h4>

$$
\delta^{3} = \delta^{4} \text{ if } o^2 \geq 0 \text{ else 0 }
$$

$$
\boxed{\delta^3 = 0}
$$

Same bad situation as in [this paragraph](#bad_situation).

<p style="color: red;">
We will make the assumption we do not fall into this
bad situation: we will not discuss the case where $ o^2 < 0 $ and assume we always have $ o^2 \geq 0 $.
</p>

## L2 interpretation

$ L2 $ is a $ Linear $ $ layer $ with 1 output **neuron**.
In the [linear layer article]({% post_url 2021-09-19-linear %}), we found:

$$
\delta^{2} = \delta^{3} . w^2
$$

and

$$
\delta w^{2} = \delta^{3} . o^1
$$

We must recall our final goal is to be able to **update** the **weights**.
In our current situation we only have $ w^{2} $ **weights** and we can already update them thanks to the formula
we recalled from [this paragraph](#update_weights):

$$
\hat{w^2} = w^2 - \alpha . \delta w^2
$$

This means we do not care about $ \delta^{2} $ anymore: we just have to compute $ \delta w^{2} $ now.

Let us cover the 3 cases coming from the [previous paragraph](#l3-interpretation):

- when $ model $ produces $ o^3 = y^{truth} $ => $ \delta^3 = 0 $
- when $ model $ produces $ o^3 < y^{truth} $ => $ \delta^3 < 0 $
- when $ model $ produces $ o^3 > y^{truth} $ => $ \delta^3 > 0 $

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 = 0 $ </h3>

As we saw in this [paragraph](#nothing_to_learn), we have nothing to learn in this situation.
Without any surprise we find:

$$
\begin{align}
\delta w^{2} &= \delta^{3} . o^1 \\
             &= 0
\end{align}
$$

Thanks to the **update** formula, we know the **weights** will be:

$$
\begin{align}
\hat{w^2} &= w^2 - \alpha . \delta w^2 \\
          &= w^2 - \alpha . \delta^{3} . o^1
\end{align}
$$

$$
\boxed{\hat{w^2} = w^2}
$$

This is on par with the fact there is nothing to learn in this situation.

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 < 0 $ </h3>

In this situation, we must beware of the sign of $ o^1 $.

<h4> $ o^1 \geq 0 $ </h4>

$$
\begin{align}
\hat{w^2} &= w^2 - \alpha . \delta w^2 \\
          &= w^2 - \alpha . \delta^{3} . o^1
\end{align}
$$

$$
\boxed{\hat{w^2} > w^2}
$$

<h4> $ o^1 < 0 $ </h4>

$$
\begin{align}
\hat{w^2} &= w^2 - \alpha . \delta w^2 \\
          &= w^2 - \alpha . \delta^{3} . o^1
\end{align}
$$

$$
\boxed{\hat{w^2} < w^2}
$$

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 > 0 $ </h3>

## L1 interpretation
