---
layout: post
title:  "Linear Network"
date:   2021-10-31
excerpt: >-
  Looking back at the simple "Example" model to illustrate the impact of the weights.
---

## Introduction

In the [linear layer]({% post_url 2021-09-19-linear %}) and
[activation layer]({% post_url 2021-10-06-activation %}) articles,
we saw how to compute the **forward pass** and the **backward pass**
of the different $ layers $ that compose the $ model $ introduced in the "Example" of
the [second article]({% post_url 2021-08-06-inside-the-model %}).

In this article we will consider this simple $ model $ in order to better illustrate the impact of
the **weights** on the final $ loss $ value
(read the [loss function article]({% post_url 2021-08-09-loss-function %})).

## Example

Here is the **neural structure** synthesis during the **forward pass** of our simple $ model $:

![Activation](/_assets/images/network/Linear1.png)

And below is the **neural structure** during its **backward pass**:

![Activation](/_assets/images/network/Linear2.png)

## Sign Flow Analysis

Let us recap the main steps of the **learning phase** we introduced in the
[first article]({% post_url 2021-08-05-general-concepts %}). These steps have been called the
**gradient descent** algorithm in [this article]({% post_url 2021-08-23-gradient-descent %}):

1. pick one **data input** in the **dataset**
2. run the **forward pass** for the $ model $ on this **data input**
3. use the $ Loss $ function to compute the error between the result produced by the $ model $ and
the expectation given by the **data output**
4. run the **backward pass** to compute:
    - the **learning flow**
    - the $ derivative $ of the $ Loss $ function according to $ W $

5. update the **weights** of $ model $

We saw in the [weights article]({% post_url 2021-08-19-weights %}) that the **learning flow** sole purpose is
to be able to compute $ \delta w $ in the **weights** **update** formula:

$$
\hat{w} = w - \alpha . \delta w
$$

In the [same article]({% post_url 2021-08-19-weights %}), we also saw how this $ -\delta w $ is the direction of
**update** and $ \alpha $ the length of the **update**.

Still, $ \alpha $ is not that important, we know it
must be very little so that the many **epochs** we run during the **gradient descent** algorithm will progressively
minimize the $ Loss $ function.

The really important part of the **update** formula is the direction of **update**: $ -\delta w $, and especially
its **sign**.

This is the reason why we will now analyze the effect of the
**sign** of the final $ loss $ on the **sign** of $ \delta w $. This analyze will illustrate the "impact" notion
we have been dealing with since the
[loss function article]({% post_url 2021-08-09-loss-function %}).

But before that, we have to follow the order of the **backward pass** because we know the importance of the
**learning flow** in order to compute $ \delta w $.

Let us first analyze the **sign** back propagation during the **backward pass** so that we will finally be able
to analyze the **sign** of the direction of the **update**:
1. [Sign Flow Analysis](#loss-sign-flow)
2. [Sign Update Analysis](#sign-update-analysis)

In our [Example](#example), the order of the **backward pass** is:
$ Loss $ -> $ L3 $ -> $ L2 $ -> $ L1 $, so let us begin with the $ Loss $ **sign flow**.

## Loss Sign Flow

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
We have 3 cases to consider for our **sign analysis**:

- when $ model $ produces $ o^3 = y^{truth} $
- when $ model $ produces $ o^3 < y^{truth} $
- when $ model $ produces $ o^3 > y^{truth} $

<hr style="width: 65%; margin: auto;">

<h3 id="nothing_to_learn" style="text-align:center; margin-top: 2%;"> $ o^3 = y^{truth} $ </h3>

The perfect situation: the $ model $ already produces the expected output, nothing to learn!

$$
\delta^4 = o^3 - y^{truth}
$$

$$
\boxed{\delta^4 = 0}
$$

<hr style="width: 65%; margin: auto;">

<h3 id="update_weights" style="text-align:center; margin-top: 2%;"> $ o^3 < y^{truth} $ </h3>

The $ model $ produces a lower than expected output.

$$
\delta^4 = o^3 - y^{truth}
$$

$$
\boxed{\delta^4 < 0}
$$

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ o^3 > y^{truth} $ </h3>

The $ model $ produces a higher than expected output. It is the same case as in the previous paragraph but with
the opposite impact!

$$
\delta^4 = o^3 - y^{truth}
$$

$$
\boxed{\delta^4 > 0}
$$

## L3 Sign Flow

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

## L2 Sign Flow

$ L2 $ is a $ Linear $ $ layer $ with 1 output **neuron**.
In the [linear layer article]({% post_url 2021-09-19-linear %}), we found:

$$
\delta^{2} = \delta^{3} . w^2
$$

Let us cover the 3 cases coming from the [previous paragraph](#l3-sign-flow):

- when $ model $ produces $ o^3 = y^{truth} $ => $ \delta^3 = 0 $
- when $ model $ produces $ o^3 < y^{truth} $ => $ \delta^3 < 0 $
- when $ model $ produces $ o^3 > y^{truth} $ => $ \delta^3 > 0 $

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 = 0 $ </h3>

As we saw in this [paragraph](#nothing_to_learn), we have nothing to learn in this situation.

$$
\delta^{2} = \delta^{3} . w^2
$$

$$
\boxed{\delta^{2} = 0}
$$

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 < 0 $ </h3>

In this situation, we must beware of the sign of $ w^2 $.

$$
sign(\delta^{2}) = sign(\delta^{3}) . sign(w^2)
$$

$$
\boxed{sign(\delta^{2}) = -sign(w^2)}
$$

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 > 0 $ </h3>

In this situation, we must beware of the sign of $ w^2 $.

$$
sign(\delta^{2}) = sign(\delta^{3}) . sign(w^2)
$$

$$
\boxed{sign(\delta^{2}) = sign(w^2)}
$$

## L1 Sign Flow

$ L1 $ is an $ Input \text{ } 1D $ $ layer $ with 3 output **neurons**.
In the [activation layer article]({% post_url 2021-10-06-activation %}), we found:

$$
\delta^{1} = \delta^{2}
$$

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^2 = 0 $ </h3>

$$
\delta^{1} = \delta^{2}
$$

$$
\boxed{\delta^{1} = 0}
$$

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^2 < 0 $ </h3>

$$
\delta^{1} = \delta^{2}
$$

$$
\boxed{\delta^{1} < 0}
$$

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^2 > 0 $ </h3>

$$
\delta^{1} = \delta^{2}
$$

$$
\boxed{\delta^{1} > 0}
$$

## Sign Update Analysis

We have been able to back propagate the **sign** of having lower results than expected or
higher results than expected on the different $ layers $ in the order of the **backward pass**.

We are now ready to analyze the impact of these lower/higher results on the
**sign** of the direction of the **update**: $ -\delta w $. This direction will enable us decreasing
the final $ loss $ value.

In our current $ model $ we only have **weights** in the $ L2 $ $ layer $. Thus we have to **update** them with
the formula we already saw:

$$
\hat{w^2} = w^2 - \alpha . \delta w^2
$$

## L2 Sign Update

$ L2 $ is a $ Linear $ $ layer $ with 1 output **neuron**.
In the [linear layer article]({% post_url 2021-09-19-linear %}), we found:

$$
\delta w^{2} = \delta^{3} . o^1
$$

We are now able to replace $ \delta w^2 $ in the below formula:

$$
\hat{w^2} = w^2 - \alpha . \delta w^2
$$

Let us cover the 3 cases coming from the [sign flow analysis](#l3-sign-flow):

- when $ model $ produces $ o^3 = y^{truth} $ => $ \delta^3 = 0 $
- when $ model $ produces $ o^3 < y^{truth} $ => $ \delta^3 < 0 $
- when $ model $ produces $ o^3 > y^{truth} $ => $ \delta^3 > 0 $

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 = 0 $ </h3>

As we saw in this [paragraph](#nothing_to_learn), we have nothing to learn in this situation.

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

This is on par with the fact that there is nothing to learn in this situation.

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
\boxed{\hat{w^2} \geq w^2}
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

In this situation, we must beware of the sign of $ o^1 $.

<h4> $ o^1 \geq 0 $ </h4>

$$
\begin{align}
\hat{w^2} &= w^2 - \alpha . \delta w^2 \\
          &= w^2 - \alpha . \delta^{3} . o^1
\end{align}
$$

$$
\boxed{\hat{w^2} \leq w^2}
$$

<h4> $ o^1 < 0 $ </h4>

$$
\begin{align}
\hat{w^2} &= w^2 - \alpha . \delta w^2 \\
          &= w^2 - \alpha . \delta^{3} . o^1
\end{align}
$$

$$
\boxed{\hat{w^2} > w^2}
$$

## Interpretation

In this paragraph we illustrate the intuition behind the **weights** **updates**
computed in the [previous paragraph](#l2-sign-update).

First of all let us recap the different situations we have for our $ \hat{w^2} $ **update**:

| situation                          | model result          | $ \hat{w^2} $     |
| :--------------------------------: | :-------------------: | :---------------: |
| $ delta^3 = 0 $                    | as expected           | keep same $ w^2 $ |
| $ delta^3 < 0 $ and $ o^1 \geq 0 $ | lower than expected   | increase $ w^2 $  |
| $ delta^3 < 0 $ and $ o^1 < 0 $    | lower than expected   | decrease $ w^2 $  |
| $ delta^3 > 0 $ and $ o^1 \geq 0 $ | higher than expected | decrease $ w^2 $  |
| $ delta^3 > 0 $ and $ o^1 < 0 $    | higher than expected | increase $ w^2 $  |

Let us go back to the $ L2 $ $ layer $ definition:

$$
\begin{align}
    L2(X^2, W^2) &= W^2 . X^2          & \text{ with } X^2 = (X^2_1, X^2_2, X^2_3) \\
                 &                     & \text{ and } W^2 = (W^2_1, W^2_2, W^2_3) \\
                 &= W^2_1 . X^2_1 + W^2_2 . X^2_2 + W^2_3 . X^2_3 \\
\end{align}
$$

Said differently:

$$
o^2 = w^2_1 . o^1_1 + w^2_2 . o^1_2 + w^2_3 . o^1_3
$$

In the array above, we spoke about $ w^2 $ but in fact there are several **weights**:
$ w^2_1 $, $ w^2_2 $, $ w^2_3 $.
In order to fix the ideas, we will concentrate on one of them: $ w^2_1 $. The exact same logic applies for the others.

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 = 0 $ </h3>

This is the ideal situation, it is no wonder we must keep the same value for $ w^2_1 $.

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 < 0 $ and $ o^1 \geq 0 $ </h3>

The $ model $ produces a lower than expected output.
The **learning flow** together with the **update** formula tell us we must increase $ w^2_1 $.

This is logical considering that:

- $ \delta^3 < 0 $ literally means: $ o^2 $ is not big enough => $ o^2 $ must be increased
- $ o^1_1 \geq 0 $
- the only part that we can modify is $ w^2_1 $

=> $ w^2_1 $ must be increased so that $ o^2 = w^2_1 . o^1_1 + w^2_2 . o^1_2 + w^2_3 . o^1_3 $ increases.

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 < 0 $ and $ o^1 < 0 $ </h3>

The $ model $ produces a lower than expected output.
The **learning flow** together with the **update** formula tell us we must decrease $ w^2_1 $.

This is logical considering that:

- $ \delta^3 < 0 $ literally means: $ o^2 $ is not big enough => $ o^2 $ must be increased
- $ o^1_1 < 0 $
- the only part that we can modify is $ w^2_1 $

=> $ w^2_1 $ must be decreased so that $ o^2 = w^2_1 . o^1_1 + w^2_2 . o^1_2 + w^2_3 . o^1_3 $ increases.

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 > 0 $ and $ o^1 \geq 0 $ </h3>

The $ model $ produces a higher than expected output.
The **learning flow** together with the **update** formula tell us we must decrease $ w^2_1 $.

This is logical considering that:

- $ \delta^3 > 0 $ literally means: $ o^2 $ is too big => $ o^2 $ must be decreased
- $ o^1_1 \geq 0 $
- the only part that we can modify is $ w^2_1 $

=> $ w^2_1 $ must be decreased so that $ o^2 = w^2_1 . o^1_1 + w^2_2 . o^1_2 + w^2_3 . o^1_3 $ decreases.

<hr style="width: 65%; margin: auto;">

<h3 style="text-align:center; margin-top: 2%;"> $ \delta^3 > 0 $ and $ o^1 < 0 $ </h3>

The $ model $ produces a higher than expected output.
The **learning flow** together with the **update** formula tell us we must increase $ w^2_1 $.

This is logical considering that:

- $ \delta^3 > 0 $ literally means: $ o^2 $ is too big => $ o^2 $ must be decreased
- $ o^1_1 < 0 $
- the only part that we can modify is $ w^2_1 $

=> $ w^2_1 $ must be increased so that $ o^2 = w^2_1 . o^1_1 + w^2_2 . o^1_2 + w^2_3 . o^1_3 $ decreases.

## Example

We can go further in our simple $ model $.
Indeed, we have remarked that $ \delta^3 $ has the same **sign** as the **learning flow** of $ Loss $: $ \delta^4 $.
This means that whatever the final result, the intermediate result $ o^2 $ has the same tendency. The other
way round is also true: if $ o^2 $ appears to be too high,
then the final result will be too high aswell (compared to the expected result).

We may also consider that $ L1 $ is an $ Input \text{ } 1D $ $ layer $. This means that $ o^1 = data $.

Let us recap the $ L2 $ formula with the **weights** values of the [weights article]({% post_url 2021-08-19-weights %}):

$$
\begin{align}
o^2 &= L2(o^1) \\
    &= w^2 * o^1 \\
    &= \frac{1}{200} * o^1_1 - \frac{3 000}{11 600 000} * o^1_2 + \frac{1}{5 800} * o^1_3 \\
    &= \frac{1}{200} * data_1 - \frac{3 000}{11 600 000} * data_2 + \frac{1}{5 800} * data_3
\end{align}
$$

This brings us to the conclusion that $ data_1 $ and $ data_3 $ have a positive impact on the final result because:
$ w^2_1 = \frac{1}{200} > 0 $ and $ w^2_3 = \frac{1}{5 800} > 0 $ while $ data_2 $ has a negative impact on the
final result because $ w^2_2 = - \frac{3 000}{11 600 000} < 0 $.

We can "verify" this by considering what the **data** was
in the [first article]({% post_url 2021-08-05-general-concepts %}):

$$
data = (\text{broccoli}, \text{Tagada strawberries}, \text{workout hours})
$$

It seems coherent that "broccoli" and "workout hours" tend to increase the result (1 => good shape),
while "Tagada strawberries" tend to decrease the final result (0 => bad shape).

## Back to the Learning Flow

There are 2 paragraphs that were not used during our [sign update analysis](#l2-sign-update):
the [L2 Sign Flow](#l2-sign-flow) and the [L1 Sign Flow](#l1-sign-flow) paragraphs.

In fact we have already seen this aspect but the **learning flow**'s only purpose is to be able to compute $ \delta w $.
In our current example, the $ L2 $ $ layer $ is the only one to have **weights**. Hence, the **learning flow**
back propagation is necessary until we get $ \delta^3 $.

If we look back at the last diagram of the [first paragraph](#example),
it is clear we just have to compute the **learning flow**
for $ Loss $ and for the $ L3 $ $ layer $.
This means we could have skipped the computation of the **learning flow** for $ L2 $ and for $ L1 $ in the
[backward pass article]({% post_url 2021-08-13-backward-pass %}) :smiling_imp:

## Conclusion

In the [interpretation paragraph](#interpretation), we illustrated that the **weights** **update** comes
from the fact that the final result is too high or too low compared to the expected result
and that the **weights** are the only "moving part"
(see the [weights article]({% post_url 2021-08-19-weights %})) to compensate.
From there, the **learning flow** just helps cascading the impact on the different intermediate levels.
