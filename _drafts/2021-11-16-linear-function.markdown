---
layout: post
title:  "Linear Representations"
category: layer
date:   2021-10-31
excerpt: >-
  Looking back at the simple "Example" model to illustrate the weights' balancing over time.
---

## Introduction

## Example

We can go further in our analysis of the simple $ model $. 
Indeed, we have remarked that $ \delta^3 $ has the same **sign** as the **learning flow** of $ Loss $: $ \delta^4 $ 
(see [this paragraph](#l3-sign-flow)). 
This means that whatever the final result, the intermediate result $ o^2 $ has the same tendency: if the final 
result is too high compared to the expected result ($ \delta^4 > 0 $), 
then $ o^2 $ will be too high aswell ($ \delta^3 > 0 $). The other 
way round is also true: if $ o^2 $ appears to be too high, 
then the final result will be too high aswell.

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

## Conclusion
