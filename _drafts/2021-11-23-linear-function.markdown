---
layout: post
title:  "Linear Function"
category: network
date:   2021-11-23
excerpt: >-
  Investigating the global function of the linear network.
---

## Introduction

In the [previous article]({% post_url 2021-11-17-weights-balancing %}), we illustrated the balance that occurs 
after each **weights** **update** of the [gradient descent algorithm]({% post_url 2021-08-23-gradient-descent %}). 

In this article we will investigate the purpose of the $ Linear $ **network**. 

Note that a **network** and a $ model $ are 
very similar. We use the word **network** to refer to the general structure of the $ model $. 

Also note that the simple 
$ model $ used so far (see the [second article]({% post_url 2021-08-06-inside-the-model %})) 
is not linear in the mathematical sense because of the use of 
$ activation $ $ layers $ (which imply a non linearity, 
see the [activation layer article]({% post_url 2021-10-06-activation %})). 
We call linear in the sense that the main **learning** $ layer $ is a $ Linear $ one 
(see the [linear layer article]({% post_url 2021-09-19-linear %})).

## From a Neuron Perspective

Let us go back to one situation we saw in the [linear layer article]({% post_url 2021-09-19-linear %}). 
Let us consider one $ L^k $ $ Linear $ $ layer $ and suppose it has 2 output **neurons**: $ o^{k}_1 $ and $ o^{k}_2 $. 
Let us suppose $ L^{k-1} $ produces 3 output **neurons**: $ o^{k-1}_1 $, $ o^{k-1}_2 $ and $ o^{k-1}_3 $. 

During the **forward pass** of $ L^k $ we expect to have the formula: 

$$ 
\begin{align}
o^{k}_1 &= w^{k, 1}_1 . o^{k-1}_1 + w^{k, 1}_2 . o^{k-1}_2 + w^{k, 1}_3 . o^{k-1}_3 + b^{k}_1 \\
o^{k}_2 &= w^{k, 2}_1 . o^{k-1}_1 + w^{k, 2}_2 . o^{k-1}_2 + w^{k, 2}_3 . o^{k-1}_3 + b^{k}_2
\end{align}
$$ 

In this paragraph we want to illustrate the role of one **neuron** as an individual. For the moment we will suppose 
that $ b^{k}_1 = 0 $ and $ b^{k}_2 = 0 $, our formula become: 

$$ 
\begin{align}
o^{k}_1 &= w^{k, 1}_1 . o^{k-1}_1 + w^{k, 1}_2 . o^{k-1}_2 + w^{k, 1}_3 . o^{k-1}_3 \\
o^{k}_2 &= w^{k, 2}_1 . o^{k-1}_1 + w^{k, 2}_2 . o^{k-1}_2 + w^{k, 2}_3 . o^{k-1}_3
\end{align}
$$ 

Our main question here is: what makes $ o^{k}_1 $ and $ o^{k}_2 $ different ?

They both share the same immediate "**data input**": $ o^{k-1}_1 $, $ o^{k-1}_2 $ and $ o^{k-1}_3 $. 
But they create different "meaning" of these "**data input**": this "meaning" is what we called 
**representation** in the [second article]({% post_url 2021-08-06-inside-the-model %}).

We immediately understand that what actually build the different **representations** are the **weights**.
Thanks to the **weights**, the different output **neurons** will be differently correlated to the immediate 
"**data input**" of the previous $ L^{k-1} $ $ layer $.

For example, the **weights** for $ o^{k}_1 $ could be: $ w^{k, 1}_1 > 0 $, $ w^{k, 1}_2 < 0 $ and $ w^{k, 1}_3 > 0 $.
Thanks to the $ o^{k}_1 $ formula above, this means that $ o^{k}_1 $ increases when the immediate 
"**data input**" $ o^{k-1}_1 $ is $ (o^{k-1}_1 > 0, o^{k-1}_2 < 0, o^{k-1}_3 > 0) $. 

To be more specific in the previous case: 
- locking $ o^{k-1}_2 $ and $ o^{k-1}_3 $ we have: $ o^{k-1}_1 $ <span style="color:green">↑</span> => $ o^{k}_1 $ <span style="color:green">↑</span> and $ o^{k-1}_1 $ <span style="color:red">↓</span> => $ o^{k}_1 $ <span style="color:red">↓</span>
- locking $ o^{k-1}_1 $ and $ o^{k-1}_3 $ we have: $ o^{k-1}_2 $ <span style="color:red">↓</span> => $ o^{k}_1 $ <span style="color:green">↑</span> and $ o^{k-1}_2 $ <span style="color:green">↑</span> => $ o^{k}_1 $ <span style="color:red">↓</span>
- locking $ o^{k-1}_1 $ and $ o^{k-1}_2 $ we have: $ o^{k-1}_3 $ <span style="color:green">↑</span> => $ o^{k}_1 $ <span style="color:green">↑</span> and $ o^{k-1}_3 $ <span style="color:red">↓</span> => $ o^{k}_1 $ <span style="color:red">↓</span>

We can now artificially attribute a "meaning" to our different **neurons**. 
According to the special "meaning" attributed to $ o^{k-1} $, we can think about the "meaning" of $ o^{k}_1 $.
For example, if $ o^{k-1} $ "means": 
$ (\text{eat vegetables}, \text{ had a rare operation as a child}, \text{ workout regularly}) $, then 
the "meaning" of the $ o^k_1 $ output **neuron** could be: "be in good shape" because of the **weights**' signs 
fixed earlier and the "logic" of the immediate "**data input**".

In the same example, the **weights** for $ o^k_2 $ could be: 
$ w^{k, 2}_1 < 0 $, $ w^{k, 2}_2 = 0 $ and $ w^{k, 2}_3 < 0 $. 
In that way the "meaning" for the $ o^k_2 $ output **neuron** could be: 
"do not have a regular habit" because $ o^k_2 $ is linked negatively to $ o^{k-1}_1 $ (eat vegetables), 
$ o^{k-1}_3 $ (workout regularly) and not linked at all to $ o^{k-1}_2 $ (had a rare operation as a child).

What we must keep in mind here is that the **weights** give some sort of uniqueness to every **neuron**. 
We could even give some "meaning" to them but that is the hardest part :smiling_imp:

## Link with the Activation Potential in Biology

In the [previous paragraph](#from-a-neuron-perspective), we saw the natural correlation linking the output **neurons** 
to the input **neurons** thanks to the **weights** of the $ Linear $ $ layer $.

Now we want to elaborate on the role of an $ activation $ $ layer $ that would follow the previous $ L^k $ $ Linear $ 
$ layer $.

Adding the non linearity of the $ activation $ function produces a sort of decision making linked to a threshold 
(set to 0 in the $ ReLU $ case).
In our example, the "final" output $ neuron $ would not activate until we have enough quantities of the **data input**.

## Backward Balancing

![Linear](/_assets/images/layers/Linear6.png)

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
