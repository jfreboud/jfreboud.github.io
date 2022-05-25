---
layout: post
title:  "Normalization Layer"
category: layer
date:   2022-05-14
excerpt: >-
  Normalization layer allows us to stabilize learning. 
---

## Introduction

In the [batch learning article]({% post_url 2021-08-24-batch-learning %}), we saw a method that helps stabilizing 
the **learning  process**. 
It is a global method that modifies the behavior of the **gradient descent** algorithm itself.

In this article we will discuss a much more localised method: applying **normalization** on the $ layer $ scope.

## The Formula

The idea is that instead of considering one global methodology that "stabilizes" **gradients**, we will include 
a specific operation in some $ layers $ of our Deep learning $ model $ in order to "stabilize" the output **neurons** 
ot these specific chosen $ layers $.

First of all, how should we "stabilize" the output **neurons** ?

In order to "stabilize", we will compute a "norm", representing the typical average output of the 
**neurons** of our chosen $ layer $. Let us note this $ mean $ value $ \mu $: 

$$ 
\mu = \frac{1}{\textbf{nb elements}} . \sum_{elem=0}^{\textbf{nb elements} - 1} o_{elem}
$$

We also want to know about the typical difference we might observe between one output **neuron** of our 
considered $ layer $ and the $ mean $ above. This is called the **standard deviation**, noted $ \sigma $: 

$$ 
\sigma = \frac{1}{\textbf{nb elements}} . \sqrt{\sum_{elem=0}^{\textbf{nb elements} - 1} (o_{elem} - \mu)^2 + \epsilon}
$$

with $$ \epsilon \approx 0 $$.

## The Shapes of Normalization

We have the two principal elements needed to "stabilize" the output **neurons** of one $ layer $. But there is 
still an important problem to solve...

In the [linear function]({% post_url 2021-12-12-linear-function %}) and 
the [second dimension]({% post_url 2022-01-22-second-dimension %}) articles, we respectively saw how the 
$ Linear $ and the $ Convolution $ $ layers $ build **representations**.

For the $ Linear $ $ layer $, the "meaning" is hold by each of the output **neurons** themselves. 
For the $ Convolution $ $ layer $, the "meaning" is hold by the different **channels**, composed of output **neurons** 
organized in a grid.

During the **normalization** process we will have to preserve these "meaning" intact, avoiding to mix output 
**neurons** that do not **represent** "the same thing".

### The Linear Layer

Let us suppose $ L^{k-1} $ is a $ Linear $ $ layer $ that produces 3 output **neurons**. We want to 
compute its "norm" elements as in the [previous paragraph](#the-formula).

![BN](/_assets/images/layers/BN1.png)

In the diagram above, we are computing the "norm" elements mixing different output **neurons** which is a bad idea. 

In order to cope with this problem, we are going to use the idea of 
the [batch learning article]({% post_url 2021-08-24-batch-learning %}) and compute the "norm" elements thanks 
to output **neurons** that are in the same "meaning" but different **batch**.

In the diagram below, we have drawn the same $ layer $ as before but for a **batch** size of 3.

![BN](/_assets/images/layers/BN2.png)

### The Convolution Layer 

Let us suppose $ L^{k-1} $ is a $ Convolution $ $ layer $ that produces 3 output **channels** of a certain size 
$ (width, height) $. We want to compute its "norm" elements as in the [previous paragraph](#the-formula).

![BN](/_assets/images/layers/BN3.png)

In the diagram above, the proposition was to compute the "norm" elements on each "pixel" so that: 

$$ 
\begin{align}
\mu_{k,0,0}    &= & \frac{1}{3} . (ch_{k-1,1,0,0} + ch_{k-1,2,0,0} + ch_{k-1,3,0,0}) \\ 
\sigma_{k,0,0} &= & \frac{1}{3} . \sqrt{(ch_{k-1,1,0,0} - \mu_{k,0,0})^2 + (ch_{k-1,2,0,0} - \mu_{k,0,0})^2 + (ch_{k-1,3,0,0} - \mu_{k,0,0})^2 + \epsilon}
\end{align}
$$

Once more, we have the same problem as in the [previous paragraph](#the-linear-layer): we are computing the "norm" 
elements mixing output **neurons** of different **channels**.

In order to cope with this problem, we may use the same idea as before but there is a more interesting move to do here 
:smiling_imp:

Indeed, as each of our **channels** is made of multiple **neurons**, the "pixels", we may use these different 
"pixels" to compute the "norm" elements for each **channel**. That way, we are not mixing **neurons** from different 
**channels** together.

![BN](/_assets/images/layers/BN4.png)

In the diagram above, the proposition was to compute the "norm" elements on each **channel** so that: 

$$ 
\begin{align}
\mu_{k,1}    &= & \frac{1}{width . height} . (ch_{k-1,1,0,0} + ... + ch_{k-1,1,0,width-1} \\
             &  & + ... \\
             &  & + ch_{k-1,1,height-1,0} + ... + ch_{k-1,1,height-1,width-1}) \\
\sigma_{k,1,tmp}^2 &= & (ch_{k-1,1,0,0} - \mu_{k,1})^2 + ... + (ch_{k-1,1,0,width-1} - \mu_{k,1})^2 \\
                   &  & + ... \\
                   &  & + (ch_{k-1,1,height-1,0} - \mu_{k,1})^2 + ... + (ch_{k-1,1,height-1,width-1} - \mu_{k,1})^2 \\
\sigma_{k,1} &= & \frac{1}{width . height} . \sqrt{\sigma_{k,1,tmp}^2 + \epsilon}
\end{align}
$$

Finally, the method used in the [previous paragraph](#the-linear-layer) also applies in a **batch learning** setting. 
In the diagram below, we have drawn the same $ layer $ as before but for a **batch** size of 2.

![BN](/_assets/images/layers/BN5.png)

## Forward Pass

From now on we will make the assumption we have already chosen the good shape for **normalization** as mentioned 
in the [previous paragraph](#the-shapes-of-normalization): let us suppose that we are looking at 3 elements that 
do not mix different output **neurons**. These 3 elements may be the 3 **batches** of one output **neuron** of 
a $ Linear $ $ layer $ or the 3 **batches** of one really small **channel** of size (1,1) 
in a $ Convolution $ $ layer $.

We already know how to compute the "norm" elements for these 3 elements, noted $ o^1_{k-1} $, $ o^2_{k-1} $ 
and $ o^3_{k-1} $:

$$ 
\begin{align}
\mu_k    &= & \frac{1}{3} . (o^1_{k-1} + o^2_{k-1} + o^3_{k-1}) \\ 
\sigma_k &= & \frac{1}{3} . \sqrt{(o^1_{k-1} - \mu_k)^2 + (o^2_{k-1} - \mu_k)^2 + (o^3_{k-1} - \mu_k)^2 + \epsilon}
\end{align}
$$

The question is now: how do we use these "norm" elements in order to build the "normalized" output **neurons** of our 
$ L^{k} $ $ Normalization $ layer $.

We are going to transform each of these output **neurons** so that the new average output and standard deviation for 
these elements are "compensated" by the "norm" elements computed before: 

$$ 
\begin{align}
o^1_{k'}   &= &  \frac{o^1_{k-1} - \mu_k}{\sigma_k} \\ 
o^2_{k'}   &= &  \frac{o^2_{k-1} - \mu_k}{\sigma_k} \\ 
o^3_{k'}   &= &  \frac{o^3_{k-1} - \mu_k}{\sigma_k} 
\end{align}
$$

### The New "Norm" Elements

We have transformed our **neurons** $$ o^1_{k-1} $$, $$ o^2_{k-1} $$ and $$ o^3_{k-1} $$ of "norm" elements 
($$ \mu_k $$, $$ \sigma_k $$ ) into $$ o^1_{k'} $$, $$ o^2_{k'} $$ and $$ o^3_{k'} $$ of "norm" elements 
($$ \mu_{k'} $$, $$ \sigma_{k'} $$ ). Let us compute these new "norm" elements.

$$ 
\begin{align}
\mu_{k'}    &= & \frac{1}{3} . (o^1_{k'} + o^2_{k'} + o^3_{k'}) \\ 
            &= & \frac{1}{3} . (\frac{o^1_{k-1} - \mu_k}{\sigma_k + \epsilon} + 
                 \frac{o^2_{k-1} - \mu_k}{\sigma_k + \epsilon} + 
                 \frac{o^3_{k-1} - \mu_k}{\sigma_k + \epsilon}) \\
            &= & \frac{1}{\sigma_k + \epsilon} . (\frac{1}{3} . (o^1_{k-1} + o^2_{k-1} + o^3_{k-1}) - 
                                                  \frac{1}{3} . 3\mu_k) \\
            &= & \frac{1}{\sigma_k + \epsilon} . (\mu_k - \mu_k) \\
            &= & 0
\end{align}
$$

$$ 
\begin{align}
\sigma_{k'} &= & \frac{1}{3} . \sqrt{(o^1_{k'} - \mu_{k'})^2 + (o^2_{k'} - \mu_{k'})^2 + (o^3_{k'} - \mu_{k'})^2 + \epsilon} \\
            &= & \frac{1}{3} . \sqrt{(o^1_{k'})^2 + (o^2_{k'})^2 + (o^3_{k'})^2 + \epsilon} \\
            &= & \frac{1}{3} . \sqrt{(\frac{o^1_{k-1} - \mu_k}{\sigma_k})^2 + 
                                     (\frac{o^2_{k-1} - \mu_k}{\sigma_k})^2 + 
                                     (\frac{o^3_{k-1} - \mu_k}{\sigma_k})^2 + \epsilon} \\ 
            &= & \frac{1}{\sigma_k} . \frac{1}{3} . \sqrt{(o^1_{k-1} - \mu_k)^2 + 
                                                          (o^2_{k-1} - \mu_k)^2 + 
                                                          (o^3_{k-1} - \mu_k)^2 + \epsilon} \\ 
            &= & \frac{1}{\sigma_k} . \sigma_k \\ 
            &= & 1 
\end{align}
$$

The average and standard deviation of our new output **neurons** are respectively 0 and 1. 

### The Final Shift

We have transformed the output **neurons** from "norm" elements ($$ \mu_k $$, $$ \sigma_k $$) to (0, 1).
 
We could stop the transformation here, but it would be forgetting that we can insert some magical deep learning move 
:smiling_imp:

What if the new "norm" elements do not represent what is best for our $ L^k $ ? What if we could find some new 
average and standard deviation that serve our $ model $ better ?

This is what we will fix with 2 **weights** $$ \Gamma_{k+1} $$ and $$ B_{k+1} $$: 

- $$ B_{k+1} $$ variable of value $$ \beta_k $$, the new average
- $$ \Gamma_{k+1} $$ variable of value $$ \gamma_k $$, the new standard deviation

We have already spoken about **weights** in the [weights article]({% post_url 2021-08-19-weights %}), 
these 2 **weights** will be modified during the **learning phase** so that their modification better suits the 
$ model $'s needs.

The final transform is:

$$ 
\begin{align}
o^1_{k}   &= &  \beta_k + \gamma_k . o^1_{k'} \\ 
o^2_{k}   &= &  \beta_k + \gamma_k . o^2_{k'} \\ 
o^3_{k}   &= &  \beta_k + \gamma_k . o^3_{k'}
\end{align}
$$

Without doing any computation, the new "norm" elements of these final output **neurons** are ($$ \beta $$, $$ \gamma $$).

Let us recap our $ L^k $ $ Normalization $ $ layer $.

<a id="BN-structure" class="anchor">
![BN](/_assets/images/layers/BN6.png)
</a>

## Backward Pass for the Learning Flow 

We will have to be very careful to compute the **backward pass** for the $ Normalization $ $ layer $ because 
as we saw in [this diagram](#BN-structure), we have several intermediate elements that are $ function $ of other 
elements. In order to compute the different **impacts** correctly, we have to follow the order of the 
[backward pass]({% post_url 2021-08-13-backward-pass %}) once more.

First of all, let us see the different $ function $ that appear in the [previous diagram](#BN-structure): 

- $$ X^1_k $$, $$ X^2_k $$, $$ X^3_k $$: these functions only depend on elements that come in a previous $ layer $. 
Computing their own **back propagation** would be the task of the $$ L^{k-1} $$ $$ layer $$, not the current 
$$ L^k $$ one.
- $$ M_{k+1} $$ clearly depends on $$ X^1_k $$, $$ X^2_k $$, $$ X^3_k $$.
- $$ \Sigma_{k+1} $$ depends on $$ X^1_k $$, $$ X^2_k $$, $$ X^3_k $$ and on $$ M_{k+1} $$.
- $$ X^1_{k+1} $$, $$ X^2_{k+1} $$, $$ X^3_{k+1} $$ depend on: 
$$ X^1_k $$, $$ X^2_k $$, $$ X^3_k $$, $$ M_{k+1} $$, $$ \Sigma_{k+1} $$, $$ \beta_{k+1} $$ and $$ \gamma_{k+1} $$.
- $$ \Gamma_{k+1} $$, $$ B_{k+1} $$ hopefully do not depend on anything, they will be **updated** in a later 
paragraph.

Thus, in order to take into account the different **impacts**, we have to compute our 
**learning flow** in the following order: 

1. $$ \delta X^1_{k+1} $$, $$ \delta X^2_{k+1} $$, $$ \delta X^3_{k+1} $$ are given by construction of 
the **backward pass** (considering we are looking to back propagate the layer $ L^k $).
2. $ \delta \Sigma_{k+1} $
3. $ \delta M_{k+1} $
4. $$ \delta X^1_k $$, $$ \delta X^2_k $$, $$ \delta X^3_k $$

### $$ \delta \Sigma_{k+1} $$ 

As we already know, the only functions that depend on $ \Sigma_{k+1} $ are: 
$$ X^1_{k+1} $$, $$ X^2_{k+1} $$, $$ X^3_{k+1} $$. Said differently, $ \Sigma_{k+1} $ **impacts**: 
$$ X^1_{k+1} $$, $$ X^2_{k+1} $$ and $$ X^3_{k+1} $$.

We have computed several "**impact** formula" since the 
[linear layer article]({% post_url 2021-09-19-linear %}), let us get straight to the point: 

$$ 
\delta \sigma_{k} = \delta^{k+1}_1 . \frac{\partial X^1_{k+1}}{\partial \Sigma_{k+1}}(.) + 
\delta^{k+1}_2 . \frac{\partial X^2_{k+1}}{\partial \Sigma_{k+1}}(.) + 
\delta^{k+1}_3 . \frac{\partial X^2_{k+1}}{\partial \Sigma_{k+1}}(.)
$$

We have put $$ (.) $$ to remind we must evaluate the $ derivative $ $ function $. 
Yet, we hide the value where the $ function $ must be evaluated, 
see the [linear layer article]({% post_url 2021-09-19-linear %}).

Let us compute $$ \frac{\partial X^1_{k+1}}{\partial \Sigma_{k+1}} $$:

$$ 
\begin{align}
\frac{\partial X^1_{k+1}}{\partial \Sigma_{k+1}} &= 
\frac{\partial (B_{k+1} + \Gamma_{k+1} . X^1_{k+1})}{\partial \Sigma_{k+1}} \\
&= \frac{\partial (B_{k+1} + \Gamma_{k+1} . \frac{X^1_{k} - M_{k+1}}{\Sigma_{k+1}})}{\partial \Sigma_{k+1}} \\
&= \Gamma_{k+1} . (X^1_{k} - M_{k+1}) . \frac{\partial (\frac{1}{\Sigma_{k+1}})}{\partial \Sigma_{k+1}} \\
&= \Gamma_{k+1} . (X^1_{k} - M_{k+1}) . \frac{-1}{(\Sigma_{k+1})^2} \\
\frac{\partial X^1_{k+1}}{\partial \Sigma_{k+1}} &= 
- \frac{\Gamma_{k+1}}{(\Sigma_{k+1})^2} . (X^1_{k} - M_{k+1})
\end{align}
$$

After having evaluated the previous $ derivative $ $ function $, we obtain the "**impact** formula" updated:

$$ 
\boxed{\delta \sigma_{k} = - \frac{\gamma_k}{(\sigma_k)^2} . 
(\delta^{k+1}_1 . (o^1_{k-1} - \mu_{k}) + 
 \delta^{k+1}_2 . (o^2_{k-1} - \mu_{k}) + 
 \delta^{k+1}_3 . (o^3_{k-1} - \mu_{k}))}
$$

### $$ \delta M_{k+1} $$

$ M_{k+1} $ **impacts** $$ X^1_{k+1} $$, $$ X^2_{k+1} $$, $$ X^3_{k+1} $$ and $$ \Sigma_{k+1} $$.

We have the "**impact** formula":

$$ 
\delta \mu_{k} = \delta^{k+1}_1 . \frac{\partial X^1_{k+1}}{\partial M_{k+1}}(.) + 
\delta^{k+1}_2 . \frac{\partial X^2_{k+1}}{\partial M_{k+1}}(.) + 
\delta^{k+1}_3 . \frac{\partial X^2_{k+1}}{\partial M_{k+1}}(.) +
\delta \sigma_{k} . \frac{\partial \Sigma_{k+1}}{\partial M_{k+1}}(.)
$$

Let us compute $$ \frac{\partial X^1_{k+1}}{\partial M_{k+1}} $$: 

$$ 
\begin{align}
\frac{\partial X^1_{k+1}}{\partial M_{k+1}} &= 
\frac{\partial (B_{k+1} + \Gamma_{k+1} . X^1_{k+1})}{\partial M_{k+1}} \\
&= \frac{\partial (B_{k+1} + \Gamma_{k+1} . \frac{X^1_{k} - M_{k+1}}{\Sigma_{k+1}})}{\partial M_{k+1}} \\
&= \Gamma_{k+1} . \frac{\partial (\frac{-M_{k+1}}{\Sigma_{k+1}})}{\partial M_{k+1}} \\
&= \Gamma_{k+1} . \frac{-1}{\Sigma_{k+1}} \\
\frac{\partial X^1_{k+1}}{\partial M_{k+1}} &= - \frac{\Gamma_{k+1}}{\Sigma_{k+1}}
\end{align}
$$

Let us also compute $$ \frac{\partial \Sigma_{k+1}}{\partial M_{k+1}} $$: 

$$ 
\begin{align}
\frac{\partial \Sigma_{k+1}}{\partial M_{k+1}} &= 
\frac{\partial (\frac{1}{3} . \sqrt{(X^1_{k} - M_{k+1})^2 + (X^2_{k} - M_{k+1})^2 + (X^3_{k} - M_{k+1})^2 + \epsilon})}{\partial M_{k+1}} \\
&= \frac{1}{3} . \frac{\frac{\partial ((X^1_{k} - M_{k+1})^2 + (X^2_{k} - M_{k+1})^2 + (X^3_{k} - M_{k+1})^2 + \epsilon)}{\partial M_{k+1}}}{2 . \sqrt{(X^1_{k} - M_{k+1})^2 + (X^2_{k} - M_{k+1})^2 + (X^3_{k} - M_{k+1})^2 + \epsilon}} \\
&= \frac{1}{6 . \sqrt(...)} . \frac{\partial ((X^1_{k} - M_{k+1})^2 + (X^2_{k} - M_{k+1})^2 + (X^3_{k} - M_{k+1})^2 + \epsilon)}{\partial M_{k+1}} \\
&= \frac{1}{6 . \sqrt(...)} . (2 . (X^1_{k} - M_{k+1}) + 2 . (X^2_{k} - M_{k+1}) + 2 . (X^3_{k} - M_{k+1}))  \\
&= \frac{1}{6 . \sqrt(...)} . (2 * 3 * \frac{1}{3} . (X^1_{k} + X^2_{k} + X^3_{k}) - 2 * 3 . M_{k+1}) \\
&= \frac{1}{6 . \sqrt(...)} . (6 . M_{k+1} - 6 . M_{k+1}) \\
\frac{\partial \Sigma_{k+1}}{\partial M_{k+1}} &= 0
\end{align}
$$

We obtain the "**impact** formula" updated:

$$ 
\delta \mu_{k} = - \frac{\gamma_{k}}{\sigma_{k}} . (\delta^{k+1}_1 + \delta^{k+1}_2 + \delta^{k+1}_3) -
\delta \sigma_{k} * 0
$$

$$ 
\boxed{\delta \mu_{k} = - \frac{\gamma_{k}}{\sigma_{k}} . (\delta^{k+1}_1 + \delta^{k+1}_2 + \delta^{k+1}_3)}
$$
