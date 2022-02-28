---
layout: post
title:  "The Convolution Layer"
category: layer
date:   2022-02-19
excerpt: >-
  Let us add the missing piece for the Convolution layer to learn. 
---

## Introduction

In the [previous article]({% post_url 2022-01-22-second-dimension %}), we explored how to build **representations** 
when the **data** input are images: we introduced the $ Convolution $ $ layer $. We used **convolution kernels** that 
capture spatial information of the **channel** they are associated to. 

The question is now: how can the $ Convolution $ $ layer $ learn anything ?

## The Convolution Neural Structure

We will work the same way as in the [linear article]({% post_url 2021-09-19-linear %}), introducing the 
$ Convolution $ **neural structure**. 

Before that, let us recall the diagram of the $ Convolution $ operation we saw at the 
[previous article]({% post_url 2022-01-22-second-dimension %}).

![Convolution](/_assets/images/layers/Convolution1.png)

In order to fix the ideas the diagram above shows an example $ L^k $ $ Convolution $ $ layer $
where $ ch^{k-1,1} $, $ ch^{k-1,2} $, $ ch^{k-1,3} $ 
are the 3 input **channels** and $ ch^{k,1} $ and $ ch^{k,2} $ are the two output **channels**.
Note that we use 6 different **convolution kernels** that correspond to the combination: 
$ 2 \textbf{ output channels } * 3 \textbf{ input channels } = 6 $. 

In the following we will zoom in this part of the previous diagram:

![Convolution](/_assets/images/layers/Convolution2.png)

The elements we called "pixels" in the [previous article]({% post_url 2022-01-22-second-dimension %}) are in fact the 
**neurons** of our $ Convolution $ $ layer $. 
Applying the **convolution kernel** $ k^{1,1} $ to the grid of **neurons** of our input **channel** $ ch^{k-1,1} $ 
allows us to compute a grid of temporary **neurons** $ tmp^{1,1} $. 

Now let us zoom on the computation of one temporary **neuron** $$ tmp^{1,1}_{4,4} $$. In order to simplify 
the diagram, we get rid of some indices, keeping only the indices relating to the positions in the grid. 
We want to compute $ tmp_{4,4} $. 

From what we saw in the [previous article]({% post_url 2022-01-22-second-dimension %}), 
we know how to proceed: we take the center of our $ ker $ **kernel** 
($ ker_{1,1} $ as $ ker $ is a **kernel** of size (3,3) in our example), 
we align it with the **neuron** in the input **channel** ($ ch_{3, 3} $ as $ ch $ is a grid of size 
(7,7) in our example) and we add the different multiplied couples together. 

Here are the different multiplied couples:

![Convolution](/_assets/images/layers/Convolution3.png)

And here we add them together to obtain:  

$$ 
\begin{align}
tmp_{4,4} &= & (ch_{2,2} * ker_{0,0}) + (ch_{2,3} * ker_{0,1}) + (ch_{2,4} * ker_{0,2}) \\
          &  & + (ch_{3,2} * ker_{1,0}) + (ch_{3,3} * ker_{1,1}) + (ch_{3,4} * ker_{1,2}) \\
          &  & + (ch_{4,2} * ker_{2,0}) + (ch_{4,3} * ker_{2,1}) + (ch_{4,4} * ker_{2,2}) 
\end{align}
$$

Now let us zoom out. We have just computed one temporary **neuron** $ tmp^{1,1}_{4,4} $. We have to do the same 
to compute every temporary **neuron** of $ tmp^{1,1} $. 

Let us zoom out again: we have computed one temporary grid $ tmp^{1,1} $, we have to do the same to compute the 
other temporary grids with the other 
combinations: $ ch^{k-1,1} $ and $ ker^{2,1} $ to obtain $ tmp^{2,1} $, 
$ ch^{k-1,2} $ and $ ker^{2,2} $ to obtain $ tmp^{2,2} $, 
$ ch^{k-1,3} $ and $ ker^{1,3} $ to obtain $ tmp^{1,3} $, 
$ ch^{k-1,3} $ and $ ker^{2,3} $ to obtain $ tmp^{2,3} $. 

Finally it is simple to obtain the output **channels**, each being a grid of output **neurons**:

$$
\boxed{
\begin{align}
ch^{k,1} = tmp^{1,1} + tmp^{1,2} + tmp^{1,3} + b^{k,1}\\
ch^{k,2} = tmp^{2,1} + tmp^{2,2} + tmp^{2,3} + b^{k,2}
\end{align}
}
$$

where $ b^{k,1} $ and $ b^{k,1} $ are **biases**.

Looking at one particular **neuron** of the grid, for example $ ch^{k,1}_{4,4} $ we have:

$$ 
\begin{align}
ch^{k,1}_{4,4} &= & (ch^{k-1,1}_{2,2} * ker^{1,1}_{0,0}) + (ch^{k-1,1}_{2,3} * ker^{1,1}_{0,1}) + (ch^{k-1,1}_{2,4} * ker^{1,1}_{0,2}) \\
               &  & + (ch^{k-1,1}_{3,2} * ker^{1,1}_{1,0}) + (ch^{k-1,1}_{3,3} * ker^{1,1}_{1,1}) + (ch^{k-1,1}_{3,4} * ker^{1,1}_{1,2}) \\
               &  & + (ch^{k-1,1}_{4,2} * ker^{1,1}_{2,0}) + (ch^{k-1,1}_{4,3} * ker^{1,1}_{2,1}) + (ch^{k-1,1}_{4,4} * ker^{1,1}_{2,2}) \\ \\
               &  & + (ch^{k-1,2}_{2,2} * ker^{1,2}_{0,0}) + (ch^{k-1,2}_{2,3} * ker^{1,2}_{0,1}) + (ch^{k-1,2}_{2,4} * ker^{1,2}_{0,2}) \\
               &  & + (ch^{k-1,2}_{3,2} * ker^{1,2}_{1,0}) + (ch^{k-1,2}_{3,3} * ker^{1,2}_{1,1}) + (ch^{k-1,2}_{3,4} * ker^{1,2}_{1,2}) \\
               &  & + (ch^{k-1,2}_{4,2} * ker^{1,2}_{2,0}) + (ch^{k-1,2}_{4,3} * ker^{1,2}_{2,1}) + (ch^{k-1,2}_{4,4} * ker^{1,2}_{2,2}) \\ \\
               &  & + (ch^{k-1,3}_{2,2} * ker^{1,3}_{0,0}) + (ch^{k-1,3}_{2,3} * ker^{1,3}_{0,1}) + (ch^{k-1,3}_{2,4} * ker^{1,3}_{0,2}) \\
               &  & + (ch^{k-1,3}_{3,2} * ker^{1,3}_{1,0}) + (ch^{k-1,3}_{3,3} * ker^{1,3}_{1,1}) + (ch^{k-1,3}_{3,4} * ker^{1,3}_{1,2}) \\
               &  & + (ch^{k-1,3}_{4,2} * ker^{1,3}_{2,0}) + (ch^{k-1,3}_{4,3} * ker^{1,3}_{2,1}) + (ch^{k-1,3}_{4,4} * ker^{1,3}_{2,2}) \\ \\
               &  & + b^{k,1}
\end{align}
$$

## The Machine Learning Paradigm

The different **neurons** in the grid correspond to the output of the $ Convolution $ $ layer $. 
Looking back at the [linear layer article]({% post_url 2021-09-19-linear %}), the **neurons** were structured 
as vector of numbers. It seems legitimate that are output are now grids.

Still, we are looking for a "moving part", such as the **weights** we introduced in the 
[weights article]({% post_url 2021-08-19-weights %}). We have a perfect place to consider **weights** variables 
in our $ Convolution $ $ layer $, right in the **convolution kernels**. 

This enables us not to choose these **kernels** at all and rely on the **learning process** operating during 
the **gradient descent** algorithm (see [this article]({% post_url 2021-08-23-gradient-descent %})). 
In a way, it is the **data** that will configure those **kernels** "automatically". 

This is the beautiful paradigm of the **machine learning**. We try to give the $ models $ the power to 
configure the required operations in order to correctly "understand" the **data**. 

Yet, for now, we must also 
keep in mind that we do not give that much power to these $ models $. We actually are just talking about some 
"moving part" parameters, the **weights**, that allow to configure some specific $ layers $. For now these 
specific $ layers $ are: [the linear layer]({% post_url 2021-09-19-linear %}) and the $ Convolution $ $ layer $. 
We see that in fact, the global structure of the $ models $ 
(the structure in $ layers $, the number of $ layers $, the nature of each $ layer $...) 
is still up to the **developer** :smiling_imp:

## Forward Pass

We have already seen how the $ Convolution $ $ layer $ computes its **forward pass**, it merely consists 
in applying the operation described in the [previous paragraph](#the-convolution-neural-structure). 

## Backward Pass

Because the $ Convolution $ $ layer $ has **weights**, the **backward pass** will be composed of:

- the **backward pass** for the **learning flow**
- the **backward pass** for the **weights**

## Backward Pass for the Learning Flow 

We are currently focusing on the $ L^{k} $ $ layer $, trying to compute:

$$ 
\delta^{k} = \frac{\partial Loss}{\partial X^{k}}(o^{k-1})
$$

In the [backward pass article]({% post_url 2021-08-13-backward-pass %}), we would use the **chain rule** in order 
to compute the explicit formula for $ \frac{\partial Loss}{\partial X^{k}} $.

We will see how to obtain this $ \delta^{k} $ with a more straight forward approach. 

The principal idea is to go back to the very structure of $ L^{k} $ in order to find the impacts of $ X^{k} $ 
on the $ Loss $ function, knowing that the "future" 
**learning flow** has already been computed (by definition of the **backward pass**). 

The structure for the $ L^{k} $ $ layer $ is: 
- 2 output **neurons** 
- 3 input **neurons**. 

$ \delta^{k+1}_1 $ and $ \delta^{k+1}_2 $ are the "future" **learning flow**: we admit they have already been 
computed.
We must back propagate the **learning flow** to $ \delta^{k}_1 $, $ \delta^{k}_2 $ and $ \delta^{k}_3 $.

![Linear](/_assets/images/layers/Linear5.png)

## Backward Pass for the Weights

In this paragraph we continue focusing on the $ L^{k} $ $ layer $. This time we want to compute 
the opposite direction of its **weights**' **update**: 

$$ 
\delta w^{k} = \frac{\partial Loss}{\partial W}(o^{k-1})
$$

We will use the exact same strategy as in the [last paragraph](#backward-pass-for-the-learning-flow). 
The principal idea is to go back to the very structure of $ L^{k} $ in order to find the impacts of $ W^{k} $ 
on the $ Loss $ function, knowing that the "future" 
**learning flow** has already been computed (by definition of the **backward pass**). 

<a id="linear-structure3" class="anchor">
![Linear](/_assets/images/layers/Linear9.png)
</a>

<a id="linear-structure4" class="anchor">
![Linear](/_assets/images/layers/Linear10.png)
</a>

There are two more **weights** we have to **update** during the **learning phase**: 
$ B^{k}_1 $ and $ B^{k}_2 $, the **biases**. 
Thus, we will have to compute $ \delta b^{k}_1 $ and $ \delta b^{k}_2 $.

## Conclusion
