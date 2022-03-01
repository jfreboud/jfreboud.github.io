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

<a id="convolution-structure1" class="anchor">
![Convolution](/_assets/images/layers/Convolution1.png)
</a>

In order to fix the ideas the diagram above shows an example $ L^k $ $ Convolution $ $ layer $
where $ ch^{k-1,1} $, $ ch^{k-1,2} $, $ ch^{k-1,3} $ 
are the 3 input **channels** and $ ch^{k,1} $ and $ ch^{k,2} $ are the two output **channels**.
Note that we use 6 different **convolution kernels** that correspond to the combination: 
$ 2 \textbf{ output channels } * 3 \textbf{ input channels } = 6 $. 

In the following we will zoom in this part of the previous diagram:

<a id="convolution-structure2" class="anchor">
![Convolution](/_assets/images/layers/Convolution2.png)
</a>

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

## Backward Pass for the Learning Flow 

Let us focus on the computation of the **learning flow** for the central input **neuron** of our 
$ L^{k} $ $ Convolution $ $ layer $: 

$$ 
\delta^{k,1}_{3,3} = \frac{\partial Loss}{\partial X^{k,1}_{3,3}}(ch^{k-1,1}_{3,3})
$$

The interesting variable is $ X^{k,1}_{3,3} $, visible in the input grid in [this diagram](#convolution-structure2). 
Let us find its **impacts** on the $ Loss $ function.

What is difficult in the $ Convolution $ case is the dual operation we saw in the 
[previous article]({% post_url 2022-01-22-second-dimension %}):

1. the spatial context which is captured by the **convolution kernels** (this is specific to the 2D case)
2. the combination of previous **representations** (this is a legacy of the 1D case)

We are trying to resolve the **impacts** of $ X^{k,1}_{3,3} $ on the output grid of our $ L^{k} $ $ layer $. 
Let us consider the diagram below coming from the **forward pass**.

![Convolution](/_assets/images/layers/Convolution4.png)

What is important is to note that during the **forward pass**, there are multiple output **neurons** that have 
captured the spatial context from the 
**neuron** we are studying $ ch^{k-1,1}_{3,3} $ during their own computation. This is due to the 1st point.

For example let us look at $$ ch^{k-1,1}_{2,2} $$. Looking at the [previous diagram](#convolution-structure1) 
there are 2 output **neurons** that have used this input **neuron**: $$ ch^{k,1}_{2,2} $$ and $$ ch^{k,2}_{2,2} $$. 
This is due to the 2nd point.

For now, we have found 2 output **neurons** of $ L^{k} $ that have used the input **neuron** $$ ch^{k-1,1}_{3,3} $$.
Said differently, $$ ch^{k-1,1}_{3,3} $$ **impacts** $$ ch^{k,1}_{2,2} $$ and $$ ch^{k,2}_{2,2} $$.

In fact there are other **impacts** !
Due to the 1st point here is the list of the **neurons** that capture context from $ ch^{k-1,1}_{3,3} $: 

$$
\begin{matrix}
ch^{k-1,1}_{2,2} & ch^{k-1,1}_{2,3} & ch^{k-1,1}_{2,4} \\
ch^{k-1,1}_{3,2} & ch^{k-1,1}_{3,3} & ch^{k-1,1}_{3,4} \\
ch^{k-1,1}_{4,2} & ch^{k-1,1}_{4,3} & ch^{k-1,1}_{4,4}
\end{matrix}
$$ 

Due to the 2nd point here is the list of the output **neurons** that have used these input **neurons**: 

$$
\begin{matrix}
ch^{k,1}_{2,2} & ch^{k,1}_{2,3} & ch^{k,1}_{2,4} \\
ch^{k,1}_{3,2} & ch^{k,1}_{3,3} & ch^{k,1}_{3,4} \\
ch^{k,1}_{4,2} & ch^{k,1}_{4,3} & ch^{k,1}_{4,4}
\end{matrix}
$$ 

$$
\begin{matrix}
ch^{k,2}_{2,2} & ch^{k,2}_{2,3} & ch^{k,2}_{2,4} \\
ch^{k,2}_{3,2} & ch^{k,2}_{3,3} & ch^{k,2}_{3,4} \\
ch^{k,2}_{4,2} & ch^{k,2}_{4,3} & ch^{k,2}_{4,4}
\end{matrix}
$$ 

We have found $ 2 * 9 = 18 $ output **neurons** that are **impacted** by $ ch^{k-1,1}_{3,3} $ !

We add these 18 **impacts**, using the **chain rule** and the "future" **learning flow** to obtain the 
"**impact** formula": 

$$ 
\begin{align}
\delta^{k,1}_{3,3} &= & \delta^{k+1,1}_{2,2} . \frac{\partial X^{k+1,1}_{2,2}}{X^{k,1}_{3,3}}(ch^{k-1,1}_{2,2}) 
                        + \delta^{k+1,1}_{2,3} . \frac{\partial X^{k+1,1}_{2,3}}{X^{k,1}_{3,3}}(ch^{k-1,1}_{2,3}) \\
                   &  & + ... + \delta^{k+1,1}_{4,4} . \frac{\partial X^{k+1,1}_{4,4}}{X^{k,1}_{3,3}}(ch^{k-1,1}_{4,4}) \\
                   &  & + \delta^{k+1,2}_{2,2} . \frac{\partial X^{k+1,2}_{2,2}}{X^{k,1}_{3,3}}(ch^{k-1,2}_{2,2}) 
                        + \delta^{k+1,2}_{2,3} . \frac{\partial X^{k+1,2}_{2,3}}{X^{k,1}_{3,3}}(ch^{k-1,2}_{2,3}) \\
                   &  & + ... + \delta^{k+1,2}_{4,4} . \frac{\partial X^{k+1,2}_{4,4}}{X^{k,1}_{3,3}}(ch^{k-1,2}_{4,4}) 
\end{align}
$$

Let us compute one of these: 

$$ 
\begin{align}
\frac{\partial X^{k+1,1}_{2,2}}{X^{k,1}_{3,3}}(ch^{k-1,1}_{2,2}) &= 
\frac{\partial (... + X^{k,1}_{3,3} * ker^{1,1}_{2,2} + ... + b^{k,1})}{\partial X^{k,1}_{3,3}} \\
                                          &= ker^{1,1}_{2,2} 
\end{align}
$$

We obtain our final "**impact** formula":

$$ 
\boxed{
\begin{align}
\delta^{k,1}_{3,3} &= & \delta^{k+1,1}_{2,2} . ker^{1,1}_{2,2} 
                        + \delta^{k+1,1}_{2,3} . ker^{1,1}_{2,1} \\ 
                   &  & + ... + \delta^{k+1,1}_{4,4} . ker^{1,1}_{0,0} \\
                   &  & + \delta^{k+1,2}_{2,2} . ker^{2,1}_{2,2} 
                        + \delta^{k+1,2}_{2,3} . ker^{2,1}_{2,1} \\  
                   &  & + ... + \delta^{k+1,2}_{4,4} . ker^{2,1}_{0,0} 
\end{align}
}
$$

We have just computed the **information flow** for one of the input **neurons**: $ \delta^{k,1}_{3,3} $. 
We have to do the same for every other input **neurons** of the first channel and then repeat for the other 
channels: $ \delta^{k,2} $ and $ \delta^{k,3} $...

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
