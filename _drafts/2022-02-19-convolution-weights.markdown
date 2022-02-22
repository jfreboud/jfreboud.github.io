---
layout: post
title:  "Convolution Weights"
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

In order to fix the ideas the diagram above shows an example $ Convolution $ 
where $ ch^{1,1} $, $ ch^{1,2} $, $ ch^{1,3} $ 
are the 3 input **channels** and we create two output **channels**: $ ch^{2,1} $ and $ ch^{2,2} $.
Note that we use 6 different **convolution kernels** that correspond to the combination: 
$ 2 \textbf{ output channels } * 3 \textbf{ input channels } = 6 $. 

In the following we will zoom in this part of the previous diagram:

![Convolution](/_assets/images/layers/Convolution2.png)

The elements we called "pixels" in the [previous article]({% post_url 2022-01-22-second-dimension %}) are in fact the 
**neurons** of our $ Convolution $ $ layer $. 
Applying the **convolution kernel** $ k^{1,1} $ to the grid of **neurons** of our input **channel** $ ch^{1,1} $ 
allows us to compute a grid of temporary **neurons** $ tmp^{1,1} $. 

Now let us zoom on the computation of one temporary **neuron** $$ tmp^{1,1}_{4,4} $$. In order to simplify 
the diagram, we get rid of some indices, keeping only the indices relating to the positions in the grid. 
We want to compute $ tmp_{4,4} $. 

From what we saw in the [previous article]({% post_url 2022-01-22-second-dimension %}), 
we know how to proceed: we take the center of our $ k $ **kernel** ($ k_{1,1} $ as $ k $ is a **kernel** of size (3,3) 
in our example), 
we align it with the **neuron** in the input **channel** ($ ch_{3, 3} $ as $ ch $ is a grid of size 
(7,7) in our example) and we add the different multiplied couples together. 

Here are the different multiplied couples:

![Convolution](/_assets/images/layers/Convolution3.png)

And here we add them together to obtain:  

$$ 
\begin{align}
tmp_{4,4} &= & (ch_{2,2} * k_{0,0}) + (ch_{2,3} * k_{0,1}) + (ch_{2,4} * k_{0,2}) \\
          &  & + (ch_{3,2} * k_{1,0}) + (ch_{3,3} * k_{1,1}) + (ch_{3,4} * k_{1,2}) \\
          &  & + (ch_{4,2} * k_{2,0}) + (ch_{4,3} * k_{2,1}) + (ch_{4,4} * k_{2,2}) 
\end{align}
$$

Now let us zoom out. We have just computed one temporary **neuron** $ tmp^{1,1}_{4,4} $. We have to do the same 
to compute every temporary **neuron** of $ tmp^{1,1} $. 

Let us zoom out again: we have computed one temporary grid $ tmp^{1,1} $, we have to do the same to compute the 
other temporary grids with the other 
combinations: $ ch^{1,1} $ and $ k^{2,1} $ to compute $ tmp^{2,1} $, 
$ ch^{1,2} $ and $ k^{2,2} $ to obtain $ tmp^{2,2} $, 
$ ch^{1,3} $ and $ k^{1,3} $ to obtain $ tmp^{1,3} $, 
$ ch^{1,3} $ and $ k^{2,3} $ to obtain $ tmp^{2,3} $. 

Finally it is simple to obtain the output **channels**, each being a grid of output **neurons**:

$$
\begin{align}
ch^{2,1} = tmp^{1,1} + tmp^{1,2} + tmp^{1,3} \\
ch^{2,2} = tmp^{2,1} + tmp^{2,2} + tmp^{2,3}
\end{align}
$$

## The Choice Paradigm

In fact, we will not choose them at all :smiling_imp:
We will rely on the **data** driven approach for the $ model $ to learn these **kernels** by itself. 
We have one last problem with the 


## Conclusion
