---
layout: post
title:  "The Second Dimension"
category: network
date:   2022-01-05
excerpt: >-
  In this article, we open the second dimension of our trip to Deep-Learning.
---

## Introduction

In the [previous article]({% post_url 2021-12-12-linear-function %}), we saw that $ Linear $ **network** are 
already good at finding correlations among structured **data**. 

In this article we will explore a field where the **data** is not structured: what solution could we find in such 
a case ? 

## Computer Vision

The typical field where **data** is not structured is **Computer Vision**. In this field, we would like  
machines to understand **images** and **video** the way humans do. 

The main problem is that humans are already used 
to talk about **images** as **representations** (introduced in the [second article]({% post_url 2021-08-06-inside-the-model %}), 
we talked about them more in the [previous article]({% post_url 2021-12-12-linear-function %})). 

For example when I see an **image** of another human, I tend to identify his/her main characteristics: 
the hair color, the eyes, the hands... And in fact each of these characteristics are some sort of **representations**. 
In my mind, I can play with these **representations**, comparing them for example: this human is taller than this one. 
In a way, I have a **representation** for the entire world and any **image** I will see will only refer to the 
**representations** I know. If I happen to interact with other senses, such as taste, I just have to enrich my 
visual **representation** knowing the taste and view refer to the same one. 

But for a machine, we must build this whole from scratch... 

## The Visual Information Flow

The first difference we should highlight between **Human Vision** and **Computer Vision** is the **information flow** 
(see the [second article]({% post_url 2021-08-06-inside-the-model %})).

In the human case the visual **information flow** is physical. It starts with light on our retina before being 
translated into **neurons** activations / non activations in our brain. 

In the machine case, we will only work with numerics. In a way it is easy: we just have to do 
some computations on those numerics to understand what happens. 
The bad news is that there will be so many computations that it will be difficult to fully grasp 
what **representations** are built. 
Hence, it is important to try and understand the different 
transformation of our **information flow** across the different **layers** of our **model**: this is our goal for 
the next articles.

Before building Deep-Learning **models**, let us start by converting our **images** into numerics!

## An Image

An image is a grid of a certain size. An image of size $ (height, width) $ will contain $ height * width $ pixels. 
Each pixel is a mix of the three primary colors $ (red, green, blue) $. 

Let us imagine a very small image of size $ (7, 7) $ and zoom in so that we observe a house ! 

![Image](/_assets/images/network/Image1.png)

<br>

You may have guessed that each circle in this image actually represents a pixel of the image. 
As we mentioned in the [previous paragraph](#the-visual-information-flow), our brain is triggered by the light 
on different colors. But the same image can also be seen as the different numbers of the pixels. 
In the below image, I have hidden the center of the image to show some pixels' values $ (red, green, blue) $.

![Image](/_assets/images/network/Image2.png)

<br>

Note that if the entire image was only numbers, you would have no way of understanding the image as a house. 
This is why we have to build a system that allows the computer to "understand" those numerics.

## Toward Visual Representations

Here is the first operation to build our numeric **representations**: instead of considering 
the grid of the previous numeric pixels, we will consider the three **channels** $ red $, $ green $ and $ blue $. 
The operation merely consists in splitting our grid into 3 **channel** grids in which each channel grid contains 
the numbers of the **channel** considered instead of pixels. 

We go from a grid of $ height * width $ pixels to 3 **channel** grids of $ height * width $ numbers.

<br>

Here is the the first **channel** grid ($ red $): 

![Image](/_assets/images/network/Image3.png)

<br>

Here is the second **channel** grid ($ green $): 

![Image](/_assets/images/network/Image4.png)

<br>

And here is the third **channel** grid ($ blue $):

![Image](/_assets/images/network/Image5.png)

<br>

Tada, we are now in the presence of 3 numeric **representations** ! 
We even know the cheat in order to go back to a real image out of these 3 different **channels**.

![Image](/_assets/images/network/Image6.png)

## Convolution Kernel

Working on images seems to complicate the structure of our **data**. 
Back to the [first article]({% post_url 2021-08-05-general-concepts%}), our **data** were vectors of numbers: 
example $ (100, 2000, 100) $. Dealing with images has increased the dimensionality of our **data**: from being 
1 dimensional (a vector of numbers), they have become 2 dimensional (a vector of vector of numbers, or said 
differently a 2D array of numbers).
 
How can we process this new type of **data** ?

Let us introduce the **convolution kernel**: a new grid that will help us elaborate an operation on 
2D arrays of numbers. The interesting part of this new grid is that it enables to capture spatial localisation 
particularities in the input **channel** (more on this in the [example](#example)). 

For now, let us take an example of a small **convolution kernel**: 

$$
\begin{bmatrix}
0 & -1 & 1 \\
-1 & 1 & 0 \\
1 & 0 & 0
\end{bmatrix}
$$

In the following I will speak about the "pixels" of the **channels** we introduced in the 
[previous paragraph](#toward-visual-representations). Bare in mind that these "pixels" are just one number 
at a precise localisation in a **channel** grid and not the pixels we talked about in the [image paragraph](#an-image). 

We want to capture spatial localisation information on every "pixel" of the different **channels** of the 
[previous paragraph](#toward-visual-representations). 
But first of all, how do we apply the **kernel** on just one "pixel" in a given **channel** ? 

By taking the center of the **kernel**, aligning it on the "pixel" to process in the 
input **channel** and adding the different multiplied couples. The result is one "pixel" of the output **channel**.

Let us see how it works with the small **kernel** defined above:

![Image](/_assets/images/network/Image7.png)

In the diagram above, only four new "pixels" are mentioned: $$ r^2_{0,0} $$, $$ r^2_{7,0} $$, $$ r^2_{0,7} $$ and 
$$ r^2_{7,7} $$. Please note that the **kernel** must be applied on **EVERY** "pixel" 
of the input **channel** in order to produce **EVERY** "pixel" of the output **channel**.

<br>

In order to be more specific, let us compute $ r^2_{7,0} $ for the red **channel**. 
An issue occurs: a part of the **kernel** does not cover the red grid, it covers an empty space. 
This is no big deal, we just replace the missing "pixels" by 0: 

![Image](/_assets/images/network/Image8.png)

Then we add together the different multiplied couples:

$$ 
\begin{align}
r^2_{7,0} &= & (0 * 0) + (-1 * 0) + (1 * 152) \\
          &  & + (-1 * 0) + (1 * 0) + (0 * 152) \\
          &  & + (1 * 0) + (0 * 0) + (0 * 0) \\
r^2_{7,0} &= 152 &
\end{align}
$$

<br>

Let us compute another example: $ r^2_{3, 3} $ for the red **channel**:

![Image](/_assets/images/network/Image9.png)

$$
\begin{align}
r^2_{3,3} &= & (0 * 147) + (-1 * 147) + (1 * 70) \\
          &  & + (-1 * 147) + (1 * 70) + (0 * 70) \\
          &  & + (1 * 207) + (0 * 207) + (0 * 207) \\
r^2_{3,3} &= 53 &
\end{align}
$$

## Convolution

We are able to transform one input **channel** into one output **channel** thanks to some operation 
with a **convolution kernel**. 
But we must not lose our goal identified in the [second article]({% post_url 2021-08-06-inside-the-model %}): 
elaborate $ layers $ that build abstract **representations**.

Let us go back to the [linear function article]({% post_url 2021-12-12-linear-function %}). We took an example where 
$ L^k $ $ Linear $ $ layer $ had 3 input **neurons** and produced 2 output **neurons**. 
We noted a particular important fact: 

"Each output **neuron** being linked with every input **neurons** via specific **weights**, the different output 
**neurons** may be seen as different new **representations** of the input **neurons**."

We want to build **representations** the same way for our images. This means we need to preserve the combination 
power of our output **neurons**.

Let us suppose that we want to build only 1 new **representation** out of the 3 **channels** of the 
[visual representations paragraph](#toward-visual-representations).

The solution is to attribute one **convolution kernel** for each and every one of the 3 input **channels** that we have. 
Then, thanks to the [previous paragraph](#convolution-kernel), we know how to apply the particular **kernel** on 
the chosen **channel** in order to build a temporary (noted tmp below) new **channel**. We now have 3 temporary 
new **channels** that we can sum together (adding together the "pixels" of the 3 temporary **channels** that 
are located at the same place in their grid) in order to build the final new **representation** noted $ rep^2_1 $ below.

![Image](/_assets/images/network/Image10.png)

$ rep^2_1 $ is a new **representation** that produces a new "meaning" thanks to the "meaning" of every input **channels** 
($ r^1 $, $ b^1 $, $ b^1 $). 

Each time we want to build a new **representation** we have to use one specific **convolution kernel** for each 
input **channels**. In a way, this is the exactly what happened to the **weights** 
in the [linear function article]({% post_url 2021-12-12-linear-function %}). 

## Example 

In this example, we will use the 3 input **channels** 
of the [visual representations paragraph](#toward-visual-representations) in order to better understand how 
the **convolution kernels** capture spatial localisation particularities and how they affect the **representations**. 

Let us build a small **convolutional model**!

### <span style="text-decoration:underline"> L1 </span>

The first layer is not the most interesting one: it plays the same role as the **Input 1D** described in 
the [activation article]({% post_url 2021-10-06-activation %}). Here it is an **Input 2D** $ layer $ that just 
receives the 3 different **channels** of our [house example](#toward-visual-representations). 

### <span style="text-decoration:underline"> L2 </span>

Much fun is happening now :smiling_imp:

This layer will be a $ Convolutional $ $ layer $. 
As we saw in the [Convolution](#convolution) paragraph, we must choose 3 different **convolution kernels** for each 
**representation** built (because $ L1 $ $ layer $ has 3 output **channels**).

We want to build 6 new **representations**, this means that we need 18 different **convolution kernels**! 
For simplicity, we will suppose that the 3 different **convolution kernels** are identical by group of 3 
(the 3 **convolution kernels** used to produce first **representation** are identical, 
the 3 **convolution kernels** used to produce the second **represetation** are identical...).

Here are the different **convolution kernels** that we will apply:

$$
\begin{bmatrix}
0 & -1 & 1 \\
-1 & 1 & 0 \\
1 & 0 & 0
\end{bmatrix}

\begin{bmatrix}
1 & -1 & 0 \\
0 & 1 & -1 \\
0 & 0 & 1
\end{bmatrix}
$$

$$
\begin{bmatrix}
1 & -1 & 1 \\
-1 & -1 & 1 \\
1 & 1 & 1
\end{bmatrix}
$$

$$
\begin{bmatrix}
-1 & 1 & 1 \\
-1 & 1 & 1 \\
-1 & 1 & 1
\end{bmatrix}

\begin{bmatrix}
1 & 1 & -1 \\
1 & 1 & -1 \\
1 & 1 & -1
\end{bmatrix}
$$

$$
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
$$

Spoiler alert: 
- the first two **kernels** will be triggered by the house's roof (note their structure in diagonal)
- the third **kernel** will detect the house's window 
- the following two **kernels** will detect the houses's walls (note their structure in vertical line)
- the last **kernel** will detect the inside of the house

<br>

Now let us apply the [convolution](#convolution) with the first **convolution kernel** 
(use the same kernel for the 3 input **channels**) to obtain our first **representation**:

![Image](/_assets/images/network/Image11.png)

The maximal values are in bright red. Observe as the two maximal numbers are precisely located on the left 
part of the roof.

<br>

Here is the second **representation** (with the second **convolution kernel** used for the 3 input **channels**):

![Image](/_assets/images/network/Image12.png)

Note as here, the maximal numbers are precisely located on the right part of the roof.

<br>

Here is the third one, corresponding to the window:

![Image](/_assets/images/network/Image13.png)

<br>

Then the two walls: 

![Image](/_assets/images/network/Image14.png)

![Image](/_assets/images/network/Image15.png)

<br>

And finally the inside of the house:

![Image](/_assets/images/network/Image16.png)

### <span style="text-decoration:underline"> L3 </span>

In this $ layer $ we will use the trick of the "Biological Neuron" we saw 
in [this article]({% post_url 2021-12-12-linear-function %}). 

First, we add **biases** to the previous **representations**.
We can make the assumption that this step is in fact included in the **convolution** 
(once again, this was already the case for the $ Linear $ case). Concretely, the **biases** are just numbers that 
are added to every "pixel" of one chosen **channel**.

As we have 6 **representations** coming from the $ L2 $ $ layer $, we must choose 6 **biases**, each 
of them being added to every "pixel" of the corresponding **representation** (the first **bias** will be 
added to every "pixel" of the first previous **representation**, the second **bias** will be 
added to every "pixel" of the second previous **representation**...).

Here are the 6 **biases**:

$$
-1030, -1030, -1990, -2000, -2000, -4059
$$

Then we use a $ ReLU $ $ activation $ $ layer $. We already saw how it works in the 1D case 
[here]({% post_url 2021-10-06-activation %}): applying the $ activation $ $ function $ to every **neuron**. 
In the 2D case, we do the same on every "pixel" of every **representations**. 

This $ L3 $ $ layer $ ($ ReLU $ $ activation $) will produce 6 new **representations**. 

<br>

Here are the two corresponding to the house's roof:

![Image](/_assets/images/network/Image17.png)

![Image](/_assets/images/network/Image18.png)

<br>

The window:

![Image](/_assets/images/network/Image19.png)

<br>

The walls:

![Image](/_assets/images/network/Image20.png)

![Image](/_assets/images/network/Image21.png)

<br>

The inside of the house:

![Image](/_assets/images/network/Image22.png)

### <span style="text-decoration:underline"> L4 </span>

$ L4 $ will be a new $ Convolutional $ $ layer $. 

We want to build 6 new **representations**, but there are 6 **representations** coming from the $ L3 $ $ layer $. 
This means that we should choose $ 6 * 6 = 36 $ different **convolution kernels**! 
We will use another trick to choose the 36 different **kernels**. 

For each group of 6 **kernels**, we will suppose that only one does not contain only 0 in it. 
To build the first new **representation**, we will suppose that the first **kernel** does not contain only 0 
(the other 5 **kernels** containing only 0).
To build the second new **representation**, we will suppose that the second **kernel** does not contain only 0 
(the other 5 **kernels** containing only 0).
Same logic for the other new **representations**.

Now for the list of the different **convolution kernels** that do not contain 0:

$$
\begin{bmatrix}
0 & 0 & 0 \\
1 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}

\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
$$

$$
\begin{bmatrix}
0 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

$$
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
1 & 0 & 0
\end{bmatrix}

\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

$$
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}
$$

Spoiler alert: 
- the first **kernel** will move "pixels" to the right 
- the second **kernel** will move "pixels" to the left  
- the third **kernel** will copy the **channel** as is 
- the fourth **kernel** will move "pixels" to the top right hand corner
- the fifth **kernel** will move "pixels" to the top left hand corner
- the sixth **kernel** will move "pixels" to the top

This $ L4 $ $ layer $ will produce 6 new **representations**.

<br>

Here are the two corresponding to the house's roof:

![Image](/_assets/images/network/Image23.png)

![Image](/_assets/images/network/Image24.png)

<br>

The window:

![Image](/_assets/images/network/Image19.png)

<br>

The walls:

![Image](/_assets/images/network/Image25.png)

![Image](/_assets/images/network/Image26.png)

<br>

The inside of the house:

![Image](/_assets/images/network/Image27.png)

### <span style="text-decoration:underline"> L5 </span>

$ L5 $ will be a new $ Convolutional $ $ layer $. 

We want to build the final most abstract **representation**. 
As there are 6 **representations** coming from the $ L4 $ $ layer $, we must choose 
$ 1 * 6 = 6 $ different **convolution kernels**.

Here are they:

$$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}

\begin{bmatrix}
0 & 0 & 1 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

$$
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
$$

$$
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
1 & 0 & 0
\end{bmatrix}

\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

$$
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}
$$

Let us recall the different **representations** coming from $ L4 $: 
- the first two keep track of the localisation of the top left roof and the top right roof 
- the third keeps track of the localisation of the window
- the fourth and fifth deep track of the localisation of the left and right walls
- the sixth keep track of the localisation of the inside of the house

We can now "translate" what the meaning of the final **representation** will be: 
- The first **kernel** is associated to the first **representation**, it means that we are looking for a left roof in the 
top left hand corner
- The second **kernel** is associated to the second **representation**, it means that we are looking for a right roof 
in the fop right hand corner
- The third **kernel** and representation** mean we are looking for window to the right
- The forth couple means we are looking for a wall in the bottom left hand corner
- The firth couple means we are looking for a wall in the bottom right hand corner
- The sixth couple means we are looking for the inside of the house in the bottom

In fact, there is a precise localisation where these 6 statements are all true: the exact center. 
Applying the **convolution** to the center "pixel", we find : $ 1 + 1 + 1 + 1 + 1 + 1 = 6 $ coming from the 
different **kernels** application on their corresponding **representations**. 

Indeed if we take a look at the final **representation**, we have: 

![Image](/_assets/images/network/Image28.png)

We could add one **bias** and a final $ ReLU $ $ layer $ to obtain a cleaner **representation** 
with just the center "pixel" triggered (same idea as $ L3 $) 
but we have already understood that it is the maximal activation 
that particularly interests us. 

<br>

We have built a $ model $ composed of 5 $ layers $, our biggest so far!

It is time to summarize what a house is for our $ model $. According to the final **representation**, a house is 
something that:

- contains a top left roof (first previous **channel**) positioned at the top left hand corner (first **kernel**)
- contains a top right roof (second previous **channel**) at the top right hand corner (second **kernel**)
- contains a window (third previous **channel**) to the right (third **kernel**)
- contains a wall (fourth previous **channel**) at the bottom left hand corner (fourth **kernel**)
- contains another wall (fifth previous **channel**) at the bottom right hand corner (fifth **kernel**)
- contains the inside of a house (sixth previous **channel**) at the bottom (sixth **kernel**) 

## Conclusion

In this article, we saw that an image is a type of **data** that is not structured in itself. In order to 
understand such low level information, one has to build "abstract" **representations** that progressively build sense. 

The $ Convolution $ is an operation that combines **representations**, capturing spatial particularities. It links 
 the semantic together with space.
 
 Yet, in this article, the different **kernels** we used were given "out of nowhere". In the next article, we 
 will see what is missing in order for these **kernels** to be learned by the $ model $ itself.
