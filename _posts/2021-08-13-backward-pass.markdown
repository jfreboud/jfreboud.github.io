---
layout: post
title:  "The Backward Pass"
date:   2021-08-14 9:00:00 +0200
excerpt: >-
  4/ The backward pass is the nemesis of the forward pass: this is the second step toward the learning process.
---

## Introduction

In the [previous article]({% post_url 2021-08-09-loss-function %}), we began to compute the different 
$ derivative $ functions of $ Loss $ but we were stuck in early. 

In this article we will talk about how to use the structure in $ layers $ of our $ model $ to compute the lasting 
$ derivative $ functions. 

## The backward pass

Before diving into some more computation, let us talk about the **forward pass** one more time 
(first referenced [here]({% post_url 2021-08-06-inside-the-model %})). It has a nemesis in the **backward pass** that 
plays the kind of the exact opposite role.

The **forward pass** lets **information flux** go through the different layers from 
the **input layer** to the **output layer**. The **backward pass** is about a reversed signal that we could call 
the **learning flux**. It goes from the **output layer** to the **input layer**.

If the **input layer** is the place where the **data input** is initialised by the developer in the **forward pass**, 
it is clear from the early computations we made ($ \delta 4 $) 
in the [previous article]({% post_url 2021-08-09-loss-function %}) that the $ Loss $ function is the place where 
the **backward pass** is initialised. 

There is a final difference between the two: the **forward pass** is run in the **learning phase** and in the 
**inferring phase** (see "Learning, inferring" in the [first article]({% post_url 2021-08-05-general-concepts %})) 
while the **backward pass** is only run during the **learning phase**, 
which comforts its signal naming of **learning flux**.

![Layers](/_assets/images/backward/Layer-2.png)

## The Chain rule

Back to the [previous article]({% post_url 2021-08-09-loss-function %}), we now have to compute the $ derivate $ functions 
of $ Loss $ according to $ X $ for $ X $ an "inner" dependency variable ($ X $ declared in the function of an 
intermediate $ layer $): 

$$
\frac{\partial Loss(X, Y^{truth})}{\partial X}
$$

We saw that when considering the $ X $ dependency for an intermediate $ layer $, the impact link to the $ Loss $ 
is indirect which explains why it is not obvious to compute the $ derivative $ function. 

In fact we have to use the **chain rule**, here on [Wikipedia](https://en.wikipedia.org/wiki/Chain_rule): 

"If a variable z depends on the variable y, which itself depends on the variable x (that is, y and z are dependent variables), 
then z depends on x as well, via the intermediate variable y. In this case, the chain rule is expressed as

$$ 
\frac{dz}{dx} = \frac{dz}{dy}.\frac{dy}{dx}
$$

and 

$$

\frac{dz}{dx} \bigg\rvert_{x} = \frac{dz}{dy} \bigg\rvert_{y(x)} . \frac{dy}{dx} \bigg\rvert_{x}

$$

for indicating at which points the derivatives have to be evaluated."

## Example (mathematically shy people should pass)

We are now able to end the computations we began in the last paragraph of the 
[previous article]({% post_url 2021-08-09-loss-function %}) !

### Computing $ \frac{\partial Loss(X^3, Y^{truth})}{\partial X^3} $ 

We are looking for a link between $ X^3 $ and $ Loss $. 
As the [backward pass](#the-backward-pass) suggests, we have to use what we have already computed: 
$ \delta 4 = \frac{\partial Loss(X^4, Y^{truth})}{\partial X^4}(o3, y^{truth}) $ and what directly uses $ X^3 $ which is $ L3 $: 
$ L3(X^3) = X^3 \text{ if } X^3 > 0 \text{, else } 0 $. 

Then we are able to use the **chain rule** with $ z = Loss $ and $ y = L3 $, the formula becomes: 

$$ 
\boxed{\frac{\partial Loss}{\partial X^3} = \frac{\partial Loss}{\partial L3} . \frac{\partial L3}{\partial X^3}}
$$

Now we compute: 

$$
\begin{align}
    \frac{\partial Loss}{\partial L3} &= \frac{\partial Loss(X^4, Y^{Truth})}{\partial L3(X^3)} \\ 
                                      &= \frac{\partial Loss(X^4, Y^{Truth})}{\partial X^4} \text{ already computed in the previous article !}
\end{align}
$$

and: 

$$
\begin{align}
    \frac{\partial L3}{\partial X^3} &= \frac{\partial L3(X^3)}{\partial X^3} \\ 
                                     &= \frac{\partial (X^3 \text{ if } X^3 > 0 \text{, else } 0)}{\partial X^3} \text{ with the definition of } L3(X^3) \\
                                     &= 1 \text{ if } X^3 > 0 \text{, else } 0
\end{align}
$$

Assembling those results: 

$$
\begin{align}
    \frac{\partial Loss}{\partial X^3} &= \frac{\partial Loss}{\partial L3} . 
                                          \frac{\partial L3}{\partial X^3} \\
                                       &= (\frac{\partial Loss(X^4, Y^{Truth})}{\partial X^4}) * 
                                          (1 \text{ if } X^3 > 0 \text{, else } 0) \\
                                       &= \frac{\partial Loss(X^4, Y^{Truth})}{\partial X^4} \text{ if } X^3 > 0 \text{, else } 0
\end{align}
$$

Do not forget that $ \frac{\partial Loss}{\partial X^3} $ is a function depending on $ X^3 $ and $ Y^{truth} $. 
Thus we can apply it on the values that produced the errors highlighted by $ Loss $: 

$$
\begin{align}
    \frac{\partial Loss}{\partial X^3}(o2, y^{truth}) &= \frac{\partial Loss(X^4, Y^{Truth})}{\partial X^4}(o3, y^{truth}) \text{ if } o2 > 0 \text{, else } 0 \\ 
                                                      &= \delta 4 \text{ if } o2 > 0 \text{, else } 0 
\end{align}
$$

We have found: 

$$ 
\boxed{\delta 3 = \frac{\partial Loss(X^3, Y^{truth})}{\partial X^3}(o2, y^{truth}) = \delta 4 \text{ if } o2 > 0 \text{, else } 0}
$$

### Computing $ \frac{\partial Loss(X^2, Y^{truth})}{\partial X^2} $ 

We are looking for a link between $ X^2 $ and $ Loss $. 
As the [backward pass](#the-backward-pass) suggests, we have to use what we have already computed: 
$ \delta 3 = \frac{\partial Loss(X^3, Y^{truth})}{\partial X^3}(o2, y^{truth}) $ and what directly uses $ X^2 $ which is $ L2 $: 
$ L2(X^2) = \frac{1}{200} X^2_1 - \frac{8 800}{11 600 000}  X^2_2 + 
        \frac{1}{5 800} X^2_3 \text{, with } X^2 = (X^2_1, X^2_2, X^2_3) $. 

Then we are able to use the **chain rule** with $ z = Loss $ and $ y = L2 $, the formula becomes: 

$$ 
\boxed{\frac{\partial Loss}{\partial X^2} = \frac{\partial Loss}{\partial L2} . \frac{\partial L2}{\partial X^2}}
$$

Now we compute: 

$$
\begin{align}
    \frac{\partial Loss}{\partial L2} &= \frac{\partial Loss(X^3, Y^{Truth})}{\partial L2(X^2)} \\ 
                                      &= \frac{\partial Loss(X^3, Y^{Truth})}{\partial X^3} \text{ already computed in the previous paragraph !}
\end{align}
$$

and: 

$$
\begin{align}
    \frac{\partial L2}{\partial X^2} &= \frac{\partial L2(X^2)}{\partial X^2} \\ 
                                     &= \frac{\partial (\frac{1}{200} X^2_1 - \frac{8 800}{11 600 000}  X^2_2 + 
        \frac{1}{5 800} X^2_3)}{\partial X^2} \text{ with the definition of } L2(X^2) \\
                                     &= (\frac{1}{200}, -\frac{8 800}{11 600 000}, \frac{1}{5 800})
\end{align}
$$

Assembling those results: 

$$
\begin{align}
    \frac{\partial Loss}{\partial X^2} &= \frac{\partial Loss}{\partial L2} . 
                                          \frac{\partial L2}{\partial X^2} \\
                                       &= (\frac{\partial Loss(X^3, Y^{Truth})}{\partial X^3}) * 
                                          (\frac{1}{200}, -\frac{8 800}{11 600 000}, \frac{1}{5 800}) 
\end{align}
$$

Do not forget that $ \frac{\partial Loss}{\partial X^2} $ is a function depending on $ X^2 $ and $ Y^{truth} $. 
Thus we can apply it on the values that produced the errors highlighted by $ Loss $: 

$$
\begin{align}
    \frac{\partial Loss}{\partial X^2}(o1, y^{truth}) &= \frac{\partial Loss(X^3, Y^{Truth})}{\partial X^3}(o2, y^{truth}) * 
                                          (\frac{1}{200}, -\frac{8 800}{11 600 000}, \frac{1}{5 800}) \\ 
                                                      &= \delta 3 * 
                                          (\frac{1}{200}, -\frac{8 800}{11 600 000}, \frac{1}{5 800})
\end{align}
$$

We have found: 

$$ 
\boxed{\delta 2 = \frac{\partial Loss(X^2, Y^{truth})}{\partial X^2}(o1, y^{truth}) = \delta 3 * 
                                          (\frac{1}{200}, -\frac{8 800}{11 600 000}, \frac{1}{5 800})}
$$

### Computing $ \frac{\partial Loss(X^1, Y^{truth})}{\partial X^1} $ 

We are looking for a link between $ X^1 $ and $ Loss $. 
As the [backward pass](#the-backward-pass) suggests, we have to use what we have already computed: 
$ \delta 2 = \frac{\partial Loss(X^2, Y^{truth})}{\partial X^2}(o1, y^{truth}) $ and what directly uses $ X^1 $ which is $ L1 $: 
$ L1(X^1) = X^1 \text{, with } X^1 = (X^1_1, X^1_2, X^1_3) $. 

Then we are able to use the **chain rule** with $ z = Loss $ and $ y = L1 $, the formula becomes: 

$$ 
\boxed{\frac{\partial Loss}{\partial X^1} = \frac{\partial Loss}{\partial L1} . \frac{\partial L1}{\partial X^1}}
$$

Now we compute: 

$$
\begin{align}
    \frac{\partial Loss}{\partial L1} &= \frac{\partial Loss(X^2, Y^{Truth})}{\partial L1(X^1)} \\ 
                                      &= \frac{\partial Loss(X^2, Y^{Truth})}{\partial X^2} \text{ already computed in the previous paragraph !}
\end{align}
$$

and: 

$$
\begin{align}
    \frac{\partial L1}{\partial X^1} &= \frac{\partial L1(X^1)}{\partial X^1} \\ 
                                     &= \frac{\partial (X^1)}{\partial X^1} \text{ with the definition of } L1(X^1) \\
                                     &= (1, 1, 1)
\end{align}
$$

Assembling those results: 

$$
\begin{align}
    \frac{\partial Loss}{\partial X^1} &= \frac{\partial Loss}{\partial L1} . 
                                          \frac{\partial L1}{\partial X^1} \\
                                       &= (\frac{\partial Loss(X^2, Y^{Truth})}{\partial X^2}) * 
                                          (1, 1, 1) 
\end{align}
$$

Do not forget that $ \frac{\partial Loss}{\partial X^1} $ is a function depending on $ X^1 $ and $ Y^{truth} $. 
Thus we can apply it on the values that produced the errors highlighted by $ Loss $: 

$$
\begin{align}
    \frac{\partial Loss}{\partial X^1}(x, y^{truth}) &= \frac{\partial Loss(X^2, Y^{Truth})}{\partial X^2}(x, y^{truth}) * 
                                          (1, 1, 1) \\ 
                                                     &= \delta 2 * (1, 1, 1)
\end{align}
$$

We have found: 

$$ 
\boxed{\delta 1 = \frac{\partial Loss(X^1, Y^{truth})}{\partial X^1}(x, y^{truth}) = \delta 2 * (1, 1, 1)}
$$

## Conclusion

What we should keep in mind from all these scary computations is that the **learning flux** has 
a rather simple form: it depends on the $ derivative $ of the current $ layer $ multiplied by the previous  
**learning flux** in the order of the **backward pass**($ \delta 4 $ => $ \delta 3 $ => $ \delta 2 $ => $ \delta 1 $ 
in the [example](#example)).

What we have to do now is to use this **learning flux** for a different purpose than just computing the 
descending **learning flux** :smiling_imp: