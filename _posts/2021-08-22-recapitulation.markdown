---
layout: post
title:  "Recapitulation for the Learning Process"
date:   2021-08-22 10:00:00 +0200
excerpt: >-
  6/ We use the different pieces to run the learning phase from scratch.
---

## Introduction

In the [previous article]({% post_url 2021-08-19-weights %}), we both introduced **weights** and 
saw how to update them. 
 
In this paragraph we are going to use numeric values to see the **learning process** in action. 

## What we have...

- a **dataset** containing (**data input**, **data output**): 
    - we want to run a $ model $ on the **data input** in order to produce results
    - we want to confront the results produced against the expectations given by the **data output**

- a $ model $ function, structured as a graph of $ layers $ which tries to understand the **data input** 

- a $ Loss $ function which systematically compares the results of $ model $ to the expectations

## What we do...

In the [first article]({% post_url 2021-08-05-general-concepts %}), we saw that the learning process 
of a deep-learning $ model $ happens during the **learning phase**. 

The different steps of the **learning phase** are: 

1. pick one **data input** in the **dataset**
2. run the **forward pass** for the $ model $ on this **data input**
3. use the $ Loss $ function to compute the error between the result produced by the $ model $ and 
the expectation given by the **data output**
4. run the **backward pass** to compute:
    - the **learning flux**
    - the $ derivative $ of the $ Loss $ function according to $ W $
    
5. update the **weights** of $ model $

Do this for all **data input** in the **dataset**, as we want our $ model $ to produce 
learn on the whole **dataset**. 

## Example: recapitulation of the state so far

We use the same example as in the previous articles. 
Let us recap the formula we have found for our $ model $ example.

### <span style="text-decoration:underline"> Data </span>

Same **data** as in the [first article]({% post_url 2021-08-05-general-concepts %}).

| data input | data output (expectation) |
| ---------------- | ----- |
| (100 broccoli, 2000 Tagada strawberries, 100 workout hours) | (bad shape)  |
| (200 broccoli,  0 Tagada strawberries, 0 workout hours)     | (good shape) |
| (0 broccoli, 2000 Tagada strawberries, 3 000 workout hours) | (good shape) |

### <span style="text-decoration:underline"> Model </span> 

Same $ model $ as in the [previous article]({% post_url 2021-08-19-weights %}).

$$
\begin{align}
    L1(X^1)  &= X^1 & \text{ with } X^1 = (X^1_1, X^1_2, X^1_3) \\
    L2(X^2, W^2) &= W^2 . X^2          & \text{ with } X^2 = (X^2_1, X^2_2, X^2_3) \\
                 &                     & \text{ and } W^2 = (W^2_1, W^2_2, W^2_3) \\
                 &= W^2_1 . X^2_1 + W^2_2 . X^2_2 + W^2_3 . X^2_3 \\
    L3(X^3)  &= X^3 \text{ if } X^3 \geq 0 \text{ else } 0 \\ \\
    model(X) &= L3(L2(L1(X))) & \text{ with } X = (X_1, X_2, X_3) \\ 
    Loss(X^4, Y^{truth})  &= \frac{1}{2} (X^4 - Y^{truth})^2 
\end{align}
$$

With the values of $ w^2 $:

$$ 
w^2 = (\frac{1}{200}, \frac{3 000}{11 600 000}, \frac{1}{5 800})
$$

### <span style="text-decoration:underline"> Run the forward pass </span>

| $ x $              | $ o1 = L1(x) $   | $ o2 = L2(o1) $ | $ o3 = L3(o2) $ |
| :----------------: | :--------------: | :-------------: | :-------------: |
| (100, 2000, 100)   | (100, 2000, 100) | (0)             | (0)             |
| (200,  0, 0)       | (200,  0, 0)     | (1)             | (1)             |
| (0, 2000, 3 000)   | (0, 2000, 3 000) | (0)             | (0)             |

| $ o3 = model(x) $ | $ y^{truth} $ = expected result | $ loss = Loss(o3, y^{truth}) $ | correct ? |
| :----: | :-----: | :-----: | :---: |
| (0) | (0) | (<span style="color:green">0</span>) | ![wrong](/_assets/images/general/right.png) |
| (1) | (1) | (<span style="color:green">0</span>) | ![wrong](/_assets/images/general/right.png) |
| (0) | (1) | (<span style="color:red">0.5</span>) | ![right](/_assets/images/general/wrong.png) |

### <span style="text-decoration:underline"> Run the backward pass </span>

![Layers](/_assets/images/backward/Layer-3.png)

$$ 
\boxed{\delta 4 = o3 - y^{truth}} 
$$

$$ 
\boxed{\delta 3 = \delta 4 \text{ if } o2 \geq 0 \text{ else } 0}
$$

$$ 
\boxed{\delta 2 = \delta 3 . w2 \text{ with } w^2 = (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})}
$$

$$ 
\boxed{\frac{\partial Loss}{\partial W^2}(o1) = \delta 3 * o1}
$$

$$
\boxed{\delta 1 = \delta 2}
$$

### <span style="text-decoration:underline"> Update the weights </span>

We have to use the update formula for $ w^2 $ : 

$$ 
\boxed{\hat{w^2} = w^2 - \alpha * \frac{\partial Loss}{\partial W^2}(o1)}
$$

## Example: learning phase from scratch

Let us run the **learning phase** with this **learning rate** $ \alpha = 1 $.

### <span style="text-decoration:underline"> Run the model on the 1st data input </span>

1: **data input**: $ x = (100, 2000, 100) $

2: run the **forward pass**: 

$$
\begin{align}
    o3 &= model(x) \\ 
       &= model((100, 2000, 100)) \\
       &= (0)
\end{align}
$$

3: compute $ loss $ 

$$ 
\begin{align}
loss &= Loss(o3, y^{truth}) \\
     &= (0) 
\end{align}
$$

4: run the **backward pass**:

$$ 
\begin{align}
\delta 4 &= o3 - y^{truth} \\
         &= (0) - (0) \\
         &= (0)
\end{align}
$$

$$ 
\begin{align}
\delta 3 &= \delta 4 \text{ if } o2 \geq 0 \text{ else } 0 \\ 
         &= (0) \text{ if } (0) \geq 0 \text{ else } 0 \\
         &= (0)
\end{align}
$$

$$ 
\begin{align}
\delta 2 &= \delta 3 . w2 \text{ with } w^2 = (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
         &= (0) * (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
         &= (0, 0, 0)
\end{align}
$$

$$ 
\begin{align}
\frac{\partial Loss}{\partial W^2}(o1) &= \delta 3 * o1 \\
                                       &= (0) * (100, 2000, 100) \\
                                       &= (0, 0, 0)
\end{align}
$$

$$
\begin{align}
\delta 1 &= \delta 2 \\
         &= (0, 0, 0)
\end{align}
$$
    
5: update the **weights** of $ model $

$$
\begin{align}
    \hat{w^2} &= w^2 - \alpha * \frac{\partial Loss}{\partial W^2}(o1) \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) - 1 * (0, 0, 0) \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})
\end{align}
$$

So it appears the new value for $ w^2 $ is still the same !
This is no wonder as for this first **data input**: $ loss = 0 $.
This $ loss $ value is typical for a $ model $ that has already produced the right result and has nothing to learn.

### <span style="text-decoration:underline"> Run the model on the 2nd data input </span>

1: **data input**: $ x = (200, 0, 0) $

2: run the **forward pass**: 

$$
\begin{align}
    o3 &= model(x) \\ 
       &= model((200, 0, 0)) \\
       &= (1)
\end{align}
$$

3: compute $ loss $ 

$$ 
\begin{align}
loss &= Loss(o3, y^{truth}) \\
     &= (0) 
\end{align}
$$

4: run the **backward pass**:

$$ 
\begin{align}
\delta 4 &= o3 - y^{truth} \\
         &= (1) - (1) \\
         &= (0)
\end{align}
$$

$$ 
\begin{align}
\delta 3 &= \delta 4 \text{ if } o2 \geq 0 \text{ else } 0 \\ 
         &= (0) \text{ if } (1) \geq 0 \text{ else } 0 \\
         &= (0)
\end{align}
$$

$$ 
\begin{align}
\delta 2 &= \delta 3 . w2 \text{ with } w^2 = (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
         &= (0) * (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
         &= (0, 0, 0)
\end{align}
$$

$$ 
\begin{align}
\frac{\partial Loss}{\partial W^2}(o1) &= \delta 3 * o1 \\
                                       &= (0) * (200, 0, 0) \\
                                       &= (0, 0, 0)
\end{align}
$$

$$
\begin{align}
\delta 1 &= \delta 2 \\
         &= (0, 0, 0)
\end{align}
$$
    
5: update the **weights** of $ model $

$$
\begin{align}
    \hat{w^2} &= w^2 - \alpha * \frac{\partial Loss}{\partial W^2}(o1) \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) - 1 * (0, 0, 0) \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})
\end{align}
$$

Once more, the new value for $ w^2 $ has not changed.
The same remark as before applies: $ loss = 0 $ means the $ model $ already produced the right result for this 
second **data input** and has nothing to learn.

### <span style="text-decoration:underline"> Run the model on the 3rd data input </span>

1: **data input**: $ x = (0, 2000, 3 000) $

2: run the **forward pass**: 

$$
\begin{align}
    o3 &= model(x) \\ 
       &= model((0, 2000, 3 000)) \\
       &= (0)
\end{align}
$$

3: compute $ loss $ 

$$ 
\begin{align}
loss &= Loss(o3, y^{truth}) \\
     &= (0.5) 
\end{align}
$$

4: run the **backward pass**:

$$ 
\begin{align}
\delta 4 &= o3 - y^{truth} \\
         &= (0) - (1) \\
         &= (-1)
\end{align}
$$

$$ 
\begin{align}
\delta 3 &= \delta 4 \text{ if } o2 \geq 0 \text{ else } 0 \\ 
         &= (-1) \text{ if } (0) \geq 0 \text{ else } 0 \\
         &= (-1)
\end{align}
$$

$$ 
\begin{align}
\delta 2 &= \delta 3 . w2 \text{ with } w^2 = (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
         &= (-1) * (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
         &= -(\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})
\end{align}
$$

$$ 
\begin{align}
\frac{\partial Loss}{\partial W^2}(o1) &= \delta 3 * o1 \\
                                       &= (-1) * (0, 2000, 3 000) \\
                                       &= -(0, 2000, 3 000)
\end{align}
$$

$$
\begin{align}
\delta 1 &= \delta 2 \\
         &= -(\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})
\end{align}
$$
    
5: update the **weights** of $ model $

$$
\begin{align}
    \hat{w^2} &= w^2 - \alpha * \frac{\partial Loss}{\partial W^2}(o1) \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) - 1 * (-(0, 2000, 3 000)) \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) + (0, 2000, 3 000) \\
              &= (\frac{1}{200}, 2000 - \frac{3 000}{11 600 000}, 3 000 + \frac{1}{5 800})
\end{align}
$$

Now we have a the new value for $ w^2 $ !
We observe that $ loss = 0.5 > 0 $: there was something to learn on this last **data input**.
Hence the modification of the **weights** to compensate the failure.

## Conclusion

We have been able to run the **learning phase** for our $ model $ on the given **dataset**.
The major steps are: 

1. run the **forward pass**
2. run the **backward pass**
3. update the **weights** 

Is it over now ?
Yes because we saw the basics of the **learning process**.
No because we have been heavy on the **learning rate** ($ \alpha $ in the formula to update **weights**). 
We will see the impact on the next article.
 