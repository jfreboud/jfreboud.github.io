---
layout: post
title:  "The Gradient Descent Algorithm"
date:   2021-08-23
excerpt: >-
  We use the different parts we have seen so far to run the learning phase from scratch.
---

## Introduction

In the [previous article]({% post_url 2021-08-19-weights %}), we both introduced **weights** and 
saw how to **update** them: this is the core of the **learning process**.

In this article we are going to use the different parts we have explored so far in order to see this 
**learning process** in action.

## What we have...

- a **dataset** containing (**data input**, **data output**): 
    - we want to run a $ model $ on the **data input** in order to produce results
    - we want to confront the results produced against the expectations given by the **data output**

- a $ model $ function, structured as a graph of $ layers $ which tries to understand the **data input** 

- a $ Loss $ function which systematically compares the results of $ model $ to the expectations

## What we do...

In the [first article]({% post_url 2021-08-05-general-concepts %}), we saw that the **learning process** 
of a deep learning $ model $ happens during the **learning phase**. 

Through the different articles: [Loss function]({% post_url 2021-08-09-loss-function %}), 
[backward pass]({% post_url 2021-08-13-backward-pass %}) and [weights]({% post_url 2021-08-19-weights %}), 
we explored the different parts that were specific to this **learning phase**.

We are now able to give the different steps of the **learning phase** in the right order: 

1. pick one **data input** in the **dataset** 
2. run the **forward pass** for the $ model $ on this **data input**
3. use the $ Loss $ function to compute the error between the result produced by the $ model $ and 
the expectation given by the **data output** 
4. run the **backward pass** to compute:
    - the **learning flow**
    - the $ derivative $ of the $ Loss $ function according to $ W $
    
5. **update** the **weights** of $ model $

As we want our $ model $ to learn on every **data** of our **dataset** we will repeat the previous points until 
we have "picked" every **data input** of our **dataset**.

## The Gradient Descent Algorithm

In the [previous article]({% post_url 2021-08-19-weights %}), we saw that in order to **update** the **weights** 
we use a formula with a direction of **update**: $ -\delta w $ 
and a length of **update**: $ \alpha $ which we called the **learning rate**. 

We mentioned how this **learning rate** had to be very small in order not to break our local "prediction" of the 
$ Loss(model(X), Y^{truth}) $ function evaluated on (**data input**, **data output**).

The remaining problem is that if we use a very small **learning rate**, it means that after each **weights** update, 
the $ model $ will not learn a lot.

This is the reason why we repeat the whole process several times in order that many 
"small understanding steps" converge to a global understanding of the whole **dataset**.

This is the **gradient descent** algorithm which mainly consists in repeating what we already know: 

1. pick one **data input** in the **dataset**
2. run the **forward pass** for the $ model $ on this **data input**
3. use the $ Loss $ function to compute the error between the result produced by the $ model $ and 
the expectation given by the **data output**
4. run the **backward pass** to compute:
    - the **learning flow**
    - the $ derivative $ of the $ Loss $ function according to $ W $
    
5. update the **weights** of $ model $

As before, we do it for all **data input** in the **dataset**, we call it an **epoch**. 

What changes now is that we continue running these points during several more **epochs**. 
That way, we have many "small understanding steps" for every **data input** of our **dataset** which helps alleviate 
the small **learning rate**. 

Where does **gradient descent** name comes from ?

From the **weights** update formula of the [last article]({% post_url 2021-08-19-weights %}): 

$$
\hat{w} = w - \alpha . \frac{\partial Loss}{\partial W}(x, y^{truth})
$$

For multivariate functions (function with multiple variables), 
the $ \frac{\partial Loss}{\partial W} $ is called the **gradient**. 

It is a **gradient descent** because we follow the direction of $ -\frac{\partial Loss}{\partial W} $ to **update** 
the **weights**. And from what we saw in the paragraph "The derivative of Loss according to W" in the 
[last article]({% post_url 2021-08-19-weights %}), 
it corresponds to the $ x $ axis direction where the tangent is descending.

![Tangent](/_assets/images/backward/tangent.png)

## Example: what we have...

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
w^2 = (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})
$$

### <span style="text-decoration:underline"> Run the forward pass </span>

| $ x $              | $ o1 = L1(x) $   | $ o2 = L2(o1) $ | $ o3 = L3(o2) $ |
| :----------------: | :--------------: | :-------------: | :-------------: |
| (100, 2000, 100)   | (100, 2000, 100) | (0)             | (0)             |
| (200,  0, 0)       | (200,  0, 0)     | (1)             | (1)             |
| (0, 2000, 3 000)   | (0, 2000, 3 000) | (0)             | (0)             |

| $ o3 = model(x) $ | $ y^{truth} $ | $ loss = Loss(o3, y^{truth}) $ | correct ? |
| :----: | :-----: | :-----: | :---: |
| (0) | (0) | (<span style="color:green">0</span>) | ![wrong](/_assets/images/general/right.png) |
| (1) | (1) | (<span style="color:green">0</span>) | ![wrong](/_assets/images/general/right.png) |
| (0) | (1) | (<span style="color:red">0.5</span>) | ![right](/_assets/images/general/wrong.png) |

### <span style="text-decoration:underline"> Run the backward pass </span>

![Layers](/_assets/images/backward/Layer-5.png)

$$ 
\boxed{\delta 4 = o3 - y^{truth}} 
$$

$$ 
\boxed{\delta 3 = \delta 4 \text{ if } o2 \geq 0 \text{ else } 0}
$$

$$ 
\boxed{\delta 2 = \delta 3 * w2 \text{ with } w^2 = (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})}
$$

$$ 
\boxed{\delta w^2 = \delta 3 * o1}
$$

$$
\boxed{\delta 1 = \delta 2}
$$

### <span style="text-decoration:underline"> Update the weights </span>

We have to use the update formula for $ w^2 $ : 

$$
\boxed{\hat{w^2} = w^2 - \alpha . \delta w^2}
$$

## Example: what we do...

Let us run the **learning phase** with a very small **learning rate** $ \alpha = 10^{-7} $.
The $ model $ has to learn on each **data input** of our **dataset**. 
Thus we will run the **learning phase** on our 3 **data input**: this will be one **epoch** of the 
**gradient descent** algorithm. 

### <span style="text-decoration:underline"> Run the learning phase on the 1st data input </span>

1. pick **data input**: $ x = (100, 2000, 100) $

2. run the **forward pass**: 

    $$
    \begin{align}
        o3 &= model(x) \\ 
           &= model((100, 2000, 100)) \\
           &= (0)
    \end{align}
    $$

3. compute $ loss $ 

    $$ 
    \begin{align}
    loss &= Loss(o3, y^{truth}) \\
         &= (0) 
    \end{align}
    $$

4. run the **backward pass**:

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
    \delta 2 &= \delta 3 * w2 \text{ with } w^2 = (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
             &= (0) * (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
             &= (0, 0, 0)
    \end{align}
    $$
    
    $$ 
    \begin{align}
    \delta w^2 &= \delta 3 * o1 \\
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
    
5. update the **weights** of $ model $

    $$
    \begin{align}
    \hat{w^2} &= w^2 - \alpha . \delta w^2 \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) - 10^{-7} * (0, 0, 0) \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})
    \end{align}
    $$

It appears the new value for $ w^2 $ is still the same !
This is no wonder as for this first **data input**: $ loss = 0 $.
This $ loss $ value is typical for a $ model $ that has already produced the right result and has nothing to learn.

### <span style="text-decoration:underline"> Run the learning phase on the 2nd data input </span>

1. pick **data input**: $ x = (200, 0, 0) $

2. run the **forward pass**: <sup>[[1]](#remark)</sup> 

    $$
    \begin{align}
        o3 &= model(x) \\ 
           &= model((200, 0, 0)) \\
           &= (1)
    \end{align}
    $$

3. compute $ loss $ 

    $$ 
    \begin{align}
    loss &= Loss(o3, y^{truth}) \\
         &= (0) 
    \end{align}
    $$

4. run the **backward pass**:

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
    \delta 2 &= \delta 3 * w2 \text{ with } w^2 = (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
             &= (0) * (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
             &= (0, 0, 0)
    \end{align}
    $$
    
    $$ 
    \begin{align}
    \delta w^2 &= \delta 3 * o1 \\
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
    
5. update the **weights** of $ model $

    $$
    \begin{align}
    \hat{w^2} &= w^2 - \alpha . \delta w^2 \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) - 10^{-7} * (0, 0, 0) \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})
    \end{align}
    $$

Once more, the new value for $ w^2 $ has not changed.
The same remark as before applies: $ loss = 0 $ means the $ model $ already produced the right result for this 
second **data input** and has nothing to learn.

### <span style="text-decoration:underline"> Run the learning phase on the 3rd data input </span>

1. pick **data input**: $ x = (0, 2000, 3 000) $

2. run the **forward pass**: <sup>[[1]](#remark)</sup>

    $$
    \begin{align}
        o3 &= model(x) \\ 
           &= model((0, 2000, 3 000)) \\
           &= (0)
    \end{align}
    $$

3. compute $ loss $ 

    $$ 
    \begin{align}
    loss &= Loss(o3, y^{truth}) \\
         &= (0.5) 
    \end{align}
    $$

4. run the **backward pass**:

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
    \delta 2 &= \delta 3 * w2 \text{ with } w^2 = (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
             &= (-1) * (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) \\
             &= -(\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800})
    \end{align}
    $$
    
    $$ 
    \begin{align}
    \delta w^2 &= \delta 3 * o1 \\
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
    
5. update the **weights** of $ model $

    $$
    \begin{align}
    \hat{w^2} &= w^2 - \alpha . \delta w^2 \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) - 10^{-7} * (-(0, 2000, 3 000)) \\
              &= (\frac{1}{200}, -\frac{3 000}{11 600 000}, \frac{1}{5 800}) + (0, 0.0002, 0.0003) \\
              &= (\frac{1}{200}, 0.0002 - \frac{3 000}{11 600 000}, 0.0003 + \frac{1}{5 800})
    \end{align}
    $$
    
    Let us keep in mind the new values we computed for $ w^2 $: 
    
    $$
    \boxed{w^2 = (\frac{1}{200}, 0.0002 - \frac{3 000}{11 600 000}, 0.0003 + \frac{1}{5 800})}
    $$

We have just run one **epoch** of the **gradient descent** algorithm on our whole **dataset**. 
Let us stop our algorithm now and check the new results when we run a new **forward pass** on every **data input** 
of our **dataset**.

### <span style="text-decoration:underline"> Run a new forward pass </span>

| $ x $              | $ o1 = L1(x) $   | $ o2 = L2(o1) $ | $ o3 = L3(o2) $ |
| :----------------: | :--------------: | :-------------: | :-------------: |
| (100, 2000, 100)   | (100, 2000, 100) | (0.43)          | (0.43)          |
| (200,  0, 0)       | (200,  0, 0)     | (1)             | (1)             |
| (0, 2000, 3 000)   | (0, 2000, 3 000) | (1.3)           | (1.3)           |

We see that our new results do not come along with (0) => (bad shape) and (1) => (good shape). 
We will add a new column in order to show the result that is aligned with (0) or (1) in order to compare with 
the expectations. Let us use a threshold to make the decision:
 
- values < 0.5 will be transformed to 0, 
- values $ \geq $ 0.5 will be transformed to 1

Now we have:

| $ x $              | $ o3 = model(x) $ | $ result $ |
| :----------------: | :---------------: | :--------: |
| (100, 2000, 100)   | (0.43)            | (0)        |
| (200,  0, 0)       | (1)               | (1)        |
| (0, 2000, 3 000)   | (1.3)             | (1)        |

We are now able to compare $ result $ with $ y^{truth} $. In order to have a more objective indicator we still 
compute $ loss $ with the real outputs of our $ model $: $ o3 $ 

| $ o3 = model(x) $ | $ result $ | $ y^{truth} $ | $ loss = Loss(o3, y^{truth}) $ | correct ? |
| :----: | :-----: | :-----: | :-----: | :---: |
| (0.43) | (0) | (0) | (0.092) | ![wrong](/_assets/images/general/right.png) |
| (1)    | (1) | (1) | (<span style="color:green">0</span>) | ![right](/_assets/images/general/right.png) |
| (1.3)  | (1) | (1) | (0.045) | ![right](/_assets/images/general/right.png) |

We observe that the $ result $ is now aligned with the expectation $ y^{truth} $ on the 3 **data input** !
The $ model $ has well learnt. Still, the $ loss $ for the first **data input** has increased. In the past its $ loss $
was 0 and now 0.092. This shows that any **learning** made on one **data input** may affect the understanding of 
other **data input** :smiling_imp:

## Conclusion

In this article we run the whole **gradient descent** algorithm and saw the **learning process** in action. 
We saw the **gradient descent** name comes from the direction followed to **update** the **weights**. 

We run only one **epoch** of the algorithm and saw the **weights**' **updates** have slightly degraded some results 
that were perfect in the original state. We could have run a second **epoch** to see how the situation evolves. 
Rather than that we will talk about an upgrade to the current algorithm. This upgrade will help stabilize 
**learning** at each iteration. Let us go to the [next article]({% post_url 2021-08-24-batch-learning %}) !

<br>

<a id="remark" class="anchor" href="#header-title">[1]:</a>

Note that it is a "coincidence" that our $ model $ did not learn anything on the previous **data input**. 
If it had learnt anything, we would have **updated** the $ w^2 $ value and the results of the **forward pass** 
would have differed from the values that we computed in the past. 
This is the reason why we should keep in mind to run the **forward pass** on one **data input** at a time when the 
goal is to **update** **weights**.
