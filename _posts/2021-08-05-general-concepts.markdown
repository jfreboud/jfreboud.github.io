---
layout: post
title:  "General Concepts"
date:   2021-08-05 20:00:00 +0200
---

## Introduction 

Deep-Learning is one of those words that is buzzing today. One might believe that the artificial intelligence 
is about building a Frankenstein that takes over its creator in order to gain its liberty.

The reality is far away from that vision. The global structure of the learning process happens in the 
deep-learning **model**. This **model** is not learnt by any means, it is set by the developer. 
Finally, the only area where the learning takes place are the **model's weights**.

![Frankenstein](/_assets/images/general/Frankenstein.png)

## Build a model

The main objective of deep-learning is to be able to **learn** something and to apply this **learning** on something 
new. This **learning** is located inside the deep-learning **model**. More precisely, it is located in the 
**model's weights** but we will talk about them later.

As we saw in the [introduction](#introduction), there is no magic: 
it is the role of the developer to actually build the **model**. We will see some considerations about it later too. 

## Run a model 

Once the **model** is built, we can run it. 

- Let **X** be the input variable of the **model**. As **model** depends on **X**, we generally note **model(X)** as the 
mathematical **model** function depending on the **X** variable. 

- Let **x** be a value for that **X** variable, we note **model(x)** the result of the **model** 
on that particular **x** value. 

We can refer to **model(X)** to speak about the theoretical **model** function depending on **X** and 
**model(x)** to speak about a real production of the **model** on some given **x** value. 

But where does this **x** value come from ? 

### Example 

We have a cohort of patients with **data**. More precisely we know how many broccoli they eat per year, how many 
Tagada strawberries they eat per year and how many hours of cardio latin dance they workout per year. 
And we as well happen to know for each of them if they are in good shape or not. 

We can say we have three **inputs**: 
quantity of broccoli, quantity of Tagada strawberries and quantity of cardio latin dance workout.
We have one **output**: good shape or not. 

So if we try to use one **model** on that we could say that: 
- **X** is a vector with 3 dimensions (broccoli, Tagada, workout)
- **model(X)** is a vector with 1 dimension (good shape or not)

Let us look at some patients' **data**: 
- (100 broccoli, 2000 Tagada strawberries, 100 workout hours), (bad shape)
- (200 broccoli,  0 Tagada strawberries, 0 workout hours), (good shape)
- (0 broccoli, 2000 Tagada strawberries, 3 000 workout hours), (good shape)

We can run our **model** on the **data** in order to produce results **model(x)** and see what happens:

| x | expected result | model(x) | correct ? |
| :----------------: | :-----: | :----: | :---: |
| (100, 2000, 100) | (<span style="color:red">bad shape</span>)    | (<span style="color:green">good shape</span>) | ![wrong](/_assets/images/general/wrong.png) |
| (200,  0, 0)     | (<span style="color:green">good shape</span>) | (<span style="color:red">bad shape</span>)    | ![wrong](/_assets/images/general/wrong.png) |
| (0, 2000, 3 000) | (<span style="color:green">good shape</span>) | (<span style="color:green">good shape</span>) | ![right](/_assets/images/general/right.png) |

In the column: "correct ?", we can see that the model have produced wrong results in 2 cases !
How is that possible ? 

That is exactly what we want to rectify as the **model** learns on the **data**...

## Learning, inferring

Let us assume we have built a **model**, we want to run it.

As we observed in this [example](#example), we may consider a **dataset** containing the associations: 
(**data input**, **data output**) where **data input** are the inputs for one patient and 
**data output** are the expectations for the same patient.

- let **X** be a variable that represents the inputs we want to **learn** from
- let **Y_truth** represent the associated expectation

Our goal is to run the **model** on **X** so that: **model(X) = Y_truth**.  

But again, we saw in the [example](#example) that what the **model** produces may be right or wrong according to 
the expectations. And indeed, we would like the **model** to give only correct answers. In a way, we would like 
to teach the **model** how to predict good results out of the **data inputs**.

In order to teach the **model**, we will in fact tell for each of its results 
whether it is right ![right](/_assets/images/general/right.png) or wrong ![wrong](/_assets/images/general/wrong.png). 
The **model** will then **learn** via its **weights**. 

=> We call it the **learning** phase.

Once we are satisfied that our **model** has learnt on the **dataset**, we can use the **model** on other **data** that 
we did not explore yet, in order to create new results. This is the final goal: the student has become the master.

=> This is the **inferring** phase.

## Conclusion

We saw in this article some general concepts that revolve around the deep-learning **model**.
**Learning** is just about modifying the **model** so that the **model** produces expected results 
on given **data inputs**. 
Then, it is possible to use this **model** to produce new results.

In the [next article]({% post_url 2021-08-06-inside-the-model %}), we will dig deeper inside the **model**. 
