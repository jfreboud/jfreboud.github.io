---
layout: page
title: Table of Contents
permalink: /table_of_contents/
---

<div class="post-content myTitles">
<h2 id="deep-learning-meta-walkthrough">Deep Learning Meta Walkthrough</h2>

<h3 id="the-foundation">The Foundation</h3>

<h4 id="1-general-concepts">1. <a href="/2021/08/05/general-concepts.html">General Concepts</a></h4>

<p>This is the first article of our walkthrough in deep learning neural networks.
First things first, we explore some general concepts of deep learning, introducing the deep learning model.</p>

<h4 id="2-inside-the-model">2. <a href="/2021/08/06/inside-the-model.html">Inside the Model</a></h4>

<p>In this article, we explore the generic structure of a deep learning model.</p>

<h3 id="the-learning-process">The Learning Process</h3>

<h4 id="1-the-loss-function">1. <a href="/2021/08/09/loss-function.html">The Loss function</a></h4>

<p>We complete the deep learning model with the loss function: this is the first step toward the learning process.</p>

<h4 id="2-the-backward-pass">2. <a href="/2021/08/14/backward-pass.html">The Backward Pass</a></h4>

<p>The backward pass is the nemesis of the forward pass: this is the second step toward the learning process.</p>

<h4 id="3-the-weights">3. <a href="/2021/08/19/weights.html">The Weights</a></h4>

<p>The weights are the learning elements of the deep learning model: the core of the learning process.</p>

<h3 id="the-deep-learning-algorithm">The Deep-Learning Algorithm</h3>

<h4 id="1-the-gradient-descent-algorithm">1. <a href="/2021/08/23/gradient-descent.html">The Gradient Descent Algorithm</a></h4>

<p>We use the different parts we have seen so far to run the learning phase from scratch.</p>

<h4 id="2-batch-learning">2. <a href="/2021/08/24/batch-learning.html">Batch Learning</a></h4>

<p>A new idea to build a more robust learning: learn on multiple data input at once.</p>
</div>

<div class="post-content myTitles">
<h2 id="from-a-layer-perspective">From a Layer Perspective</h2>

<h3 id="the-linear-layer">The Linear Layer</h3>

<h4 id="1-the-linear-layer">1. <a href="/2021/09/19/linear.html">The Linear Layer</a></h4>

<p>We explore the linear layer. It is the first step to be able to design deep learning models. 
We also speak about the neural structure and a better way to compute the backward pass.</p>

<h4 id="2-the-activation-layer">2. <a href="/2021/10/06/activation.html">The Activation Layer</a></h4>

<p>Let us reveal the neural structure for the other layers of our “Example”: the Activation and the Input 1D layers.</p>
</div>

<div class="pager">
<ul class="pagination">
  <li><div class="dot"><div class="current-page">1</div><a class="next-page" onclick="currentTitle(1)">1</a></div></li>
  <li><div class="dot"><div class="current-page">2</div><a class="next-page" onclick="currentTitle(2)">2</a></div></li>
</ul>
</div>

<script>
var titleIndex = 1;
showTitles(titleIndex);

function currentTitle(n) {
  showTitles(titleIndex = n);
}

function showTitles(n) {
  var i;
  var titles = document.getElementsByClassName("myTitles");
  var dots = document.getElementsByClassName("dot");
  if (n > titles.length) {titleIndex = 1}    
  if (n < 1) {titleIndex = titles.length}
  for (i = 0; i < titles.length; i++) {
      titles[i].style.display = "none";  
  }
  for (i = 0; i < dots.length; i++) {
      dots[i].childNodes[0].style.display = "none";
      dots[i].childNodes[1].style.display = "block";
  }
  titles[titleIndex-1].style.display = "block";  
  dots[titleIndex-1].childNodes[0].style.display = "block";
  dots[titleIndex-1].childNodes[1].style.display = "none";
}
</script>
