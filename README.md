# Chord Accompaniment using Reinforcement Learning

**POC:** Nina Rauscher (nr2861@columbia.edu)

---
<h3> Summary </h3>

In this project, we propose creating chord progressions using reinforcement learning, specifically, proximal policy optimization (PPO) with clipped objective and an actor-critic method. For the sake of simplicity, we limited potential chords to four chord types (Major, minor, dominant 7th, and minor 7th). Combining rewards from music theory and occurrences of chord progressions from a subset of Beethoven sonatas, our model has achieved results significantly better than how random chord progressions would perform. From a human evaluation standpoint, in most cases, the generated chord accompaniment is satisfying. Still, it tends to give a different character to the song except when the melody is not overly complex. Thus, future work includes improving the model for more elaborate melodies as well as expanding the scope to generate additional parts (such as a bass line) to go with the melody and chords.

<h2> Introduction </h2>

Among the subtasks of music generation, chord accompaniment corresponds to creating an optimal sequence of chords to enrich and provide harmonic and rhythmic support to a specific melody. When composing a song, composers often start by defining a melody and then determining an appropriate sequence of chords (a chord progression) to go with that melody. This process requires both the following of music theory rules while simultaneously creating a pleasing sound. However, finding the right tradeoff between the guidance of music theory and creativity is challenging. Indeed, traditional learning techniques didn’t appear appropriate for melody harmonization since even neural-network-based initiatives have failed to respect music rules and maintain long-term dependency.

In this work, we propose an innovative way to use reinforcement learning to help a composer devise a chord progression for a given input melody. To do so, our system makes use of data sets of well-known music (e.g., Beethoven sonatas). Looking at the chord changes in that music, we assume that the chord changes occurring most frequently are the most desirable chord transitions to employ. At the same time however, we want to encourage chords that will sound the best with the input melody, i.e., create pleasant-sounding intervals with the melody as dictated by standard classical music theory. Therefore, we reward the chord progressions that follow these music rules. Finally, to encourage musical phrases, we positively reward beginning and ending on the same chord.

<h2> Literature Review </h2>

The AI research community has shown a growing
interest in automatic music generation over the past
decade. Most traditional studies on chord
accompaniment leverage recurrent neural networks
(RNNs) to predict chord labels (and sometimes chord
functions). Lim et al. [2] and Yeh et al. [10] were
among the first to use bidirectional long short-term
memory (BLSTM) to generate chord progressions.

Despite being effective according to their results, these
supervised methods have shown limitations: they don’t
integrate external music theory knowledge, and the
number of generated chords is restricted, having at
most two chords per bar. Besides, most existing
models are evaluated on metrics such as the maximum
likelihood estimation (MLE) that will likely entail a
loss of diversity in the generated chords as it learns
common patterns in the corpus used for training.

Since 2018, several studies (Ji, Yang, Luo and Li,
2023 [3]; Jiang, Jin, Duan and Zhang, 2020 [4]; Kotecha,
2018 [5]; Shukla and Banka, 2018 [9]) have emerged where
these issues are attenuated thanks to the
implementation of reinforcement learning (RL)
techniques. Indeed, similar to text generation or image
generation, music generation works well with
reinforcement learning as chord accompaniment can
be seen as a sequential decision problem, and the agent
can consider music rules (concordance between chords
and melody notes). Depending on the complexity of
the RL algorithms, results vary from relatively
monotonous chord progressions (when the algorithm is
simple as suggested in Shukla and Banka, 2018) to
creations that humans evaluate as almost as good as
authentic melody harmonization (not only on
coherence but also fluency). The most satisfying
studies rely on actor-critic (AC) models (Jiang et al.,
2020 [4]) or DQN combined with conditional LSTM (Ji et
al., 2023 [3]).

<h2> Methodology </h2>
<h3> Reward System </h3>

Our methodology entails formulating melody
harmonization as an RL problem. We set specific
assumptions, define our ambitions, and create a fully
observable, deterministic RL environment to teach the
system chord transitions based on empirical music data
and theoretical rules. In designing our algorithm, we
established a reward system influenced by both
empirical frequency in known music and adherence to
music theory.

Regarding empirical frequency, we looked at how
often particular chord transitions occurred in 32 of
Beethoven’s piano sonatas. Furthermore, for
simplification, we only looked at the frequency of a
particular chord change only if both the first chord and the second chord were a major, minor, dominant 7th, or minor 7th chord (i.e. only four chord types were
considered).

For further simplification, we (1) only accepted input
melodies in 4/4 time, (2) only allowed note durations
to be eighth note multiples (e.g. C for 1.5 beats = C for
3 eighth notes), and (3) only allowed our generated
chords to fall on downbeats (in 4/4 time there are four
downbeats in a measure and two 8th notes to every
beat, one 8th note on the downbeat, one on the
upbeat). So when a chord does change, it should only
change on a downbeat, but it doesn't have to change on
every downbeat.

Thus, we rewarded chord switches that Beethoven made in
one or more of his sonatas. For the “best” chords, the
melody note should be the first, major 3rd, minor 3rd,
4th, 5th, minor 6th, major 6th, or octave of the
chord. 7ths are also tolerated for 7th chords. Chords can change on a rest (i.e., no melody note is currently
playing). These “best” chord changes were given
larger rewards. Furthermore, we gave an even
larger reward to an overall chord progression (i.e. all
the chord changes for the given melody taken as a
whole) if that chord progression would start and finish
on the same chord to encourage musical phrases that
sounded complete. All these “best” decisions are
supported by classical music theory.

<h3> Open AI Gym Environment </h3>

To train our reinforcement learning agent, we used Open AI Gym to create a custom environment adapted to our problem and representation.

First of all, we considered a setting of discrete states
and actions. The state is represented by a tuple (Chord,
Melody note), and the agent's action is to select the next
chord. There are 576 states and 48 discrete actions,
which remains reasonable in terms of calculations.

<div style="flex: 48%;">
    <p align="center">
      <img src="Chord accompaniment loop schema.png" alt="At a given step, the agent is in state t described by (Chord_chord_type, Melody_note). The agent takes an action t, consisting of choosing the next chord. The environment is defined by initial_state, melody_notes, frequencies_matrix, MaxiReward, MinReward, BonusReward and NegReward. Once action t is taken, the environment will inform the agent of the new reward r(t+1) and the new state s(t+1)." style="width: 70%;">
    </p>
</div>

When it comes to the most important part of the
environment, which is the reward structure, we
decided to consider an *empirical reward* that reflects
how good the action is based on frequencies of chord
transitions we’ve collected from musical datasets, and
a *legal reward* that captures the consistency of the
action with respect to music theory rules.

<div style="flex: 48%;">
    <p align="center">
      <img src="Chord Accompaniment Step Function.png" alt="Chord accompaniment step function and reward structure" style="width: 70%;">
    </p>
</div>

These rewards are a function of 4 hyperparameters that
we need to tune before using the environment to train
an RL agent. To do so, we focused on the values of
these parameters that allow the environment to
distinguish between *good* and *bad* chord
progressions. Below is an example of rewards given to a *good* chord progression (in blue) and a *bad* chord progression (in orange), under defined parameter values:

<div style="flex: 48%;">
    <p align="center">
      <img src="Rewards good vs bad chord progressions.png" alt="Rewards for a good vs bad chord progression" style="width: 60%;">
    </p>
</div>

On this chart, we notice that the reward at each timestep (new chord decision / action taken) is higher for the good chord progression than the bad one. The only timestep at which the rewards are pretty close is the last one as a high reward is given to chord progressions that satisfy the rule of starting and ending on the same chord, and the 2 chord progressions displayed follow this rule.

To better understand the role of each hyperparameter we introduced, here are the detailed formulas for each subcomponent of our rewards:

![Rewards formula](<Rewards formula.png>)

<h3> Algorithm Design </h3>

After defining the environment setup, we chose a suitable
algorithm for this project, the proximal policy
optimization (PPO) with a clipped objective, which has numerous advantages. First, PPO is simple to
implement, with no requirement of analytical results or
second-order derivatives. Instead, it relies on updates
via SGD algorithms, namely ADAM in this case. In
addition, since this algorithm does not rely on any
property of the initial environment, it can be used for
both discrete and continuous state and action spaces.
Besides, PPO exhibits far more stable training than a
vanilla policy gradient method (such as
REINFORCE), since it discourages large policy
updates by clipping the surrogate advantage. Finally, the complexity of the spaces can be increased without requiring any modifications to our algorithm. This was an important point at the beginning of our work as we were likely to modify the definition of our state and action spaces.

As literature review has shown the relevancy of actor-critic methods for our problem, we incorporated one in our PPO implementation. Both the actor and the critic are structured as deep neural networks with three hidden layers, each containing 64 units. The actor takes in the current state as input and outputs the probability distribution over possible actions, while the critic estimates the value of the current state and predicts the total expected reward, which helps to guide the actor’s decisions.

Besides, to ensure that our results are satisfying, we have implemented a baseline model where actions are selected randomly from a uniform random distribution (see the *Results* section).

Below is the pseudocode for the policy-based algorithm: (*sources: Achiam, 2017 and Heeswijk, 2022*)

<div style="flex: 48%;">
    <p align="center">
      <img src="Pseudo code PPO with clipped obj.png" alt="Pseudo code PPO with clipped objective" style="width: 70%;">
    </p>
</div>

As shown above, during each epoch, we gather trajectories according to policy $\pi_k$. The preliminary step involves evaluating the advantages $A_t$ under the policy $\pi_k$ using neural networks to determine the best actions at every state. The advantage $A_t$ is calculated as the difference between the actual return-to-go $R_t$, which is the discounted expected cumulative future reward from each state, and the value estimate $V_t$ provided by the critic network for each state:

$A_t$ = $R_t$ - $V_t$ with $R_t$ = $r_t$ + $\gamma$ $r_{t+1}$ + $\gamma^2$ $r_{t+2}$ + ... + $\gamma^{T-t-1}$ $r_{T}$

In the previous equation, $r_i$ refers to the reward obtained a timestep i and $\gamma$ corresponds to the discount factor applied to the rewards to progressively decrease their importance.

Then we compute our loss to get a policy update. This loss is divided into two parts. 

The first part is the actor loss:

![Actor loss formula](<Actor loss.png>)

It is calculated using the importance sampling ratio $\left. \rho_t(\pi_\theta, \pi_{\theta_k} \right)$:

![Sampling ratio formula](<Sampling ratio formula.png>)

which compares the probabilities of actions under the current and new policy iterations. 

Clipping is applied to this ratio using a hyperparameter $\epsilon$ to prevent excessively large updates, thereby stabilizing training. We used $\epsilon$ = 0.2, which empirically works well (Heeswijk, 2022). However, tuning this value is an area of improvement we could investigate for future work. The terms $\left. (1 − \epsilon) A_t \right.$ and $\left. (1 + \epsilon) A_t \right.$ are independent of $\theta$, thus resulting in a null gradient. Besides, there are six non-trivial cases for the range of the importance sampling ratio with corresponding update behavior.

The second part is the critic loss:

![Critic loss formula](<Critic loss formula.png>)

This loss is computed as the mean squared error (MSE)
between the critic value estimate $V_i$ and the return-to-go $R_i$ from each state. The purpose of the critic loss is to train the critic network to make accurate predictions about the expected returns from each state. By minimizing this loss, the critic network brings the value estimate V closer to the actual return.

Finally, we combine these two losses to get the total
loss:

![Total loss formula](<Total loss formula.png>)

By minimizing this total loss, we can update our policy taking into account all the constraints included in our initial problem.

<h3> Results </h3>

We used Weights and Biases to visualize the training and evaluation performance of our model and compare it to the baseline (random action). 

The proposed model's reward achieves convergence during training as illustrated by this chart of the training reward moving average, initially slightly negative and quickly converging to about 400:

<div style="flex: 48%;">
    <p align="center">
      <img src="Training reward moving average.png" alt="Training reward moving average" style="width: 60%;">
    </p>
</div>

Furthermore, we analyzed the differences in evaluation
rewards between the baseline model and the policy-
based model, observing a significant improvement in
the latter model, with the highest reward being above 525. The rewards of the baseline model on the other hand, are almost all negative, indicating bad chord choices:

<div style="display: flex; justify-content: space-between;">

  <div style="flex: 48%;">
    <p align="center">
      <img src="Baseline evaluation reward moving average.png" alt="Baseline evaluation reward" style="width: 60%;">
      <br>
      Baseline (random actions)
    </p>
  </div>

  <br>

  <div style="flex: 48%;">
    <p align="center">
      <img src="PPO evaluation reward moving average.png" alt="PPO evaluation reward" style="width: 60%;">
      <br>
      Our model (PPO with clipped objective)
    </p>
  </div>

  <br>

</div>

In terms of audio output, we used our policy-based
model to create chord progressions for two excerpts
from Beethoven’s “Sonata No. 1, 1st Movement, Opus
2 No.1” and for a melody we created ourselves. None of these were used to train the model, i.e., the sonata isn't part of the subset of sonatas used to compute our empirical rewards.

The first Beethoven excerpt resulted in a number of
different chords that weren't present in the original
work. Our chords sounded pleasing, but also gave a
different character to the piece. 

<div align="center">
  
  | Sonata 1 - Beginning - Original work | Sonata 1 - Beginning - Generated chord progressions |
  |----------------------|----------------------|
  | <p align="center">[Link to audio](Sonata%201%20(beginning%20-%20original%20chords).wav)|<p align="center">[Link to audio](Sonata%201%20(beginning%20-%20our%20chords).wav)|
  | <p align="center"> <img src="Sonata 1 - Beginning - Original.png" alt="Sonata 1 - Beginning - Original" style="width: 90%;">|<p align="center"><img src="Sonata 1 - Beginning - Our chords.png" alt="Sonata 1 - Beginning - Generated chords" style="width: 90%;">|

</div>

The second Beethoven excerpt sounded a bit less satisfying since our generated chord progression did not start and end on the same chord. 

<div align="center">
  
  | Sonata 1 - End - Original work | Sonata 1 - End - Generated chord progressions |
  |----------------------|----------------------|
  | <p align="center">[Link to audio](Sonata%201%20(ending%20-%20original%20chords).wav)|<p align="center">[Link to audio](Sonata%201%20(ending%20-%20our%20chords).wav)|
  | <p align="center"> <img src="Sonata 1 - End - Original.png" alt="Sonata 1 - End -  Original" style="width: 90%;">|<p align="center"><img src="Sonata 1 - End - Generated chords.png" alt="Sonata 1 - End - Generated chords" style="width: 90%;">|

</div>

The best chord progression however, was the one that our RL code created for our own melody. This can perhaps be attributed to the fact that our original melody was simpler than the Beethoven melodies and did not require as many chord changes.

<div align="center">

| Our melody | Our melody with generated chord progressions |
|----------------------|----------------------|
| <p align="center">[Link to audio](Original%20Melody.wav)|<p align="center">[Link to audio](Original%20Melody%20with%20chords.wav)|

</div>

<h2> Future Work </h2>

One area of improvement could consist of broadening our problem not only to the choice of chords for the melody, but also of other parts. For example, training an agent that would be able to determine what bass line would go with those chords and melody or what drum part would
appropriately fit the melody’s rhythm. This could produce the following excerpt (for our melody):

<div align="center">
  
  [Listen to the Original Melody with chords and future work](Original%20Melody%20with%20chords%20and%20future%20work.wav)

</div>


In addition, it would be great to extend our agent capabilities to enable the selection of a music genre (e.g. rock music, jazz, or classical music) to produce the optimal melody harmonization with respect to the selected
genre, likely by utilizing different datasets (for
different genres) to compute empirical rewards.

Finally, we could achieve better performance by fine-tuning further our hyperparameters
and exploring other algorithms besides policy gradient
based algorithms, such as DQN (as suggested by the literature review).

<h2> Conclusion </h2>

We have introduced a novel chord
accompaniment generation approach using Proximal
Policy Optimization (PPO) with an actor-critic
architecture. Our model combines insights from music
theory, empirical chord frequency analysis, and
reinforcement learning techniques to strongly outperform
random chord progressions.

Our methodology incorporates a unique reward system
based on empirical chord frequencies and adherence to
music theory rules. Leveraging the OpenAI Gym
environment, we established a discrete state and action space, facilitating the training of our PPO-based
model. The convergence of our training rewards
validates the effectiveness of our approach in learning
meaningful chord progressions.

Despite initial challenges in modeling the Markov Decision
Process and selecting an appropriate state space, we
successfully generated chord progressions for excerpts
from Beethoven's sonatas and an original melody. The
audio outputs, while mainly satisfactory, highlight the need for further refinement, especially for complex
compositions.

Looking forward, our future work includes expanding
the model's capabilities to generate additional musical
components and exploring alternative reinforcement
learning algorithms such as DQN. Continuous fine-tuning of hyperparameters and incorporating diversity in the empirical chord frequency analysis datasets for different genres are crucial aspects of our ongoing efforts.

<h2> References </h2>

[1] Achiam, J. (2017, October 11). Advanced Policy
Gradient Methods. http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_13_advanced_pg.pdf

[2] Heeswijk, W. V. (2022, November 29). Proximal
Policy Optimization (PPO) Explained.
Medium.https://towardsdatascience.com/proximal-policy-optimization-ppo-explained-abed1952457b

[3] Ji, S., Yang, X., Luo, J., & Li, J. (2023). RL-Chord:
CLSTM-Based Melody Harmonization Using Deep
Reinforcement Learning. IEEE Transactions on
Neural Networks and Learning Systems, 1–14.
https://doi.org/10.1109/TNNLS.2023.3248793

[4] Jiang, N., Jin, S., Duan, Z., & Zhang, C. (2020,
February 7). RL-Duet: Online Music Accompaniment
Generation Using Deep Reinforcement Learning
(arXiv:2002.03082). Art. arXiv:2002.03082.
https://doi.org/10.48550/arXiv.2002.03082

[5] Kotecha, N. (2018, December 3). Bach2Bach:
Generating Music Using A Deep Reinforcement
Learning Approach. ArXiv.Org.
https://arxiv.org/abs/1812.01060v1

[6] Lim, H., Rhyu, S., & Lee, K. (2017, December 4).
Chord Generation from Symbolic Melody Using
BLSTM Networks (arXiv:1712.01011). Art.
arXiv:1712.01011.
https://doi.org/10.48550/arXiv.1712.01011

[7] Ping, T. (n.d.). Functional-
harmony/BPS_FH_Dataset. GitHub, https://github.com/Tsung-Ping/functional-harmony/tree/master/BPS_FH_Dataset

[8] Schulman, J., Wolski, F., Dhariwal, P., Radford,
A., & Klimov, O. (2017). Proximal Policy
Optimization Algorithms (arXiv:1707.06347). arXiv.
http://arxiv.org/abs/1707.06347

[9] Shukla, S., & Banka, H. (2018). An Automatic
Chord Progression Generator Based On
Reinforcement Learning. 2018 International
Conference on Advances in Computing,
Communications and Informatics (ICACCI), 55–59.
https://doi.org/10.1109/ICACCI.2018.8554901

[10] Yeh, Y.-C., Hsiao, W.-Y., Fukayama, S.,
Kitahara, T., Genchel, B., Liu, H.-M., Dong, H.-W.,
Chen, Y., Leong, T., & Yang, Y.-H. (2021). Automatic
melody harmonization with triad chords: A
comparative study. Journal of New Music Research,
50(1), 37–51.
https://doi.org/10.1080/09298215.2021.1873392

---

**Note:** 
This project was conducted in December 2023 as part of my *ORCS 4529 Reinforcement Learning* class at Columbia University, along with Mohamed El Amine Lakhnichi, Phuong Anh Nguyen, Guillaume Chilla, and Gregor Zdunski Hanuschak.
