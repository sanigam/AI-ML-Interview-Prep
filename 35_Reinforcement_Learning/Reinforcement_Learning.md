# Reinforcement Learning

📺 **Video Lecture:** https://youtu.be/0d70e5V333A

## Interview Anchor
- **MDP (Markov Decision Process):** Formal framework for sequential decision-making with states, actions, rewards, and transitions
- **Value Function:** Estimate of expected future reward from a given state or state-action pair
- **Policy:** Strategy mapping states to actions, optimized to maximize cumulative reward

## Key Concepts Overview
Reinforcement Learning (RL) is the study of learning through interaction with an environment. Unlike supervised learning (learn from labeled data) or unsupervised learning (find patterns), RL agents learn by taking actions, receiving rewards/penalties, and optimizing to maximize long-term cumulative reward. RL powers game-playing (AlphaGo), robotics (robotic control), and increasingly production systems (dialogue systems, recommendation tuning with RLHF). Understanding RL is valuable because it handles sequential decision-making, deals with sparse rewards, and enables autonomous learning without labeled data. However, RL is notoriously difficult: non-stationary environments, sparse feedback, exploration challenges, and high sample complexity make it research-intensive.

---

### Q1: Explain MDPs (Markov Decision Processes) and their components.

**A:** An **MDP** formally models sequential decision-making with five components:

- **States (S):** environment configurations the agent observes.
- **Actions (A):** choices the agent can make.
- **Transition dynamics P(s′ | s, a):** probability of reaching state s′ after taking action a in state s.
- **Reward function R(s, a, s′):** immediate reward for the transition.
- **Discount factor γ:** how much future rewards matter, with 0 < γ < 1 (typically 0.9–0.99).

The defining **Markov property:** the future depends only on the current state, not the full history.

**Process:** start in state s₀, take action a₀, transition to s₁, receive reward r₁, and repeat:

```
s₀ → a₀ → s₁ → a₁ → s₂ → ... → r_T
```

**Objective:** maximize the cumulative discounted reward:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

**Concrete example — robot navigation:** states are grid positions; actions are up/down/left/right; reward is +1 for reaching the goal and −1 for obstacles; γ = 0.99. The agent learns to navigate efficiently.

MDPs are foundational — most RL problems are modeled this way. Variants include **POMDPs** (where the agent only observes part of the state) and continuous state/action spaces.

---

### Q2: What are value functions (V and Q) and Bellman equations?

**A:** Two complementary value functions:

**State value function V(s)** — expected cumulative future reward starting from state s:

```
V(s) = E[ G_t | s_t = s ]
     = E[ r_t + γ · V(s_{t+1}) ]
```

A high V means s is a "good" state — it tends to lead to high reward.

**Action value function Q(s, a)** — expected cumulative reward starting from s, taking action a, then acting optimally:

```
Q(s, a) = E[ r_t + γ · max_{a′} Q(s_{t+1}, a′) ]
```

A high Q means (s, a) is a good state-action pair.

**Bellman equation** — the recursive relationship that makes RL tractable. For a policy π:

```
V(s) = Σ_a π(a|s) · Σ_{s′,r} P(s′, r | s, a) · [ r + γ · V(s′) ]
```

In a deterministic environment this collapses to:

```
V(s) = R(s) + γ · V(s′)
```

The recursion lets you compute the value of any state from the values of its successors.

**Concrete example — chess:** the state is a board position; the value is the probability of winning. The value of the current position equals the average value of all next positions (weighted by the move-probability) plus any immediate reward.

**Why it matters:** Bellman equations enable dynamic programming — instead of expensive forward simulation, you solve the equations backward. The two key operations are:

- **Policy evaluation** — compute V for a fixed policy.
- **Policy improvement** — find a better policy using V.

Together they form **policy iteration**: evaluate → improve → repeat.

---

### Q3: Explain the difference between policy iteration and value iteration.

**A:** Both algorithms solve MDPs to find an optimal policy, but they take different routes.

**Policy iteration** alternates two steps until convergence:

1. **Evaluation** — given the current policy π, solve the Bellman equation to compute V_π.
2. **Improvement** — derive a new policy π′ by acting greedily with respect to V_π:

   ```
   π′(s) = argmax_a Q_π(s, a)
   ```

3. Repeat until the policy stops changing — that's the optimal policy π*.

**Value iteration** skips the explicit policy-evaluation step and iterates the **Bellman optimality equation** directly:

```
V(s) ← max_a Σ_{s′, r} P(s′, r | s, a) · [ r + γ · V(s′) ]
```

After V converges, extract the policy with one greedy step:

```
π(s) = argmax_a Q(s, a)
```

**Tradeoffs:**

- *Policy iteration* fully evaluates the policy each iteration — fewer total iterations, but each one is slow.
- *Value iteration* improves value every iteration — faster per iteration but typically more total iterations.

In practice, value iteration is more common because it's simpler and often faster overall. Both algorithms guarantee convergence to the optimal policy in finite state/action spaces with proper initialization, though speed depends on the problem structure.

**Examples:**

- Small gridworld → value iteration tends to be faster.
- Pacman with a large state space → policy iteration may be faster because each evaluation converges quickly.

Both assume the MDP (transition probabilities and rewards) is known. When it isn't, use model-free methods like Q-learning or policy gradients.

---

### Q4: What is the exploration-exploitation trade-off in RL?

**A:** Two competing goals at every step:

- **Exploitation:** take the action you believe is best (highest estimated Q-value).
- **Exploration:** take an action you're uncertain about to gather information.

Pure exploitation can leave you stuck with a suboptimal habit; pure exploration wastes time on bad actions. The art is balancing the two.

**Common strategies:**

- **Epsilon-greedy.** Take the greedy action with probability 1 − ε; pick a random action with probability ε. Simple and effective. Usually ε decays over time so the agent explores less as it learns.

- **Upper Confidence Bound (UCB).** Pick the action with the highest Q-value plus a confidence bonus:

  ```
  UCB(a) = Q(a) + c · √( log(t) / N(a) )
  ```

  Actions with fewer pulls (small N(a)) get a larger bonus, which encourages trying uncertain options. Has theoretically optimal regret bounds.

- **Thompson Sampling.** Maintain a posterior distribution over Q-values, sample from it, and act greedily on the sample. This naturally balances exploration (sampling uncertainty) and exploitation (high posterior means).

- **Boltzmann (softmax) exploration.** Pick actions according to a softmax over Q-values. High temperature → near-uniform (more exploration); low temperature → near-greedy (more exploitation). Anneal the temperature down over time.

**Practical wisdom:** explore aggressively early to learn the environment, then exploit later. Multi-armed bandits formalize this and the same strategies extend to full RL.

**Challenges:**

- **Delayed rewards** — an exploratory action might pay off 100 steps later, making credit hard to assign.
- **Non-stationarity** — true Q-values change as the environment or other agents evolve.
- **Regret minimization** — the goal is often to minimize cumulative regret (reward gap from the optimal policy).

Best strategy depends on the problem: epsilon-greedy is fine for simple bandits, UCB or Thompson sampling tend to work better in complex environments.

---

### Q5: What is the difference between model-based and model-free RL?

**A:** The two paradigms differ in whether the agent learns an explicit model of the environment.

**Model-based RL.** Learn (or know) the environment dynamics P(s′ | s, a) and the reward function, then use *planning* to decide actions.

1. Learn a model from interactions.
2. Use the model for planning (e.g., value or policy iteration on the learned model).
3. Execute the best planned actions.

- *Pros:* sample-efficient (learn the model from few interactions, plan extensively); enables counterfactual reasoning ("what if I did X instead?").
- *Cons:* model errors compound during planning; planning is computationally expensive.

**Model-free RL.** Skip the model — directly learn a value function or policy from interactions.

1. Take an action; observe reward and next state.
2. Update the value/policy based on the transition.
3. Repeat.

- *Pros:* simpler (no model to learn); more robust to model misspecification; often faster in practice.
- *Cons:* sample-inefficient (many interactions needed); no planning.

Methods include Q-learning (learn Q-values) and policy gradients (learn the policy directly).

**Practical takeaway:** model-based methods are more efficient when accurate models are available. Model-free methods are less efficient but more robust. Hybrid approaches are common — learn an approximate model, then refine with model-free RL. Modern deep RL is mostly model-free (it's hard to learn accurate models in complex environments), but model-based RL is an active research area.

---

### Q6: Explain Q-learning and SARSA: what's the difference?

**A:** Both are model-free temporal-difference (TD) algorithms that estimate Q-values from sampled transitions. They differ in *what next-action value they bootstrap from*.

**Q-learning** (off-policy) — bootstraps from the *best* next action, regardless of what the agent actually does:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_{a′} Q(s′, a′) − Q(s, a) ]
```

Even when the agent explores randomly, Q-learning's target uses the optimal value of the next state — so it learns the optimal policy Q* while following any exploratory policy.

**SARSA** (on-policy) — bootstraps from the action a′ the agent *actually* takes next:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · Q(s′, a′) − Q(s, a) ]
```

This learns the value of the policy being followed (including its exploration noise).

**The key difference:** Q-learning uses `max_{a′}`; SARSA uses Q(s′, a′) for the actual a′.

**Implication:**

- Q-learning is *off-policy* — learns optimal Q* while exploring however it likes.
- SARSA is *on-policy* — learns the value of the actual exploring policy, which makes it more conservative.

**Cliffwalking example:** in a gridworld with a "cliff" of large negative reward, Q-learning learns to walk right next to the cliff (optimal under deterministic execution). SARSA learns to walk further from the cliff because exploration occasionally pushes the agent over the edge — it accounts for the cost of its own exploration.

**Convergence:** both converge to the optimal Q* given sufficient exploration. Q-learning typically converges faster because it bootstraps from optimal values; SARSA is safer in the meantime.

**In practice:** Q-learning is more common because it's more sample-efficient and cleanly separates exploration from learning. **Deep Q-Networks (DQN)** extend Q-learning to deep neural networks.

---

### Q7: What are Deep Q-Networks (DQN) and how do they scale Q-learning?

**A:** **DQN** extends Q-learning to high-dimensional states (like raw images) using neural networks. Tabular Q-learning maintains an entry Q[s, a] for every state-action pair, which is infeasible for large or continuous state spaces. DQN replaces the table with a neural-net approximator:

```
Q(s, a ; θ)        # neural network with parameters θ
```

**Training loop:**

1. Forward state s through the network to get Q-values for all actions.
2. Select an action with ε-greedy.
3. Observe reward r and next state s′.
4. Compute the TD target using a separate **target network** with parameters θ⁻:

   ```
   y = r + γ · max_{a′} Q(s′, a′ ; θ⁻)
   ```

5. Update θ to minimize the squared error:

   ```
   loss = ( Q(s, a ; θ) − y )²
   ```

**Two key stabilizing innovations:**

- **Experience replay** — store transitions in a buffer and train on random minibatches. This breaks correlation between consecutive samples and stabilizes training.
- **Target network** — a periodic copy of the Q-network used to compute TD targets. Holding the target fixed for many steps avoids the instability of chasing a moving target.

**Benefits:**

- *Scalability* — handles high-dimensional states (Atari frames).
- *Generalization* — network generalizes Q-values to unseen states.
- *Stability* — replay buffer + target network make learning much more reliable.

**Known limitations:**

- *Overestimation* — the max operator systematically overestimates Q-values.
- *Off-policy drift* — the learned network can diverge from the behavior policy.
- *Sample inefficiency* — still requires many interactions.

**Modern improvements:** Double DQN (separate networks to reduce overestimation), Dueling DQN (separate value and advantage streams), Prioritized Experience Replay (sample important transitions more often).

DQN was the breakthrough that achieved human-level performance on Atari without domain knowledge — the foundation for modern deep RL.

---

### Q8: What are policy gradient methods (REINFORCE) and when to use them?

**A:** **Policy gradients** directly optimize a parameterized policy π(a | s ; θ) by following the gradient of expected return.

**REINFORCE — the basic policy gradient.** For a trajectory (s₀, a₀, r₀, ..., s_T), compute the discounted return from each step:

```
G_t = Σ_{τ=t}^{T} γ^(τ−t) · r_τ
```

Then update parameters in the direction that increases the log-probability of high-return actions:

```
θ ← θ + α · G_t · ∇_θ log π(a_t | s_t ; θ)
```

**Intuition:** if an action led to high return, increase its probability; if low, decrease it.

**Advantages:**

- *Direct optimization* — optimizes the expected return, which is what we actually care about.
- *Continuous actions* — naturally handles continuous action spaces (e.g., Gaussian policies).
- *Stochastic policies* — learns a distribution over actions, which gives natural exploration.

**Limitations:**

- *High variance* — G_t is an unbiased but very noisy estimator of value.
- *Slow convergence* — many samples needed.
- *On-policy* — must use recent data; you can't easily reuse old transitions.

**Variance reduction with a baseline.** Subtract a learned value function V(s_t) from the return:

```
θ ← θ + α · (G_t − V(s_t)) · ∇_θ log π(a_t | s_t ; θ)
```

The quantity A_t = G_t − V(s_t) is the **advantage**. It has the same expected gradient but much lower variance because it measures *relative* value. Methods that explicitly learn both an actor (policy) and a critic (value function) are called **advantage actor-critic** methods.

**When to use policy gradients:** continuous action spaces (robotics), settings where you need a stochastic policy, or problems where Q-learning fails to converge. Policy gradients are foundational for deep RL in continuous control domains.

---

### Q9: Explain actor-critic methods (A2C, A3C, PPO) and their advantages.

**A:** **Actor-critic** methods combine policy gradients (the *actor*, which picks actions) with a learned value function (the *critic*, which evaluates them).

The **advantage** is the critic's relative judgment of an action:

```
A(s, a) = Q(s, a) − V(s)
```

The two networks are updated jointly:

```
actor:    θ ← θ + α · A(s, a) · ∇_θ log π(a | s ; θ)

critic:   φ ← φ − β · ∇_φ [ r + γ · V(s′ ; φ) − V(s ; φ) ]²
```

The actor pushes probability mass toward high-advantage actions; the critic learns value estimates that reduce gradient variance.

**Common variants:**

- **A2C (Advantage Actor-Critic):** synchronous; trains on batches of trajectories from multiple parallel environments.
- **A3C (Asynchronous A2C):** asynchronous; multiple threads learn independently and share global networks.
- **PPO (Proximal Policy Optimization):** constrains the update so the new policy stays close to the old one. The clipped objective is:

  ```
  L_CLIP(θ) = E[ min( r_t(θ) · A_t,  clip(r_t(θ), 1 − ε, 1 + ε) · A_t ) ]
  ```

  where r_t(θ) = π_new(a_t | s_t) / π_old(a_t | s_t) is the importance ratio. The clip prevents huge policy jumps and keeps training stable. PPO is the de facto standard in modern deep RL — including RLHF for LLM training.

- **SAC (Soft Actor-Critic):** maximum-entropy RL — optimizes both reward and policy entropy to keep exploration alive. Popular in robotics.

**Quick comparison:**

- *REINFORCE* — simplest, but high variance.
- *A2C / A3C* — lower variance, more complex.
- *PPO* — stable and sample-efficient; the practical default.
- *SAC* — adds an entropy bonus, especially good for continuous control.

---

### Q10: What is reward shaping and sparse rewards problem?

**A:** **Sparse rewards** — the agent receives a reward only occasionally (e.g., on reaching a goal), not at every step.

Example: in chess, the only reward signal is at the game's end (win/loss/draw). Sparse rewards make learning hard — thousands of steps with one reward at the end means it's very hard to figure out which steps actually mattered.

**Reward shaping** augments sparse rewards with dense intermediate rewards. In chess, you might add +0.1 for capturing a piece and +1.0 for checkmate. Dense rewards guide learning and accelerate convergence. **The catch:** poorly shaped rewards bias learning — the agent might learn to capture pieces but never finish the game (local optimum).

**Potential-based shaping** is a principled approach. Define a potential function Φ(s) (often a value-function estimate of goal proximity), then add a shaping term that telescopes:

```
r′(s, a, s′) = r(s, a, s′) + γ · Φ(s′) − Φ(s)
```

This provably preserves the optimal policy while still providing dense intermediate guidance.

**Challenges with reward design:**

- *Hand-designed* — picking good rewards requires real domain knowledge.
- *Suboptimal guidance* — wrong rewards steer learning toward the wrong objective.

**Alternatives to manual reward design:**

- **Learning from demonstrations** — observe an expert and extract a reward function.
- **Inverse RL** — explicitly learn the reward function that explains observed expert behavior.
- **Curriculum learning** — start with easy tasks (dense rewards) and progress to hard ones (sparse).
- **Intrinsic motivation** — bonus for visiting novel states (curiosity-driven exploration).

Sparse rewards are genuinely hard; most solutions require either careful engineering or learning from data.

---

### Q11: What are multi-agent RL basics and challenges?

**A:** Multi-agent RL has multiple agents learning simultaneously in a shared environment. Compared to single-agent, this introduces several complications:

- **Non-stationarity** — the environment changes as other agents learn.
- **Partial observability** — agents don't see other agents' actions or states completely.
- **Scalability** — the joint state/action space explodes with the number of agents.
- **Cooperation vs. competition** — agents may help or hinder each other.

A canonical example is AlphaStar (StarCraft II), which trains two agents competing — the environment is highly non-stationary because the opponent improves over time.

**Approaches:**

- **Self-play** — an agent plays against itself, iteratively improving. AlphaGo trains a model, plays it against a previous version, and whichever wins becomes the new champion.
- **Independent learners** — each agent learns separately, treating the others as part of the environment. Simple, but convergence isn't guaranteed.
- **Centralized training, decentralized execution** — train with access to the full state, but execute with only local observations. Enables learned coordination without runtime communication.
- **Learned communication** — agents learn to communicate to coordinate (emergent communication).

**Key challenges:**

- **Credit assignment** — which agent deserves credit for a shared reward?
- **Cooperation vs. equilibrium** — Prisoner's-dilemma-style situations where agents may not cooperate even when cooperation would be best.
- **Scalability** — the joint action space is exponential in the number of agents.

This is an active research area; AlphaStar's self-play approach was a major breakthrough. Applications include multi-robot coordination, game playing, and traffic control.

---

### Q12: What is Reinforcement Learning from Human Feedback (RLHF) and its role in LLM alignment?

**A:** **RLHF** trains models using human preferences, aligning outputs with what humans actually want.

**Three-stage pipeline:**

1. **Behavioral cloning** — fine-tune the language model on high-quality human demonstrations.
2. **Reward model** — show humans pairs of completions (A vs B), collect their preferences, and train a classifier to predict which completion a human would prefer.
3. **RL training** — use the reward model as the reward signal and fine-tune the LLM with RL (typically PPO) to maximize predicted human preference.

**Concrete example: GPT-3 → ChatGPT.** Generate four completions, humans rank them, train a reward model on the rankings, then use PPO to fine-tune the LLM to maximize the reward model's score.

**Benefits:**

- **Alignment** — outputs match what humans actually prefer.
- **Safety** — the model can be trained to refuse harmful requests.
- **Expressiveness** — the model learns nuanced preferences that are hard to specify by hand.

**Limitations:**

- **Reward hacking** — the LLM learns to exploit quirks of the reward model, producing text that scores high without being genuinely better.
- **Costly human feedback** — ranking thousands of completions is expensive.
- **Preference diversity** — humans disagree, raising the question of whose preferences count.

Modern variants supplement human feedback with **AI feedback** (rule-based or learned evaluators) to reduce human cost, and approaches like **Constitutional AI** train the model against an explicit set of values rather than direct human ratings.

RLHF is standard in modern LLM training (GPT, Claude, and others) and is crucial for deployment because it reduces harmful outputs and improves helpfulness.

---

### Q13: What is sim-to-real transfer and why is it important for robotics?

**A:** **Sim-to-real:** train an agent in simulation, then deploy on a real robot. Simulation is cheap and fast (thousands of episodes per day); real robots are expensive in both hardware and time.

**The core challenge — the simulation gap.** Simulators are imperfect: real physics has friction, inertia, and sensor noise that simulators don't capture exactly. An agent trained in simulation can fail on a real robot. (Train in MuJoCo to grasp objects; deploy on a real robot, grasps fail because the friction model is slightly off.)

**Solutions:**

- **Domain randomization** — randomize simulator parameters (friction, mass, colors, lighting) during training. The hope is the real robot lies inside the distribution of simulators you trained on. Effective but requires careful parameter choices.
- **Domain adaptation** — train in simulation, then fine-tune on a small amount of real data.
- **Robust policies** — train explicitly for robustness using adversarial perturbations at training time. Sacrifice some performance for stability.
- **Better simulators** — invest in more realistic physics (compute-intensive).
- **Curriculum learning** — start easy (e.g., high friction, slow speed) and progress to harder settings.

**Practical recipe:** randomize broadly during simulation training, then fine-tune on the real robot if possible. Robotics research has shown domain randomization works surprisingly well — OpenAI's dexterous-hand work was trained entirely in simulation with domain randomization.

There's no perfect solution. Best practice: randomize broadly and plan for some fine-tuning on the real robot.

---

### Q14: What is offline RL and why does it matter?

**A:** **Offline RL** trains on a fixed dataset of past interactions, with no additional environment interaction during training. Traditional RL is online — collect data, update, repeat. Offline RL is given a fixed log and asked to extract the best possible policy from it.

**Why it matters:**

- **Safety** — can't risk learning on a real environment (medical, robotics, autonomous driving).
- **Cost** — collecting online data may be prohibitively expensive.
- **Existing logs** — companies often have huge logs of user behavior (e.g., recommender systems) that can be reused.

**The core challenge — distribution shift.** The learned policy may take actions outside the dataset's distribution, where Q-value estimates are unreliable. (Dataset only contains actions {up, right}; the trained policy wants to output {down}; the estimate has no support.)

**Solutions:**

- **Conservative learning** — constrain the policy to stay close to the data distribution. Penalize actions far from what the dataset contains.
- **Pessimistic Q-learning** — use a minimum (rather than max) over next-state Q-values to be conservative about unseen actions:

  ```
  Q(s, a) ← Q(s, a) + α · [ r + γ · min_{a′} Q(s′, a′) − Q(s, a) ]
  ```

- **Behavior cloning** — train a supervised model to imitate dataset actions, then fine-tune with RL while constraining deviation.
- **Inverse RL** — learn a reward function from the data, then fine-tune a policy.

This is an active research area without a clear consensus winner; results typically underperform online RL because of distribution shift. **Applications:** recommender systems (user-interaction logs), healthcare (patient records), robotics (demonstration videos). Offline RL is a practical necessity in high-stakes domains.

---

### Q15: What are advanced RL topics and recent developments?

**A:** A whirlwind tour of frontier RL.

**Advanced topics:**

- **Meta-RL** — *learn to learn*. Train an agent on a distribution of tasks; at test time, the agent adapts to a new task with very few interactions. (Train on 100 maze variants, then adapt to a new maze in just 5 episodes.)
- **Hierarchical RL** — decompose complex tasks into sub-tasks. Learn a high-level policy that selects sub-goals and low-level policies that execute them. Enables more complex behaviors.
- **Imitation learning** — learn from demonstrations rather than rewards. Variants include behavioral cloning (supervised) and inverse RL (recover the reward from expert behavior).
- **Multimodal policies** — many problems have multiple valid solutions. Standard RL picks one; multimodal RL learns a *distribution* over solutions.
- **World models** — learn a generative model of the environment and use it for imagination-based planning (predict future observations given actions, then plan inside the model).
- **Language-guided RL** — use natural-language instructions to specify goals; the agent learns to follow instructions, which transfers to new tasks.

**Recent developments:**

- **Large models as RL backbones** — use transformers / LLMs as the policy or value-function backbone, enabling transfer learning across tasks.
- **Diffusion models for action generation** — capture continuous and multimodal action distributions.
- **AI feedback** — use LLMs to evaluate RL rewards, reducing the cost of human feedback.
- **Vision transformers** — better visual representations for RL agents.

The frontier is combining RL with foundation models (LLMs, vision transformers) to build more general and more capable agents.

---

## Interview Cheatsheet

**Key Terms:**
- **MDP:** Markov Decision Process, formal model with states, actions, rewards, transitions
- **Value Function:** Expected cumulative future reward from state (V) or state-action (Q)
- **Policy:** Mapping from states to actions, optimized to maximize reward
- **Bellman Equation:** Recursive relation for value functions enabling dynamic programming
- **Exploration-Exploitation:** Trade-off between trying new actions and using knowledge
- **Q-Learning:** Off-policy algorithm learning optimal action values
- **Policy Gradient:** Direct optimization of policy using gradient
- **Actor-Critic:** Combines policy (actor) and value function (critic)
- **RLHF:** Reinforcement Learning from Human Feedback, training with human preferences

**Rapid-Fire Q&A:**
- **Q: What's better, Q-learning or policy gradients?** **A:** Q-learning for discrete actions, policy gradients for continuous. Both useful; depends on problem.
- **Q: How do you handle sparse rewards?** **A:** Reward shaping, curriculum learning, learning from demos, or intrinsic motivation.
- **Q: Why is exploration critical in RL?** **A:** Without exploration, agent gets stuck in local optima, misses better strategies.
- **Q: What causes instability in deep RL?** **A:** Target network changes during training. Fix: use target network (slow copy), experience replay.
- **Q: Why is RLHF important for LLMs?** **A:** Aligns outputs with human preferences, reduces harmful outputs, makes models more helpful.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*
