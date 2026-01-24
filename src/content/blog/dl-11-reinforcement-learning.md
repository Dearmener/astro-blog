---
title: '深度学习完全指南（十一）：强化学习基础'
description: '从马尔可夫决策过程到DQN、PPO，全面掌握强化学习的核心概念与算法实现'
pubDate: '2024-02-11'
heroImage: ''
category: '技术'
tags: ['深度学习', '强化学习', 'DQN', 'PPO']
series: '深度学习完全指南'
seriesOrder: 11
---

## 强化学习概述

强化学习（Reinforcement Learning, RL）是机器学习的第三大范式，智能体通过与环境交互学习最优策略。

### 与监督学习的区别

| 特点 | 监督学习 | 强化学习 |
|------|---------|---------|
| 数据 | 标注数据集 | 交互产生 |
| 反馈 | 即时、确定 | 延迟、稀疏 |
| 目标 | 最小化误差 | 最大化累积奖励 |
| 挑战 | 泛化 | 探索-利用平衡 |

### 经典应用

- **游戏AI**: AlphaGo、OpenAI Five、AlphaStar
- **机器人控制**: 行走、抓取、导航
- **自动驾驶**: 决策规划
- **推荐系统**: 长期用户价值优化
- **资源调度**: 数据中心、交通信号

---

## 马尔可夫决策过程（MDP）

MDP是强化学习的数学框架。

### MDP五元组

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

- $\mathcal{S}$: 状态空间
- $\mathcal{A}$: 动作空间
- $P(s'|s, a)$: 状态转移概率
- $R(s, a, s')$: 奖励函数
- $\gamma \in [0, 1]$: 折扣因子

### 核心概念

```python
import numpy as np

class MDP:
    """马尔可夫决策过程"""
    def __init__(self, states, actions, transition_probs, rewards, gamma=0.99):
        self.states = states
        self.actions = actions
        self.P = transition_probs  # P[s][a] = [(prob, next_state, reward, done), ...]
        self.R = rewards
        self.gamma = gamma
    
    def step(self, state, action):
        """执行动作，返回下一状态和奖励"""
        transitions = self.P[state][action]
        probs = [t[0] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        prob, next_state, reward, done = transitions[idx]
        return next_state, reward, done

# 策略
# 确定性策略: π(s) -> a
# 随机策略: π(a|s) -> probability

# 价值函数
# 状态价值: V^π(s) = E[Σ γ^t * r_t | s_0 = s]
# 动作价值: Q^π(s, a) = E[Σ γ^t * r_t | s_0 = s, a_0 = a]
```

### Bellman方程

```python
# Bellman期望方程
# V^π(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V^π(s')]

# Bellman最优方程
# V*(s) = max_a Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V*(s')]
# Q*(s,a) = Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * max_{a'} Q*(s',a')]

def bellman_update(V, P, R, gamma, state):
    """Bellman最优更新"""
    values = []
    for action in P[state]:
        value = 0
        for prob, next_state, reward, done in P[state][action]:
            value += prob * (reward + gamma * V[next_state] * (1 - done))
        values.append(value)
    return max(values)
```

---

## 值迭代与策略迭代

### 值迭代（Value Iteration）

```python
def value_iteration(mdp, theta=1e-6, max_iterations=1000):
    """
    值迭代算法
    """
    V = {s: 0 for s in mdp.states}
    
    for i in range(max_iterations):
        delta = 0
        for s in mdp.states:
            v = V[s]
            V[s] = bellman_update(V, mdp.P, mdp.R, mdp.gamma, s)
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            print(f"收敛于第 {i+1} 次迭代")
            break
    
    # 提取最优策略
    policy = {}
    for s in mdp.states:
        best_action = None
        best_value = float('-inf')
        for a in mdp.actions:
            value = sum(p * (r + mdp.gamma * V[s_]) 
                       for p, s_, r, _ in mdp.P[s][a])
            if value > best_value:
                best_value = value
                best_action = a
        policy[s] = best_action
    
    return V, policy
```

### 策略迭代（Policy Iteration）

```python
def policy_iteration(mdp, theta=1e-6):
    """
    策略迭代算法
    """
    # 初始化随机策略
    policy = {s: np.random.choice(mdp.actions) for s in mdp.states}
    V = {s: 0 for s in mdp.states}
    
    while True:
        # 策略评估
        while True:
            delta = 0
            for s in mdp.states:
                v = V[s]
                a = policy[s]
                V[s] = sum(p * (r + mdp.gamma * V[s_]) 
                          for p, s_, r, _ in mdp.P[s][a])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        
        # 策略改进
        policy_stable = True
        for s in mdp.states:
            old_action = policy[s]
            
            # 选择最优动作
            best_action = max(mdp.actions, 
                            key=lambda a: sum(p * (r + mdp.gamma * V[s_]) 
                                            for p, s_, r, _ in mdp.P[s][a]))
            policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        if policy_stable:
            break
    
    return V, policy
```

---

## Q-Learning

Q-Learning是经典的无模型RL算法，不需要知道环境的转移概率。

### 算法实现

```python
import numpy as np
from collections import defaultdict

class QLearning:
    def __init__(self, actions, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(len(actions)))
    
    def get_action(self, state):
        """ε-greedy策略"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        return self.actions[np.argmax(self.Q[state])]
    
    def update(self, state, action, reward, next_state, done):
        """Q值更新"""
        action_idx = self.actions.index(action)
        
        # TD目标
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        
        # TD误差
        td_error = td_target - self.Q[state][action_idx]
        
        # 更新Q值
        self.Q[state][action_idx] += self.lr * td_error

# 训练循环
def train_qlearning(env, agent, num_episodes=1000):
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        
        # 衰减探索率
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
    
    return rewards_history
```

### SARSA

SARSA是on-policy版本的Q-Learning：

```python
class SARSA:
    def __init__(self, actions, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(len(actions)))
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        return self.actions[np.argmax(self.Q[state])]
    
    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA更新：使用实际采取的下一个动作"""
        action_idx = self.actions.index(action)
        next_action_idx = self.actions.index(next_action)
        
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.Q[next_state][next_action_idx]
        
        td_error = td_target - self.Q[state][action_idx]
        self.Q[state][action_idx] += self.lr * td_error
```

---

## 深度Q网络（DQN）

DQN用神经网络近似Q函数，解决高维状态空间问题。

### 核心技术

1. **经验回放（Experience Replay）**: 打破样本相关性
2. **目标网络（Target Network）**: 稳定训练
3. **奖励裁剪（Reward Clipping）**: 稳定梯度

### DQN实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 target_update=10, buffer_size=10000, batch_size=64):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.batch_size = batch_size
        self.update_count = 0
        
        # Q网络和目标网络
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        
        # 采样batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

# 训练
def train_dqn(env, agent, num_episodes=500):
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history
```

### Double DQN

解决DQN的过估计问题：

```python
class DoubleDQN(DQN):
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            # Double DQN: 用q_network选择动作，用target_network评估
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

### Dueling DQN

分离状态价值和优势函数：

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # 共享特征层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势函数流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
```

---

## 策略梯度方法

直接优化策略参数，不需要学习价值函数。

### REINFORCE

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.saved_log_probs = []
        self.rewards = []
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return action.item()
    
    def update(self):
        # 计算折扣回报
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算策略梯度损失
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 清空缓存
        self.saved_log_probs = []
        self.rewards = []
```

### Actor-Critic

结合策略梯度和价值函数：

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class A2C:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, _ = self.model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update(self, states, actions, rewards, next_states, dones, log_probs):
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        log_probs = torch.stack(log_probs)
        
        # 计算价值
        _, values = self.model(states)
        _, next_values = self.model(next_states)
        values = values.squeeze()
        next_values = next_values.squeeze()
        
        # 计算优势和目标
        targets = rewards + self.gamma * next_values * (1 - dones)
        advantages = targets - values
        
        # Actor损失
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic损失
        critic_loss = nn.MSELoss()(values, targets.detach())
        
        # 总损失
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## PPO（Proximal Policy Optimization）

PPO是目前最流行的策略梯度算法，结合了稳定性和样本效率。

### PPO-Clip实现

```python
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_ratio=0.2, epochs=10):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs, value = self.actor_critic(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()
    
    def compute_gae(self, rewards, values, next_values, dones):
        """计算广义优势估计（GAE）"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages)
    
    def update(self, states, actions, old_log_probs, rewards, next_states, dones, values):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        values = torch.FloatTensor(values)
        
        # 计算next_values
        with torch.no_grad():
            next_states_t = torch.FloatTensor(next_states)
            _, next_values = self.actor_critic(next_states_t)
            next_values = next_values.squeeze().numpy()
        
        # GAE
        advantages = self.compute_gae(rewards.numpy(), values.numpy(), next_values, dones.numpy())
        returns = advantages + values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        for _ in range(self.epochs):
            probs, new_values = self.actor_critic(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 重要性采样比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO-Clip目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic损失
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            
            # 总损失（包含熵正则化）
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()

# 训练循环
def train_ppo(env, agent, num_episodes=1000, rollout_length=2048):
    for episode in range(num_episodes):
        states, actions, log_probs, rewards, next_states, dones, values = [], [], [], [], [], [], []
        
        state = env.reset()
        
        for _ in range(rollout_length):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            values.append(value)
            
            state = next_state if not done else env.reset()
        
        # 批量更新
        agent.update(states, actions, log_probs, rewards, next_states, dones, values)
```

---

## 连续动作空间

### SAC（Soft Actor-Critic）

适用于连续动作空间的最大熵强化学习：

```python
class GaussianPolicy(nn.Module):
    """高斯策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = self.shared(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 重参数化采样
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        
        # Tanh压缩到[-1, 1]
        action = torch.tanh(x)
        
        # 计算log_prob（考虑Tanh变换）
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

class SAC:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # 策略网络
        self.policy = GaussianPolicy(state_dim, action_dim)
        
        # 双Q网络
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 目标Q网络
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        
        self.buffer = ReplayBuffer(100000)
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy.sample(state)
        return action.squeeze().numpy()
    
    def update(self, batch_size=256):
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # 更新Q网络
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(torch.cat([next_states, next_actions], dim=1))
            q2_next = self.q2_target(torch.cat([next_states, next_actions], dim=1))
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * q_next
        
        q1 = self.q1(torch.cat([states, actions], dim=1))
        q2 = self.q2(torch.cat([states, actions], dim=1))
        
        q_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # 更新策略网络
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(torch.cat([states, new_actions], dim=1))
        q2_new = self.q2(torch.cat([states, new_actions], dim=1))
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

---

## 模仿学习

从专家示范中学习策略。

### 行为克隆（Behavior Cloning）

```python
class BehaviorCloning:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def train(self, expert_states, expert_actions, epochs=100, batch_size=64):
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(expert_states),
            torch.FloatTensor(expert_actions)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for states, actions in dataloader:
                pred_actions = self.policy(states)
                loss = nn.MSELoss()(pred_actions, actions)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

### GAIL（Generative Adversarial Imitation Learning）

```python
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

class GAIL:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.discriminator = Discriminator(state_dim, action_dim)
        self.policy = GaussianPolicy(state_dim, action_dim)
        
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def update_discriminator(self, expert_states, expert_actions, policy_states, policy_actions):
        # 专家数据标签为1，策略数据标签为0
        expert_labels = torch.ones(expert_states.size(0), 1)
        policy_labels = torch.zeros(policy_states.size(0), 1)
        
        expert_preds = self.discriminator(expert_states, expert_actions)
        policy_preds = self.discriminator(policy_states, policy_actions)
        
        loss = nn.BCELoss()(expert_preds, expert_labels) + nn.BCELoss()(policy_preds, policy_labels)
        
        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()
        
        return loss.item()
    
    def get_reward(self, state, action):
        """使用判别器输出作为奖励"""
        with torch.no_grad():
            d = self.discriminator(state, action)
            reward = -torch.log(1 - d + 1e-8)  # 奖励塑形
        return reward
```

---

## 常用环境

### OpenAI Gym

```python
import gym

# 创建环境
env = gym.make('CartPole-v1')

# 基本接口
state = env.reset()
action = env.action_space.sample()  # 随机动作
next_state, reward, done, info = env.step(action)

# 离散动作空间
print(f"动作空间: {env.action_space}")  # Discrete(2)

# 连续动作空间
env2 = gym.make('Pendulum-v1')
print(f"动作空间: {env2.action_space}")  # Box(-2.0, 2.0, (1,))
```

### PyBullet

```python
import pybullet_envs

# 机器人控制环境
env = gym.make('HalfCheetahBulletEnv-v0')
```

### Gymnasium（Gym继任者）

```python
import gymnasium as gym

env = gym.make('LunarLander-v2', render_mode='human')
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

---

## 调试技巧

### 奖励曲线可视化

```python
import matplotlib.pyplot as plt

def plot_rewards(rewards, window=100):
    plt.figure(figsize=(10, 5))
    
    # 原始奖励
    plt.plot(rewards, alpha=0.3, label='Raw')
    
    # 移动平均
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), smoothed, label=f'MA-{window}')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Training Rewards')
    plt.show()
```

### 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 奖励不增长 | 学习率过大/小 | 调整lr (1e-4 ~ 1e-3) |
| 训练不稳定 | 目标网络更新太频繁 | 增大target_update |
| 过拟合特定状态 | 探索不足 | 增大epsilon/entropy |
| 动作总是相同 | 策略崩塌 | 添加熵正则化 |

---

## 小结

| 算法 | 类型 | 适用场景 |
|------|------|---------|
| Q-Learning | Value-based | 小状态空间、离散动作 |
| DQN | Value-based | 高维状态、离散动作 |
| REINFORCE | Policy Gradient | 简单任务 |
| A2C/A3C | Actor-Critic | 通用 |
| PPO | Policy Gradient | 通用、稳定 |
| SAC | Actor-Critic | 连续动作、样本效率高 |
| GAIL | Imitation | 无奖励函数 |

**下一篇**：模型训练与优化技巧，包括优化器选择、学习率调度、正则化等。
