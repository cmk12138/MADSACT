# -*- coding: utf-8 -*-
import os
import time
import pickle
from typing import NamedTuple, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from flax import linen as nn
from flax.training.train_state import TrainState
import matplotlib.pyplot as plt

# ==========================================
# 0. 环境导入处理
# ==========================================
try:
    from rl_env.path_env import RlGame
except ImportError:
    try:
        from path_env import RlGame
    except ImportError:
        print("Error: 找不到 'path_env.py'。请确保该文件在当前目录下或在 PYTHONPATH 中。")
        exit(1)

# ==========================================
# 1. 配置 (Config)
# ==========================================
class Config:
    # 模式开关
    TEST_MODE = True    # True: 进行测试并输出指标; False: 进行训练
    TEST_MODEL_PATH = "checkpoints/dsact_jax_final.pkl" # 模型路径

    # Environment
    N_AGENT = 1
    M_ENEMY = 1
    RENDER = True      # 训练时:关闭False 测试时:True
    TEST_RENDER = True  # 测试时开启 (若不想看动画可改为 False 加速)
    
    # Training
    SEED = 42
    EP_MAX = 50000       
    EP_LEN = 1000       
    BATCH_SIZE = 128
    BUFFER_SIZE = 100000
    GAMMA = 0.99
    
    # Optimization
    ACTOR_LR = 3e-4
    CRITIC_LR = 3e-4
    ALPHA_LR = 3e-4
    TAU = 0.01          
    
    # DSAC-T Hyperparameters
    TAU_B = 0.01        
    STD_BIAS = 0.1
    HUBER_DELTA = 50.0
    TD_BOUND_SCALE = 3.0
    
    # Updates
    START_STEPS = 2000
    UPDATE_EVERY = 1
    DELAY_UPDATE = 2
    
    # Saving
    SAVE_DIR = "./checkpoints"
    SAVE_EVERY = 10000 
    
    # Testing Metrics
    TEST_EPISODES = 100  # 测试总回合数，建议100以获得准确统计

# ==========================================
# 2. 工具函数 & Buffer
# ==========================================
def unwrap_step(step_out):
    """
    处理 path_env.py 返回值。
    path_env返回: hero_state, r, done, win, team_counter, (可能还有 dis...)
    """
    if isinstance(step_out, (list, tuple)):
        obs = step_out[0]
        r = step_out[1]
        done = step_out[2]
        # 尝试获取 win 和 team_counter，如果没有则给默认值
        win = step_out[3] if len(step_out) > 3 else False
        team_counter = step_out[4] if len(step_out) > 4 else 0
        return obs, r, done, win, team_counter
    return step_out

def save_model(agents, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    params_list = [jax.device_get(agent.actor.params) for agent in agents]
    with open(filepath, 'wb') as f:
        pickle.dump(params_list, f)
    print(f"模型已保存: {filepath}")

def load_model(agents, filepath):
    print(f"正在加载模型: {filepath} ...")
    with open(filepath, 'rb') as f:
        params_list = pickle.load(f)
    
    # params_list 的长度通常是训练时的智能体数量 (比如 2: [Leader, Follower])
    # agents 的长度是当前测试环境的智能体数量 (比如 5: [Leader, Follower1, Follower2, Follower3, Follower4])
    
    train_num = len(params_list)
    test_num = len(agents)
    
    print(f"  - 训练模型包含智能体数量: {train_num}")
    print(f"  - 当前测试环境智能体数量: {test_num}")

    new_agents = []
    for i, agent in enumerate(agents):
        if i < train_num:
            # 如果索引在训练数量范围内，直接加载对应参数
            # (Leader 加载 Leader, Follower1 加载 Follower1)
            target_params = params_list[i]
        else:
            # [关键] 如果是新增的跟随者，复用训练好的最后一个跟随者的参数
            # 假设 index 0 是 Leader，index 1 是 Follower，那么 index 2,3,4 都复用 index 1
            print(f"  - Agent {i} (Extra Follower) 复用 Agent {train_num - 1} 的参数")
            target_params = params_list[train_num - 1]
            
        new_actor = agent.actor.replace(params=target_params)
        new_agents.append(agent._replace(actor=new_actor))
        
    print("模型加载完成！")
    return new_agents

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, num_agents):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((capacity, num_agents * obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, num_agents * act_dim), dtype=np.float32)
        self.r = np.zeros((capacity, num_agents), dtype=np.float32)
        self.d = np.zeros((capacity, num_agents), dtype=np.float32)
        self.s_next = np.zeros((capacity, num_agents * obs_dim), dtype=np.float32)

    def add(self, s, a, r, d, s_next):
        idx = self.ptr
        self.s[idx] = s.reshape(-1)
        self.a[idx] = a.reshape(-1)
        self.r[idx] = r.reshape(-1)
        self.d[idx] = d.reshape(-1)
        self.s_next[idx] = s_next.reshape(-1)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            jnp.array(self.s[idx]),
            jnp.array(self.a[idx]),
            jnp.array(self.r[idx]),
            jnp.array(self.d[idx]),
            jnp.array(self.s_next[idx])
        )

# ==========================================
# 3. 网络定义 (Flax)
# ==========================================
class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.act_dim)(x)
        log_std = nn.Dense(self.act_dim)(x)
        log_std = jnp.clip(log_std, -20, 2)
        
        base_dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        bijector = distrax.Block(distrax.Tanh(), ndims=1)
        dist = distrax.Transformed(distribution=base_dist, bijector=bijector)
        return dist

class Critic(nn.Module):
    @nn.compact
    def __call__(self, s_global, a_all):
        x = jnp.concatenate([s_global, a_all], axis=-1)
        # Q1
        q1 = nn.Dense(512)(x)
        q1 = nn.relu(q1)
        q1 = nn.Dense(512)(q1)
        q1 = nn.relu(q1)
        q1_mean = nn.Dense(1)(q1)
        q1_log_std = nn.Dense(1)(q1)
        # Q2
        q2 = nn.Dense(512)(x)
        q2 = nn.relu(q2)
        q2 = nn.Dense(512)(q2)
        q2 = nn.relu(q2)
        q2_mean = nn.Dense(1)(q2)
        q2_log_std = nn.Dense(1)(q2)

        q1_log_std = jnp.clip(q1_log_std, -5.0, 2.0)
        q2_log_std = jnp.clip(q2_log_std, -5.0, 2.0)
        q1_std = nn.softplus(q1_log_std) + 1e-4
        q2_std = nn.softplus(q2_log_std) + 1e-4
        return q1_mean, q1_std, q2_mean, q2_std

# ==========================================
# 4. TrainStates 管理
# ==========================================
class AgentState(NamedTuple):
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    log_alpha: TrainState
    running_mean_std1: jnp.ndarray
    running_mean_std2: jnp.ndarray
    rng: jnp.ndarray

def create_agent_state(rng, obs_dim, act_dim, cfg: Config, num_agents):
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    
    actor_def = Actor(act_dim)
    s_local_mock = jnp.ones((1, obs_dim))
    actor_params = actor_def.init(actor_key, s_local_mock)
    actor_state = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optax.adam(cfg.ACTOR_LR))
    
    global_dim = obs_dim * num_agents + act_dim * num_agents
    critic_def = Critic()
    s_global_mock = jnp.ones((1, obs_dim * num_agents))
    a_global_mock = jnp.ones((1, act_dim * num_agents))
    critic_params = critic_def.init(critic_key, s_global_mock, a_global_mock)
    
    critic_state = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(cfg.CRITIC_LR))
    target_critic_state = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=optax.set_to_zero())
    
    log_alpha = jnp.array([0.0], dtype=jnp.float32)
    alpha_state = TrainState.create(apply_fn=None, params=log_alpha, tx=optax.adam(cfg.ALPHA_LR))
    
    return AgentState(
        actor=actor_state, critic=critic_state, target_critic=target_critic_state,
        log_alpha=alpha_state, running_mean_std1=jnp.array(1.0), running_mean_std2=jnp.array(1.0), rng=rng
    )

# ==========================================
# 5. 核心算法逻辑 (JIT Compiled)
# ==========================================
@jax.jit
def huber_loss(pred, target, delta):
    diff = pred - target
    abs_diff = jnp.abs(diff)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic ** 2 + delta * linear

@jax.jit
def update_step(agent_states, batch, cfg_vals, global_step):
    b_s, b_a, b_r, b_d, b_s_next = batch
    num_agents = len(agent_states)
    new_agent_states = []
    
    obs_dim = b_s.shape[1] // num_agents
    act_dim = b_a.shape[1] // num_agents
    def get_slice(data, idx, dim): return data[:, idx*dim : (idx+1)*dim]

    next_actions_list = []
    next_log_probs_list = []
    current_rngs = []
    
    # 1. Target Policy
    for i in range(num_agents):
        state = agent_states[i]
        rng_now, key = jax.random.split(state.rng)
        current_rngs.append(rng_now)
        s_next_i = get_slice(b_s_next, i, obs_dim)
        dist = state.actor.apply_fn(state.actor.params, s_next_i)
        a_next, log_pi_next = dist.sample_and_log_prob(seed=key)
        next_actions_list.append(a_next)
        next_log_probs_list.append(log_pi_next)

    b_a_next_all = jnp.concatenate(next_actions_list, axis=-1)

    # 2. Update
    for i in range(num_agents):
        st = agent_states[i]
        s_i = get_slice(b_s, i, obs_dim)
        r_i = b_r[:, i:i+1]
        d_i = b_d[:, i:i+1]
        alpha = jnp.exp(st.log_alpha.params[0])
        
        def critic_loss_fn(c_params, run_std1, run_std2):
            q1_m, q1_s, q2_m, q2_s = st.critic.apply_fn(c_params, b_s, b_a)
            q1n_m, q1n_s, q2n_m, q2n_s = st.target_critic.apply_fn(st.target_critic.params, b_s_next, b_a_next_all)
            q_next_mean = jnp.minimum(q1n_m, q2n_m)
            target_q = r_i + cfg_vals['GAMMA'] * (1 - d_i) * (q_next_mean - alpha * next_log_probs_list[i])
            
            td_bound1 = cfg_vals['TD_BOUND_SCALE'] * run_std1
            target_q_bound1 = q1_m + jnp.clip(target_q - q1_m, -td_bound1, td_bound1)
            td_bound2 = cfg_vals['TD_BOUND_SCALE'] * run_std2
            target_q_bound2 = q2_m + jnp.clip(target_q - q2_m, -td_bound2, td_bound2)

            q1_s_det = jax.lax.stop_gradient(q1_s)
            q2_s_det = jax.lax.stop_gradient(q2_s)
            ratio1 = jnp.clip((run_std1**2)/(q1_s_det**2 + cfg_vals['STD_BIAS']), 0.1, 10.0)
            ratio2 = jnp.clip((run_std2**2)/(q2_s_det**2 + cfg_vals['STD_BIAS']), 0.1, 10.0)
            
            l1 = huber_loss(q1_m, jax.lax.stop_gradient(target_q), cfg_vals['HUBER_DELTA'])
            l1_var = q1_s_det**2 - huber_loss(jax.lax.stop_gradient(q1_m), jax.lax.stop_gradient(target_q_bound1), cfg_vals['HUBER_DELTA'])
            loss_q1 = ratio1 * (l1 + q1_s * l1_var / (q1_s_det + cfg_vals['STD_BIAS']))
            l2 = huber_loss(q2_m, jax.lax.stop_gradient(target_q), cfg_vals['HUBER_DELTA'])
            l2_var = q2_s_det**2 - huber_loss(jax.lax.stop_gradient(q2_m), jax.lax.stop_gradient(target_q_bound2), cfg_vals['HUBER_DELTA'])
            loss_q2 = ratio2 * (l2 + q2_s * l2_var / (q2_s_det + cfg_vals['STD_BIAS']))
            return jnp.mean(loss_q1 + loss_q2), (q1_s, q2_s)

        grad_c_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        (c_loss, (curr_q1_s, curr_q2_s)), c_grads = grad_c_fn(st.critic.params, st.running_mean_std1, st.running_mean_std2)
        new_critic_state = st.critic.apply_gradients(grads=c_grads)
        
        new_run_std1 = (1 - cfg_vals['TAU_B']) * st.running_mean_std1 + cfg_vals['TAU_B'] * jnp.mean(curr_q1_s)
        new_run_std2 = (1 - cfg_vals['TAU_B']) * st.running_mean_std2 + cfg_vals['TAU_B'] * jnp.mean(curr_q2_s)

        def update_actor_alpha(actor_st, alpha_st, critic_p):
            target_entropy = -float(act_dim)
            def alpha_loss_fn(log_a_params):
                dist = actor_st.apply_fn(actor_st.params, s_i)
                _, log_pi_curr = dist.sample_and_log_prob(seed=current_rngs[i]) 
                return -jnp.mean(log_a_params * (log_pi_curr + target_entropy))

            alpha_l, alpha_grads = jax.value_and_grad(alpha_loss_fn)(alpha_st.params)
            new_alpha_st = alpha_st.apply_gradients(grads=alpha_grads)
            curr_alpha = jnp.exp(new_alpha_st.params[0])
            
            def actor_loss_fn(a_params):
                dist = actor_st.apply_fn(a_params, s_i)
                a_curr, log_pi_curr = dist.sample_and_log_prob(seed=current_rngs[i])
                b_a_replaced = b_a.at[:, i*act_dim:(i+1)*act_dim].set(a_curr)
                q1_pi, _, q2_pi, _ = st.critic.apply_fn(critic_p, b_s, b_a_replaced)
                return jnp.mean(curr_alpha * log_pi_curr - jnp.minimum(q1_pi, q2_pi))
            
            act_l, act_grads = jax.value_and_grad(actor_loss_fn)(actor_st.params)
            new_act_st = actor_st.apply_gradients(grads=act_grads)
            return new_act_st, new_alpha_st

        is_update = (global_step % cfg_vals['DELAY_UPDATE'] == 0)
        new_actor_state, new_alpha_state = jax.lax.cond(
            is_update, update_actor_alpha, lambda a,b,c: (a,b), st.actor, st.log_alpha, st.critic.params
        )

        new_target_params = optax.incremental_update(new_critic_state.params, st.target_critic.params, cfg_vals['TAU'])
        new_target_critic = st.target_critic.replace(params=new_target_params)

        new_agent_states.append(AgentState(
            actor=new_actor_state, critic=new_critic_state, target_critic=new_target_critic,
            log_alpha=new_alpha_state, running_mean_std1=new_run_std1, running_mean_std2=new_run_std2, rng=current_rngs[i]
        ))
        
    return new_agent_states

# ==========================================
# 6. 训练与测试逻辑
# ==========================================
def train(cfg: Config):
    print(f"初始化环境 (Train Mode): {cfg.N_AGENT} Agents ...")
    env = RlGame(n=cfg.N_AGENT, m=cfg.M_ENEMY, render=cfg.RENDER)
    if hasattr(env, 'unwrapped'): env = env.unwrapped
    
    obs = env.reset()
    obs = np.array(obs, dtype=np.float32)
    num_agents = cfg.N_AGENT + cfg.M_ENEMY
    obs_dim = obs.shape[-1]
    act_dim = env.action_space.shape[0]
    
    rng = jax.random.PRNGKey(cfg.SEED)
    agents = []
    for i in range(num_agents):
        rng, key = jax.random.split(rng)
        agents.append(create_agent_state(key, obs_dim, act_dim, cfg, num_agents))
    
    buffer = ReplayBuffer(cfg.BUFFER_SIZE, obs_dim, act_dim, num_agents)
    cfg_dict = {'GAMMA': cfg.GAMMA, 'TAU': cfg.TAU, 'TAU_B': cfg.TAU_B, 'STD_BIAS': cfg.STD_BIAS, 
                'HUBER_DELTA': cfg.HUBER_DELTA, 'TD_BOUND_SCALE': cfg.TD_BOUND_SCALE, 'DELAY_UPDATE': cfg.DELAY_UPDATE}
    
    global_step = 0
    total_rewards_history = []
    
    print("开始训练...")
    for ep in range(cfg.EP_MAX):
        obs = env.reset()
        obs = np.array(obs, dtype=np.float32)
        ep_ret = np.zeros(num_agents)
        
        for t in range(cfg.EP_LEN):
            global_step += 1
            actions = []
            for i in range(num_agents):
                obs_tensor = jnp.expand_dims(jnp.array(obs[i]), 0)
                dist = agents[i].actor.apply_fn(agents[i].actor.params, obs_tensor)
                rng_act, key_act = jax.random.split(agents[i].rng)
                agents[i] = agents[i]._replace(rng=rng_act)
                action = dist.sample(seed=key_act)
                action = np.array(action).squeeze(0)
                if global_step < cfg.START_STEPS: action = env.action_space.sample()
                actions.append(action)
            
            actions = np.array(actions)
            step_res = env.step(actions)
            # 训练时可以忽略 win/team_counter
            obs_next, r, done, _, _ = unwrap_step(step_res)
            
            obs_next = np.array(obs_next, dtype=np.float32)
            r = np.array(r, dtype=np.float32).flatten()
            if isinstance(done, (bool, int, np.bool_, np.integer)): d = np.array([float(done)]*num_agents)
            else: d = np.array(done, dtype=np.float32)

            buffer.add(obs, actions, r, d, obs_next)
            obs = obs_next
            ep_ret += r
            
            if buffer.size >= cfg.BATCH_SIZE and global_step > cfg.START_STEPS:
                batch = buffer.sample(cfg.BATCH_SIZE)
                agents = update_step(agents, batch, cfg_dict, global_step)

            if np.any(d): break
        
        total_rewards_history.append(np.sum(ep_ret))
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:04d} | Reward: {np.sum(ep_ret):.2f}")
        
        if (ep + 1) % cfg.SAVE_EVERY == 0:
            save_path = os.path.join(cfg.SAVE_DIR, f"dsact_jax_ep{ep+1}.pkl")
            save_model(agents, save_path)

    final_path = os.path.join(cfg.SAVE_DIR, "dsact_jax_final.pkl")
    save_model(agents, final_path)
    
    plt.figure()
    plt.plot(total_rewards_history)
    plt.title('Training Reward')
    plt.savefig('reward_curve.png')
    plt.close()
    env.close()

def test(cfg: Config):
    print(f"进入测试模式: Loading from {cfg.TEST_MODEL_PATH}")
    if not os.path.exists(cfg.TEST_MODEL_PATH):
        print(f"Error: 模型文件不存在 {cfg.TEST_MODEL_PATH}")
        return

    env = RlGame(n=cfg.N_AGENT, m=cfg.M_ENEMY, render=cfg.TEST_RENDER)
    if hasattr(env, 'unwrapped'): env = env.unwrapped
    
    obs = env.reset()
    obs = np.array(obs, dtype=np.float32)
    num_agents = cfg.N_AGENT + cfg.M_ENEMY
    obs_dim = obs.shape[-1]
    act_dim = env.action_space.shape[0]

    # 初始化空网络
    rng = jax.random.PRNGKey(cfg.SEED)
    agents = []
    for i in range(num_agents):
        rng, key = jax.random.split(rng)
        agents.append(create_agent_state(key, obs_dim, act_dim, cfg, num_agents))
    
    # 加载权重
    agents = load_model(agents, cfg.TEST_MODEL_PATH)
    
    # 性能指标统计
    total_episodes = cfg.TEST_EPISODES
    success_count = 0
    total_fkr = 0.0          # Formation Keeping Rate
    total_time = 0.0         # Shortest Time (sum of successful steps)
    total_dist = 0.0         # Shortest Path (sum of trajectory length)
    total_energy = 0.0       # Minimum Energy (sum of control input)
    
    success_episodes_count = 0 # 记录成功的总次数用于计算特定指标平均值

    print(f"\n开始测试 {total_episodes} 回合...")
    for i in range(total_episodes):
        obs = env.reset()
        obs = np.array(obs, dtype=np.float32)
        
        ep_steps = 0
        ep_dist = 0.0
        ep_energy = 0.0
        
        # 记录上一步位置用于计算距离，假设 obs 中 [0,1] 是归一化的 x,y
        # path_env 中 obs = [x/1000, y/1000, ...]
        last_pos = [ob[:2] * 1000.0 for ob in obs] 

        # 编队计数
        episode_team_counter = 0

        for t in range(cfg.EP_LEN):
            actions = []
            for j in range(num_agents):
                obs_tensor = jnp.expand_dims(jnp.array(obs[j]), 0)
                dist = agents[j].actor.apply_fn(agents[j].actor.params, obs_tensor)
                # Deterministic (mean) for testing
                # TanhTransformedDist 均值计算较难，采样方差小，直接采样即可
                rng, key = jax.random.split(rng)
                action = dist.sample(seed=key) 
                actions.append(np.array(action).squeeze(0))
            
            actions_np = np.array(actions)
            
            # 计算能耗: sum(|u| + |w|)
            step_energy = np.sum(np.abs(actions_np))
            ep_energy += step_energy

            step_res = env.step(actions_np)
            obs_next, r, done, win, team_ctr = unwrap_step(step_res)
            
            # 记录编队计数 (env返回的是累积值还是单步flag? path_env中是累积值 self.team_counter)
            # 这里我们使用环境返回的最新 counter 值
            episode_team_counter = team_ctr 

            # 计算路程
            obs_next = np.array(obs_next, dtype=np.float32)
            curr_pos = [ob[:2] * 1000.0 for ob in obs_next]
            
            # 累加所有智能体的移动距离
            step_dist = 0
            for k in range(num_agents):
                d = np.linalg.norm(curr_pos[k] - last_pos[k])
                step_dist += d
            ep_dist += step_dist
            
            obs = obs_next
            last_pos = curr_pos
            ep_steps += 1
            
            if cfg.TEST_RENDER:
                env.render()
                # time.sleep(0.01) # 可选延时
            
            if np.any(done):
                # 检查是否成功
                # win 在 path_env 中是单个 bool 还是 list? unwrap_step 返回的是单个
                # 如果是多智能体，只要有一个 done 且 win=True 就算成功?
                # 根据 path_env 逻辑，win 是 hero0.win
                if win:
                    success_count += 1
                    success_episodes_count += 1
                    total_time += ep_steps
                    total_dist += ep_dist
                    total_energy += ep_energy
                break
        
        # 编队保持率 = 保持编队步数 / 总步数
        fkr = episode_team_counter / max(ep_steps, 1)
        total_fkr += fkr

        print(f"Test Ep {i+1} | Steps: {ep_steps} | Dist: {ep_dist:.1f} | Energy: {ep_energy:.1f} | Win: {win} | FKR: {fkr:.2%}")

    env.close()

    # --- 计算统计指标 ---
    mcr = (success_count / total_episodes) * 100
    avg_fkr = (total_fkr / total_episodes) * 100
    
    # 仅针对成功回合计算平均时间和路程能耗
    if success_episodes_count > 0:
        avg_time = total_time / total_episodes
        avg_dist = total_dist / total_episodes
        avg_energy = total_energy / total_episodes
    else:
        avg_time = 0
        avg_dist = 0
        avg_energy = 0

    print("\n" + "="*40)
    print("           测试结果汇总            ")
    print("="*40)
    print(f"任务完成率 (MCR):            {mcr:.2f}%")
    print(f"平均编队保持率 (FKR):        {avg_fkr:.2f}%")
    print(f"平均飞行时间 (Steps):        {avg_time:.2f}")
    print(f"平均飞行路程 (Distance):     {avg_dist:.2f}")
    print(f"平均能量损耗 (Energy):       {avg_energy:.2f}")
    #print(f"成功回合数:            {success_episodes_count:.2f}")
    print("="*40)

if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    cfg = Config()
    
    if cfg.TEST_MODE:
        test(cfg)
    else:
        train(cfg)