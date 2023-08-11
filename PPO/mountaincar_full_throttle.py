import sys
import numpy as np
import gym

env = gym.make('MountainCarContinuous-v0')


for _ in range(100):  # 循环100次，表示进行100回合（episodes）
  s = env.reset()  # 重置环境，获取初始状态s。每个回合开始时，都需要执行这个操作，确保环境的初始状态是一致的。
  done = False  # 将标志变量初始化为False,表示回合尚未结束。每个回合开始时，都需要这个操作，以便在后续循环中判断回合是否结束。

  max_pos = -1.0  # 每个回合开始时，将回合内的最大位置值初始化为一个较小的值，然后在每个时间步中根据当前状态更新为更大的位置值
  max_speed = 0.0  # 每个回合开始时，将回合内的最大速度值初始化为0.0，然后在每个时间步中根据当前状态更新为更大的速度值
  ep_reward = 0.0  # 每个回合开始时，将回合累积奖励初始化为0.0，然后在每个时间步中将当前时间步的奖励添加到累积奖励中。

  while not done:  # 回合结束，done=True
    env.render()  # 渲染，可视化智能体与环境的交互过程
    a = [1.0] # step on throttle  # 执行一个固定的动作，表示马力全开
    s_, r, done, _ = env.step(a)  # 使用环境的step方法执行动作a,获取下一个状态s_、奖励 r、是否结束标志 done 和其他信息。

    if s_[0] > max_pos: max_pos = s_[0]  # 更新最大位置
    if s_[1] > max_speed: max_speed = s_[1]   # 更新最大速度
    ep_reward += r  # 累积回合奖励

  print("ep_reward: ", ep_reward, "| max_pos: ", max_pos, "| max_speed: ", max_speed)  # 每个回合结束，打印累积奖励、最大位置和最大速度
  

