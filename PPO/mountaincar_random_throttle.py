import sys  # 导入系统模块，用于与系统交互和处理输入输出
import numpy as np  # 用于数值计算和数值操作
import gym  # 用于创建强化学习环境和进行实验

env = gym.make('MountainCarContinuous-v0')


for _ in range(100):  # 循环执行100次，表示进行100个回合（episodes）
  s = env.reset()  # 环境重置，并获取初始状态s。在每个回合开始时，都需要执行这个操作，确保环境初始状态是一致的。
  done = False  # 将标志变量done初始化为False,表示回合尚未结束。在每个回合开始时，都需要执行这个操作，以便在后续循环中判断回合是否结束。

  max_pos = -1.0  # 用于记录回合内达到的最大位置值。每个回合开始时，将其初始化为一个较小的值，然后在每个时间步中根据当前状态更新为更大的位置值。
  max_speed = 0.0  # 用于记录回合内达到的最大速度值。每个回合开始时，将其初始化为0.0，然后在每个时间步中根据当前状态更新为更大的速度值
  ep_reward = 0.0  # 用于累积回合内的奖励，以计算整个回合的累积奖励。每个回合开始时，将其初始化为0.0，然后在每个时间步中将当前时间步的奖励添加到累积奖励中。

  while not done:  # 在每个回合中，循环直到回合结束（done = True）
    env.render()  # 环境渲染可视化智能体与环境的交互过程。
    a = [-1.0 + 2.0*np.random.uniform()]  # 在[-1.0, 1.0]内生成一个均匀分布的随机数，然后将其作为动作作用于环境中的智能体。随机发力。
    s_, r, done, _ = env.step(a)  # 使用环境的step方法执行动作a，获取下一个状态s_、奖励 r、是否结束标志 done 和其他信息。

    if s_[0] > max_pos: max_pos = s_[0]  # 更新最大位置
    if s_[1] > max_speed: max_speed = s_[1]   # 更新最大速度
    ep_reward += r  # 累积回合奖励

  print("ep_reward: ", ep_reward, "| max_pos: ", max_pos, "| max_speed: ", max_speed)  # 每个回合结束后，打印
  

