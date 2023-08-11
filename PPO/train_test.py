import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import sys
import time

from class_ppo import *

#------------------------------------------------------------------------------------

def reward_shaping(s_):

     r = 0.0

     if s_[0] > -0.4:
          r += 5.0*(s_[0] + 0.4)
     if s_[0] > 0.1: 
          r += 100.0*s_[0]
     if s_[0] < -0.7:
          r += 5.0*(-0.7 - s_[0])
     if s_[0] < 0.3 and np.abs(s_[1]) > 0.02:
          r += 4000.0*(np.abs(s_[1]) - 0.02)

     return r


#----------------------------------------------------------------------------------------

env = gym.make('MountainCarContinuous-v0')


EP_MAX = 1000
GAMMA = 0.9

A_LR = 2e-4
C_LR = 2e-4

BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]

print("S_DIM: ", S_DIM, "| A_DIM: ", A_DIM)

CLIP_METHOD = dict(name='clip', epsilon=0.1)

# train_test = 0 for train; =1 for test
train_test = 0

# irestart = 0 for fresh restart; =1 for restart from ckpt file
# irestart控制算法的重启选项。当为0时，进行新的训练启动；当为1时，表示从之前的检查点文件恢复训练，此时迭代次数可能会在检查点文件中加载。
irestart = 0

# 迭代次数计数，初始化为0
iter_num = 0

# 启动重启选项，迭代次数设置为0。表示从头开始进行新的训练。
if (irestart == 0):
  iter_num = 0

#----------------------------------------------------------------------------------------

# 创建一个TensorFlow会话，称为sess
sess = tf.Session()  

ppo = PPO(sess, S_DIM, A_DIM, A_LR, C_LR, A_UPDATE_STEPS, C_UPDATE_STEPS, CLIP_METHOD)  # 创建PPO类的一个实例，称为ppo

saver = tf.train.Saver()  # 创建一个TensorFlow的存储器


if (train_test == 0 and irestart == 0):  # 如果从头开始训练
  sess.run(tf.global_variables_initializer())  # 初始化所有模型参数
else:
  saver.restore(sess, "ckpt/model")  # 从保存的agent继续训练或测试，从ckpt/model路径恢复参数

#----------------------------------------------------------------------------------------

# 在每个训练回合中，初始化状态、缓冲区和一些变量，准备开始与环境交互并收集数据。
for ep in range(iter_num, EP_MAX):

    print("-"*70)
   
    s = env.reset()  # 重置环境，并获取初始状态s。表示一个回合的开始。

    buffer_s, buffer_a, buffer_r = [], [], []  # 初始化存储当前回合数据的缓冲区。存储状态、动作和奖励。
    ep_r = 0  # 当前回合的累积奖励初始化为0

    max_pos = -1.0  # 初始化变量。用于跟踪回合中达到的最大位置。
    max_speed = 0.0  # 初始化变量。用于跟踪回合中达到的最大速度。
    done = False  # 初始化标志变量，表示回合是否结束。
    t = 0  # 初始化计时变量。用于跟踪当前回合中的时间步数。

    # 在外循环内部，有随时间步长变化的内部while循环。
    while not done:    
       
        env.render()  # 渲染环境，可视化智能体与环境的交互过程，方便观察算法的运行情况。

        # sticky actions
        #if (t == 0 or np.random.uniform() < 0.125): 
        if (t % 8 ==0):  # 每隔8个时间步长进行一次动作选择。因为短时间内汽车可能不会显著移动，此处使用粘性操作。同时可以控制选择动作的频率。
          a = ppo.choose_action(s) 

        # small noise for exploration
        a += 0.1 * np.random.randn()  # 在选择的动作上添加一个小的随机噪声，以鼓励探索不同的动作策略。

        # clip
        a = np.clip(a, -1.0, 1.0)  # 将动作a剪切到范围[-1.0, 1.0]，确保动作值在合理范围内。

        # take step  
        s_, r, done, _ = env.step(a)  # 执行动作a与环境交互，获取下一个状态s_、r、是否结束标志 done以及其他信息。
       
        if s_[0] > 0.4:  # 条件检查，用于检查s_的第一个元素。如果状态的第一个元素超过一定阈值。
            print("nearing flag: ", s_, a)   # 打印相关信息，可能用于在达到特定条件时输出调试信息。

        if s_[0] > 0.45:
          print("reached flag on mountain! ", s_, a) 
          if done == False:
             print("something wrong! ", s_, done, r, a)
             sys.exit()   

        # reward shaping 
        if train_test == 0:
          r += reward_shaping(s_)

        if s_[0] > max_pos:  #  检查状态s_ 的第一个元素是否大于之前跟踪的最大位置
           max_pos = s_[0]   # 如果是，更新最大位置
        if s_[1] > max_speed:  # 检查状态s_的第二个元素是否大于之前跟踪的最大速度
           max_speed = s_[1]  # 如果是，更新最大速度

        # 如果在训练模式中，状态、动作和奖励会被添加到缓冲区。用于后续的批量数据训练。
        if (train_test == 0):
          buffer_s.append(s)
          buffer_a.append(a)
          buffer_r.append(r)    

        s = s_  # 新状态被设置为当前状态
        ep_r += r  # 将当前时间步获得的奖励r累积到当前回合的累积奖励ep_r，以记录整个回合的累积奖励
        t += 1  # 将时间步数变量t增加1，表示经过了一个时间步。

        if (train_test == 0):
          if (t+1) % BATCH == 0 or done == True:
              v_s_ = ppo.get_v(s_)
              discounted_r = []
              for r in buffer_r[::-1]:
                  v_s_ = r + GAMMA * v_s_
                  discounted_r.append(v_s_)
              discounted_r.reverse()

              bs = np.array(np.vstack(buffer_s))
              ba = np.array(np.vstack(buffer_a))  
              br = np.array(discounted_r)[:, np.newaxis]

              buffer_s, buffer_a, buffer_r = [], [], []
             
              ppo.update(bs, ba, br)

        if (train_test == 1):
              time.sleep(0.1)

        if (done  == True):
             print("values at done: ", s_, a)
             break

    print("episode: ", ep, "| episode reward: ", round(ep_r,4), "| time steps: ", t)
    print("max_pos: ", max_pos, "| max_speed:", max_speed)

    if (train_test == 0):
      with open("performance.txt", "a") as myfile:
        myfile.write(str(ep) + " " + str(round(ep_r,4)) + " " + str(round(max_pos,4)) + " " + str(round(max_speed,4)) + "\n")

    if (train_test == 0 and ep%10 == 0):
      saver.save(sess, "ckpt/model")




