# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:30:14 2021

@author: Dmytro Pokidin
"""

import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent_n_bootstrap import Agent
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')
# parameters
BRAIN_NAME = 'ReacherBrain'
NUM_AGENTS = 20
STATE_SIZE = 33
ACTION_SIZE = 4

agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE, random_seed=10, n_agents = NUM_AGENTS)
def ddpg(n_episodes=1000, print_every=50):
    """DDQN Algorithm.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        print_every (int): frequency of printing and loging information throughout iteration """
    
    scores = []
    scores_deque = deque(maxlen=print_every)
    log = open("log.txt","w+")
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[BRAIN_NAME]
        agent.reset()
        state = env_info.vector_observations                               # get the current state
        score = np.zeros(NUM_AGENTS)[:, None] 
        
        while True:
            action = agent.act(state)                                      # select an action
            env_info = env.step(action)[BRAIN_NAME]                        # send the action to the environment
            next_state = env_info.vector_observations                      # get the next state
            reward = np.array(env_info.rewards)[:, None]                   # get the reward
            done = np.array(env_info.local_done )[:, None]                 # see if episode has finished
            agent.step(state, action, reward, next_state, done)            # take step with agent (including learning)
            score += reward                                                # update the score
            state = next_state                                             # roll over the state to next time step
            if np.any(done):                                     
                break
        scores_deque.append(score.mean())      
        scores.append(score.mean())             
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            string ='\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque))
            print(string)
            log.write(string)
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        
        if np.mean(scores_deque)>=30.0 and i_episode>=100:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    log.close()
    return scores
scores = ddpg(n_episodes = 500)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

