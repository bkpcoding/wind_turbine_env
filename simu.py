#!/usr/bin/env pythnon3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Paul Aubin
# Created Date: 2022/09/28
# ---------------------------------------------------------------------------
""" Setup a simulation of a wind turbine agent """
# ---------------------------------------------------------------------------
import numpy as np
from math_utils import wrap_to_m180_p180
from wind_turbine import Wind_turbine, Wind


class Basic_agent:
	__threshold = 5 					# deg, corresponds to the wind deadzone in which no action is taken

	def __init__(self):
		pass

	def policy(self, rel_wind_heading) -> int:
		'''
		Define the policy of the agent
		Input  : relative wind heading between the wind and the wind turbine.
		Ouptut : an int corresponding to the selected action : 0 rotate clockwise, 1 do nothgin, 2 rotate trigo
		'''
		rel_wind_heading = wrap_to_m180_p180(rel_wind_heading)
		# If the relative angle to the wind is low do nothing
		if np.abs(rel_wind_heading) - self.__threshold < 0:
			return 1
		# Else follow the wind
		else:
			return np.sign(rel_wind_heading) + 1

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)


class Random_agent:
	def __init__(self):
		pass

	def policy(self, rel_wind_heading) -> int:
		'''
		Define the policy of the agent, as a random agent it selects a random action given a uniform probability distribution
		Ouptut : an int corresponding to the selected action : 0 rotate clockwise, 1 do nothgin, 2 rotate trigo
		'''
		return np.random.choice([0, 1, 2])

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)


# An agent to be completed
class Custom_agent:
	def __init__(self):
		pass

	def policy(self, rel_wind_heading) -> int:
		pass

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)


class Simu:
	#TODO : create a non-episodic environment, as autonomous agent in the real world will not be reseted
	#TODO : add a reward function, a reward function as mentioned in this paper is a good start : 
	#"Artificial Neural Network Based Reinforcement Learning for Wind Turbine Yaw Control"
	# The reward function is the difference between the power output and power output without yaw control,
	# normalized by the power output with accurate yaw control with no sensor error.
	
	power_output_log = [] 			# MW
	action_log = []
	rel_wind_heading_log = []		# deg
	true_rel_wind_heading_log = [] 	# deg
	wd_heading_log = [] 			# deg
	step_count = 0
	episode_length = 7*24           # hours

	def __init__(self, agent=None, wind_model=None, wind_turbine_model=None, max_steps=None):
		self.wd = Wind(10, 0, 1, 'OU') if wind_model is None else wind_model
		self.wt = Wind_turbine(0, False) if wind_turbine_model is None else wind_turbine_model
		self.agent = Basic_agent() if agent is None else agent
		self.max_steps = 24*3600 if max_steps is None else max_steps 
		self.power_output_log = self.max_steps * [0]
		self.power_output_max_log = self.max_steps * [0]
		self.power_output_no_control_log = self.max_steps * [0]
		self.action_log = self.max_steps * [1]
		self.reward_log = self.max_steps * [0]
		self.rel_wind_heading_log = self.max_steps * [0]
		self.true_rel_wind_heading_log = self.max_steps * [0]
		self.wd_heading_log = self.max_steps * [0]
	
	def reset(self):
		# clear all the logs, reset the step count, however, we keep the wind turbine in the same state
		# as it leads to more diverse initial conditions.
		self.power_output_log = self.max_steps * [0]
		self.power_output_max_log = self.max_steps * [0]
		self.power_output_no_control_log = self.max_steps * [0]
		self.action_log = self.max_steps * [0]
		self.reward_log = self.max_steps * [0]
		self.rel_wind_heading_log = self.max_steps * [0]
		self.true_rel_wind_heading_log = self.max_steps * [0]
		self.wd_heading_log = self.max_steps * [0]
		self.step_count = 0
		self.next_state = wrap_to_m180_p180(self.wd.heading) - wrap_to_m180_p180(self.wt.heading)
		self.next_state = np.array([self.next_state])
		return self.next_state

	def step(self, action):
		# Log the estimated wind
		self.rel_wind_heading_log[self.step_count] = wrap_to_m180_p180(self.wd.heading - self.wt.heading)

		# Log the true wind
		self.true_rel_wind_heading_log[self.step_count] = wrap_to_m180_p180(self.wd.heading - self.wt.true_heading)
		self.wd_heading_log[self.step_count] = self.wd.heading

		# Get action
		#self.action_log[self.step_count] = self.agent.policy(self.rel_wind_heading_log[self.step_count])
		self.action_log[self.step_count] = action

		# Apply action and get power output
		self.power_output_log[self.step_count], self.power_output_max_log[self.step_count], \
		self.power_output_no_control_log[self.step_count]	= \
		self.wt.step(self.wd.speed, self.wd.heading, self.action_log[self.step_count])

		# check if self.power_output_max_log[self.step_count] is not 0
		if self.power_output_max_log[self.step_count] == 0:
			self.reward_log[self.step_count] = 0
		else:
		# calculate the reward and log it
			self.reward_log[self.step_count] = (self.power_output_log[self.step_count] - \
											self.power_output_no_control_log[self.step_count]) / \
											self.power_output_max_log[self.step_count]
				

		# Generate new wind
		self.wd.step()
		self.next_state = wrap_to_m180_p180(self.wd.heading) - wrap_to_m180_p180(self.wt.heading)
		self.next_state = np.array([self.next_state])
		#print("Current state vs next state and action taken: ", self.rel_wind_heading_log[self.step_count], self.next_state, self.action_log[self.step_count])
		self.step_count += 1
		#if self.step_count % 1 == 0:
		#	print("Average reward over the last hour: ", np.mean(self.reward_log[self.step_count-3600:self.step_count]))
		#	print("Reward in the last hour:", self.reward_log[self.step_count-1])
		#	print("Step count: ", self.step_count)
		#	print("Time: ", self.wd.get_time()/ 3600, "h")
		# return next state, reward, done
		return self.next_state, self.reward_log[self.step_count-1], self.step_count - 1>= self.episode_length, {}

	def run_simu(self):
		while self.step_count < self.max_steps:
			self.step()
			self.step_count += 1

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

