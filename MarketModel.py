# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:11:29 2021

@author: bbarber
"""

import numpy as np

class MarketModel: #Market Simulator Class

  #Initialize the market model by specifying the number of time periods, the trade friction parameter, the covariance matrix, the market impact factors, the drift, and the number of assets
  def __init__(self, T=5, eta=0, M=np.array([[1,0],[0,1]]), c=np.array([0,0]), mu=np.array([0,0]), d=2): 
    self.d = d
    self.T = T
    self.r_cur = 1
    self.s0 = 100*np.ones((self.d))
    self.w_cur = (1/self.d)*np.ones((self.d))
    self.eta = eta
    self.M = M
    self.c = c
    self.mu = mu
    self.reset()

  def reset(self): #reset all the internal arrays to the unsimulated value
    self.t = 0
    self.s_hist = np.zeros((self.d, self.T+1))
    self.w_hist = np.zeros((self.d, self.T+1))
    self.r_hist = np.zeros((self.T+1))
    self.P_hist = np.zeros((self.T+1))
    self.R_hist = np.zeros((self.T+1))
    self.s_cur = self.s0
    self.s_hist[:,0] = self.s0
    self.w_hist[:,0] = np.ones((self.d))/self.d
    self.R_hist[0] = 1

  def step(self, w): #Move the market model forward one time step using the portfolio w
    if self.t >= self.T:
      print("Trading period terminated (t = T), no more actions may be taken!")
    
    if np.shape(w)[0] != self.d : 
      print("Provided weights are not the correct size: Need more weights")
    
    elif np.shape(np.shape(w))[0] != 1:
      print("Provided weights are not the correct size: Incorrect Formatting")

    else: # Safety check, don't execute anything if the weights are mis-sized

      # NEEDS TO CHECK FOR THE SIMPLEX BULLSHIT
      # if not in simplex, make it simplex

      # ASSUMES w IS ONE DIMENSIONAL VECTOR (hence the above elif)
      if (w<0).any or np.abs(np.sum(w)-1)<0.00001: # negative weights or not normalized?
        w=forceSimplex(w)

      # Step Forward
      self.t = self.t + 1

      # Update weights
      self.w_cur = w;
      self.w_hist[:, self.t] = self.w_cur

      if self.t == 1:
        u_delta = 0
        self.P_hist[0] = 1

      else:
        self.P_hist[self.t] = ( 1 + self.r_hist[self.t-1])* self.P_hist[self.t - 1]     
        u_cur = self.w_cur * self.P_hist[self.t]/self.s_cur 
        u_last = self.w_hist[:,self.t-1]*self.P_hist[self.t-1]/self.s_hist[:,self.t-1]
        u_delta = u_cur-u_last

      xi_cur = np.random.normal(loc=0.0, scale=1.0, size=self.d)
      self.s_last = self.s_cur
      self.s_cur = self.genPriceStep(du=u_delta, xi=xi_cur)
      self.s_hist[:,self.t] = self.s_cur
      r_s_cur = (self.s_cur-self.s_last)/self.s_last;
      self.r_cur = np.dot(self.w_cur,r_s_cur) - self.eta*np.dot(u_delta,u_delta) # Updated 3-18-21: takes change in positions rather than changes in weights
      self.r_hist[self.t] = self.r_cur
      self.R_hist[self.t] = self.getTotalReturn()
      self.P_hist[self.t] = self.P_hist[self.t-1]*(1+self.r_cur)

  def getTotalReturn(self):
    return (np.prod(self.r_hist+1))

  def genPriceStep (self, du, xi):
    S = np.log(self.s_cur)
    dS = self.mu+self.marketImpact(du)+self.M @ xi
    return np.exp(S+dS)

  def marketImpact(self, du):
    return self.c*np.sign(du)*np.sqrt(np.abs(du))

  def getState(self):
    stateDictionary = {"t":self.t, "s_hist":self.s_hist[:,0:self.t+1], "w_hist":self.w_hist[:,0:self.t+1], "r_hist":self.r_hist[0:self.t+1], "cum_Return":self.R_hist[0:self.t+1]}
    return stateDictionary

  def makeTuple(self):
    stateTuple = {"s":tuple(self.s_hist[:,self.t-1]),"a":tuple(self.w_hist[:,self.t]),"r":self.r_hist[self.t],"sprime":tuple(self.s_hist[:,self.t])}
    return stateTuple
    
  def printHistory(self):
    print("R:",self.R_hist)
    print("r:",self.r_hist)
    print("P:",self.P_hist)
    print("w:",self.w_hist)
    print("S:",self.s_hist)

def forceSimplex(weights): # Force all weights to be positive, and normalize all weights
  weights[weights<0]=0
  weights = weights/np.sum(weights)
  return weights
  
class agent: #Agent that interacts with the environment

  def __init__(self, pol, dim = 3):
    self.dimensions = dim
    self.policy = pol

  def updatePolicy(self, pol):
    self.policy = pol

  def makeWeights(self, state, epsilon=0):
    if (np.random.uniform()<epsilon):
      return self.randomWeights()
    else:
      return self.policy(state)

  def randomWeights(self):
    weights = np.abs(np.random.normal(size=self.dimensions))
    return forceSimplex(weights)
    
def randomizedWeights(state): #generate random weights
  weights = np.abs(np.random.normal(size=np.shape(state)[0]))
  return forceSimplex(weights)

def flatPortfolio(state): #weight everything the same
  weights = np.ones(shape =np.shape(state)[0])
  return forceSimplex(weights)
