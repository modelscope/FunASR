#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.w_1 = model.w_1
		self.w_2 = model.w_2
		self.activation = model.activation
	
	def forward(self, x):
		x = self.activation(self.w_1(x))
		x = self.w_2(x)
		return x


class PositionwiseFeedForwardDecoderSANM(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.w_1 = model.w_1
		self.w_2 = model.w_2
		self.activation = model.activation
		self.norm = model.norm
	
	def forward(self, x):
		x = self.activation(self.w_1(x))
		x = self.w_2(self.norm(x))
		return x