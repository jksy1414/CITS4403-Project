# CITS4403 Project – Social Media Fake vs Real News Diffusion Model

## Overview
This project is part of the **CITS4403 Computational Modelling** unit at UWA.  
It simulates how **fake** and **real** news spread across a social network, using different network structures:

- **Erdős–Rényi (ER)** random graphs  
- **Watts–Strogatz (WS)** small-world graphs  
- **Barabási–Albert (BA)** scale-free graphs  

Each **user** in the network is treated as an **agent** with states:
- **UNAWARE** – hasn’t seen the story  
- **SEEN** – has seen but not shared  
- **POSTED** – shared with friends  
- **QUARANTINED** – (fake only) flagged by moderation  
- **IMMUNE** – lost interest, won’t share further  

This allows us to explore how **network topology**, **platform policies** (e.g., flagging, correction), and **user behaviour** affect the competition between fake and real news.

---

## Features
- Two competing contagions (Fake vs Real)
- Moderation via fake post flagging
- Correction effect (real news reduces fake sharing)
- Attention limits and clickbait bias
- User heterogeneity (credulity drawn from a Beta distribution)
- Comparison across ER, WS, and BA networks

---

## Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/jksy1414/CITS4403-Project.git
cd CITS4403-Project
pip install networkx numpy matplotlib
