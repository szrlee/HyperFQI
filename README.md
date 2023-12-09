Official repo for the article ["Efficient and scalable reinforcement learning via Hypermodel"](https://richardli.xyz).

# Installation
```
cd HyperFQI
pip install -e .
```

# Usage
To reproduce the results of Atari:
```
cd experiments
sh experiments/start_atari.sh 0 Pong
```

To reproduce the results of DeepSea:
```
sh experiments/start_deepsea.sh 0 20
```
