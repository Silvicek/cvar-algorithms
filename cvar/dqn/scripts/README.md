# CVaR DQN Scripts

Run your training and testing from here.

For baseline IceLake benchmarks run

    python3 train_simple.py --env IceLake --nb-atoms 100 --run-alpha -1 --num-steps 2000000 --buffer-size 200000

For IceLakeRGB run

    python3 train_ice.py 

Also try Atari with

    python3 train_atari.py --help

or faster Atari benchmark

    python3 train_pong.py


After learning, run

    python3 enjoy_[{simple, ice, pong, atari}].py

for visualizations.
