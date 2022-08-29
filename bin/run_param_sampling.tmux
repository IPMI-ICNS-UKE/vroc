new-session -s vroc
new-window -n run_1
send-keys 'conda activate vroc && python bin/sample_parameter_space.py /home/administrator/data/learn2reg/NLST /home/administrator/data/learn2reg/results/param_sampling.sqlite --n-worker 5 --i-worker 0 --device cuda:1 ' Enter

new-window -n run_2
send-keys 'conda activate vroc && python bin/sample_parameter_space.py /home/administrator/data/learn2reg/NLST /home/administrator/data/learn2reg/results/param_sampling.sqlite --n-worker 5 --i-worker 1 --device cuda:2 ' Enter

new-window -n run_3
send-keys 'conda activate vroc && python bin/sample_parameter_space.py /home/administrator/data/learn2reg/NLST /home/administrator/data/learn2reg/results/param_sampling.sqlite --n-worker 5 --i-worker 2 --device cuda:3 ' Enter

new-window -n run_4
send-keys 'conda activate vroc && python bin/sample_parameter_space.py /home/administrator/data/learn2reg/NLST /home/administrator/data/learn2reg/results/param_sampling.sqlite --n-worker 5 --i-worker 3 --device cuda:4 ' Enter

new-window -n run_5
send-keys 'conda activate vroc && python bin/sample_parameter_space.py /home/administrator/data/learn2reg/NLST /home/administrator/data/learn2reg/results/param_sampling.sqlite --n-worker 5 --i-worker 4 --device cuda:5 ' Enter