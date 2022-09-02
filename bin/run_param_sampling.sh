SESSION="VROC"

SESSIONEXISTS=$(tmux list-sessions | grep $SESSION)

# Only create tmux session if it doesn't already exist
if [ "$SESSIONEXISTS" = "" ]; then
  # start new session
  tmux new-session -d -s $SESSION
  tmux rename-window -t 0 'Main'

  # create all windows and run script
  tmux new-window -t $SESSION:1 -n 'VROC 1'
  tmux send-keys -t 'VROC 1' 'conda activate varreg_on_crack' C-m 'python sample_parameter_space.py /home/administrator/data/learn2reg/NLST /home/administrator/data/learn2reg/results/param_sampling.sqlite --n-worker 5 --i-worker 0 --device cuda:1' C-m

  tmux new-window -t $SESSION:2 -n 'VROC 2'
  tmux send-keys -t 'VROC 2' 'conda activate varreg_on_crack' C-m 'python sample_parameter_space.py /home/administrator/data/learn2reg/NLST /home/administrator/data/learn2reg/results/param_sampling.sqlite --n-worker 5 --i-worker 1 --device cuda:2' C-m

  tmux new-window -t $SESSION:3 -n 'VROC 3'
  tmux send-keys -t 'VROC 3' 'conda activate varreg_on_crack' C-m 'python sample_parameter_space.py /home/administrator/data/learn2reg/NLST /home/administrator/data/learn2reg/results/param_sampling.sqlite --n-worker 5 --i-worker 2 --device cuda:3' C-m

  tmux new-window -t $SESSION:3 -n 'VROC 4'
  tmux send-keys -t 'VROC 4' 'conda activate varreg_on_crack' C-m 'python sample_parameter_space.py /home/administrator/data/learn2reg/NLST /home/administrator/data/learn2reg/results/param_sampling.sqlite --n-worker 5 --i-worker 3 --device cuda:4' C-m

  tmux new-window -t $SESSION:3 -n 'VROC 5'
  tmux send-keys -t 'VROC 5' 'conda activate varreg_on_crack' C-m 'python sample_parameter_space.py /home/administrator/data/learn2reg/NLST /home/administrator/data/learn2reg/results/param_sampling.sqlite --n-worker 5 --i-worker 4 --device cuda:5' C-m
fi

tmux attach-session -t $SESSION:0
