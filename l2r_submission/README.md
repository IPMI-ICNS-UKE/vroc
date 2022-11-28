# L2R submission: Team VROC
Frederic Madesta (f.madesta@uke.de, corresponding author)
Thilo Sentker (t.sentker@uke.de)
René Werner (r.werner@uke.de)

## Training
No training needed.

## Testing
Similar to `l2r_scb` Docker (argument 0 is JSON filepath, argument 1 is GPU index, sorted by PCI slot):
```shell
docker run --rm --gpus all -v [INSERT PATH]/output:/l2r/output/ -v [INSERT PATH]/NLST_testdata:/l2r/data:ro --entrypoint ./test.sh l2r_vroc data/NLST_dataset.json 1
```

A detailed `output.log` is created in `/l2r/output/output.log`.
