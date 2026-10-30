# Snapshots

tvclient saves a checkpoint with `fork()`, starts from it with `start()`, and deletes it
with `stop()`. The original container manager can't save a checkpoint: `/fork` only returns
a running copy. So our fork of the manager adds three endpoints:

- `/snapshot` saves a terminal as Docker images. Nothing extra runs.
- `/restore` starts a new terminal from a snapshot. It can be called any number of times.
- `/delete_snapshot` removes the images.

Two Docker behaviors had to be worked around. Stopping a terminal deletes the images it
started from, so each restore uses its own image name. And `docker commit` copies a Compose
label that made stopping the original terminal delete its snapshots, so snapshot images get
their own label.

Without these endpoints, use `snapshot_mode="parked"`. Then a snapshot is a running copy that
nothing ever runs commands in. It works, but each snapshot keeps a container running, and
saving or restoring takes about 1.3 s instead of about 0.4 s.

A snapshot keeps files, the current directory, and exported variables. It doesn't keep running
programs, unexported variables, or Docker volumes. Results only hold for tasks that don't depend
on those. Multi-container tasks haven't been tested yet.
