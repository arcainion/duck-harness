# Agent Notes

## Docker / Kaggle development container

- Docker Desktop is installed per-user. In PowerShell sessions where `docker`
  is not on `PATH`, invoke the CLI directly at:
  `C:\Users\arcai\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe`.
- Do not assume Docker is installed inside WSL; use the Windows CLI above.
- The persistent local Kaggle development container is named
  `duck-kaggle-dev` and uses `kaggle/python:latest`.
- The repository is bind-mounted in that container at
  `/workspace/duck-harness`, which is also its working directory.
- The container was created with `--gpus all`. Confirm the currently exposed
  GPU with `nvidia-smi` before running GPU-dependent tests; local hardware is
  not identical to Kaggle's requested T4 environment.
- Use container names rather than transient container IDs. For example:

  ```powershell
  $dockerCli = 'C:\Users\arcai\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe'
  & $dockerCli ps -a
  & $dockerCli start duck-kaggle-dev
  & $dockerCli exec duck-kaggle-dev sh -lc 'cd /workspace/duck-harness && nvidia-smi'
  ```

- The older `nervous_mirzakhani` container is a stopped one-shot container
  with no persistent command, repository mount, or GPU request. Do not use it
  for Duck deployment testing.
