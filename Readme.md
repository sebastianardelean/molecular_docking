## Build the image

```sh
docker build -t jupyter .
```

## Run in interactive mode

```sh
docker run -p 8888:8888 -it -v <path_to_mapped_folder>:/app --entrypoint /bin/bash jupyter
```

<path_to_mapped_folder> is the path to the work folder from the repository.

After a successfull conection, you can start the jupyter lab using:

```sh
jupyter-lab --allow-root
```
