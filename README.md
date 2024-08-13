# Spatial filtering controller by neuro-diffuse algorythm 
This repo contains the code for a controller with the capability to align a spatial filter in X-Y-Z plane using neural 
networks and diffuse control.

> The Z axis is not developed yet.

# Setting the environment

## MySQL CUDA environment 

Create a new image using the dockerfile in  `/From_0/python_env_mysql` executing in terminal 
`docker build -t <image_name:tag> /From_0/python_env_mysql`. and is ready to use the image as interpreter in PyCharm.

> Remember edit the run configurations, add `--gpus all` in **Docker container setting** section.

## MySQL container

The container with the sql server is based in the image generated by the dockerfile in `/From_0/MySQL_server_image`
To build a docker image use from terminal  `docker build -t <image_name:tag> /From_0/MySQL_server_image`.

### Run and configuration

To run use `docker run --name <name_of_container> -e MYSQL_ROOT_PASSWORD=<password> -p 3307:3306 -d <image_name>`. 
This creates a detached container and forward port 3306 from the container into port 3307 of localhost.
Use `docker stop <container name>` to stop the container.
Use `docker exec -it <container name> bash` to enter the container's bash.

### Connect to the server

The connection is the same as the os installation, so can connect from pycharm as usually is done. Check that in this 
case the port to make the connection is  ***3307*** instead of ***3306.*** Means that port 3306 of the container is exposed to port 3307 of the host. Can add 127.0.0.1:3307:3306 
to be shure that the port 3307 is in localhost.