# VM ininitialization command list

## Create sudo user
[Source](https://www.digitalocean.com/community/tutorials/how-to-create-a-sudo-user-on-ubuntu-quickstart)

```
    adduser tmc
    usermod -aG sudo tmc
    su - tmc
    sudo ls -la root
```

## Clone and change branch
- ```git clone ...```
- ```git fetch --all```
- ```git checkout ...```

## Run docker
Run docker image:    
- ```sudo docker run -it tensorflow/tensorflow:latest-gpu bash```

Run docker image with gpu:s    
- ```sudo docker run --gpus all -it tensorflow/tensorflow:latest-gpu bash```

Run docker image with gpu:s and mount current folder (```$PWD```) as /temp on docker image.    
```sudo docker run --gpus all -v $PWD:/temp -it tensorflow/tensorflow:latest-gpu bash```

## create ssh key
```ssh-keygen -t rsa -b 4096```

## Correct .ssh/ access rights
Directly copied from [this gist](https://gist.github.com/grenade/6318301)

    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_rsa
    ssh-add ~/.ssh/github_rsa
    ssh-add ~/.ssh/mozilla_rsa

    chmod 700 ~/.ssh
    chmod 644 ~/.ssh/authorized_keys
    chmod 644 ~/.ssh/known_hosts
    chmod 644 ~/.ssh/config
    chmod 600 ~/.ssh/id_rsa
    chmod 644 ~/.ssh/id_rsa.pub
    chmod 600 ~/.ssh/github_rsa
    chmod 644 ~/.ssh/github_rsa.pub
    chmod 600 ~/.ssh/mozilla_rsa
    chmod 644 ~/.ssh/mozilla_rsa.pub

## Graphic card related

### Check linux version
    cat /etc/os-release

### Check if card is connected to vm
    lspci

or

    lspci | grep 'NVIDIA'

### install gpu drivers
https://cloud.google.com/ai-platform/deep-learning-vm/docs/creating-images

Check if gpu drivers are installed

    nvidia-smi



