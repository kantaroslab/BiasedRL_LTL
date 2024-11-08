import yaml 


def load_config(filename="./config/ground_robot.yaml"):
    with open(filename, "r") as yamlfile:
        aio_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read {} as dictionary config.".format(filename))
    return aio_config 
