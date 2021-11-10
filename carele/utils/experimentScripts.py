from carele.utils.manager_scripts import pathManager
from carele.utils.utils import create_environment


def setupExperiment(algorithm, config, cuda):
    manager = pathManager(config)
    env, agent, train_func = startExperiment(config, manager)
    return env, agent, manager, train_func


def startExperiment(config, manager):
    env = create_environment(config['env']['name'])
    agent = createAgent(config)
    train_fun = createAlgorithm(config)
    return env, agent, train_fun


def createAgent(config):
    import carele.agents as agents
    target_class = config['agent']['class']
    if hasattr(agents, target_class):
        agentClass = getattr(agents, target_class)
    else:
        raise AssertionError('There is no Agent called {}'.format(target_class))
    return agentClass(config)


def createAlgorithm(config):
    import carele.algorithms as algorithms
    target_class = config['train']['algorithm']
    if hasattr(algorithms, target_class):
        trainfuncClass = getattr(algorithms, target_class)
    else:
        raise AssertionError('There is no Train Function called {}'.format(target_class))
    return trainfuncClass
