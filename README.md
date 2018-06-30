# Research Foundations

# Setup and install
Follow instructions at [pysc2 github](https://github.com/deepmind/pysc2)

# Creating a new agent
create a python file in agent dir and inherit from pysc2.agents.base_agent similar to [pysc2_agents.py](agent/pysc2_agents.py)

e.g. [a2c.py](agent/a2c.py) (this is just a copy of a scripted agent for now)

# Running an agent
### Running from bash
```bash
export SC2PATH=/[your path to]/StarCraftII/
# e.g. run the scripted agent CollectMineralShards from the pysc2_agents.py file in the agent dir 
python -m pysc2.bin.agent --map CollectMineralShards --agent agent.pysc2_agents.CollectMineralShards
```
### Running in pycharm - using deepmind agent runner
configure runner similar to below, using your own interpreter ofcourse
![run config](images/pycharm_run_config.png)


### Running in pycharm - using local agent runner
configure the Defaults in run configuration to be similar to above 
(for the Environment section only)  
change the agent and map by modifying this lines in custom_agent.py
```python
flags.DEFINE_string("agent", "agent.a2c.A2CMoveToBeacon",
...
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
```
run custom_agent.py from pycharm "right click -> run". 