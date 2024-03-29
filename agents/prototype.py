""" Scripted agent -- based on scripted_agent.py """

#from pysc2 import agent
import numpy as np
from PIL import Image
from scipy.misc import imresize

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from time import sleep

from pysc2.env.environment import StepType

import threading

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from pysc2.lib import app
import gflags as flags


FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_agent_steps", 2500, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

#flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
#                    "Which agent to run")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(),
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.mark_flag_as_required("map")


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class MoveToBeacon(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""

  def step(self, obs):
    super(MoveToBeacon, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      if not neutral_y.any():
        return actions.FunctionCall(_NO_OP, [])
      target = [int(neutral_x.mean()), int(neutral_y.mean())]
      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
    else:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""

  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      if not neutral_y.any() or not player_y.any():
        return actions.FunctionCall(_NO_OP, [])
      player = [int(player_x.mean()), int(player_y.mean())]
      closest, min_dist = None, None
      for p in zip(neutral_x, neutral_y):
        dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist
      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
    else:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


class DefeatRoaches(base_agent.BaseAgent):
  """An agent specifically for solving the DefeatRoaches map."""

  def step(self, obs):
    super(DefeatRoaches, self).step(obs)
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      roach_y, roach_x = (player_relative == _PLAYER_HOSTILE).nonzero()
      if not roach_y.any():
        return actions.FunctionCall(_NO_OP, [])
      index = numpy.argmax(roach_y)
      target = [roach_x[index], roach_y[index]]
      return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
    elif _SELECT_ARMY in obs.observation["available_actions"]:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    else:
      return actions.FunctionCall(_NO_OP, [])

class RandomAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""

  def step(self, obs):
    #print(obs)
    #sdf
    sleep(.500)
    super(RandomAgent, self).step(obs)
    function_id = numpy.random.choice(obs.observation["available_actions"])
    args = [[numpy.random.randint(0, size) for size in arg.sizes]
            for arg in self.action_spec.functions[function_id].args]
    return actions.FunctionCall(function_id, args)

class Prototype(base_agent.BaseAgent):

    """
    Information is bottom-up, goals are given top-down, control is 
    group/individual
    """
    
    def __init__(self):
        super(Prototype, self).__init__()
        self.action_queue = []
        self.action_meta = []
        self.map_history = None
        self.camera_size = None
        self.minimap_scale = None
    
    def initialize(self, minimap, screen_size):
        visibility = minimap[1]
        # TODO: We need to know what the max dims are for the map
        camera = minimap[3]
        camera_size = int(np.sqrt(np.sum(camera))) # Note: sqrt will not be correct if not square
        
        self.camera_size = camera_size
        
        valid_coords = np.where(minimap[0])
        
        y_max = np.max(valid_coords[0])
        x_max = np.max(valid_coords[1])
        y_min = np.min(valid_coords[0])
        x_min = np.min(valid_coords[1])
        
        x_range = list(range(x_min + camera_size//2, x_max - camera_size//2, camera_size)) + [x_max - camera_size//2]
        y_range = list(range(y_min + camera_size//2, y_max - camera_size//2, camera_size)) + [y_max - camera_size//2]
        
        #print(camera_size)
        #print(minimap.shape)
        #print(np.array(minimap.shape[1:3])/camera_size)
        
        # This isn't working because camera size changes dependent on the
        # starting location of the player.
        #map_size = np.array(screen_size) * minimap.shape[1:3]/camera_size
        
        # This assumes the camera covers the entire minimap
        map_size = np.array(screen_size) #* minimap.shape[1:3]/camera_size
        
        #print(map_size)
        self.map_history = np.zeros((*map_size.astype('uint64'), 13))
        
        self.minimap_scale = np.array(map_size) / minimap.shape[1:3]
        
        # TODO: only send camera to places of current visibility
        # TODO: in future, only update when there is a change on the minimap
        # TODO: Constant update where there is significant action
        # Brute force update for now
        #mm_size = 64
        #x_range = [i * camera_size for i in range(mm_size//camera_size)]
        #y_range = [i * camera_size for i in range(mm_size//camera_size)]

        for y in y_range:
            for x in x_range:
                # For some reason x and y seem to be switched
                action = actions.FunctionCall(1, [(x, y)]) # "move_camera"
                self.action_queue.append(action)
                self.action_meta.append((x, y))

    # Strategists
    # TODO: Return analysis (including score) and proposed action sequence
    # TODO: Mutally exclusive actions, e.g., build + attack should be performed
    # TODO: Competing actions, e.g., build multiple (limited resources) must be prioritized
    def resource_manager(self):
        pass
    
    def vision_manager(self, minimap):
        pass
        
    
    def defence_manager(self):
        pass
    
    def offence_manager(self):
        pass
    
    def creativity_manager(self):
        pass
    
    # Planners
    def short_term_planner(self):
        pass
    
    def long_term_planner(self):
        pass
    
    def local_planner(self):
        pass
    
    def global_planner(self):
        pass
    
    # Interconnect
    def communication_protocol(self):
        pass
    
    # Decision makers
    def builder(self):
        pass
    
    def assigner(self):
        pass
    
    def mover(self):
        pass
    
    
    # MM
    # height
    # visibility
    # creep
    # camera
    # player_id
    # player_relative
    # selected
    
    # Screen
    # height
    # visibility
    # creep
    # power
    # player_id
    # player_relative
    # unit_type
    # selected
    # hit_points
    # energy
    # shields
    # density
    # density_aa
    
    

    def step(self, obs):
        super(Prototype, self).step(obs)
        
        minimap = obs.observation['minimap']
        screen = obs.observation['screen']
    
        if obs.step_type == StepType.FIRST:
            # Initialize our view of the world
            screen_size = (screen.shape[1], screen.shape[2])
            self.initialize(minimap, screen_size)
            #print(obs.observation['screen'].shape)
        else:
            #x, y = self.action_meta.pop() # Assumes only coords for now
            #print(x)
            #print(y)
            #x, y = x * self.minimap_scale, y * self.minimap_scale
            #print(x)
            #print(y)
            #print(self.minimap_scale)
            
            #map_coords = np.where(minimap[3]) * self.minimap_scale
            
            map_coords = imresize((minimap[3]>0).astype('uint8'), self.map_history.shape)
            
            #print(map_coords)
            #print(map_coords.shape)
            #print(np.rollaxis(screen, 0, 3).shape)
            print(screen.shape)
            self.map_history[map_coords>0] = np.rollaxis(screen, 0, 3).reshape((np.prod(screen.shape[1:]), 13))
            
            #Image.fromarray((minimap[1] > 0).astype('uint8')*255).show()
            #Image.fromarray((screen[4] > 0).astype('uint8')*255).show()
            #Image.fromarray((minimap[3] > 0).astype('uint8')*255).show()
            Image.fromarray((self.map_history[:, :, 4] > 0).astype('uint8')*255).show()
        
        
            sleep(200)
        
        #print(obs.observation['minimap'].shape)
        #print(obs.observation['screen'].shape)
        #print(obs.observation["available_actions"])
        #sfd
        
        #self.vision_manager(obs.observation['minimap'])
        
        #function_id = np.random.choice(obs.observation["available_actions"])
        #function_id = 1
        #print(self.action_spec.functions[function_id].args)
        
        #args = [[np.random.randint(0, size) for size in arg.sizes]
        #        for arg in self.action_spec.functions[function_id].args]
        #return actions.FunctionCall(function_id, args)
        #return actions.FunctionCall(_NO_OP, [])
        return self.action_queue.pop()

def run_thread(agent_cls, map_name, visualize):
  with sc2_env.SC2Env(
      map_name=map_name,
      agent_race=FLAGS.agent_race,
      bot_race=FLAGS.bot_race,
      difficulty=FLAGS.difficulty,
      step_mul=FLAGS.step_mul,
      game_steps_per_episode=FLAGS.game_steps_per_episode,
      screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
      minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
      visualize=visualize,
      camera_width_world_units=128) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    agent = agent_cls()
    run_loop.run_loop([agent], env, FLAGS.max_agent_steps)
    if FLAGS.save_replay:
      env.save_replay(agent_cls.__name__)


def main(unused_argv):
  """Run an agent."""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map)  # Assert the map exists.

  #agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = Prototype#getattr(importlib.import_module(agent_module), agent_name)

  threads = []
  for _ in range(FLAGS.parallel - 1):
    t = threading.Thread(target=run_thread, args=(agent_cls, FLAGS.map, False))
    threads.append(t)
    t.start()

  run_thread(agent_cls, FLAGS.map, FLAGS.render)

  for t in threads:
    t.join()

  if FLAGS.profile:
    print(stopwatch.sw)



if __name__ == "__main__":
  app.run(main)
