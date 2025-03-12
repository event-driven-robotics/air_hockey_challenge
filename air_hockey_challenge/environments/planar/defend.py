import numpy as np
import math 
from air_hockey_challenge.environments.planar.single import AirHockeySingle


class AirHockeyDefend(AirHockeySingle):
    """
    Class for the air hockey defending task.
    The agent should stop the puck at the line x=-0.6.
    """

    def __init__(self, gamma=0.99, horizon=500, viewer_params={}):

        self.init_velocity_range = (1, 3)
        self.got_reward = False
        self.absorbing = False
        self.reward_value = 0

        self.start_range = np.array([[0.29, 0.65], [-0.4, 0.4]])  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame
        
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

    def setup(self, state=None):
        
        # possibili_posizioni = np.array([[0, -0.25], [0, 0], [0, 0.25]])
        possibili_posizioni = [np.array([-0.7, -0.3])]

        # Seleziona casualmente un indice
        indice_posizione = np.random.randint(0, len(possibili_posizioni))

        # Ottieni la posizione corrispondente all'indice selezionato
        puck_pos = possibili_posizioni[indice_posizione]

        lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
        # angle = np.random.uniform(-0.5, 0.5)
        angle=0
        puck_vel = np.zeros(3)
        
        # puck_vel[0] = np.random.uniform(-1, -1, 1)
        
        puck_vel[0] = np.random.uniform(0, 0, 1)
   
        possibili_vel = [1.0, 1.1, 0.9, 1.2, 0.8]
     
        indice_vel= np.random.randint(0, len(possibili_vel))
        puck_vel[1] 
        
        puck_vel[1]= possibili_vel[indice_vel]
        # puck_vel[1]= np.random.uniform(1.3, 1.8, 1)
        # puck_vel[1]= np.random.uniform(0.9, 1.2, 1)
        # print("puck_vel[1]: ", puck_vel[1])
        
        puck_vel[2] = np.random.uniform(0, 0, 1)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyDefend, self).setup(state)
        
        
    def computeEuclideanDist(self, v1, v2):
        dist = math.sqrt((v1[0]-v2[0])*(v1[0]-v2[0]) + (v1[1]-v2[1])*(v1[1]-v2[1]))
        return dist
  

    def reward(self, state, action, next_state, absorbing):
        puck_pos, puck_vel = self.get_puck(state)
        ee_pos, ee_vel = self.get_ee()
        global reward
        dist = self.computeEuclideanDist(ee_pos, puck_pos)
        
        if self.absorbing:
            self.reward_value = 1
            return self.reward_value
        
        

        if puck_vel[0] > 0.01 and puck_pos[0] > -0.6 :
            print("HIT      ", dist)
            self.got_reward = True
            self.reward_value = 250
            return self.reward_value
        
        
        # elif self.joint<0.0001 :
        #     # print ("waiting")
        #     self.reward_value = 5
        else:
            self.reward_value = -1
                
        return self.reward_value
    
    
        
    def is_absorbing(self, state):
        puck_pos, puck_vel = self.get_puck(state)
     
        ee_pos, _ = self.get_ee()
        dist = self.computeEuclideanDist(ee_pos, puck_pos)

        if self.got_reward and puck_vel[0] > 0.01 and puck_pos[0] > -0.60:
            self.absorbing = True
            self.reward_value = 100
            return True
        else:
            self.absorbing = False
            
       
        return super().is_absorbing(state)

if __name__ == '__main__':
    env = AirHockeyDefend()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    env.reset()
    while True:
        action = np.zeros(3)
        observation, reward, done, info = env.step(action)
        env.render()
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1

        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
