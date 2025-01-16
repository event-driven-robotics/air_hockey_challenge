import numpy as np
import math 
from air_hockey_challenge.air_hockey_challenge.environments.planar.single import AirHockeySingle

class AirHockeyDefend(AirHockeySingle):
    """
    Class for the air hockey defending task.
    The agent should stop the puck at the line x=-0.6.
    """

    def __init__(self, gamma=0.99, horizon=500, viewer_params={}):

        self.init_velocity_range = (1, 3)

        self.start_range = np.array([[0.29, 0.65], [-0.4, 0.4]])  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

    def setup(self, state=None):
        
        # possibili_posizioni = np.array([[0, -0.25], [0, 0], [0, 0.25]])
        possibili_posizioni = [np.array([-0.6, -0.3])]

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
   
        possibili_vel = [1.2]
     
        indice_vel= np.random.randint(0, len(possibili_vel))

        puck_vel[1]= possibili_vel[indice_vel]
        
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
        ee_pos, _ = self.get_ee()
        dist = self.computeEuclideanDist(ee_pos, puck_pos)
        mod_vel =  math.sqrt(puck_vel[0]*puck_vel[0] + puck_vel[1]*puck_vel[1])
        
        # if dist<=0.085 and puck_vel[0]>0.1:
        if puck_vel[0]>0 and puck_pos[0]>-0.6:
            return 1
        else:
            return 0

    def is_absorbing(self, state):
        puck_pos, puck_vel = self.get_puck(state)
        ee_pos, _ = self.get_ee()
        dist = self.computeEuclideanDist(ee_pos, puck_pos)
        
        # If puck is over the middle line and moving towards opponent
        if puck_vel[0]>0 and puck_pos[0]>-0.55:
            return True

        # if np.linalg.norm(puck_vel[:2]) < 0.1:
        #     return True
        

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
