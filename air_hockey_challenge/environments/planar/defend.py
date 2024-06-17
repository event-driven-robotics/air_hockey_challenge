import numpy as np
import math
from air_hockey_challenge.environments.planar.single import AirHockeySingle
from air_hockey_challenge.utils import inverse_kinematics, world_to_robot
class AirHockeyDefend(AirHockeySingle):
    """
    Class for the air hockey defending task.
    The agent tries to stop the puck at the line x=-0.6.
    If the puck get into the goal, it will get a punishment.
    """
    
    def __init__(self, gamma=0.99, horizon=500, viewer_params={}):

        # self.init_velocity_range = (1, 3)
        self.init_velocity_range = (0.0, 0.0)
        self.start_range = np.array([[0.29, 0.65], [-0.4, 0.4]])  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame
        self.min_contact =  0.03165+0.04815  #in base.py
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

    def setup(self, state=None):
     
        # # possibili_posizioni = [ np.array([-0.25, 0]), np.array([-0.25, 0.1]), np.array([-0.2, 0.3])]
        possibili_posizioni = [np.array([-0.5, -0.3])]
        # x_puck =np.random.uniform(-0.5, -0.5, 1)
        # y_puck= np.random.uniform(-0.3, 0.3, 1)
        # possibili_posizioni = [np.array([x_puck[0], y_puck[0]])]
        # Seleziona casualmente un indice
        indice_posizione = np.random.randint(0, len(possibili_posizioni))

        # Ottieni la posizione corrispondente all'indice selezionato
        puck_pos = possibili_posizioni[indice_posizione]

        lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
        # angle = np.random.uniform(-0.5, 0.5)
        angle=math.pi/2
        puck_vel = np.zeros(3)
        puck_vel[0] = 0
        # # # # puck_vel[1] = np.sin(angle) * lin_vel
        # # #[1.5, 1.8, 1.6]
        # possibili_vel = [1.5, 1.8]
        # # Seleziona casualmente un indice
        # indice_vel= np.random.randint(0, len(possibili_vel))

        # # Ottieni la posizione corrispondente all'indice selezionato
        # puck_vel[1]= possibili_vel[indice_vel]
        puck_vel[1] = 1.3
        puck_vel[2] = np.random.uniform(0, 0, 1)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])
        
        super(AirHockeyDefend, self).setup(state)

    def setAction(self, action_idx):
        global index
        index = action_idx
        
    def getAction(self):
        return self.index

    global check_rew
    check_rew=False

    
    def sigmoid(self, x):
        sigmoid=((1 / (math.exp(-abs(10 * x)) + 1)) * 2) - 1
        return sigmoid
    
    def line(self, x):
        line=(x/1.1)-(0.1/1.1)
        return line
    
    def computeEuclideanDist(self, v1, v2):
        dist = math.sqrt((v1[0]-v2[0])*(v1[0]-v2[0]) + (v1[1]-v2[1])*(v1[1]-v2[1]))
        return dist
    
    def reward(self, state, action, next_state, absorbing):
        global check_rew
        
        check_rew=False
        
        reward_local = 0   
        puck_pos, puck_vel = self.get_puck(state)
        ee_pos, _ = self.get_ee()
        dist = self.computeEuclideanDist(ee_pos, puck_pos)
        mod_vel =  math.sqrt(puck_vel[0]*puck_vel[0] + puck_vel[1]*puck_vel[1])
        #  puck_vel[0] > 0:
        
        if dist<=0.085 and puck_vel[0]>0:
            
            check_rew = True
            # reward_local=self.line(mod_vel)
            reward_local=1
            

        # if (puck_vel[0] < 0):
        #    reward_local=-0.001  
        # # if puck_pos[0]>0 and puck_vel[0]>0 :
        # #     check_rew=True
        
  
        return reward_local
    
    def is_absorbing(self, state):
        global check_rew
        global reward_local
        global prev_rev
        if check_rew == True:
            check_rew = False

            return True
        
        else:
            return False
    

         
     

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