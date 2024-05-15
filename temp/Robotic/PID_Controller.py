import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from root_tip import img_handler
import cv2
from stable_baselines3 import PPO
from ot2_env_wrapper import OT2Env
import numpy as np


env = OT2Env(render=True)


observation = env.reset()

observation = observation[0]

verbose = 0


def img_process(image_name):
    im = cv2.imread(image_name)

    x_end, y_end = img_handler(im)

    img = 2764, 2764
    size = 0.149, 0.149

    dens_x= img[0] / size[0]
    dens_y = img[1] / size[1]

    x_end_real = []
    y_end_real = []
    for x in range(len(x_end)):
        
        if x_end[x] == 0 or y_end[x] == 0:
            pass
        else:
            x_end_real.append(round(x_end[x]/ dens_x + 0.062, 3))
            y_end_real.append(round(y_end[x] / dens_y + 0.10775, 3))

    if verbose == 1:
        print(x_end_real)
        print(y_end_real)
    
    return x_end_real, y_end_real


actions = np.zeros(3)
integral_error = np.zeros(3)
prev_error = np.zeros(3)

kp = 10
ki = 0
kd = 0

done = False

def PIP_calc(pip_loc, goal, prev_error, integral_error):


    error = goal - pip_loc
    derivative_error = error - prev_error
    integral_error += error



    action = kp * error + derivative_error * kd + integral_error * ki
    return action, error, integral_error


while True:
    done = False
    print("Enter name of image to read(e.g.): test_image_2.tif. Press Enter to use get_img")
    image_name = input()

    if image_name == '':
        image_name = env.get_img()
        print(image_name)

    if image_name == "reset":
        env.reset()
    else:
    
        x_end_real, y_end_real = img_process(image_name)

        for x in range(len(x_end_real)):
            done = False
            location = y_end_real[x], x_end_real[x], 0.175


            print(f"Move to plant: {(x+1)}")
            while not done:

                pip_location = observation[:3]


                for x in range(3):
                    actions[x], prev_error[x], integral_error[x] = PIP_calc(pip_location[x], location[x], prev_error[x], integral_error[x])
                action = [[actions[0], actions[1], actions[2], 0]]

                observation, reward, terminated, truncated, info = env.step(action, location)
                
                if terminated == True:
                    done = True
                    env.drop()

                    
                    if verbose == 1:
                        print(f"Finished in {info[1]} steps")
                        print(f"Euc distance: {info[0]}")
                        print(f"Goal position: {observation[3:]} End position: {observation[:3]}")
                if truncated == True:
                    done = True
                    if verbose == 1:
                        print("Sorry, couldn't reach that.")
                        print(f"Eucl distance: {info[0]}")
                        print(f"Goal position: {observation[3:]} End position: {observation[:3]}")
                    env.reset()
        env.rendering()


env.close()



