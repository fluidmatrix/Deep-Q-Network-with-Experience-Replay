This project uses Reinforcement learning to land a Lunar Rover, where there are three
actions, namely, Do nothing, Main Thruster, left Thruster and Right Thruster. 

The reward system: After every step, a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

For each step, the reward:

is increased/decreased the closer/further the lander is to the landing pad.
is increased/decreased the slower/faster the lander is moving.
is decreased the more the lander is tilted (angle not horizontal).
is increased by 10 points for each leg that is in contact with the ground.
is decreased by 0.03 points each frame a side engine is firing.
is decreased by 0.3 points each frame the main engine is firing.
The episode receives an additional reward of -100 or +100 points for crashing or landing safely respectively.

An episode terminates when the lunar lander comes in contact with surface or when it leaves the boundaries of the 
left or right border.

The course is the working implementation as taught by the Instructor Andrew NG, from collboration of DeepLearning.AI and
Stanford Online.