import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
# Import here to test new policy
from student_submissions.s2210xxx.policy2210xxx import ColumnGenerationPolicy,LP,mygreed,oneLP,myColumn
import numpy as np
# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
copyenv = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
# Make the window movable
if env.render_mode == "human":
    import pygame
    pygame.display.set_mode((600, 600), pygame.RESIZABLE)
if __name__ == "__main__":
    # Reset the environment
    seed = 811123
    np.random.seed(seed)
    observation, info = env.reset(seed=seed)
    np.random.seed(seed)
    copobservation, copinfo = copyenv.reset(seed=seed)
    # Write products input to input.txt
    with open('input.txt', 'w') as f:
        total_items = sum(product["quantity"] for product in observation["products"])
        f.write(f"SUM OF ITEM: {total_items}\n")
        for product in observation["products"]:
            f.write(f"Product {product['size']} - Remaining quantity: {product['quantity']}\n")
    with open('inputcopy.txt', 'w') as f:
        total_items = sum(product["quantity"] for product in copobservation["products"])
        f.write(f"SUM OF ITEM: {total_items}\n")
        for product in copobservation["products"]:
            f.write(f"Product {product['size']} - Remaining quantity: {product['quantity']}\n")
    # Change this line to test your policy
    policy2210xxx = myColumn()
    with open('stock.txt', 'w') as f:
        for idx,stock in enumerate(observation["stocks"]):
            f.write(f"{idx} - {policy2210xxx._get_stock_size_(stock)}\n")
    while True:
        action = policy2210xxx.get_action(observation, info) # Get the action from the policy
        observation, reward, terminated, truncated, info = env.step(action) # Take the action
        # print(f"Action: {action}") # Print the action taken by the policy
        # print(info) # Print the how much stock get used after taking the action
        # # Print the products and their remaining quantity after each action
        # for product in observation["products"]:
        #     print(f"Product {product['size']} - Remaining quantity: {product['quantity']}")
        if terminated:
            print(info)
            print("ratio:",policy2210xxx.stock_area/policy2210xxx.prod_area)
            break
    screen = pygame.display.get_surface()   # Get the screen surface
    pygame.image.save(screen, "result.png") # Save the last frame of the environment to result.png

    greedy = mygreed()
    while True:
        action = greedy.get_action(copobservation, copinfo) # Get the action from the policy
        copobservation, reward, terminated, truncated, copinfo = copyenv.step(action) # Take the action
        # print(f"Action: {action}") # Print the action taken by the policy
        # print(info) # Print the how much stock get used after taking the action
        # # Print the products and their remaining quantity after each action
        # for product in observation["products"]:
        #     print(f"Product {product['size']} - Remaining quantity: {product['quantity']}")
        if terminated:
            print(copinfo)
            print("ratio:",greedy.stock_area/greedy.prod_area)
            break
    screen = pygame.display.get_surface()   # Get the screen surface
    pygame.image.save(screen, "greedy.png") # Save the last frame of the environment to greedy.png
env.close()
