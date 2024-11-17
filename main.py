import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx, BendersDecompositionPolicy,ISHP_Policy,ApproximationPolicy,KenyonRemilaPolicy,ConstraintProgrammingPolicy

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
###########################################################
if env.render_mode == "human":
    import pygame
    pygame.display.set_mode((600, 600), pygame.RESIZABLE)
###########################################################
NUM_EPISODES = 100

if __name__ == "__main__":
    # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test GreedyPolicy
    # gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = gd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1

    # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1

    # Uncomment the following code to test your policy
    # Reset the environment
    observation, info = env.reset(seed=42)
    print(info)
    # policy2210xxx = Policy2210xxx()
    # print(env.__str__)
    # while True:
    #     action = policy2210xxx.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(info)
    #     if terminated:
    #         break
    #     if truncated:
    #         observation, info = env.reset()

    observation, info = env.reset(seed=42)
    #print("Observation:", observation)
    with open('input.txt', 'w') as f:
        f.write(str(info) + '\n')
        total_items = sum(product["quantity"] for product in observation["products"])
        f.write(f"SUM OF ITEM: {total_items}\n")
        for product in observation["products"]:
            f.write(f"Product {product['size']} - Remaining quantity: {product['quantity']}\n")
    policy2210xxx = Policy2210xxx()
    while True:
        action = policy2210xxx.get_action(observation, info)
        print(f"Action: {action}")
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)
        for product in observation["products"]:
            print(f"Product {product['size']} - Remaining quantity: {product['quantity']}")

        if truncated:
            observation, info = env.reset()
        if terminated:
            break

env.close()
