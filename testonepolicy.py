import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import mygreed,myColumn
# Create the environment
def run_policy(NUM_EPISODES = 10,policy = "greedy"):
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        # render_mode="human",  # Comment this line to disable rendering
    )
    if env.render_mode == "human":
        import pygame
        pygame.display.set_mode((600, 600), pygame.RESIZABLE)
    if __name__ == "__main__":
        # Reset the environment
        observation, info = env.reset(seed=42)
        total_items = sum(product["quantity"] for product in observation["products"])
        if policy == "greedy":
            print("GREEDY POLICY")
            mypolicy = mygreed()
        else:
            print("COLUMN GENERATION POLICY")
            mypolicy = myColumn()
        ep = 0
        total_loss = 0
        while ep < NUM_EPISODES:
            action = mypolicy.get_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"SUM ITEM {ep}: {total_items}")
                print(info)
                if env.render_mode == "human":
                    screen = pygame.display.get_surface()
                    pygame.image.save(screen, f"res/result{ep}.png")
                mypolicy.create = True
                total_loss += info["trim_loss"]
                observation, info = env.reset(seed=ep)
                total_items = sum(product["quantity"] for product in observation["products"])
                ep += 1
        print(f"Average loss: {total_loss / NUM_EPISODES}")
    env.close()
if __name__ == "__main__":
    NUM_EPISODES = 100
    run_policy(NUM_EPISODES,"d")

