import pygame
import random
import math
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Define colors
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
width, height = 800, 600

flock_predator_catch_rates = []
flock_survival_times = []
flock_avg_distances = []
solo_predator_catch_rates = []
solo_survival_times = []
solo_avg_distances = []


class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))  # Random initial velocity
        self.speed = 1  # Adjust the speed as needed
        self.neighbor_distance = 100  # Adjust the neighbor distance as needed

    def update(self):
        # Update agent's position based on its velocity
        self.x += self.velocity.x * self.speed
        self.y += self.velocity.y * self.speed

        # Wrap around the screen edges
        self.x %= width
        self.y %= height

    def draw(self, screen):
        # Draw the agent as a circle
        pygame.draw.circle(screen, blue, (int(self.x), int(self.y)), 8)


    def flock(self, agents, predator, with_flock, num_closest_agents_to_avoid=30):
        alignment = pygame.Vector2(0, 0)
        cohesion = pygame.Vector2(0, 0)
        separation = pygame.Vector2(0, 0)
        predator_avoidance = pygame.Vector2(0, 0)  # New vector for predator avoidance
        neighbor_count = 0

        # Calculate distances to all agents
        agent_distances = [(other, math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)) for other in agents
                           if other != self]

        agent_distances.sort(key=lambda item: item[1])  # Sort agents by distance

        for i, (other, distance) in enumerate(agent_distances):
            if distance < self.neighbor_distance:
                alignment += other.velocity
                cohesion += pygame.Vector2(other.x, other.y)
                separation += (pygame.Vector2(self.x, self.y) - pygame.Vector2(other.x, other.y)) / distance
                neighbor_count += 1

            if i < num_closest_agents_to_avoid and distance < 40:
                predator_avoidance += pygame.Vector2(self.x - predator.x, self.y - predator.y).normalize()

        if neighbor_count > 0:
            alignment /= neighbor_count
            alignment = alignment.normalize() * self.speed

            cohesion /= neighbor_count
            cohesion = (cohesion - pygame.Vector2(self.x, self.y)).normalize() * self.speed

            separation /= neighbor_count
            separation = (separation.normalize() * self.speed) / neighbor_count  # Normalize and scale by neighbor count

        # Limit the maximum velocity
        max_velocity = 5.0  # Adjust the maximum velocity as needed
        if self.velocity.length() > max_velocity:
            self.velocity = self.velocity.normalize() * max_velocity

        if (with_flock):
            self.velocity += alignment * 0.2 + cohesion * 0.1 + separation * 0.15 + predator_avoidance * 0.1
        else:
            self.velocity += (predator_avoidance * 0.1)


class Predator:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed
        self.last_removal_time = 0  # Initialize the last removal time
        self.catch_count = 0  # Initialize catch count

    def update(self, agents):
        # Find the nearest agent
        nearest_agent = min(agents, key=lambda agent: math.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2))

        # Calculate direction towards the nearest agent
        direction = pygame.Vector2(nearest_agent.x - self.x, nearest_agent.y - self.y).normalize()

        # Update predator's position
        self.x += direction.x * self.speed
        self.y += direction.y * self.speed

        # Check for collision with agents and apply 0.5-second delay
        current_time = time.time()
        if current_time - self.last_removal_time >= 0.5:  # Check if 0.5 seconds have passed
            for agent in agents:
                distance = math.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
                if distance < 10:  # Adjust collision distance as needed
                    agents.remove(agent)
                    self.catch_count += 1  # Increment catch count
                    self.last_removal_time = current_time  # Update the last removal time

    def draw(self, screen):
        pygame.draw.circle(screen, red, (int(self.x), int(self.y)), 12)


def sim(flock_status=True):
    # Initialize pygame
    pygame.init()

    # Set up display
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Flocking and Predator Simulation")

    # Create a predator
    predator = Predator(random.randint(0, width), random.randint(0, height), 4)  # Adjust predator speed as needed

    # Create agents
    num_agents = 30
    agents = [Agent(random.randint(0, width), random.randint(0, height)) for _ in range(num_agents)]

    # Main simulation loop
    running = True
    clock = pygame.time.Clock()

    catch_count = 0
    start_time = time.time()

    # Initialize arrays to store statistics
    predator_catch_rates = []
    survival_times = []
    avg_distances = []

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update agents
        for agent in agents:
            agent.flock(agents, predator, flock_status)
            agent.update()

        # Update predator
        predator.update(agents)

        # Calculate Predator Catch Rate
        catch_count = predator.catch_count

        if len(agents) == 3:
            break

        # Calculate Survival Time
        survival_time = time.time() - start_time

        # Calculate Distance from Predator
        distances = [math.sqrt((agent.x - predator.x) ** 2 + (agent.y - predator.y) ** 2) for agent in agents]
        if len(agents) <= 3 or len(distances) == 0:
            break
        avg_distance = sum(distances) / len(distances)

        # Append statistics to arrays
        predator_catch_rates.append(catch_count)
        survival_times.append("{:.4f}".format(float(survival_time)))
        avg_distances.append("{:.4f}".format(float(avg_distance)))

        # Clear screen
        screen.fill((255, 255, 255))

        # Draw agents
        for agent in agents:
            agent.draw(screen)

        # Draw predator
        predator.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    if (flock_status):
        flock_predator_catch_rates.append(predator_catch_rates)
        flock_survival_times.append(survival_times)
        flock_avg_distances.append(avg_distances)
    else:
        solo_predator_catch_rates.append(predator_catch_rates)
        solo_survival_times.append(survival_times)
        solo_avg_distances.append(avg_distances)

    pygame.quit()



def distance_analysis(flock_array, solo_array):
    float_arr1 = str_to_float(flock_array)
    float_arr2 = str_to_float(solo_array)
    flock_data = np.array(float_arr1)
    solo_data = np.array(float_arr2)

    print('flock mean dist: ', flock_data.mean())
    print('flock max dist: ', flock_data.max())
    print('flock min dist: ', flock_data.min())

    print('solo mean dist: ', solo_data.mean())
    print('solo max dist: ', solo_data.max())
    print('solo min dist: ', solo_data.min())

    plt.hist([flock_data, solo_data], bins='auto', color=['blue', 'red'], label=['Flock', 'Solo'])
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Histogram')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.bar(x=['Flock', 'Solo'], height=[flock_data.mean(), solo_data.mean()], color=['blue', 'red'])
    plt.title('Mean Average Distance From Predator')
    plt.xlabel('Scenarios')
    plt.ylabel('In-Game Units')
    plt.show()



def catch_rate_analysis(flock_array, solo_array):

    float_arr1 = str_to_float(flock_array)
    float_arr2 = str_to_float(solo_array)
    flock_data = np.array(float_arr1)
    solo_data = np.array(float_arr2)

    print("flock mean catch rate: ", flock_data.mean())
    print("solo mean catch rate: ", solo_data.mean())

    plt.bar(x=['Flock', 'Solo'], height=[flock_data.mean(), solo_data.mean()], color=['blue', 'red'])
    plt.title('Mean Catch Rate Of Predator Per Cycle')
    plt.xlabel('Scenarios')
    plt.ylabel('Pray Caught')
    plt.show()


def survival_time_analysis(flock_array, solo_array):
    float_arr1 = str_to_float(flock_array)
    float_arr2 = str_to_float(solo_array)
    flock_data = np.array(float_arr1)
    solo_data = np.array(float_arr2)

    print('flock mean survival time: ', flock_data.mean())
    print('flock max survival time: ', flock_data.max())

    print('solo mean survival time: ', solo_data.mean())
    print('solo max survival time: ', solo_data.max())


    # Pad or truncate arrays to have the same length
    max_length = max(len(flock_data), len(solo_data))
    padded_flock_data = np.pad(flock_data, (0, max(0, len(solo_data) - len(flock_data))), mode='constant',
                               constant_values=np.nan)
    padded_solo_data = np.pad(solo_data, (0, max(0, len(flock_data) - len(solo_data))), mode='constant',
                              constant_values=np.nan)

    # Create a DataFrame from the padded or truncated arrays
    df = pd.DataFrame({'Flock': padded_flock_data, 'Solo': padded_solo_data})

    # Create a box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, palette=['blue', 'red'])
    plt.xlabel('Scenario')
    plt.ylabel('Time Survived')
    plt.title('Time Survived Box Plot')
    plt.grid(True)
    plt.show()

    plt.bar(x=['Flock', 'Solo'], height=[flock_data.mean(), solo_data.mean()], color=['blue', 'red'])
    plt.title('Mean Time Survived In Simulation')
    plt.xlabel('Scenarios')
    plt.ylabel('In-Game time')
    plt.show()


def str_to_float(array):
    float_arr = []
    for i in range(len(array)):
        float_arr.append(float(array[i]))
    return float_arr


if __name__ == '__main__':
    for i in range(1):
        print("round number:", i)
        sim(True)
        sim(False)

    distance_analysis(np.concatenate(flock_avg_distances), np.concatenate(solo_avg_distances))
    catch_rate_analysis(np.concatenate(flock_predator_catch_rates), np.concatenate(solo_predator_catch_rates))
    survival_time_analysis(np.concatenate(flock_survival_times), np.concatenate(solo_survival_times))

