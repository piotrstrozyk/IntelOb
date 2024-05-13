import math
import random
import matplotlib.pyplot as plt

GRAVITY = 9.81  # Przyspieszenie ziemskie [m/s^2]
TREBUCHET_HEIGHT = 100  # Wysokość trebusza [m]
INITIAL_VELOCITY = 50   # Początkowa prędkość pocisku [m/s]
TARGET_MARGIN = 5  # Margines błędu trafienia [m]

def calculate_distance(angle):
    angle_rad = math.radians(angle)
    time_to_ground = (INITIAL_VELOCITY * math.sin(angle_rad) + math.sqrt((INITIAL_VELOCITY * math.sin(angle_rad)) ** 2 + 2 * GRAVITY * TREBUCHET_HEIGHT)) / GRAVITY
    max_distance = INITIAL_VELOCITY * math.cos(angle_rad) * time_to_ground
    return max_distance, time_to_ground

def plot_trajectory(angle, time):
    angle_rad = math.radians(angle)
    times = [i * time / 100 for i in range(101)]
    x_values = [INITIAL_VELOCITY * math.cos(angle_rad) * t for t in times]
    y_values = [INITIAL_VELOCITY * math.sin(angle_rad) * t - 0.5 * GRAVITY * t**2 for t in times]
    
    plt.plot(x_values, y_values, color='blue')
    plt.xlabel('Odległość [m]')
    plt.ylabel('Wysokość [m]')
    plt.title('Trajektoria pocisku Warwolf')
    plt.grid(True)
    plt.savefig("trajektoria.png")
    plt.show()

def main():
    target = random.randint(50, 340)
    print(f"Cel znajduje się na odległości: {target} m")
   
    attempts = 0
    while True:
        attempts += 1
        angle = float(input("Podaj kąt strzału (w stopniach): ")) 

        distance, time = calculate_distance(angle)
        print(f"Pocisk doleci na odległość: {distance} m")
        
        if target - TARGET_MARGIN <= distance <= target + TARGET_MARGIN:
            print(f"Gratulacje! Trafiono cel po {attempts} próbach.")
            plot_trajectory(angle, time)
            break
        else:
            print("Pudło. Spróbuj ponownie.")

if __name__ == "__main__":
    main()