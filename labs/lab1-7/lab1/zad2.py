import math
import random
import matplotlib.pyplot as plt


def calculate_distance(angle):
    
    g = 9.81  
    h = 100  
    v0 = 50   
    
    angle_rad = math.radians(angle)
    
    t = (v0 * math.sin(angle_rad) + math.sqrt((v0 * math.sin(angle_rad)) ** 2 + 2 * g * h)) / g

    d = v0 * math.cos(angle_rad) * t
    
    return d, t

def plot_trajectory(angle, time):
    g = 9.81  
    v0 = 50   
    
    angle_rad = math.radians(angle)
    
    t = time
    times = [i * t / 100 for i in range(101)]
    
    x_values = [v0 * math.cos(angle_rad) * t for t in times]
    y_values = [v0 * math.sin(angle_rad) * t - 0.5 * g * t**2 for t in times]
    
    plt.plot(x_values, y_values, color='blue')
    plt.xlabel('Distance [m]')
    plt.ylabel('Height [m]')
    plt.title('Trajectory of the Warwolf projectile')
    plt.grid(True)
    plt.savefig("trajektoria.png")
    plt.show()

def main():
   target = random.randint(50, 340)
   print(f"Target distance: {target} m")
   
   attempts = 0
   while True:
        
        attempts += 1
        angle = float(input("Angle: ")) 

        distance, time = calculate_distance(angle)
        print(f"Distance: {distance} m")
        
        if target - 5 <= distance <= target + 5:
            print(f"Bullseye! You hit the target in {attempts} attempts.")
            plot_trajectory(angle, time)
            break
        else:
            print("You missed the target.")
        
if __name__ == "__main__":
    main()