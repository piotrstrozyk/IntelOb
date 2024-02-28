# from datetime import datetime
# from math import sin
# import math

# def biorythm(year, month, day):
#     today = datetime.now()
#     birth = datetime(year, month, day)
#     days = (today - birth).days
    
#     phys = sin(2 * math.pi * days / 23)
#     emo = sin(2 * math.pi * days / 28)
#     intel = sin(2 * math.pi * days / 33)
    
#     return phys, emo, intel
    
# def main():
#     name = input("Name: ")
#     year = int(input("Year: "))
#     month = int(input("Month: "))
#     day = int(input("Day: "))
    
#     phys, emo, intel = biorythm(year, month, day)
    
#     print(f"Today is {datetime.now().strftime('%Y-%m-%d')}, day of your life.")
#     print(f"Your results:")
#     print(f"- Physical: {phys}")
#     print(f"- Emotional: {emo}")
#     print(f"- Intellectual: {intel}")
    
#     biorythms = [phys, emo, intel]
    
#     name = ["Physical", "Emotional", "Intellectual"]
    
#     for name, value in zip(name, biorythms):
#         if value > 0.5:
#             print(f"Congratulations! Your {name} biorhythm is high today.")
#         elif value < -0.5:
#             print(f"Don't worry. It's just a bad day for your {name} biorhythm.")
#         else:
#             print(f"Your {name} biorhythm is neutral today.")


# if __name__ == "__main__":
#     main()

from datetime import datetime
from math import sin, pi

def biorythm(year, month, day):
    today = datetime.now()
    birth = datetime(year, month, day)
    days = (today - birth).days
    
    phys = sin(2 * pi * days / 23)
    emo = sin(2 * pi * days / 28)
    intel = sin(2 * pi * days / 33)
    
    return phys, emo, intel
    
def main():
    name = input("Name: ")
    year = int(input("Year: "))
    month = int(input("Month: "))
    day = int(input("Day: "))
    
    phys, emo, intel = biorythm(year, month, day)
    
    print(f"Today is {(datetime.now().date() - datetime(year, month, day).date()).days + 1}, day of your life.")
    print("Your results:")
    
    biorythms = [("Physical", phys), ("Emotional", emo), ("Intellectual", intel)]
    
    for name, value in biorythms:
        if value > 0.5:
            print(f"Congratulations! Your {name} biorhythm is high today.")
        elif value < -0.5:
            print(f"Don't worry. It's just a bad day for your {name} biorhythm.")
        else:
            print(f"Your {name} biorhythm is neutral today.")

if __name__ == "__main__":
    main()  
    
#Samodzielnie: ~10 min
#Z chatem: ~5 min
#Kompletny program: ~15 minut