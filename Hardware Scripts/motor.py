import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)



GPIO.setup(31, GPIO.OUT)  # input 1
GPIO.setup(29, GPIO.OUT)  # input 2
GPIO.setup(33, GPIO.OUT)  # PWM enable motor1 (front right)
GPIO.setup(37, GPIO.OUT)  # input 3
GPIO.setup(36, GPIO.OUT)  # input 4
GPIO.setup(35, GPIO.OUT)  # PWM enable motor2 (front left)

GPIO.setup(32, GPIO.OUT)  # PWM enable motor3 (rear right)
GPIO.setup(18, GPIO.OUT)  # input 1
GPIO.setup(16, GPIO.OUT)  # input 2
GPIO.setup(15, GPIO.OUT)  # input 3
GPIO.setup(13, GPIO.OUT)  # input 4
GPIO.setup(12, GPIO.OUT)  # PWM enable motor4 (rear left)

pwm = GPIO.PWM(33, 100)  # front right
pwm2 = GPIO.PWM(32, 100)  # rear left
pwm3 = GPIO.PWM(35, 100)  # front left
pwm4 = GPIO.PWM(12, 100)  # rear right

pwm.start(0)
pwm2.start(0)
pwm3.start(0)
pwm4.start(0)

#Functions to move the car.
def front_right():
    GPIO.output(29, True)
    GPIO.output(31, False)  # Motor 1 move forward (front right)
    
def front_right_reverse():
    GPIO.output(29, False)
    GPIO.output(31, True)  # Motor 1 move backward (front right)    

def front_left():
    GPIO.output(36, True)
    GPIO.output(37, False)  # Motor 2 move forward (front left)
    
def front_left_reverse():
    GPIO.output(36, False)
    GPIO.output(37, True)  # Motor 2 move backward (front left)    

def rear_left():
    GPIO.output(18, True)
    GPIO.output(16, False)  # Motor 3 move forward (rear right)
    
def rear_left_reverse():
    GPIO.output(18, False)
    GPIO.output(16, True)  # Motor 3 move backward (rear right)        

def rear_right():
    GPIO.output(15, True)
    GPIO.output(13, False)  # Motor 4 move forward (rear left)

def rear_right_reverse():
    GPIO.output(15, False)
    GPIO.output(13, True)  # Motor 4 move backward (rear left)

def move_back():
    GPIO.output(29, False)
    GPIO.output(31, True)   # motor 1 backward

    GPIO.output(36, False)
    GPIO.output(37, True)   # motor 2 backward

    GPIO.output(18, False)
    GPIO.output(16, True)   # motor 3 backward

    GPIO.output(15, False)
    GPIO.output(13, True)  # motor 4 backward


def stop_motors():
    GPIO.output(29, False)
    GPIO.output(31, False)   # motor 1 stop

    GPIO.output(36, False)
    GPIO.output(37, False)   # motor 2 stop

    GPIO.output(18, False)
    GPIO.output(16, False)   # motor 3 stop

    GPIO.output(15, False)
    GPIO.output(13, False)  # motor 4 stop


def duty_cycle(motor, speed):
    if speed < 25 or speed > 100:
        return "select speed from 25-100"
    switcher = {
        1: pwm.ChangeDutyCycle(speed),
        2: pwm2.ChangeDutyCycle(speed),
        3: pwm3.ChangeDutyCycle(speed),
        4: pwm4.ChangeDutyCycle(speed)
    }
    return switcher.get(motor, "Invalid motor selection")


def move_car(direction, speed):  # forward 1, right 2, left 3, backward 4, 5 stop
    #if direction != (1 or 2 or 3 or 4 or 5):

        if direction == 1: #forward
            front_right();  duty_cycle(1, speed)
            front_left();   duty_cycle(2, speed)
            rear_left();    duty_cycle(3, speed)
            rear_right();   duty_cycle(4, speed)
        if direction == 2: #right
            front_left();   duty_cycle(2, speed)
            rear_left();  duty_cycle(3, speed)
            front_right_reverse();  duty_cycle(1, speed)
            rear_right_reverse();  duty_cycle(4, speed*1.2)
        if direction == 3: #left
            front_right();  duty_cycle(1, speed)
            rear_right();   duty_cycle(4, speed)
            front_left_reverse();  duty_cycle(2, speed)
            rear_left_reverse();   duty_cycle(3, speed*1.4)
        if direction == 4:
            move_back() #backwards
            duty_cycle(1, speed)
            duty_cycle(2, speed)
            duty_cycle(3, speed)
            duty_cycle(4, speed)
        if direction == 5:
            stop_motors()
            duty_cycle(1, speed)
            duty_cycle(2, speed)
            duty_cycle(3, speed)
            duty_cycle(4, speed)


def enable_pwm():
    GPIO.output(33, True)
    GPIO.output(32, True)  # start enable
    GPIO.output(35, True)
    GPIO.output(12, True)  # start enable


def disable_pwm():
    GPIO.output(33, False)
    GPIO.output(32, False)
    GPIO.output(35, False)
    GPIO.output(12, False)


def cleanup_gpio():
    pwm.stop()
    pwm2.stop()
    pwm3.stop()
    pwm4.stop()
    GPIO.cleanup()



# TEST PROGRAM
if __name__ == '__main__':
    move_car(10, 50)    # verify no bad input is put in
    sleep(1)

    move_car(1, 55) # Moves left 2 seconds
    enable_pwm()
    sleep(1)
    move_car(3,55)
    sleep(1.)
    move_car(1,55)
    sleep(2)
    move_car(2,55)
    sleep(1.3)
    move_car(1,55)
    sleep(1.5)
    disable_pwm()
    
    move_car(5, 0)  #n h Stop 1 second
    enable_pwm()
    sleep(1)
    disable_pwm()
   
    cleanup_gpio()
