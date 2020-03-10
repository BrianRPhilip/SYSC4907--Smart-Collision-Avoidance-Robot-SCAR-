import Rpi.GPIO as GPIO
import time
GPIO.setmode(GPIO.board)

echo = 3
trig = 5

GPIO.setup(trig, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# clear trig pin
GPIO.output(trig, False)
time.sleep(1)


def distance():
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    while GPIO.input(echo) == 0:
        pulse_start = time.time()

    while GPIO.input(echo) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    d = pulse_duration * 17150
    d = round(distance, 2)
    return d


if __name__ == '__main__':
    try:
        while True:
            dist = distance()
            print("Measured Distance = %.1f cm" % dist)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Ultrasonic program stopped. Exiting!")
        GPIO.cleanup()
