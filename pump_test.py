from gpiozero import DigitalOutputDevice
from time import sleep

in3 = DigitalOutputDevice(17)
in4 = DigitalOutputDevice(27)

try:
    print("pump on")

    in3.on()
    in4.off()

    sleep(2)

    print("pump off")

    in3.off()
    in4.off()

    sleep(2)

finally:
    in3.off()
    in4.off()
