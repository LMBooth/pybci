#include <Servo.h>
Servo handservos[5];  // create an array of servo objects
#define ANALOG_PIN A0 
void SetHand(Servo servos[], int pos[]){
  for (int i = 0; i < 5; i++) {
    servos[i].write(pos[i]);
  }
}
int handPos[6][5] ={
  {0, 0, 0, 0, 0},// open
  {180,180,180,180,180}, // closed
  {0,180,180,0,180}, // rock
  {180,180,0,0,180}, //  peace
  {0,180,180,180,180}, // pinky
  {180,180,180,180,0} // thumb
  };

void setup() {
  Serial.begin(9600);  // start serial communication at 9600bps
  handservos[0].attach(3); //little
  handservos[1].attach(5);
  handservos[2].attach(6); // middle
  handservos[3].attach(9);
  handservos[4].attach(10); // thumb
  SetHand(handservos, handPos[0]);
}

void loop() {
  if (Serial.available() > 0) {  // if there's data available to read
    char handIndex = Serial.read();  // read the incoming byte as a char
    //Serial.print(handIndex);
    if (handIndex >= '0' && handIndex <= '5') {
      int incomingInt = handIndex - '0';
      SetHand(handservos, handPos[incomingInt]);
    }
  }
  int analogValue = analogRead(ANALOG_PIN);  // read the input on analog pin
  Serial.println(analogValue); 
}
