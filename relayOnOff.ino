void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(0, OUTPUT);
  pinMode(1, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  int sa = Serial.available();
//   if(sa > 0){
//     String incoming_data = Serial.readString();
//     Serial.print("Serial av: ");
//     Serial.println(sa);
//     Serial.print("Serial recived: ");
//     Serial.println(incoming_data);
//   }
  bool state = digitalRead(1);
  if(state == LOW){
    digitalWrite(0, HIGH);
    Serial.println(1);
  }else{
    digitalWrite(0, LOW);
    Serial.println(0);
  }
}
