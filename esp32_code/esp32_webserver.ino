// Telegram Bot Ini
// #include <WiFi.h>
// #include <FastBot2.h>

// #define WIFI_SSID "HERE SHOULD BE WIFI NAME"
// #define WIFI_PASS "HERE SHOULD BE WIFI PASSWORD"
// #define BOT_TOKEN "HERE SHOULD BE TELEGRAM BOT TOKEN"

// FastBot2 bot;

// TensorflowLite Ini
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
// #include "conv3_3x3_dense.h"
#include "flatten_dense20_dense.h"
#include "init_input.h"

namespace {
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;

  int inference_count = 0;
  constexpr int kTensorArenaSize = 17 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

// void update(fb::Update& u) {
//   Serial.println(u.message().text());
  
//   auto chat_id = u.message().from().id();
//   auto text = u.message().text();

//   fb::Message msg;
//   msg.text = text;
//   msg.chatID = chat_id;

//   bot.sendMessage(msg);
// }

void init_tf() {
  model = tflite::GetModel(models_flatten_dense20_dense_flatten_dense20_dense_tflite);
  static tflite::MicroMutableOpResolver<3> resolver;

  if(resolver.AddReshape() != kTfLiteOk) {
    Serial.println("Error in AddReshape");
    return;
  }

  if(resolver.AddFullyConnected() != kTfLiteOk) {
    Serial.println("Error in AddFullyConnected");
    return;
  }

  if(resolver.AddSoftmax() != kTfLiteOk) {
    Serial.println("Error in AddSoftmax");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocates_status = interpreter->AllocateTensors();
  if(allocates_status != kTfLiteOk) {
    Serial.println("Erorr in Alloc");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void setup() {
  // ==============
  Serial.begin(9600);
  // WiFi.mode(WIFI_STA);
  // WiFi.begin(WIFI_SSID, WIFI_PASS);
  // while (WiFi.status() != WL_CONNECTED) {
  //     delay(500);
  //     Serial.println(".");
  // }
  // Serial.println("Connected");
  // // ==============

  // bot.setToken(BOT_TOKEN);
  // bot.attachUpdate(update);
  // bot.setPollMode(fb::Poll::Long, 30000);

  init_tf();
}

void loop() {
  init_input(input);

  interpreter->Invoke();

  for(int i = 0; i < output->dims->data[1]; i++){
    Serial.print(output->data.f[i]);
    Serial.print(" ");
  }
  Serial.println("\n");
  delay(2000);
}
