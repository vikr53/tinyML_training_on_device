/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <U8x8lib.h>
#include <Arduino.h>
#include <string.h>

// UNSIGNED by appending U to the end
// Ref: https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Fmemory.html&anchor=memory__fig
#define NVMC_BASE     (0x4001E000U)
#define NVMC_READY    (NVMC_BASE + 0x400U)
#define NVMC_CONFIG   (NVMC_BASE + 0x504U)
#define NVMC_ERASEPAGE  (NVMC_BASE + 0x508U)


const float myFlash2[2] __attribute__ ((section("FLASH"), aligned(0x1000))) = {0};

int j = 0;
U8X8_SH1106_128X64_NONAME_HW_I2C u8x8(U8X8_PIN_NONE);

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8 quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
// Prev. 116640
constexpr int kTensorArenaSize = 115408;
static uint8_t tensor_arena[kTensorArenaSize];

// Fully connected layer. First 256 - x1, Second 256 - x2
float fc_w[512];
float fc_b[2];

// Pin Setup
const int train_0_btn = 4;
int train_0_btn_pressed = 0;
const int train_1_btn = 5;
int train_1_btn_pressed = 0;

}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() { 
  // for (int i = 0; i < 514; i++) {
  //   myFlash[i] = 0.0;
  // }
  // Set up display
  u8x8.setI2CAddress(0x7A);
  u8x8.begin();
  u8x8.setFont(u8x8_font_amstrad_cpc_extended_f); // 8 X 8 font
  u8x8.drawString(0, 1, "Starting up...");
  delay(1000);
  u8x8.clear();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                               tflite::ops::micro::Register_AVERAGE_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  // Initialize transfer learning weights to some random small numbers and biases to 0 (ok since it is a small network i.e. vainishing gradients not a problem)
  for (int i=0; i < 512; i++) {
    fc_w[i] = random(0,100)*0.0001;
    if (i < 2) {
      fc_b[i] = 0;
    }
  }

  // Pin Setup - LEDS are on when LOW
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);

  // TRAIN_0 Pin Setup
  pinMode(train_0_btn, INPUT);
  digitalWrite(train_0_btn, HIGH);

  // TRAIN_0 Pin Setup
  pinMode(train_1_btn, INPUT);
  digitalWrite(train_1_btn, HIGH);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  if (j == 4) {
    writeWeightsToFlash(fc_w, fc_b);
  }
  
  Serial.print("Value at myFlash: ");
  hexDumpMemory((float *)(myFlash), 5);
  Serial.print("MYFLASH add: "); Serial.println((unsigned long)(myFlash), HEX);
  for (int i = 0; i < 2; i++) {
    Serial.print("MYFLASH: ");
    Serial.println(*(float *)(myFlash+i));
  }
  j = j + 1;
  /*** FORWARD PASS ***/
  // Get image from provider
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }
  u8x8.clear();
  u8x8.drawString(0, 1, "Image Retrieved");

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  u8x8.drawString(0, 3, "Model Invoked");
  
  // Output tensor has dim 1x1x1x256
  TfLiteTensor* output = interpreter->output(0);

  float x[256];
  for (int i=0; i<256; ++i) {
    x[i] = output->data.uint8[i];
  }

  // Normalize inputs
  float sum_x = 0;
  float mean_x = 0;
  float var_sum = 0;
  float var = 0;
  float epsilon = 0.00001; // Prevent divide by 0

  // Calculate mean
  for (int i = 0; i < 256; i++) {
    sum_x = sum_x + x[i];
  }
  mean_x = sum_x/256;

  // Calculate variance
  for (int i = 0; i < 256; i++) {
    var_sum = var_sum + pow((x[i]-mean_x),2);
  }
  var = var_sum/256;

  // Normalize x
  for (int i = 0; i < 256; i++) {
    x[i] = (x[i]-mean_x)/pow((var+epsilon),0.5);
  }
  
  // FC layer
  float z[] = {0, 0};
  for (int i = 0; i < 2; i++){
    for (int j = 0; j < 256; j++) {
      z[i] = z[i] + fc_w[j+i*256]*x[j];
    }
    z[i] = z[i] + fc_b[i];
  }

  // Softmax
  float y_pred[2]; // y_pred[0] = person/mask ; y_pred[1] = no person/mask
  y_pred[0] = exp(z[0])/(exp(z[0])+exp(z[1]));
  y_pred[1] = exp(z[1])/(exp(z[0])+exp(z[1]));
 
  float ce_loss = 0;

  // Read train 0, 1 btn
  int train_0_btn_state = digitalRead(train_0_btn);
  int train_1_btn_state = digitalRead(train_1_btn);

  if (train_0_btn_state == LOW && train_1_btn_pressed % 2 != 1) {
    train_0_btn_pressed = train_0_btn_pressed + 1;
  } else if (train_0_btn_state == LOW && train_1_btn_pressed % 2 == 1) {
    //Serial.println("Can't train both at same time. Sorry!");
  }

  if (train_1_btn_state == LOW && train_0_btn_pressed % 2 != 1) {
    train_1_btn_pressed = train_1_btn_pressed + 1;
  } else if (train_1_btn_state == LOW && train_0_btn_pressed % 2 == 1) {
    //Serial.println("Can't train both at same time. Sorry!");
  }

  // TRAIN 0
  if (train_0_btn_pressed % 2 == 1) {
    // Train 0 ACTIVE
    // Set LED Inidcator
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDB, LOW);
    Serial.println("TRAINING 0");

    u8x8.clear();
    u8x8.drawString(0, 0, "TRAINING 0");
    u8x8.drawString(0, 1, "------------");
    ce_loss = -log(y_pred[0]);
    u8x8.setCursor(2, 1);
    u8x8.print("Loss: " + String(ce_loss));
    
    // Train
    // 1. Calculate gradients
    // 2. Apply grad update
    // NOTE: Training 0 so y = {1,0}
    float grad_w[512];
    float grad_b[2];
    
    // Calculate gradients (weights and biases)
    for (int i = 0; i < 512; i++) {
      if (i < 256) {
        grad_w[i] = -(1-y_pred[0])*x[i];
      } else if (i >= 256) {
        grad_w[i] = y_pred[1]*x[i-256];
      }
    }
    grad_b[0] = -(1-y_pred[0]);
    grad_b[1] = y_pred[1];

    // Apply gradient update
    float eta = 0.01; // Learning rate
    for (int i = 0; i < 512; i++) {
      fc_w[i] = fc_w[i] - eta * grad_w[i];
      // Serial.println("GRAD "+ String(i) + " " + String(grad_w[i]));
      // Serial.println("fc_w "+ String(i) + " " + String(fc_w[i]));
      if (i<2) {
        fc_b[i] = fc_b[i] - eta * grad_b[i];
        Serial.println("z "+ String(i) + " " + String(z[i]));
      }
      if (i < 256) {
        // Serial.println("x "+ String(i) + " " + String(x[i]));
      }
    }

  } else {
    // TRAIN 0 NOT ACTIVE
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
  }

  // TRAIN 1
  if (train_1_btn_pressed % 2 == 1) {
    // Train 1 ACTIVE
    // Set LED Inidcator
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDB, LOW);

    u8x8.clear();
    u8x8.drawString(0, 0, "TRAINING 1");
    u8x8.drawString(0, 1, "------------");
    ce_loss = -log(y_pred[1]);
    u8x8.setCursor(2, 2);
    u8x8.print("Loss: " + String(ce_loss));

    // Train
    // NOTE: Training 1 so y = {0, 1}
    float grad_w[512];
    float grad_b[2];
    
    // Calculate gradients (weights and biases)
    for (int i = 0; i < 512; i++) {
      if (i < 256) {
        grad_w[i] = y_pred[0]*x[i];
      } else if (i >= 256) {
        grad_w[i] = -(1-y_pred[1])*x[i-256];
      }
    }
    grad_b[0] = y_pred[0];
    grad_b[1] = -(1-y_pred[1]);

    // Apply gradient update
    float eta = 0.01; // Learning rate
    for (int i = 0; i < 512; i++) {
      fc_w[i] = fc_w[i] - eta * grad_w[i];
      // Serial.println("GRAD "+ String(i) + " " + String(grad_w[i]));
      // Serial.println("fc_w "+ String(i) + " " + String(fc_w[i]));
      if (i<2) {
        fc_b[i] = fc_b[i] - eta * grad_b[i];
        // Serial.println("z "+ String(i) + " " + String(z[i]));
      }
    }
  } else {
    // TRAIN 1 NOT ACTIVE
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
  }

  // Process inference results
  // RespondToDetection(error_reporter, y_pred[0], y_pred[1]);
  if (y_pred[0] > y_pred[1]) {
    u8x8.drawString(2,5,"Person?!");
  } else if (y_pred[1] > y_pred[0]) {
    u8x8.drawString(2,5,"Not person?!");
  }

  for (int i=0; i<2; ++i) {
    Serial.println(y_pred[i]);
  }
  
  Serial.println(train_0_btn_pressed);
  Serial.println(train_1_btn_pressed);

  Serial.println(ce_loss);

  // Process the inference results.
  // int8_t person_score = output->data.uint8[kPersonIndex];
  // int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  // RespondToDetection(error_reporter, person_score, no_person_score);
}

void writeWeightsToFlash(float w[], float b[]) {
  // With GCC, create a section called FLASH that is 0x1000 bytes (4kB - page) aligned
  float res[514];
  const float myFlash[2] __attribute__ ((aligned(0x1000))) = {0.1, 0.2};
  memcpy(res, w, sizeof(w));
  memcpy(res + 512, b, sizeof(b));
  Serial.print("res size"); Serial.println(sizeof(res)/sizeof(res[0])); 

  // Clear last page with saved gradient values
  if(*(uint32_t *)NVMC_READY == 1) {
    Serial.println("Erasing...");
    *(uint32_t *)NVMC_CONFIG = 0x02;
    *(float *)NVMC_ERASEPAGE = (uint32_t)(tmpFlash);
    while(*(uint32_t *)NVMC_READY == 0)
      delay(85);
    *(uint32_t *)NVMC_CONFIG = 0x00;
    Serial.println("...Erased");
  } else {
    Serial.println("... Flash Not Ready to Erase! ...");
  }

  hexDumpMemory((float *)(myFlash), 5);

  // Write to cleared last page
  Serial.print("NVMC READY: "); Serial.println(*(uint32_t *)(NVMC_READY), HEX);
  if (*(uint32_t *)NVMC_READY == 1) {
    Serial.println("Flashing...");
    // Set NVMC_CONFIG to 0x01 (write)
    *(uint32_t *)NVMC_CONFIG = 0x01;
    for (int i = 0; i < 2; i++) {
      *(uint32_t *)(&myFlash) = 0x00;
      //*((tmpFlash)+i) = 3; // t[i] == *(t + i)
      Serial.print("Flashed "); Serial.println(i);
    }
    
    // Wait till NVMC_READY is back to 1 to ensure the write has compeleted
    while(*(uint32_t *)NVMC_READY == 0)
      delayMicroseconds(50);
    // Set NVMC_CONFIG to Ox00 (back to read-only)
    *(uint32_t *)NVMC_CONFIG = 0x00;
    Serial.println("...Flashed 1");
  }
  
}

void hexDumpMemory(float *memStart, const unsigned int nbrOfBytes) {
  /* hex dump memory to the serial interface starting at memStart for nbrOfBytes */

  float *ptr;
  int bytesPerLine = 15;
  int bytesOnLine = 0;
  
  ptr = memStart;

  Serial.print("Memory dump of: "); Serial.println((unsigned long)(memStart), HEX);
  
  for (int i = 0; i < nbrOfBytes; i++) {
    if (bytesOnLine == 0) {
      Serial.print((unsigned long)(ptr+i), HEX);
      Serial.print(": ");
    }
    
    if (*(ptr+i) < 0x10)  // print a leading 0
      Serial.print("0");
    Serial.print(*(ptr+i)); Serial.print(" ");
    
    if (bytesOnLine == bytesPerLine) {
      Serial.println(" ");
      bytesOnLine = 0;
    } else {
      bytesOnLine += 1;
    }
  }
  Serial.println("");
}

