// RUN: imc-opt %s -split-input-file -imc-tile-crossbars="rows=256 cols=256" -verify-diagnostics | FileCheck %s

// K = 600 -> 3 row tiles (256, 256, 88); N = 300 -> 2 column tiles (256, 44).
// Weight-stationary: each of the 6 tiles is programmed once, and its partial
// sums are accumulated down K before the column block is inserted.

// CHECK-LABEL: func.func @fc
// CHECK-NOT: imc.matmul
// CHECK: %[[W0:.*]] = tensor.extract_slice %arg1[0, 0] [256, 256] [1, 1]
// CHECK: %[[X0:.*]] = imc.program_weights %[[W0]] : tensor<256x256xi8> -> !imc.xbar<256, 256>
// CHECK: imc.mvm %[[X0]], %{{.*}} {adc_bits = 6 : i64, adc_shift = 3 : i64} : !imc.xbar<256, 256>, tensor<8x256xi8> -> tensor<8x256xi32>
// CHECK: imc.accumulate
// CHECK: tensor.extract_slice %arg1[512, 0] [88, 256]
// CHECK: imc.accumulate
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [8, 256]
// CHECK: tensor.extract_slice %arg1[0, 256] [256, 44]
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 256] [8, 44]
func.func @fc(%a: tensor<8x600xi8>, %w: tensor<600x300xi8>) -> tensor<8x300xi32> {
  %y = imc.matmul %a, %w {adc_bits = 6 : i64, adc_shift = 3 : i64}
     : tensor<8x600xi8>, tensor<600x300xi8> -> tensor<8x300xi32>
  return %y : tensor<8x300xi32>
}

// -----

// CHECK-LABEL: func.func @count
// CHECK-COUNT-6: imc.program_weights
// CHECK-NOT: imc.program_weights
// CHECK: return
func.func @count(%a: tensor<8x600xi8>, %w: tensor<600x300xi8>) -> tensor<8x300xi32> {
  %y = imc.matmul %a, %w {adc_bits = 6 : i64, adc_shift = 3 : i64}
     : tensor<8x600xi8>, tensor<600x300xi8> -> tensor<8x300xi32>
  return %y : tensor<8x300xi32>
}

// -----

func.func @no_adc(%a: tensor<8x16xi8>, %w: tensor<16x4xi8>) -> tensor<8x4xi32> {
  // expected-error @+1 {{imc.matmul needs adc_bits/adc_shift before it can be mapped onto crossbars}}
  %y = imc.matmul %a, %w : tensor<8x16xi8>, tensor<16x4xi8> -> tensor<8x4xi32>
  return %y : tensor<8x4xi32>
}
