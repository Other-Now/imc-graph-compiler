// RUN: imc-opt %s -split-input-file -imc-lower-to-linalg -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @mvm
// CHECK-NOT: imc.
// CHECK: linalg.fill
// CHECK: linalg.matmul ins(%arg1, %arg0 : tensor<4x64xi8>, tensor<64x8xi8>) outs(%{{.*}} : tensor<4x8xi32>)
// CHECK: linalg.generic
// CHECK:   arith.addi %{{.*}}, %c4_i32
// CHECK:   arith.shrsi %{{.*}}, %c3_i32
// CHECK:   arith.maxsi %{{.*}}, %c-32_i32
// CHECK:   arith.minsi %{{.*}}, %c31_i32
// CHECK:   arith.shli %{{.*}}, %c3_i32
// CHECK: linalg.add
func.func @mvm(%w: tensor<64x8xi8>, %a: tensor<4x64xi8>, %p: tensor<4x8xi32>) -> tensor<4x8xi32> {
  %x = imc.program_weights %w : tensor<64x8xi8> -> !imc.xbar<256, 256>
  %y = imc.mvm %x, %a {adc_bits = 6 : i64, adc_shift = 3 : i64} : !imc.xbar<256, 256>, tensor<4x64xi8> -> tensor<4x8xi32>
  %z = imc.accumulate %y, %p : tensor<4x8xi32>
  return %z : tensor<4x8xi32>
}

// -----

// An exact (no-ADC) imc.matmul lowers directly, with no ADC generic.
// CHECK-LABEL: func.func @exact
// CHECK: linalg.matmul
// CHECK-NOT: linalg.generic
func.func @exact(%a: tensor<4x64xi8>, %w: tensor<64x8xi8>) -> tensor<4x8xi32> {
  %y = imc.matmul %a, %w : tensor<4x64xi8>, tensor<64x8xi8> -> tensor<4x8xi32>
  return %y : tensor<4x8xi32>
}

// -----

func.func @untiled_adc(%a: tensor<4x64xi8>, %w: tensor<64x8xi8>) -> tensor<4x8xi32> {
  // expected-error @+1 {{imc.matmul with an ADC has no meaning until it is tiled}}
  %y = imc.matmul %a, %w {adc_bits = 6 : i64, adc_shift = 3 : i64} : tensor<4x64xi8>, tensor<64x8xi8> -> tensor<4x8xi32>
  return %y : tensor<4x8xi32>
}
