// RUN: imc-opt %s -split-input-file -verify-diagnostics

func.func @tile_too_big(%w: tensor<300x64xi8>) {
  // expected-error @+1 {{a 300x64 weight tile does not fit a 256x256 crossbar}}
  %x = imc.program_weights %w : tensor<300x64xi8> -> !imc.xbar<256, 256>
  return
}

// -----

func.func @float_weights(%w: tensor<64x64xf32>) {
  // expected-error @+1 {{weights must have i8 elements}}
  %x = imc.program_weights %w : tensor<64x64xf32> -> !imc.xbar<256, 256>
  return
}

// -----

func.func @int4_activations(%a: tensor<4x64xi4>, %w: tensor<64x8xi8>) -> tensor<4x8xi32> {
  // expected-error @+1 {{lhs must have i8 elements}}
  %y = imc.matmul %a, %w : tensor<4x64xi4>, tensor<64x8xi8> -> tensor<4x8xi32>
  return %y : tensor<4x8xi32>
}

// -----

func.func @contraction(%a: tensor<4x63xi8>, %w: tensor<64x8xi8>) -> tensor<4x8xi32> {
  // expected-error @+1 {{contraction mismatch: lhs is 4x63 but rhs is 64x8}}
  %y = imc.matmul %a, %w : tensor<4x63xi8>, tensor<64x8xi8> -> tensor<4x8xi32>
  return %y : tensor<4x8xi32>
}

// -----

func.func @half_adc(%a: tensor<4x64xi8>, %w: tensor<64x8xi8>) -> tensor<4x8xi32> {
  // expected-error @+1 {{adc_bits and adc_shift must be given together}}
  %y = imc.matmul %a, %w {adc_bits = 6 : i64} : tensor<4x64xi8>, tensor<64x8xi8> -> tensor<4x8xi32>
  return %y : tensor<4x8xi32>
}

// -----

func.func @adc_range(%w: tensor<64x8xi8>, %a: tensor<4x64xi8>) -> tensor<4x8xi32> {
  %x = imc.program_weights %w : tensor<64x8xi8> -> !imc.xbar<256, 256>
  // expected-error @+1 {{adc_bits must be in [1, 16], got 0}}
  %y = imc.mvm %x, %a {adc_bits = 0 : i64, adc_shift = 0 : i64} : !imc.xbar<256, 256>, tensor<4x64xi8> -> tensor<4x8xi32>
  return %y : tensor<4x8xi32>
}

// -----

func.func @mvm_shape(%w: tensor<64x8xi8>, %a: tensor<4x32xi8>) -> tensor<4x8xi32> {
  %x = imc.program_weights %w : tensor<64x8xi8> -> !imc.xbar<256, 256>
  // expected-error @+1 {{crossbar was programmed with a 64x8 tile but is used as 32x8}}
  %y = imc.mvm %x, %a {adc_bits = 8 : i64, adc_shift = 0 : i64} : !imc.xbar<256, 256>, tensor<4x32xi8> -> tensor<4x8xi32>
  return %y : tensor<4x8xi32>
}

// -----

func.func @mvm_overdrive(%x: !imc.xbar<128, 128>, %a: tensor<4x200xi8>) -> tensor<4x8xi32> {
  // expected-error @+1 {{input drives 200 rows but the crossbar has 128}}
  %y = imc.mvm %x, %a {adc_bits = 8 : i64, adc_shift = 0 : i64} : !imc.xbar<128, 128>, tensor<4x200xi8> -> tensor<4x8xi32>
  return %y : tensor<4x8xi32>
}

// -----

func.func @acc_type(%a: tensor<4x8xi16>, %b: tensor<4x8xi16>) -> tensor<4x8xi16> {
  // expected-error @+1 {{result must have i32 elements}}
  %y = imc.accumulate %a, %b : tensor<4x8xi16>
  return %y : tensor<4x8xi16>
}
