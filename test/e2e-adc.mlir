// End to end: tile a 3x5 * 5x3 int8 matmul onto 2x2 crossbars, lower, run.
// RUN: imc-opt %s -imc-tile-crossbars="rows=2 cols=2" -imc-lower-to-linalg \
// RUN: | %lower | %run | FileCheck %s

func.func private @printMemrefI32(tensor<*xi32>)

func.func @main() {
  %x = arith.constant dense<[[1, 2, 3, 4, 5], [-1, -2, -3, -4, -5], [10, 0, 0, 0, 10]]> : tensor<3x5xi8>
  %w = arith.constant dense<[[1, 0, 2], [1, 1, 1], [1, 0, 3], [1, 1, 1], [1, 2, 0]]> : tensor<5x3xi8>
  // 4-bit ADC, step 2: every crossbar's partial sum is rounded to an even
  // number and clamped to [-16, 14] before the digital accumulate.
  %y = imc.matmul %x, %w {adc_bits = 4 : i64, adc_shift = 1 : i64}
     : tensor<3x5xi8>, tensor<5x3xi8> -> tensor<3x3xi32>
  %u = tensor.cast %y : tensor<3x3xi32> to tensor<*xi32>
  call @printMemrefI32(%u) : (tensor<*xi32>) -> ()
  return
}

// Hand-checked: row 2, column 1 is exactly 10*2 = 20, but that partial sum
// sits on one crossbar and saturates the 4-bit ADC at 7 << 1 = 14.
// CHECK: [18, 16, 18]
// CHECK: [-12, -16, -16]
// CHECK: [20, 14, 14]
