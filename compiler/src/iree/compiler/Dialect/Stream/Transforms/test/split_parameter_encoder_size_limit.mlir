// RUN: iree-opt --split-input-file --iree-stream-resource-max-allocation-size=1500 --iree-stream-split-parameter-encoder %s | FileCheck %s

// Tests that the slab allocation in the parameter encoder is split into
// multiple batches when outputs exceed the max allocation size.
// With a 1500-byte limit and three 1024-byte outputs, each output gets its own
// slab batch.

// The encoder module should contain three resource.alloca calls (one per batch)
// and three resource.dealloca calls.
// CHECK: module @encoder
// CHECK: util.func public @__encode_parameters_all
// CHECK: stream.resource.alloca
// CHECK: stream.resource.dealloca
// CHECK: stream.resource.alloca
// CHECK: stream.resource.dealloca
// CHECK: stream.resource.alloca
// CHECK: stream.resource.dealloca

// CHECK-LABEL: module {
// CHECK-DAG: util.global private @slab_split_a : !stream.resource<constant>
util.global private @slab_split_a : !stream.resource<constant>
// CHECK-DAG: util.global private @slab_split_b : !stream.resource<constant>
util.global private @slab_split_b : !stream.resource<constant>
// CHECK-DAG: util.global private @slab_split_c : !stream.resource<constant>
util.global private @slab_split_c : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  // Single parameter used by multiple transformations.
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"shared_weights"> : vector<1024xi8>

  // Three transformations, each producing a 1024-byte output.
  %c10_i32 = arith.constant 10 : i32
  %out_a = stream.async.fill %c10_i32, %param[%c0 to %c256 for %c256] :
    i32 -> %param as !stream.resource<constant>{%c1024}

  %c20_i32 = arith.constant 20 : i32
  %out_b = stream.async.fill %c20_i32, %param[%c0 to %c256 for %c256] :
    i32 -> %param as !stream.resource<constant>{%c1024}

  %c30_i32 = arith.constant 30 : i32
  %out_c = stream.async.fill %c30_i32, %param[%c0 to %c256 for %c256] :
    i32 -> %param as !stream.resource<constant>{%c1024}

  // CHECK-DAG: util.global.store {{.+}}, @slab_split_a : !stream.resource<constant>
  util.global.store %out_a, @slab_split_a : !stream.resource<constant>
  // CHECK-DAG: util.global.store {{.+}}, @slab_split_b : !stream.resource<constant>
  util.global.store %out_b, @slab_split_b : !stream.resource<constant>
  // CHECK-DAG: util.global.store {{.+}}, @slab_split_c : !stream.resource<constant>
  util.global.store %out_c, @slab_split_c : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}
