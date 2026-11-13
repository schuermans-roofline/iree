// RUN: iree-opt --split-input-file --iree-dispatch-creation-elementwise-op-fusion %s | FileCheck %s --check-prefix=FUSE
// RUN: iree-opt --split-input-file --iree-dispatch-creation-elementwise-op-fusion="max-binding-bytes=100000" %s | FileCheck %s --check-prefix=FUSE-LIMIT
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-fuse-multi-use-elementwise-producer))" %s | FileCheck %s --check-prefix=MULTI
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-fuse-multi-use-elementwise-producer{max-binding-bytes=100000}))" %s | FileCheck %s --check-prefix=MULTI-LIMIT
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions))" %s | FileCheck %s --check-prefix=FORM
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{max-binding-bytes=100000}))" %s | FileCheck %s --check-prefix=FORM-LIMIT

// Each global below is 512x64xf16, 65536 bytes, so a dispatch reading two of
// them binds more than the 100000 byte limit.

// An embedding lookup split over two tables (as by
// iree-global-opt-split-large-constants): elementwise fusion would merge the
// two lookups into one op reading both tables.

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
util.global private @shard0 = dense<1.0> : tensor<512x64xf16>
util.global private @shard1 = dense<2.0> : tensor<512x64xf16>
util.func public @gather_chain(%ids: tensor<8xi64>) -> tensor<8x64xf16> {
  %c511 = arith.constant 511 : index
  %c512 = arith.constant 512 : index
  %shard0 = util.global.load @shard0 : tensor<512x64xf16>
  %shard1 = util.global.load @shard1 : tensor<512x64xf16>
  %empty = tensor.empty() : tensor<8x64xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}
      ins(%ids : tensor<8xi64>) outs(%empty : tensor<8x64xf16>) {
  ^bb0(%in: i64, %out: f16):
    %i = arith.index_cast %in : i64 to index
    %j = linalg.index 1 : index
    %clamped = arith.minui %i, %c511 : index
    %v = tensor.extract %shard0[%clamped, %j] : tensor<512x64xf16>
    linalg.yield %v : f16
  } -> tensor<8x64xf16>
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]}
      ins(%ids, %0 : tensor<8xi64>, tensor<8x64xf16>) outs(%empty : tensor<8x64xf16>) {
  ^bb0(%in: i64, %prev: f16, %out: f16):
    %i0 = arith.index_cast %in : i64 to index
    %i = arith.subi %i0, %c512 : index
    %j = linalg.index 1 : index
    %inShard = arith.cmpi ult, %i, %c512 : index
    %clamped = arith.minui %i, %c511 : index
    %v = tensor.extract %shard1[%clamped, %j] : tensor<512x64xf16>
    %r = arith.select %inShard, %v, %prev : f16
    linalg.yield %r : f16
  } -> tensor<8x64xf16>
  util.return %1 : tensor<8x64xf16>
}

// FUSE-LABEL: util.func public @gather_chain
//       FUSE:   linalg.generic
//       FUSE:     tensor.extract %{{.+}}[
//       FUSE:     tensor.extract %{{.+}}[
//   FUSE-NOT:   linalg.generic
//       FUSE:   util.return

// FUSE-LIMIT-LABEL: util.func public @gather_chain
//   FUSE-LIMIT-DAG:   %[[SHARD0:.+]] = util.global.load @shard0
//   FUSE-LIMIT-DAG:   %[[SHARD1:.+]] = util.global.load @shard1
//       FUSE-LIMIT:   %[[G0:.+]] = linalg.generic
//       FUSE-LIMIT:     tensor.extract %[[SHARD0]]
//   FUSE-LIMIT-NOT:     tensor.extract %[[SHARD1]]
//       FUSE-LIMIT:   %[[G1:.+]] = linalg.generic
//  FUSE-LIMIT-SAME:       ins(%{{.+}}, %[[G0]] :
//       FUSE-LIMIT:     tensor.extract %[[SHARD1]]
//       FUSE-LIMIT:   util.return %[[G1]]

// -----

// A fusion that does not grow a dispatch already over the limit is still
// done: both ops read the same table, so the fused op binds less than the
// consumer alone.

#map = affine_map<(d0, d1) -> (d0, d1)>
util.global private @table = dense<1.0> : tensor<512x64xf16>
util.func public @shrinking_fusion() -> tensor<512x64xf16> {
  %table = util.global.load @table : tensor<512x64xf16>
  %empty = tensor.empty() : tensor<512x64xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%table : tensor<512x64xf16>) outs(%empty : tensor<512x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %neg = arith.negf %in : f16
    linalg.yield %neg : f16
  } -> tensor<512x64xf16>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%0, %table : tensor<512x64xf16>, tensor<512x64xf16>) outs(%empty : tensor<512x64xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %add = arith.addf %a, %b : f16
    linalg.yield %add : f16
  } -> tensor<512x64xf16>
  util.return %1 : tensor<512x64xf16>
}

// FUSE-LIMIT-LABEL: util.func public @shrinking_fusion
//       FUSE-LIMIT:   linalg.generic
//   FUSE-LIMIT-NOT:   linalg.generic
//       FUSE-LIMIT:   util.return

// -----

// Multi-use producer fusion would merge two lookups into different tables.

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
util.global private @table0 = dense<1.0> : tensor<512x64xf16>
util.global private @table1 = dense<2.0> : tensor<512x64xf16>
util.func public @multi_use(%ids: tensor<8xi64>) -> (tensor<8x64xf16>, tensor<8x64xf16>) {
  %table0 = util.global.load @table0 : tensor<512x64xf16>
  %table1 = util.global.load @table1 : tensor<512x64xf16>
  %empty = tensor.empty() : tensor<8x64xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}
      ins(%ids : tensor<8xi64>) outs(%empty : tensor<8x64xf16>) {
  ^bb0(%in: i64, %out: f16):
    %i = arith.index_cast %in : i64 to index
    %j = linalg.index 1 : index
    %v = tensor.extract %table0[%i, %j] : tensor<512x64xf16>
    linalg.yield %v : f16
  } -> tensor<8x64xf16>
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]}
      ins(%ids, %0 : tensor<8xi64>, tensor<8x64xf16>) outs(%empty : tensor<8x64xf16>) {
  ^bb0(%in: i64, %a: f16, %out: f16):
    %i = arith.index_cast %in : i64 to index
    %j = linalg.index 1 : index
    %v = tensor.extract %table1[%i, %j] : tensor<512x64xf16>
    %add = arith.addf %a, %v : f16
    linalg.yield %add : f16
  } -> tensor<8x64xf16>
  util.return %0, %1 : tensor<8x64xf16>, tensor<8x64xf16>
}

// MULTI-LABEL: util.func public @multi_use
//       MULTI:   %[[FUSED:.+]]:2 = linalg.generic
//   MULTI-NOT:   linalg.generic
//       MULTI:   util.return %[[FUSED]]#0, %[[FUSED]]#1

// MULTI-LIMIT-LABEL: util.func public @multi_use
//       MULTI-LIMIT:   %[[G0:.+]] = linalg.generic
//       MULTI-LIMIT:     tensor.extract %table0
//       MULTI-LIMIT:   %[[G1:.+]] = linalg.generic
//  MULTI-LIMIT-SAME:       ins(%{{.+}}, %[[G0]] :
//       MULTI-LIMIT:     tensor.extract %table1
//       MULTI-LIMIT:   util.return %[[G0]], %[[G1]]

// -----

// Dispatch region formation would fuse a consumer reading a second table into
// the dispatch of a matmul with a large weight.

#map = affine_map<(d0, d1) -> (d0, d1)>
util.global private @weight = dense<1.0> : tensor<64x512xf16>
util.global private @table = dense<2.0> : tensor<512x64xf16>
util.func public @consumer_fusion(%lhs: tensor<8x64xf16>) -> tensor<8x512xf32> {
  %cst = arith.constant 0.0 : f32
  %weight = util.global.load @weight : tensor<64x512xf16>
  %table = util.global.load @table : tensor<512x64xf16>
  %empty = tensor.empty() : tensor<8x512xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x512xf32>) -> tensor<8x512xf32>
  %0 = linalg.matmul ins(%lhs, %weight : tensor<8x64xf16>, tensor<64x512xf16>)
      outs(%fill : tensor<8x512xf32>) -> tensor<8x512xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%0 : tensor<8x512xf32>) outs(%empty : tensor<8x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %v = tensor.extract %table[%j, %i] : tensor<512x64xf16>
    %ext = arith.extf %v : f16 to f32
    %add = arith.addf %in, %ext : f32
    linalg.yield %add : f32
  } -> tensor<8x512xf32>
  util.return %1 : tensor<8x512xf32>
}

// FORM-LABEL: util.func public @consumer_fusion
//       FORM:   %[[DISPATCH:.+]] = flow.dispatch.region
//       FORM:     %[[MATMUL:.+]] = linalg.matmul
//       FORM:     %[[ADD:.+]] = linalg.generic
//  FORM-SAME:         ins(%[[MATMUL]] :
//       FORM:     flow.return %[[ADD]]
//   FORM-NOT:   flow.dispatch.region
//       FORM:   util.return %[[DISPATCH]]

// FORM-LIMIT-LABEL: util.func public @consumer_fusion
//       FORM-LIMIT:   %[[MATMUL_DISPATCH:.+]] = flow.dispatch.region
//       FORM-LIMIT:     %[[MATMUL:.+]] = linalg.matmul
//       FORM-LIMIT:     flow.return %[[MATMUL]]
//       FORM-LIMIT:   %[[ADD_DISPATCH:.+]] = flow.dispatch.region
//       FORM-LIMIT:     linalg.generic
//  FORM-LIMIT-SAME:         ins(%[[MATMUL_DISPATCH]] :
//       FORM-LIMIT:   util.return %[[ADD_DISPATCH]]
