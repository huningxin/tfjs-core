/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import * as seedrandom from 'seedrandom';

import {ENV} from '../environment';
// import * as axis_util from '../ops/axis_util';
// import * as broadcast_util from '../ops/broadcast_util';
// import * as concat_util from '../ops/concat_util';
import {Conv2DInfo} from '../ops/conv_util';
// import * as erf_util from '../ops/erf_util';
// import * as ops from '../ops/ops';
// import {buffer, tensor3d, tensor4d} from '../ops/ops';
import {tensor3d} from '../ops/ops';
// import * as selu_util from '../ops/selu_util';
// import {getStridedSlicedInfo} from '../ops/slice_util';
// tslint:disable-next-line:max-line-length
import {DataId, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import * as types from '../types';
import {DataType, DataTypeMap, TypedArray} from '../types';
import * as util from '../util';

import {BackendTimingInfo, KernelBackend} from './backend';
import * as backend_util from './backend_util';

class DataInfo {
  shape: number[];
  dtype: DataType;
}

export class MathBackendWebML implements KernelBackend {
  private data = new WeakMap<DataId, DataTypeMap[DataType]>();
  private dataInfo = new WeakMap<DataId, DataInfo>();
  private canvas: HTMLCanvasElement;

  constructor() {
    if (typeof document !== 'undefined') {
      this.canvas = document.createElement('canvas');
    }
  }

  register(dataId: DataId, shape: number[], dtype: DataType): void {
    // console.log(`register ${dataId} ${shape} ${dtype}`);
    if (this.data.has(dataId)) {
      throw new Error(`Data buffer is already registered`);
    }
    this.data.set(dataId, null);
    this.dataInfo.set(dataId, {shape: shape, dtype: dtype});
  }
  write(dataId: DataId, values: TypedArray): void {
    console.log(`write ${dataId} ${values.length}`);
    if (values == null) {
      throw new Error('MathBackendWebML.write(): values can not be null');
    }
    this.throwIfNoData(dataId);
    this.data.set(dataId, values);
  }
  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    if (pixels == null) {
      throw new Error('MathBackendWebML.writePixels(): pixels can not be null');
    }
    let vals: Uint8ClampedArray;
    if (pixels instanceof ImageData) {
      vals = pixels.data;
    } else if (pixels instanceof HTMLCanvasElement) {
      vals = pixels.getContext('2d')
                 .getImageData(0, 0, pixels.width, pixels.height)
                 .data;
    } else if (
        pixels instanceof HTMLImageElement ||
        pixels instanceof HTMLVideoElement) {
      if (this.canvas == null) {
        throw new Error(
            'Can\'t read pixels from HTMLImageElement outside ' +
            'the browser.');
      }
      this.canvas.width = pixels.width;
      this.canvas.height = pixels.height;
      this.canvas.getContext('2d').drawImage(
          pixels, 0, 0, pixels.width, pixels.height);
      vals = this.canvas.getContext('2d')
                 .getImageData(0, 0, pixels.width, pixels.height)
                 .data;
    } else {
      throw new Error(
          `pixels is of unknown type: ${(pixels as {}).constructor.name}`);
    }
    let values: Int32Array;
    if (numChannels === 4) {
      values = new Int32Array(vals);
    } else {
      const numPixels = pixels.width * pixels.height;
      values = new Int32Array(numPixels * numChannels);
      for (let i = 0; i < numPixels; i++) {
        for (let channel = 0; channel < numChannels; ++channel) {
          values[i * numChannels + channel] = vals[i * 4 + channel];
        }
      }
    }
    const outShape: [number, number, number] =
        [pixels.height, pixels.width, numChannels];
    return tensor3d(values, outShape, 'int32');
  }
  async read(dataId: DataId): Promise<TypedArray> {
    return this.readSync(dataId);
  }
  readSync(dataId: DataId): TypedArray {
    console.log(`readSync ${dataId}`);
    this.throwIfNoData(dataId);
    if (this.data.get(dataId) === null) {
      let dataInfo = this.dataInfo.get(dataId);
      const size = util.sizeFromShape(dataInfo.shape);
      const values = util.getTypedArrayFromDType(dataInfo.dtype, size);
      // console.log(`getTypedArrayFromDType ${size} ${dataInfo.dtype}`);
      this.data.set(dataId, values);
    }
    return this.data.get(dataId);
  }

  disposeData(dataId: DataId): void {
    // console.log(`dispose ${dataId}`);
    if (this.data.has(dataId)) {
      this.data.delete(dataId);
    }
  }

  async time(f: () => void): Promise<BackendTimingInfo> {
    const start = performance.now();
    f();
    const kernelMs = performance.now() - start;
    return {kernelMs};
  }
  memory() {
    return {
      // Unreliable due to automatic gc. The numbers above are cumulative.
      unreliable: true
    };
  }

  private throwIfNoData(dataId: DataId) {
    if (!this.data.has(dataId)) {
      throw new Error(
          `CPU backend: No data found for this tensor. ` +
          `Did you change your backend in the middle of the program? ` +
          `New backends can't use Tensors created with previous backends`);
    }
  }

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    throw Error("not implement");
  }

  stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[],
      beginMask: number, endMask: number): T {
    throw Error("not implement");
  }

  reverse<T extends Tensor>(x: T, axis: number[]): T {
    throw Error("not implement");
  }

  // Concats 2d tensors along axis=1. See comments in MathBackend.concat().
  concat(a: Tensor2D, b: Tensor2D): Tensor2D {
    throw Error("not implement");
  }

  neg<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  add(a: Tensor, b: Tensor): Tensor {
    let c = Tensor.make(a.shape, {}, a.dtype);
    console.log(`op: add, inputs: [${a.id}, ${b.id}], outputs: [${c.id}]`);
    return c;
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    let c = Tensor.make(a.shape, {}, a.dtype);
    console.log(`op: substract, inputs: [${a.id}, ${b.id}], outputs: [${c.id}]`);
    return c;
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    throw Error("not implement");
  }

  matMul(a: Tensor2D, b: Tensor2D, transposeA: boolean, transposeB: boolean):
      Tensor2D {
    throw Error("not implement");
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  divide(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  sum(x: Tensor, axes: number[]): Tensor {
    const y = Tensor.make(x.shape, {}, x.dtype) as Tensor;
    console.log(`op: sum, inputs: [${x.id}], outputs: [${y.id}], attrs: {${axes}}`)
    return y;
  }

  argMin(x: Tensor, axis: number): Tensor {
    throw Error("not implement");
  }

  argMax(x: Tensor, axis: number): Tensor {
    throw Error("not implement");
  }

  cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean):
      Tensor {
    throw Error("not implement");
  }

  equal(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  notEqual(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  less(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  lessEqual(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  greater(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  greaterEqual(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  logicalNot<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  logicalAnd(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  logicalOr(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  where(condition: Tensor, a: Tensor, b: Tensor, dtype: DataType): Tensor {
    throw Error("not implement");
  }

  topKValues<T extends Tensor>(x: T, k: number): Tensor1D {
    throw Error("not implement");
  }

  topKIndices(x: Tensor, k: number): Tensor1D {
    throw Error("not implement");
  }

  min(x: Tensor, axes: number[]): Tensor {
    throw Error("not implement");
  }

  minimum(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  mod(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  max(x: Tensor, axes: number[]): Tensor {
    const y = Tensor.make(x.shape, {}, x.dtype) as Tensor;
    console.log(`op: max, inputs: [${x.id}], outputs: [${y.id}], attrs: {${axes}}`)
    return y;
  }

  maximum(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  squaredDifference(a: Tensor, b: Tensor): Tensor {
    throw Error("not implement");
  }

  ceil<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  floor<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  sign<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  round<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  exp<T extends Tensor>(x: T): T {
    let y = Tensor.make(x.shape, {}, x.dtype) as T;
    console.log(`op: exp, inputs: [${x.id}], outputs: [${y.id}]`);
    return y;
  }

  expm1<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  log<T extends Tensor>(x: T): T {
    let y = Tensor.make(x.shape, {}, x.dtype) as T;
    console.log(`op: log, inputs: [${x.id}], outputs: [${y.id}]`);
    return y;
  }

  log1p<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  sqrt<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  rsqrt<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  square<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  reciprocal<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  relu<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  elu<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  eluDer<T extends Tensor>(dy: T, y: T): T {
    throw Error("not implement");
  }

  selu<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  clip<T extends Tensor>(x: T, min: number, max: number): T {
    let y = Tensor.make(x.shape, {}, x.dtype) as T;
    console.log(`op: clip, inputs: [${x.id}], outputs: [${y.id}], attrs: {min: ${min}, max: ${max}}`);
    return y;
  }

  abs<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  int<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  sigmoid<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  softplus<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  sin<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  cos<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  tan<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  asin<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  acos<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  atan<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  atan2<T extends Tensor>(a: T, b: T): T {
    throw Error("not implement");
  }

  sinh<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  cosh<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  tanh<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  asinh<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  acosh<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  atanh<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  erf<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  step<T extends Tensor>(x: T, alpha = 0): T {
    throw Error("not implement");
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    //const dilationHeight = convInfo.dilationHeight;
    //const dilationWidth = convInfo.dilationWidth;
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;
    const y = Tensor.make(convInfo.outShape, {}, x.dtype) as Tensor4D;
    console.log(`op: conv2d, inputs: [${x.id}, ${filter.id}], outputs: [${y.id}], attrs: {filterHeight: ${filterHeight}, filterWidth: ${filterWidth}, padLeft: ${padLeft}, padTop: ${padTop}}`);
    return y;
  }

  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    throw Error("not implement");
  }

  conv2dDerFilter(x: Tensor4D, dy: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    throw Error("not implement");
  }

  depthwiseConv2D(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    // const dilationHeight = convInfo.dilationHeight;
    // const dilationWidth = convInfo.dilationWidth;
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;
    const chMul = convInfo.outChannels / convInfo.inChannels;
    const y = Tensor.make(convInfo.outShape, {}, x.dtype) as Tensor4D;
    console.log(`op: depthwiseConv2D, inputs: [${x.id}, ${filter.id}], outputs: [${y.id}], attrs: {filterHeight: ${filterHeight}, filterWidth: ${filterWidth}, padLeft: ${padLeft}, padTop: ${padTop}, chMul: ${chMul}}`);
    return y;
  }

  tile<T extends Tensor>(x: T, reps: number[]): T {
    throw Error("not implement");
  }

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    throw Error("not implement");
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    throw Error("not implement");
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    throw Error("not implement");
  }

  private pool(x: Tensor4D, convInfo: Conv2DInfo, poolType: 'max'|'avg'):
      Tensor4D {
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const y = Tensor.make(convInfo.outShape, {}, x.dtype) as Tensor4D;
    console.log(`op: ${poolType}pool, inputs: [${x.id}], ouputs: [${y.id}] attrs: {strideHeight: ${strideHeight}, strideWidth: ${strideWidth}, filterHeight: ${filterHeight}, filterWidth: ${filterWidth}, padTop: ${padTop}, padLeft: ${padLeft}}`);
    return y;
  }

  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return this.pool(x, convInfo, 'max').toFloat();
  }

  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    throw Error("not implement");
  }

  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    throw Error("not implement");
  }

  cast<T extends Tensor<types.Rank>>(x: T, dtype: DataType): T {
    return backend_util.castTensor(x, dtype, this);
  }

  reshape<T extends Tensor<types.Rank>, R extends types.Rank>(
      x: T, shape: types.ShapeMap[R]): Tensor<R> {
    const y = backend_util.reshapeTensor(x, shape);
    console.log(`op: reshape, inputs: [${x.id}], outputs: [${y.id}]`);
    return y;
  }

  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return this.pool(x, convInfo, 'avg').toFloat();
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    throw Error("not implement");
  }

  resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean): Tensor4D {
    throw Error("not implement");
  }

  resizeNearestNeighbor(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    throw Error("not implement");
  }

  batchNormalization(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D {
    throw Error("not implement");
  }

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    throw Error("not implement");
  }

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    throw Error("not implement");
  }

  oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number):
      Tensor2D {
    throw Error("not implement");
  }

  dispose() {}
}

ENV.registerBackend('webml', () => new MathBackendWebML(), 1 /* priority */);
