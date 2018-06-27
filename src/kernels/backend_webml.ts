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

let webml = require('webml-polyfill');

class DataInfo {
  shape: number[];
  dtype: DataType;
}

class Operation {
  op: string;
  attrs: any;
  inputs: DataId[];
  outputs: DataId[];
}

export class MathBackendWebML implements KernelBackend {
  private data = new WeakMap<DataId, DataTypeMap[DataType]>();
  private dataInfo = new WeakMap<DataId, DataInfo>();
  private operations: Operation[];
  private compiledOps: Operation[];
  private opIndex: number;
  private graphCompiled: boolean;
  private canvas: HTMLCanvasElement;
  private nn: any;
  private model: any;
  private compilation: any;
  private execution: any;

  constructor() {
    if (typeof document !== 'undefined') {
      this.canvas = document.createElement('canvas');
    }
    this.operations = [];
    this.opIndex = 0;
    this.graphCompiled = false;
    console.log(webml);
    this.nn = (navigator as any).ml.getNeuralNetworkContext();
    console.log(this.nn);
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
    await this.compileGraph();
    const data = this.readSync(dataId);
    await this.executeGraph();
    console.log(data);
    return data;
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

  addScalarInt32(operandIndex: number, value: number) {
    const scalarInt32Type = { type: this.nn.INT32 };
    this.model.addOperand(scalarInt32Type);
    this.model.setOperandValue(operandIndex, new Int32Array([value]));
  }

  addScalarFloat32(operandIndex: number, value: number) {
    const scalarInt32Type = { type: this.nn.FLOAT32 };
    this.model.addOperand(scalarInt32Type);
    this.model.setOperandValue(operandIndex, new Float32Array([value]));
  }

  async compileGraph() {
    // reset the op tracing.
    this.opIndex = 0;

    // compiled or nothing to compile
    if (this.graphCompiled || this.operations.length === 0) {
      return;
    }
    console.log('compile graph starts');
    this.model = await this.nn.createModel({ useWebGL2: false });
    let operands = new WeakMap<DataId, number>();
    let graphInputs = new Set<number>();
    let graphOutputs = new Set<number>();
    let operandIndex = 0;
    for (let i = 0; i < this.operations.length; ++i) {
      let op = this.operations[i];
      let opCode;
      let inputs: number[] = [];
      let outputs: number[] = [];
      console.log(op);
      if (op.op === 'reshape') {
        opCode = this.nn.RESHAPE;
        const x = op.inputs[0];
        let xIndex = operands.get(x);
        if (!xIndex) {
          xIndex = operandIndex++;
          const xInfo = this.dataInfo.get(x);
          console.log(`add operand id: ${xIndex}, shape: [${xInfo.shape}], dtype: ${xInfo.dtype}`);
          this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: xInfo.shape });
          operands.set(x, xIndex);
          graphInputs.add(xIndex);
        }
        graphOutputs.delete(xIndex);
        inputs.push(xIndex);

        const shape = op.attrs;
        const shapeIndex = operandIndex++;
        console.log(`add operand id: ${shapeIndex}, shape: [${shape.length}], dtype: int32`);
        console.log(`set operand value id: ${shapeIndex}, value: [${shape}]`);
        this.model.addOperand({ type: this.nn.TENSOR_INT32, dimensions: [shape.length] });
        this.model.setOperandValue(shapeIndex, new Int32Array(shape));
        inputs.push(shapeIndex);

        const y = op.outputs[0];
        const yIndex = operandIndex++;
        const yInfo = this.dataInfo.get(y);
        console.log(`add operand id: ${yIndex}, shape: [${yInfo.shape}], dtype: ${yInfo.dtype}`);
        this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: yInfo.shape });
        operands.set(y, yIndex);
        graphOutputs.add(yIndex);
        outputs.push(yIndex);
      } else if (op.op === 'conv2d' || op.op === 'depthwiseConv2D' || op.op === 'avgpool') {
        switch (op.op) {
          case 'conv2d': {
            opCode = this.nn.CONV_2D;
          } break;
          case 'depthwiseConv2D': {
            opCode = this.nn.DEPTHWISE_CONV_2D;
          } break;
          case 'avgpool': {
            opCode = this.nn.AVERAGE_POOL_2D;
          } break;
        }
        const x = op.inputs[0];
        let xIndex = operands.get(x);
        if (!xIndex) {
          xIndex = operandIndex++;
          const xInfo = this.dataInfo.get(x);
          console.log(`add operand id: ${xIndex}, shape: [${xInfo.shape}], dtype: ${xInfo.dtype}`);
          this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: xInfo.shape });
          operands.set(x, xIndex);
          graphInputs.add(xIndex);
        }
        graphOutputs.delete(xIndex);
        inputs.push(xIndex);

        const convInfo: Conv2DInfo = op.attrs;
        if (op.op === 'conv2d' || op.op === 'depthwiseConv2D') {
          const weights = op.inputs[1];
          const weightsIndex = operandIndex++;
          const weightsInfo = this.dataInfo.get(weights);
          const weightsData = this.data.get(weights);
          // hack on weigths shape from [h, w, in, out] to [out, h, w, in]
          const [height, width, inChannel, outChannel] = weightsInfo.shape;
          const length = inChannel * height * width * outChannel;
          const transposedWeights = new Float32Array(length);
          for (let h = 0; h < height; ++h) {
            for (let w = 0; w < width; ++w) {
              for (let i = 0; i < inChannel; ++i) {
                for (let o = 0; o < outChannel; ++o) {
                  transposedWeights[o * height * width * inChannel + h * width * inChannel + w * inChannel + i] =
                    weightsData[h * width * inChannel * outChannel + w * inChannel * outChannel + i * outChannel + o];
                }
              }
            }
          }
          const transposedShape = [outChannel, height, width, inChannel];
          console.log(`add operand id: ${weightsIndex}, shape: [${transposedShape}], dtype: ${weightsInfo.dtype}`);
          this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: transposedShape });
          console.log(`set operand value id: ${weightsIndex}, value length: ${this.data.get(weights).length}`);
          this.model.setOperandValue(weightsIndex, transposedWeights);
          inputs.push(weightsIndex);

          let biasIndex;
          if (this.operations[i+1].op === 'add') {
            // handle bias
            op = this.operations[++i];
            const bias = op.inputs[1];
            biasIndex = operandIndex++;
            const biasInfo = this.dataInfo.get(bias);
            console.log(`add operand id: ${biasIndex}, shape: [${biasInfo.shape}], dtype: ${biasInfo.dtype}`);
            this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: biasInfo.shape });
            console.log(`set operand value id: ${biasIndex}, value length: ${this.data.get(bias).length}`);
            this.model.setOperandValue(biasIndex, this.data.get(bias));
            inputs.push(biasIndex);
          } else {
            console.warn(`handle zero bias`);
          }
        }

        const paddingLeftIndex = operandIndex++;
        console.log(`add operand id: ${paddingLeftIndex}, shape: [1], dtype: int32`);
        console.log(`set operand value id: ${paddingLeftIndex}, value: ${convInfo.padInfo.left}`);
        this.addScalarInt32(paddingLeftIndex, convInfo.padInfo.left);
        inputs.push(paddingLeftIndex);

        const paddingRightIndex = operandIndex++;
        console.log(`add operand id: ${paddingRightIndex}, shape: [1], dtype: int32`);
        console.log(`set operand value id: ${paddingRightIndex}, value: ${convInfo.padInfo.right}`);
        this.addScalarInt32(paddingRightIndex, convInfo.padInfo.right);
        inputs.push(paddingRightIndex);

        const paddingTopIndex = operandIndex++;
        console.log(`add operand id: ${paddingTopIndex}, shape: [1], dtype: int32`);
        console.log(`set operand value id: ${paddingTopIndex}, value: ${convInfo.padInfo.top}`);
        this.addScalarInt32(paddingTopIndex, convInfo.padInfo.top);
        inputs.push(paddingTopIndex);

        const paddingBottomIndex = operandIndex++;
        console.log(`add operand id: ${paddingBottomIndex}, shape: [1], dtype: int32`);
        console.log(`set operand value id: ${paddingBottomIndex}, value: ${convInfo.padInfo.bottom}`);
        this.addScalarInt32(paddingBottomIndex, convInfo.padInfo.bottom);
        inputs.push(paddingBottomIndex);

        const strideWidthIndex = operandIndex++;
        console.log(`add operand id: ${strideWidthIndex}, shape: [1], dtype: int32`);
        console.log(`set operand value id: ${strideWidthIndex}, value: ${convInfo.strideWidth}`);
        this.addScalarInt32(strideWidthIndex, convInfo.strideWidth);
        inputs.push(strideWidthIndex);

        const strideHeightIndex = operandIndex++;
        console.log(`add operand id: ${strideHeightIndex}, shape: [1], dtype: int32`);
        console.log(`set operand value id: ${strideHeightIndex}, value: ${convInfo.strideHeight}`);
        this.addScalarInt32(strideHeightIndex, convInfo.strideHeight);
        inputs.push(strideHeightIndex);

        if (opCode === this.nn.DEPTHWISE_CONV_2D) {
          const multiplierIndex = operandIndex++;
          this.addScalarInt32(multiplierIndex, 1);
          inputs.push(multiplierIndex);
        } else if (opCode === this.nn.AVERAGE_POOL_2D) {
          const filterWidthIndex = operandIndex++;
          this.addScalarInt32(filterWidthIndex, convInfo.filterWidth);
          inputs.push(filterWidthIndex);
          const filterHeightIndex = operandIndex++;
          this.addScalarInt32(filterHeightIndex, convInfo.filterHeight);
          inputs.push(filterHeightIndex);
        }

        let fuseIndex;
        let fuseCode = this.nn.FUSED_NONE;
        if (this.operations[i+1].op === 'clip') {
          // handle relu fusion
          op = this.operations[++i];
          if (op.attrs.max === 1) {
            fuseCode = this.nn.FUSED_RELU1;
          } else if (op.attrs.max === 6) {
            fuseCode = this.nn.FUSED_RELU6;
          } else {
            fuseCode = this.nn.FUSED_RELU;
          }
        }
        fuseIndex = operandIndex++;
        console.log(`add operand id: ${fuseIndex}, shape: [1], dtype: int32`);
        console.log(`set operand value id: ${fuseIndex}, value: ${fuseCode}`);
        this.addScalarInt32(fuseIndex, fuseCode);
        inputs.push(fuseIndex);

        const y = op.outputs[0];
        const yIndex = operandIndex++;
        const yInfo = this.dataInfo.get(y);
        console.log(`add operand id: ${yIndex}, shape: [${yInfo.shape}], dtype: ${yInfo.dtype}`);
        this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: yInfo.shape });
        operands.set(y, yIndex);
        graphOutputs.add(yIndex);
        outputs.push(yIndex);
      } else if (op.op === 'max' && this.operations[i+1].op === 'reshape' && this.operations[i+2].op === 'substract' &&
          this.operations[i+3].op === 'exp' && this.operations[i+4].op === 'sum' &&
          this.operations[i+5].op === 'log' && this.operations[i+6].op === 'reshape' &&
          this.operations[i+7].op === 'add' && this.operations[i+8].op === 'reshape' &&
          this.operations[i+9].op === 'substract' && this.operations[i+10].op === 'exp') {
        opCode = this.nn.SOFTMAX;
        const x = op.inputs[0];
        let xIndex = operands.get(x);
        if (!xIndex) {
          xIndex = operandIndex++;
          const xInfo = this.dataInfo.get(x);
          console.log(`add operand id: ${xIndex}, shape: [${xInfo.shape}], dtype: ${xInfo.dtype}`);
          this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: xInfo.shape });
          operands.set(x, xIndex);
          graphInputs.add(xIndex);
        }
        graphOutputs.delete(xIndex);
        inputs.push(xIndex);

        const betaIndex = operandIndex++;
        console.log(`add operand id: ${betaIndex}, shape: [1], dtype: float32`);
        console.log(`set operand value id: ${betaIndex}, value: ${1.0}`);
        this.addScalarFloat32(betaIndex, 1.0);
        inputs.push(betaIndex);

        i = i + 10;
        this.operations[i++];

        const y = op.outputs[0];
        const yIndex = operandIndex++;
        const yInfo = this.dataInfo.get(y);
        console.log(`add operand id: ${yIndex}, shape: [${yInfo.shape}], dtype: ${yInfo.dtype}`);
        this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: yInfo.shape });
        operands.set(y, yIndex);
        graphOutputs.add(yIndex);
        outputs.push(yIndex);
      } else {
        console.warn(`unsupported op ${op.op}`);
      }

      if (opCode) {
        console.log(`add operation code: ${opCode}, inputs: [${inputs}], outputs: [${outputs}]`);
        this.model.addOperation(opCode, inputs, outputs);
      }
    }

    console.log(`graph inputs: [${Array.from(graphInputs)}], outputs: [${Array.from(graphOutputs)}]`);
    this.model.identifyInputsAndOutputs(Array.from(graphInputs), Array.from(graphOutputs));
    await this.model.finish();
    this.compilation = await this.model.createCompilation();
    this.compilation.setPreference(this.nn.PREFER_FAST_SINGLE_ANSWER);
    await this.compilation.finish();
    this.execution = await this.compilation.createExecution();
    console.log(this.execution);
    console.log(`compile graph ends`);
    this.graphCompiled = true;
    this.compiledOps = this.operations;
    this.operations = [];
  }

  async executeGraph() {
    if (!this.graphCompiled) {
      console.log('no compiled graph')
      return;
    }
    console.log('execute graph starts');
    // TODO: trace the inputs and outputs
    const inputData = this.data.get(this.compiledOps[0].inputs[0]);
    this.execution.setInput(0, inputData);
    const outputData = this.data.get(this.compiledOps[this.compiledOps.length-1].outputs[0]);
    this.execution.setOutput(0, outputData);
    const error = await this.execution.startCompute();
    if (error) {
      throw new Error(error);
    }
    console.log('execute graph ends');
  }

  traceOp(op: string, attrs: any, inputs: DataId[], outputs: DataId[]) {
    if (!this.graphCompiled) {
      //console.log(`op: ${op}, attrs: ${attrs}, inputs: [${inputs}], outputs: [${outputs}]`);
      this.operations.push({
        op: op,
        attrs: attrs,
        inputs: inputs,
        outputs: outputs
      });
    } else {
      if (this.compiledOps[this.opIndex].op !== op) {
        console.warn(`graph changes ${this.compiledOps[this.opIndex].op} !== ${op}`);
        this.graphCompiled = false;
        this.compiledOps = [];
      } else {
        // console.log(`op ${op} hit cache ${this.opIndex}`);
        this.compiledOps[this.opIndex].inputs = inputs;
        this.compiledOps[this.opIndex].outputs = outputs;
      }
    }
    this.opIndex++;
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
    // console.log(`op: add, inputs: [${a.id}, ${b.id}], outputs: [${c.id}]`);
    this.traceOp('add', null, [a.dataId, b.dataId], [c.dataId]);
    return c;
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    let c = Tensor.make(a.shape, {}, a.dtype);
    // console.log(`op: substract, inputs: [${a.id}, ${b.id}], outputs: [${c.id}]`);
    this.traceOp('substract', null, [a.dataId, b.dataId], [c.dataId]);
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
    // console.log(`op: sum, inputs: [${x.id}], outputs: [${y.id}], axes: [${axes}]`);
    this.traceOp('sum', axes, [x.dataId], [y.dataId]);
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
    // console.log(`op: max, inputs: [${x.id}], outputs: [${y.id}], axes: [${axes}]`)
    this.traceOp('max', axes, [x.dataId], [y.dataId]);
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
    // console.log(`op: exp, inputs: [${x.id}], outputs: [${y.id}]`);
    this.traceOp('exp', null, [x.dataId], [y.dataId]);
    return y;
  }

  expm1<T extends Tensor>(x: T): T {
    throw Error("not implement");
  }

  log<T extends Tensor>(x: T): T {
    let y = Tensor.make(x.shape, {}, x.dtype) as T;
    // console.log(`op: log, inputs: [${x.id}], outputs: [${y.id}]`);
    this.traceOp('log', null, [x.dataId], [y.dataId]);
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
    // console.log(`op: clip, inputs: [${x.id}], outputs: [${y.id}], min: ${min}, max: ${max}`);
    this.traceOp('clip', {min: min, max: max}, [x.dataId], [y.dataId]);
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
    const y = Tensor.make(convInfo.outShape, {}, x.dtype) as Tensor4D;
    // console.log(`op: conv2d, inputs: [${x.id}, ${filter.id}], outputs: [${y.id}], convInfo: ${JSON.stringify(convInfo)}`);
    this.traceOp('conv2d', convInfo, [x.dataId, filter.dataId], [y.dataId]);
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
    const y = Tensor.make(convInfo.outShape, {}, x.dtype) as Tensor4D;
    // console.log(`op: depthwiseConv2D, inputs: [${x.id}, ${filter.id}], outputs: [${y.id}], convInfo: ${JSON.stringify(convInfo)}`);
    this.traceOp('depthwiseConv2D', convInfo, [x.dataId, filter.dataId], [y.dataId]);
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
    const y = Tensor.make(convInfo.outShape, {}, x.dtype) as Tensor4D;
    // console.log(`op: ${poolType}pool, inputs: [${x.id}], ouputs: [${y.id}], convInfo: ${JSON.stringify(convInfo)}`);
    this.traceOp(poolType+'pool', convInfo, [x.dataId], [y.dataId]);
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
    const y = Tensor.make(shape, {}, x.dtype);
    // console.log(`op: reshape, inputs: [${x.id}], outputs: [${y.id}], attrs: ${shape}`);
    this.traceOp('reshape', shape, [x.dataId], [y.dataId]);
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
