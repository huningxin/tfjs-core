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

import {TimingInfo} from '../engine';
import {ENV} from '../environment';
import * as axis_util from '../ops/axis_util';
import {Conv2DInfo} from '../ops/conv_util';
import * as ops from '../ops/ops';
import * as reduce_util from '../ops/reduce_util';
import {getStridedSlicedInfo} from '../ops/slice_util';
// tslint:disable-next-line:max-line-length
import {DataId, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import * as types from '../types';
import {DataType, DataTypeMap, RecursiveArray, TypedArray} from '../types';
import * as util from '../util';

import {KernelBackend} from './backend';
import * as backend_util from './backend_util';
import {ArgMinMaxProgram} from './webgl/argminmax_gpu';
import {AvgPool2DBackpropProgram} from './webgl/avg_pool_backprop_gpu';
import {BatchNormProgram} from './webgl/batchnorm_gpu';
import * as binaryop_gpu from './webgl/binaryop_gpu';
import {BinaryOpProgram} from './webgl/binaryop_gpu';
import {ClipProgram} from './webgl/clip_gpu';
import {ConcatProgram} from './webgl/concat_gpu';
// tslint:disable-next-line:max-line-length
import {Conv2DDerFilterProgram, Conv2DDerInputProgram} from './webgl/conv_backprop_gpu';
import {Conv2DProgram} from './webgl/conv_gpu';
import {DepthwiseConv2DProgram} from './webgl/conv_gpu_depthwise';
import {CumSumProgram} from './webgl/cumsum_gpu';
import {FromPixelsProgram} from './webgl/from_pixels_gpu';
import {GatherProgram} from './webgl/gather_gpu';
import {GPGPUContext} from './webgl/gpgpu_context';
import * as gpgpu_math from './webgl/gpgpu_math';
import {GPGPUBinary, GPGPUProgram, TensorData} from './webgl/gpgpu_math';
import * as gpgpu_util from './webgl/gpgpu_util';
import {WhereProgram} from './webgl/logical_gpu';
import {LRNProgram} from './webgl/lrn_gpu';
import {MaxPool2DBackpropProgram} from './webgl/max_pool_backprop_gpu';
import {MatMulProgram} from './webgl/mulmat_gpu';
import {MultinomialProgram} from './webgl/multinomial_gpu';
import {OneHotProgram} from './webgl/onehot_gpu';
import {PadProgram} from './webgl/pad_gpu';
import {Pool2DProgram} from './webgl/pool_gpu';
import {ReduceProgram} from './webgl/reduce_gpu';
// tslint:disable-next-line:max-line-length
import {ResizeBilinearBackpropProgram} from './webgl/resize_bilinear_backprop_gpu';
import {ResizeBilinearProgram} from './webgl/resize_bilinear_gpu';
// tslint:disable-next-line:max-line-length
import {ResizeNearestNeighborProgram} from './webgl/resize_nearest_neighbor_gpu';
import {ReverseProgram} from './webgl/reverse_gpu';
import {SliceProgram} from './webgl/slice_gpu';
import {StridedSliceProgram} from './webgl/strided_slice_gpu';
import {TextureData, TextureType} from './webgl/tex_util';
import {TextureManager} from './webgl/texture_manager';
import {TileProgram} from './webgl/tile_gpu';
import {TransposeProgram} from './webgl/transpose_gpu';
import * as unary_op from './webgl/unaryop_gpu';
import {UnaryOpProgram} from './webgl/unaryop_gpu';
import {WebGLQuery} from './webgl/webgl_types';
import * as webgl_util from './webgl/webgl_util';

let webml = require('webml-polyfill');

type TimerNode = RecursiveArray<Promise<number>>|Promise<number>;
export interface CPUTimerQuery {
  startMs: number;
  endMs?: number;
}

export interface WebGLTimingInfo extends TimingInfo {
  uploadWaitMs: number;
  downloadWaitMs: number;
}

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

export class MathBackendWebGL implements KernelBackend {
  private texData = new WeakMap<DataId, TextureData>();
  private canvas: HTMLCanvasElement;

  private programTimersStack: TimerNode[];
  private activeTimers: TimerNode[];
  // Accumulated time spent (including blocking) in uploading data to webgl.
  private uploadWaitMs = 0;
  // Accumulated time spent (including blocking in downloading data from webgl.
  private downloadWaitMs = 0;

  private useWebML = false;
  private dataInfo = new WeakMap<DataId, DataInfo>();
  private operations: Operation[];
  private compiledOps: Operation[];
  private opIndex: number;
  private graphCompiled: boolean;
  private nn: any;
  private model: any;
  private compilation: any;
  private execution: any;

  setEnableWebML(enabled: boolean): void {
    this.useWebML = enabled;
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
    this.model = await this.nn.createModel({ useWebGL2: true });
    let operands = new WeakMap<DataId, number>();
    let graphInputs = new Set<number>();
    let graphOutputs = new Set<number>();
    let operandIndex = 0;
    for (let i = 0; i < this.operations.length; ++i) {
      let op = this.operations[i];
      let opCode;
      let inputs: number[] = [];
      let outputs: number[] = [];
      // console.log(op);
      if (op.op === 'reshape' && this.operations[i+1].op !== 'max') {
        opCode = this.nn.RESHAPE;
        const x = op.inputs[0];
        let xIndex = operands.get(x);
        if (!xIndex) {
          xIndex = operandIndex++;
          const xInfo = this.dataInfo.get(x);
          // console.log(`add operand id: ${xIndex}, shape: [${xInfo.shape}], dtype: ${xInfo.dtype}`);
          this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: xInfo.shape });
          operands.set(x, xIndex);
          graphInputs.add(xIndex);
        }
        graphOutputs.delete(xIndex);
        inputs.push(xIndex);

        const shape = op.attrs;
        const shapeIndex = operandIndex++;
        // console.log(`add operand id: ${shapeIndex}, shape: [${shape.length}], dtype: int32`);
        // console.log(`set operand value id: ${shapeIndex}, value: [${shape}]`);
        this.model.addOperand({ type: this.nn.TENSOR_INT32, dimensions: [shape.length] });
        this.model.setOperandValue(shapeIndex, new Int32Array(shape));
        inputs.push(shapeIndex);

        const y = op.outputs[0];
        const yIndex = operandIndex++;
        const yInfo = this.dataInfo.get(y);
        // console.log(`add operand id: ${yIndex}, shape: [${yInfo.shape}], dtype: ${yInfo.dtype}`);
        this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: yInfo.shape });
        operands.set(y, yIndex);
        graphOutputs.add(yIndex);
        outputs.push(yIndex);
        this.compiledOps.push(op);
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
        this.compiledOps.push(op);
        const x = op.inputs[0];
        let xIndex = operands.get(x);
        if (!xIndex) {
          xIndex = operandIndex++;
          const xInfo = this.dataInfo.get(x);
          // console.log(`add operand id: ${xIndex}, shape: [${xInfo.shape}], dtype: ${xInfo.dtype}`);
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
          const weightsData = this.texData.get(weights).values;
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
          // console.log(`add operand id: ${weightsIndex}, shape: [${transposedShape}], dtype: ${weightsInfo.dtype}`);
          this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: transposedShape });
          // console.log(`set operand value id: ${weightsIndex}, value length: ${this.texData.get(weights).values.length}`);
          this.model.setOperandValue(weightsIndex, transposedWeights);
          inputs.push(weightsIndex);

          let biasIndex;
          if (this.operations[i+1].op === 'add') {
            // handle bias
            op = this.operations[++i];
            this.compiledOps.push(op);
            const bias = op.inputs[1];
            biasIndex = operandIndex++;
            const biasInfo = this.dataInfo.get(bias);
            // console.log(`add operand id: ${biasIndex}, shape: [${biasInfo.shape}], dtype: ${biasInfo.dtype}`);
            this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: biasInfo.shape });
            // console.log(`set operand value id: ${biasIndex}, value length: ${this.texData.get(bias).values.length}`);
            this.model.setOperandValue(biasIndex, this.texData.get(bias).values);
            inputs.push(biasIndex);
          } else {
            console.warn(`handle zero bias`);
          }
        }

        const paddingLeftIndex = operandIndex++;
        // console.log(`add operand id: ${paddingLeftIndex}, shape: [1], dtype: int32`);
        // console.log(`set operand value id: ${paddingLeftIndex}, value: ${convInfo.padInfo.left}`);
        this.addScalarInt32(paddingLeftIndex, convInfo.padInfo.left);
        inputs.push(paddingLeftIndex);

        const paddingRightIndex = operandIndex++;
        // console.log(`add operand id: ${paddingRightIndex}, shape: [1], dtype: int32`);
        // console.log(`set operand value id: ${paddingRightIndex}, value: ${convInfo.padInfo.right}`);
        this.addScalarInt32(paddingRightIndex, convInfo.padInfo.right);
        inputs.push(paddingRightIndex);

        const paddingTopIndex = operandIndex++;
        // console.log(`add operand id: ${paddingTopIndex}, shape: [1], dtype: int32`);
        // console.log(`set operand value id: ${paddingTopIndex}, value: ${convInfo.padInfo.top}`);
        this.addScalarInt32(paddingTopIndex, convInfo.padInfo.top);
        inputs.push(paddingTopIndex);

        const paddingBottomIndex = operandIndex++;
        // console.log(`add operand id: ${paddingBottomIndex}, shape: [1], dtype: int32`);
        // console.log(`set operand value id: ${paddingBottomIndex}, value: ${convInfo.padInfo.bottom}`);
        this.addScalarInt32(paddingBottomIndex, convInfo.padInfo.bottom);
        inputs.push(paddingBottomIndex);

        const strideWidthIndex = operandIndex++;
        // console.log(`add operand id: ${strideWidthIndex}, shape: [1], dtype: int32`);
        // console.log(`set operand value id: ${strideWidthIndex}, value: ${convInfo.strideWidth}`);
        this.addScalarInt32(strideWidthIndex, convInfo.strideWidth);
        inputs.push(strideWidthIndex);

        const strideHeightIndex = operandIndex++;
        // console.log(`add operand id: ${strideHeightIndex}, shape: [1], dtype: int32`);
        // console.log(`set operand value id: ${strideHeightIndex}, value: ${convInfo.strideHeight}`);
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
          this.compiledOps.push(op);
          if (op.attrs.max === 1) {
            fuseCode = this.nn.FUSED_RELU1;
          } else if (op.attrs.max === 6) {
            fuseCode = this.nn.FUSED_RELU6;
          } else {
            fuseCode = this.nn.FUSED_RELU;
          }
        }
        fuseIndex = operandIndex++;
        // console.log(`add operand id: ${fuseIndex}, shape: [1], dtype: int32`);
        // console.log(`set operand value id: ${fuseIndex}, value: ${fuseCode}`);
        this.addScalarInt32(fuseIndex, fuseCode);
        inputs.push(fuseIndex);

        const y = op.outputs[0];
        const yIndex = operandIndex++;
        const yInfo = this.dataInfo.get(y);
        // console.log(`add operand id: ${yIndex}, shape: [${yInfo.shape}], dtype: ${yInfo.dtype}`);
        this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: yInfo.shape });
        operands.set(y, yIndex);
        graphOutputs.add(yIndex);
        outputs.push(yIndex);
      } else if (op.op === 'reshape' &&
          this.operations[i+1].op === 'max' &&
          this.operations[i+2].op === 'max' &&
          this.operations[i+3].op === 'reshape' &&
          this.operations[i+4].op === 'reshape' &&
          this.operations[i+5].op === 'substract' &&
          this.operations[i+6].op === 'exp' &&
          this.operations[i+7].op === 'reshape' &&
          this.operations[i+8].op === 'sum' &&
          this.operations[i+9].op === 'sum' &&
          this.operations[i+10].op === 'reshape' &&
          this.operations[i+11].op === 'log' &&
          this.operations[i+12].op === 'reshape' &&
          this.operations[i+13].op === 'add' &&
          this.operations[i+14].op === 'reshape' &&
          this.operations[i+15].op === 'substract' &&
          this.operations[i+16].op === 'exp') {
        opCode = this.nn.SOFTMAX;
        const x = op.inputs[0];
        let xIndex = operands.get(x);
        if (!xIndex) {
          xIndex = operandIndex++;
          const xInfo = this.dataInfo.get(x);
          // console.log(`add operand id: ${xIndex}, shape: [${xInfo.shape}], dtype: ${xInfo.dtype}`);
          this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: xInfo.shape });
          operands.set(x, xIndex);
          graphInputs.add(xIndex);
        }
        graphOutputs.delete(xIndex);
        inputs.push(xIndex);

        const betaIndex = operandIndex++;
        // console.log(`add operand id: ${betaIndex}, shape: [1], dtype: float32`);
        // console.log(`set operand value id: ${betaIndex}, value: ${1.0}`);
        this.addScalarFloat32(betaIndex, 1.0);
        inputs.push(betaIndex);

        const y = this.operations[i+16].outputs[0];
        const yIndex = operandIndex++;
        const yInfo = this.dataInfo.get(y);
        // console.log(`add operand id: ${yIndex}, shape: [${yInfo.shape}], dtype: ${yInfo.dtype}`);
        this.model.addOperand({ type: this.nn.TENSOR_FLOAT32, dimensions: yInfo.shape });
        operands.set(y, yIndex);
        graphOutputs.add(yIndex);
        outputs.push(yIndex);
        for (let j = i; j < i + 17; j++) {
          this.compiledOps.push(this.operations[j]);
        }
        i = i + 16;
      } else {
        console.warn(`unsupported op ${op.op}`);
        continue;
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
    this.operations = [];
  }

  async executeGraph() {
    if (!this.graphCompiled) {
      console.log('no compiled graph')
      return;
    }
    console.log('execute graph starts');
    // TODO: trace the inputs and outputs
    const inputData = this.readSync(this.compiledOps[0].inputs[0]);
    this.execution.setInput(0, inputData);
    const outputDataId = this.compiledOps[this.compiledOps.length-1].outputs[0];
    if (this.texData.get(outputDataId).values === null) {
      const size = util.sizeFromShape(this.texData.get(outputDataId).shape);
      const values = util.getTypedArrayFromDType(this.texData.get(outputDataId).dtype, size);
      // console.log(`getTypedArrayFromDType ${size} ${dataInfo.dtype}`);
      this.texData.get(outputDataId).values = values;
    }
    const outputData = this.texData.get(outputDataId).values;
    this.execution.setOutput(0, outputData);
    const error = await this.execution.startCompute();
    if (error) {
      throw new Error(error);
    }
    console.log('execute graph ends');
  }

  traceOp(op: string, attrs: any, inputs: DataId[], outputs: DataId[]): boolean {
    if (!this.useWebML) {
      return false;
    }
    if (!this.graphCompiled) {
      this.operations.push({
        op: op,
        attrs: attrs,
        inputs: inputs,
        outputs: outputs
      });
      console.log(`[${this.operations.length}] op: ${op}, attrs: ${attrs}, inputs: [${inputs}], outputs: [${outputs}]`);
      return false;
    } else {
      if (op === 'substract' && this.opIndex === 0)
        return false;

      if (this.compiledOps[this.opIndex].op !== op) {
        console.warn(`graph changes ${this.compiledOps[this.opIndex].op} !== ${op}`);
        this.graphCompiled = false;
        this.compiledOps = [];
        return false;
      } else {
        console.log(`op ${op} hit cache ${this.opIndex}`);
        this.compiledOps[this.opIndex].inputs = inputs;
        this.compiledOps[this.opIndex].outputs = outputs;
        this.opIndex++;
        return true;
      }
    }
  }

  register(dataId: DataId, shape: number[], dtype: DataType): void {
    // console.log(`register ${dataId} ${shape} ${dtype}`);
    if (this.texData.has(dataId)) {
      throw new Error('Data buffer is already registered');
    }
    this.texData.set(dataId, {
      shape,
      dtype,
      values: null,
      texture: null,
      texShape: null,
      texType: TextureType.FLOAT
    });
    this.dataInfo.set(dataId, {shape: shape, dtype: dtype});
  }
  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    if (pixels == null) {
      throw new Error('MathBackendWebGL.writePixels(): pixels can not be null');
    }
    const texShape: [number, number] = [pixels.height, pixels.width];
    const outShape = [pixels.height, pixels.width, numChannels];

    if (pixels instanceof HTMLVideoElement) {
      if (this.canvas == null) {
        throw new Error(
            'Can\'t read pixels from HTMLImageElement outside ' +
            'the browser.');
      }
      this.canvas.width = pixels.width;
      this.canvas.height = pixels.height;
      this.canvas.getContext('2d').drawImage(
          pixels, 0, 0, pixels.width, pixels.height);
      pixels = this.canvas;
    }
    const tempPixelArray = Tensor.make(texShape, {}, 'int32');

    // This is a byte texture with pixels.
    this.texData.get(tempPixelArray.dataId).texType = TextureType.UNSIGNED_BYTE;
    this.gpgpu.uploadPixelDataToTexture(
        this.getTexture(tempPixelArray.dataId), pixels);
    const program = new FromPixelsProgram(outShape);
    const res = this.compileAndRun(program, [tempPixelArray]);

    tempPixelArray.dispose();

    return res as Tensor3D;
  }
  write(dataId: DataId, values: TypedArray): void {
    console.log(`write ${dataId} ${values.length}`);
    if (values == null) {
      throw new Error('MathBackendWebGL.write(): values can not be null');
    }
    this.throwIfNoData(dataId);

    const texData = this.texData.get(dataId);
    const {texture, texShape, texType} = texData;
    if (texture != null) {
      // Release the old texture.
      this.textureManager.releaseTexture(texture, texShape, texType);
      texData.texture = null;
      texData.texShape = null;
    }
    texData.values = values;

    if (!this.delayedStorage) {
      this.uploadToGPU(dataId);
    }
  }
  readSync(dataId: DataId): TypedArray {
    // console.log(`readSync ${dataId}`);
    this.throwIfNoData(dataId);
    const texData = this.texData.get(dataId);
    const {texture, values, texShape} = texData;
    if (values != null) {
      this.cacheOnCPU(dataId);
      return values;
    }
    const shouldTimeProgram = this.activeTimers != null;
    let start: number;
    if (shouldTimeProgram) {
      start = performance.now();
    }
    const float32Values =
        this.gpgpu.downloadMatrixFromTexture(texture, texShape[0], texShape[1]);
    if (shouldTimeProgram) {
      this.downloadWaitMs += performance.now() - start;
    }
    this.cacheOnCPU(dataId, float32Values);
    return texData.values;
  }
  async read(dataId: DataId): Promise<TypedArray> {
    // console.log(`read ${dataId}`);
    if (this.useWebML) {
      await this.compileGraph();
      await this.executeGraph();
      const data = this.texData.get(dataId).values;
      // console.log(data);
      return data;
    }

    this.throwIfNoData(dataId);
    const texData = this.texData.get(dataId);
    const {texture, values, texShape} = texData;
    if (values != null) {
      this.cacheOnCPU(dataId);
      return values;
    }

    if (ENV.get('WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED')) {
      const float32Values = await this.gpgpu.downloadMatrixFromTextureAsync(
          texture, texShape[0], texShape[1]);
      this.cacheOnCPU(dataId, float32Values);
      return texData.values;
    }

    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 0) {
      return this.readSync(dataId);
    }

    // Construct an empty query. We're just interested in getting a callback
    // when the GPU command queue has executed until this point in time.
    await this.gpgpu.runQuery(() => {});
    return this.readSync(dataId);
  }

  async time(f: () => void): Promise<WebGLTimingInfo> {
    const oldActiveTimers = this.activeTimers;
    const newActiveTimers: TimerNode[] = [];

    let outerMostTime = false;
    if (this.programTimersStack == null) {
      this.programTimersStack = newActiveTimers;
      outerMostTime = true;
    } else {
      this.activeTimers.push(newActiveTimers);
    }
    this.activeTimers = newActiveTimers;

    f();

    const flattenedActiveTimers = util.flatten(this.activeTimers);
    this.activeTimers = oldActiveTimers;

    if (outerMostTime) {
      this.programTimersStack = null;
    }

    const kernelMs = await Promise.all(flattenedActiveTimers).then(results => {
      let sum = 0;
      results.forEach(result => sum += result);
      return sum;
    });
    const res: WebGLTimingInfo = {
      uploadWaitMs: this.uploadWaitMs,
      downloadWaitMs: this.downloadWaitMs,
      kernelMs,
      wallMs: null  // will be filled by the engine
    };
    this.uploadWaitMs = 0;
    this.downloadWaitMs = 0;
    return res;
  }
  memory() {
    return {unreliable: false};
  }

  private startTimer(): WebGLQuery|CPUTimerQuery {
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
      return this.gpgpu.beginQuery();
    }
    return {startMs: performance.now(), endMs: null};
  }

  private endTimer(query: WebGLQuery|CPUTimerQuery): WebGLQuery|
      {startMs: number, endMs: number} {
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
      this.gpgpu.endQuery();
      return query;
    }
    (query as CPUTimerQuery).endMs = performance.now();
    return query;
  }

  private async getQueryTime(query: WebGLQuery|CPUTimerQuery): Promise<number> {
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
      return this.gpgpu.pollQueryTime(query);
    }
    const timerQuery = query as CPUTimerQuery;
    return timerQuery.endMs - timerQuery.startMs;
  }

  disposeData(dataId: DataId): void {
    if (this.texData.has(dataId)) {
      const {texture, texShape, texType} = this.texData.get(dataId);
      if (texture != null) {
        this.textureManager.releaseTexture(texture, texShape, texType);
      }
      this.texData.delete(dataId);
    }
  }

  getTexture(dataId: DataId): WebGLTexture {
    this.uploadToGPU(dataId);
    return this.texData.get(dataId).texture;
  }

  getTextureData(dataId: DataId): TextureData {
    this.uploadToGPU(dataId);
    return this.texData.get(dataId);
  }

  private textureManager: TextureManager;
  private binaryCache: {[key: string]: GPGPUBinary} = {};
  private gpgpuCreatedLocally: boolean;

  constructor(private gpgpu?: GPGPUContext, private delayedStorage = true) {
    if (ENV.get('WEBGL_VERSION') < 1) {
      throw new Error('WebGL is not supported on this device');
    }
    if (typeof document !== 'undefined') {
      this.canvas = document.createElement('canvas');
    }
    if (gpgpu == null) {
      this.gpgpu = new GPGPUContext(gpgpu_util.createWebGLContext(this.canvas));
      this.gpgpuCreatedLocally = true;
    } else {
      this.gpgpuCreatedLocally = false;
    }

    this.textureManager = new TextureManager(this.gpgpu);

    this.operations = [];
    this.compiledOps = [];
    this.opIndex = 0;
    this.graphCompiled = false;
    console.log(webml);
    this.nn = (navigator as any).ml.getNeuralNetworkContext();
    // console.log(this.nn);
  }

  getGPGPUContext(): GPGPUContext {
    return this.gpgpu;
  }
  getCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[],
      beginMask: number, endMask: number): T {
    const [beginIndex, size] =
        getStridedSlicedInfo(x.shape, begin, end, strides, beginMask, endMask);

    if (size.some(axis => axis === 0)) {
      return ops.tensor([], size) as T;
    }

    const program = new StridedSliceProgram(beginIndex, strides, size);
    return this.compileAndRun(program, [x]);
  }

  reverse<T extends Tensor>(x: T, axis: number[]): T {
    const program = new ReverseProgram(x.shape, axis);
    return this.compileAndRun(program, [x]);
  }

  // Concats 2d tensors along axis=1. See comments in MathBackend.concat().
  concat(a: Tensor2D, b: Tensor2D): Tensor2D {
    const program = new ConcatProgram(a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  neg<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.NEG);
    return this.compileAndRun(program, [x]) as T;
  }

  matMul(a: Tensor2D, b: Tensor2D, transposeA: boolean, transposeB: boolean):
      Tensor2D {
    const program = new MatMulProgram(a.shape, b.shape, transposeA, transposeB);
    return this.compileAndRun<Tensor2D, Tensor2D>(program, [a, b]);
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.MUL, a.shape, b.shape);
    const output =
        this.makeOutputArray(
            program.outputShape, types.upcastType(a.dtype, b.dtype)) as Tensor;
    return this.compileAndRun(program, [a, b], output) as Tensor;
  }

  batchNormalization(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D {
    const inputs = [x, mean, variance];

    let offsetShape = null;
    if (offset != null) {
      offsetShape = offset.shape;
      inputs.push(offset);
    }

    let scaleShape = null;
    if (scale != null) {
      scaleShape = scale.shape;
      inputs.push(scale);
    }

    const program = new BatchNormProgram(
        x.shape, mean.shape, variance.shape, offsetShape, scaleShape,
        varianceEpsilon);
    return this.compileAndRun(program, inputs);
  }

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    const program = new LRNProgram(x.shape, radius, bias, alpha, beta);
    return this.compileAndRun(program, [x]);
  }

  tile<T extends Tensor>(x: T, reps: number[]): T {
    const program = new TileProgram(x.shape, reps);
    return this.compileAndRun(program, [x]);
  }

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    const program = new PadProgram(x.shape, paddings, constantValue);
    return this.compileAndRun(program, [x]);
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    const program = new TransposeProgram(x.shape, perm);
    return this.compileAndRun(program, [x]);
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    const program = new GatherProgram(x.shape, indices.size, axis);
    return this.compileAndRun(program, [x, indices]);
  }

  private reduce(x: Tensor2D, reduceType: 'max'|'min'|'sum', dtype: DataType):
      Tensor2D {
    const batchSize = x.shape[0];
    const inSize = x.shape[1];
    const windowSize = reduce_util.computeOptimalWindowSize(inSize);
    const reduceInfo = {windowSize, inSize, batchSize};
    const program = new ReduceProgram(reduceInfo, reduceType);
    const [rows, cols] = program.outputShape;
    const output = this.makeOutputArray<Tensor2D>([rows, cols], dtype);
    if (this.traceOp(reduceType, null, [x.dataId], [output.dataId])) {
      if (output.shape[1] === 1) {
        return output;
      }
      return this.reduce(output, reduceType, dtype);
    }
    this.compileAndRun(program, [x], output);
    // No need to run another GPGPU program.
    if (output.shape[1] === 1) {
      return output;
    }
    return this.reduce(output, reduceType, dtype);
  }

  private argReduce(
      x: Tensor2D, reduceType: 'max'|'min',
      bestIndicesA: Tensor2D = null): Tensor2D {
    let batchSize = x.shape[0];
    let inSize = x.shape[1];
    if (bestIndicesA != null) {
      batchSize = bestIndicesA.shape[0];
      inSize = bestIndicesA.shape[1];
    }
    const windowSize = reduce_util.computeOptimalWindowSize(inSize);
    const reduceInfo = {windowSize, inSize, batchSize};
    const program =
        new ArgMinMaxProgram(reduceInfo, reduceType, bestIndicesA == null);
    const [rows, cols] = program.outputShape;
    const output = this.makeOutputArray<Tensor2D>([rows, cols], 'int32');
    const inputs = [x];
    if (bestIndicesA != null) {
      inputs.push(bestIndicesA);
    }
    this.compileAndRun(program, inputs, output);
    // No need to run another GPGPU program.
    if (output.shape[1] === 1) {
      return output;
    }
    return this.argReduce(x, reduceType, output);
  }

  sum(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    const outputDType = types.sumOutType(x.dtype);
    return this.reduce(a2D, 'sum', outputDType).reshape(outShape);
  }

  argMin(x: Tensor, axis: number): Tensor {
    const axes = [axis];
    axis_util.assertAxesAreInnerMostDims('argMin', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.argReduce(a2D, 'min').reshape(outShape);
  }

  argMax(x: Tensor, axis: number): Tensor {
    const axes = [axis];
    axis_util.assertAxesAreInnerMostDims('argMax', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.argReduce(a2D, 'max').reshape(outShape);
  }

  cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean):
      Tensor {
    if (axis !== x.rank - 1) {
      throw new Error(
          `WebGL cumsum shader expects an inner-most axis=${x.rank - 1} ` +
          `but got axis=${axis}`);
    }
    const program = new CumSumProgram(x.shape, exclusive, reverse);
    return this.compileAndRun(program, [x]);
  }

  equal(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  notEqual(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.NOT_EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  less(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.LESS, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  lessEqual(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.LESS_EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  greater(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.GREATER, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  greaterEqual(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.GREATER_EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  logicalNot<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LOGICAL_NOT);
    return this.compileAndRun(program, [x]) as T;
  }

  logicalAnd(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.LOGICAL_AND, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  logicalOr(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.LOGICAL_OR, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, 'bool');
    return this.compileAndRun(program, [a, b], output);
  }

  where(condition: Tensor, a: Tensor, b: Tensor, dtype: DataType): Tensor {
    const program = new WhereProgram(condition.rank, a.shape, a.rank);
    const output = this.makeOutputArray(program.outputShape, dtype);
    return this.compileAndRun(program, [condition, a, b], output);
  }

  topKValues<T extends Tensor>(x: T, k: number): Tensor1D {
    throw new Error('topKValues GPU not yet implemented!');
  }

  topKIndices(x: Tensor, k: number): Tensor1D {
    throw new Error('topKIndices GPU not yet implemented!');
  }

  min(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('min', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, 'min', a2D.dtype).reshape(outShape);
  }

  minimum(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.MIN, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  mod(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.MOD, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  max(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('max', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, 'max', a2D.dtype).reshape(outShape);
  }

  maximum(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.MAX, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  squaredDifference(a: Tensor, b: Tensor): Tensor {
    const program =
        new BinaryOpProgram(binaryop_gpu.SQUARED_DIFFERENCE, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  divide(a: Tensor, b: Tensor): Tensor {
    let op: string;
    let outputDtype: 'float32'|'int32';
    if (a.dtype === 'int32' && b.dtype === 'int32') {
      op = binaryop_gpu.INT_DIV;
      outputDtype = 'int32';
    } else {
      op = binaryop_gpu.DIV;
      outputDtype = 'float32';
    }

    const program = new BinaryOpProgram(op, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, outputDtype);
    return this.compileAndRun<Tensor, Tensor>(program, [a, b], output);
  }

  add(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.ADD, a.shape, b.shape);
    const output =
        this.makeOutputArray(
            program.outputShape, types.upcastType(a.dtype, b.dtype)) as Tensor;
    if (this.traceOp('add', null, [a.dataId, b.dataId], [output.dataId]))
          return output;
    return this.compileAndRun<Tensor, Tensor>(program, [a, b], output);
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    const program = new BinaryOpProgram(binaryop_gpu.SUB, a.shape, b.shape);
    const output =
        this.makeOutputArray(
            program.outputShape, types.upcastType(a.dtype, b.dtype)) as Tensor;
    if (this.traceOp('substract', null, [a.dataId, b.dataId], [output.dataId]))
          return output;
    return this.compileAndRun<Tensor, Tensor>(program, [a, b], output);
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    const program = new BinaryOpProgram(binaryop_gpu.POW, a.shape, b.shape);
    const output =
        this.makeOutputArray(
            program.outputShape, types.upcastType(a.dtype, b.dtype)) as T;
    return this.compileAndRun<Tensor, T>(program, [a, b], output);
  }

  ceil<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.CEIL);
    return this.compileAndRun(program, [x]) as T;
  }

  floor<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.FLOOR);
    return this.compileAndRun(program, [x]) as T;
  }

  sign<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIGN);
    return this.compileAndRun(program, [x]) as T;
  }

  round<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ROUND);
    return this.compileAndRun(program, [x]) as T;
  }

  exp<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.EXP);
    const output = this.makeOutputArray(program.outputShape, x.dtype) as T;
    if (this.traceOp('exp', null, [x.dataId], [output.dataId]))
      return output as T;
    return this.compileAndRun(program, [x], output) as T;
  }

  expm1<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.EXPM1);
    return this.compileAndRun(program, [x]) as T;
  }

  log<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LOG);
    const output = this.makeOutputArray(program.outputShape, x.dtype) as T;
    if (this.traceOp('log', null, [x.dataId], [output.dataId]))
      return output as T;
    return this.compileAndRun(program, [x], output) as T;
  }

  log1p<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LOG1P);
    return this.compileAndRun(program, [x]) as T;
  }

  sqrt<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SQRT);
    return this.compileAndRun(program, [x]) as T;
  }

  rsqrt<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.RSQRT);
    return this.compileAndRun(program, [x]) as T;
  }

  square<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SQUARE);
    return this.compileAndRun(program, [x]) as T;
  }

  reciprocal<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.RECIPROCAL);
    return this.compileAndRun(program, [x]) as T;
  }

  relu<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.RELU);
    return this.compileAndRun(program, [x]) as T;
  }

  elu<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ELU);
    return this.compileAndRun(program, [x]) as T;
  }

  eluDer<T extends Tensor>(dy: T, y: T): T {
    const program =
        new BinaryOpProgram(binaryop_gpu.ELU_DER, dy.shape, y.shape);
    return this.compileAndRun(program, [dy, y]) as T;
  }

  selu<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SELU);
    return this.compileAndRun(program, [x]) as T;
  }

  int<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TO_INT);
    const output = this.makeOutputArray(program.outputShape, 'int32');
    return this.compileAndRun(program, [x], output) as T;
  }

  clip<T extends Tensor>(x: T, min: number, max: number): T {
    const program = new ClipProgram(x.shape, min, max);
    const output = this.makeOutputArray(program.outputShape, x.dtype) as T;
    if (this.traceOp('clip', {min: min, max: max}, [x.dataId], [output.dataId]))
      return output as T;
    return this.compileAndRun(program, [x], output) as T;
  }

  abs<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ABS);
    return this.compileAndRun(program, [x]) as T;
  }

  sigmoid<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIGMOID);
    return this.compileAndRun(program, [x]) as T;
  }

  softplus<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SOFTPLUS);
    return this.compileAndRun(program, [x]) as T;
  }

  sin<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIN);
    return this.compileAndRun(program, [x]) as T;
  }

  cos<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.COS);
    return this.compileAndRun(program, [x]) as T;
  }

  tan<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TAN);
    return this.compileAndRun(program, [x]) as T;
  }

  asin<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ASIN);
    return this.compileAndRun(program, [x]) as T;
  }

  acos<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ACOS);
    return this.compileAndRun(program, [x]) as T;
  }

  atan<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ATAN);
    return this.compileAndRun(program, [x]) as T;
  }

  atan2<T extends Tensor>(a: T, b: T): T {
    const program = new BinaryOpProgram(binaryop_gpu.ATAN2, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]) as T;
  }

  sinh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SINH);
    return this.compileAndRun(program, [x]) as T;
  }

  cosh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.COSH);
    return this.compileAndRun(program, [x]) as T;
  }

  tanh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TANH);
    return this.compileAndRun(program, [x]) as T;
  }

  asinh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ASINH);
    return this.compileAndRun(program, [x]) as T;
  }

  acosh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ACOSH);
    return this.compileAndRun(program, [x]) as T;
  }

  atanh<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ATANH);
    return this.compileAndRun(program, [x]) as T;
  }

  erf<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ERF);
    return this.compileAndRun(program, [x]) as T;
  }

  step<T extends Tensor>(x: T, alpha: number): T {
    const program = new UnaryOpProgram(x.shape, unary_op.STEP(alpha));
    return this.compileAndRun(program, [x]) as T;
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const program = new Conv2DProgram(convInfo);
    const output = this.makeOutputArray(program.outputShape, x.dtype) as Tensor4D;
    if (this.traceOp('conv2d', convInfo, [x.dataId, filter.dataId], [output.dataId]))
      return output;
    return this.compileAndRun(program, [x, filter], output);
  }

  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const program = new Conv2DDerInputProgram(convInfo);
    return this.compileAndRun(program, [dy, filter]);
  }

  conv2dDerFilter(x: Tensor4D, dy: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const program = new Conv2DDerFilterProgram(convInfo);
    return this.compileAndRun(program, [x, dy]);
  }

  depthwiseConv2D(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const program = new DepthwiseConv2DProgram(convInfo);
    const output = this.makeOutputArray(program.outputShape, x.dtype) as Tensor4D;
    if (this.traceOp('depthwiseConv2D', convInfo, [x.dataId, filter.dataId], [output.dataId]))
        return output;
    return this.compileAndRun(program, [x, filter], output);
  }

  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const program = new Pool2DProgram(convInfo, 'max', false);
    const output =
        this.makeOutputArray(program.outputShape, x.dtype) as Tensor4D;
    return this.compileAndRun(program, [x], output);
  }

  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const program = new Pool2DProgram(convInfo, 'avg', false);
    const output = this.makeOutputArray(program.outputShape, 'float32');
    if (this.traceOp('avgpool', convInfo, [x.dataId], [output.dataId]))
      return output as Tensor4D;
    return this.compileAndRun(program, [x], output) as Tensor4D;
  }

  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const getPositions = true;
    const maxPoolPositionsProgram =
        new Pool2DProgram(convInfo, 'max', getPositions);
    const maxPoolPositions: Tensor4D =
        this.compileAndRun(maxPoolPositionsProgram, [x]);

    const maxPoolBackPropProgram = new MaxPool2DBackpropProgram(convInfo);
    const output =
        this.makeOutputArray(maxPoolBackPropProgram.outputShape, x.dtype);
    const result = this.compileAndRun(
        maxPoolBackPropProgram, [dy, maxPoolPositions], output);
    maxPoolPositions.dispose();
    return result as Tensor4D;
  }

  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const avgPoolBackpropProgram = new AvgPool2DBackpropProgram(convInfo);
    const output =
        this.makeOutputArray(avgPoolBackpropProgram.outputShape, x.dtype);
    return this.compileAndRun(avgPoolBackpropProgram, [dy], output) as Tensor4D;
  }

  cast<T extends Tensor<types.Rank>>(x: T, dtype: DataType): T {
    return backend_util.castTensor(x, dtype, this);
  }

  reshape<T extends Tensor<types.Rank>, R extends types.Rank>(
      x: T, shape: types.ShapeMap[R]): Tensor<R> {
    const output = backend_util.reshapeTensor(x, shape);
    if (this.traceOp('reshape', shape, [x.dataId], [output.dataId]))
        return output;
    return output;
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const program =
        new ResizeBilinearProgram(x.shape, newHeight, newWidth, alignCorners);
    return this.compileAndRun(program, [x]);
  }

  resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean):
      Tensor4D {
    const program = new ResizeBilinearBackpropProgram(dy, x, alignCorners);

    return this.compileAndRun(program, [dy]);
  }

  resizeNearestNeighbor(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const program = new ResizeNearestNeighborProgram(
        x.shape, newHeight, newWidth, alignCorners);
    return this.compileAndRun(program, [x]);
  }

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    const probs = normalized ? logits : ops.softmax(logits);
    const batchSize = probs.shape[0];
    const numOutcomes = probs.shape[1];
    const program = new MultinomialProgram(batchSize, numOutcomes, numSamples);
    const output =
        this.makeOutputArray(program.outputShape, 'int32') as Tensor2D;
    const customSetup = program.getCustomSetupFunc(seed);
    return this.compileAndRun(program, [probs], output, customSetup);
  }

  oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number):
      Tensor2D {
    const program = new OneHotProgram(indices.size, depth, onValue, offValue);
    return this.compileAndRun(program, [indices]);
  }

  private makeOutputArray<T extends Tensor>(shape: number[], dtype: DataType):
      T {
    return Tensor.make(shape, {}, dtype) as T;
  }

  private compileAndRun<T extends Tensor, K extends Tensor>(
      program: GPGPUProgram, inputs: T[], output?: K,
      customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void):
      K {
    if (output == null) {
      output = this.makeOutputArray(program.outputShape, inputs[0].dtype);
    }
    const inputsData: Array<TensorData<T>> = inputs.map(input => {
      this.uploadToGPU(input.dataId);
      return {tensor: input, texData: this.texData.get(input.dataId)};
    });
    this.uploadToGPU(output.dataId);
    const outputData = {
      tensor: output,
      texData: this.texData.get(output.dataId)
    };
    const key = gpgpu_math.makeShaderKey(program, inputsData, outputData);
    const binary = this.getAndSaveBinary(key, () => {
      return gpgpu_math.compileProgram(
          this.gpgpu, program, inputsData, outputData);
    });
    const shouldTimeProgram = this.activeTimers != null;
    let query: WebGLQuery|CPUTimerQuery;
    if (shouldTimeProgram) {
      query = this.startTimer();
    }

    gpgpu_math.runProgram(binary, inputsData, outputData, customSetup);

    if (shouldTimeProgram) {
      query = this.endTimer(query);
      this.activeTimers.push(this.getQueryTime(query));
    }
    return output;
  }

  private getAndSaveBinary(key: string, getBinary: () => GPGPUBinary):
      GPGPUBinary {
    if (!(key in this.binaryCache)) {
      this.binaryCache[key] = getBinary();
    }
    return this.binaryCache[key];
  }

  getTextureManager(): TextureManager {
    return this.textureManager;
  }

  private disposed = false;

  dispose() {
    if (this.disposed) {
      return;
    }
    for (const key in this.binaryCache) {
      this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
    }
    this.textureManager.dispose();
    this.canvas.remove();
    if (this.gpgpuCreatedLocally) {
      this.gpgpu.dispose();
    }
    this.disposed = true;
  }

  private throwIfNoData(dataId: DataId) {
    if (!this.texData.has(dataId)) {
      throw new Error(
          `WebGL backend: No data found for this tensor. ` +
          `Did you change your backend in the middle of the program? ` +
          `New backends can't use Tensors created with previous backends`);
    }
  }

  private uploadToGPU(dataId: DataId): void {
    this.throwIfNoData(dataId);
    const texData = this.texData.get(dataId);
    const {shape, values, texture, dtype, texType} = texData;
    if (texture != null) {
      // Array is already on GPU. No-op.
      return;
    }
    const shouldTimeProgram = this.activeTimers != null;
    let start: number;
    if (shouldTimeProgram) {
      start = performance.now();
    }
    const texShape =
        webgl_util.getTextureShapeFromLogicalShape(this.gpgpu.gl, shape);
    texData.texShape = texShape;
    const newTexture = this.textureManager.acquireTexture(texShape, texType);
    texData.texture = newTexture;
    if (values != null) {
      this.gpgpu.uploadMatrixToTexture(
          newTexture, texShape[0],
          // TODO(smilkov): Propagate the original typed array to gpgpu.
          texShape[1], typedArrayToFloat32(values, dtype));
      // Once uploaded, don't store the values on cpu.
      // texData.values = null;
      if (shouldTimeProgram) {
        this.uploadWaitMs += performance.now() - start;
      }
    }
  }

  private cacheOnCPU(dataId: DataId, float32Values?: Float32Array) {
    // In delayed storage mode, when the user reads data, we don't keep a copy
    // on the gpu, to minimize likelihood of memory leak. We re-upload to gpu
    // the next time a gpgpu program needs the texture.
    const dontKeepCopyOnGPU = this.delayedStorage;
    const texData = this.texData.get(dataId);
    const {texture, texShape, dtype, texType} = texData;
    if (dontKeepCopyOnGPU && texture != null) {
      this.textureManager.releaseTexture(texture, texShape, texType);
      texData.texture = null;
      texData.texShape = null;
    }
    if (float32Values != null) {
      texData.values = float32ToTypedArray(float32Values, dtype);
    }
  }
}

ENV.registerBackend('webgl', () => new MathBackendWebGL(), 2 /* priority */);

function float32ToTypedArray<D extends DataType>(
    a: Float32Array, dtype: D): DataTypeMap[D] {
  if (dtype === 'float32') {
    return a;
  } else if (dtype === 'int32' || dtype === 'bool') {
    const result = (dtype === 'int32') ? new Int32Array(a.length) :
                                         new Uint8Array(a.length);
    for (let i = 0; i < result.length; ++i) {
      result[i] = Math.round(a[i]);
    }
    return result;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

function typedArrayToFloat32<D extends DataType>(
    a: DataTypeMap[D], dtype: D): Float32Array {
  return (a instanceof Float32Array) ? a : new Float32Array(a);
}
