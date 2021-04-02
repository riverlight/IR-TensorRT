# IR-TensorRT
使用 TensorRT 做推理引擎的 image restoration

## 使用说明
1. 项目建议放在 TensorRT 7.0 的 samples 目录下，因为用到了 samples/common 中的 logger.cpp 
2. 请使用 vs2017 打开
3. 工程中用到了 opencv4，使用了 opencv4_include 和 opencv4_lib 两个环境变量，请正确配置，当然，相关代码注释了也ok能跑

## 当前主要功能
1. 把 *.onnx 文件转成 *.trt 文件
2. 支持 *.onnx 的动态输入 shape，这个还挺麻烦的，网上资料都很散，我这个算是个 sample

