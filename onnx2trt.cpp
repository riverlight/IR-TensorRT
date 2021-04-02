#include <time.h>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "parserOnnxConfig.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"
#include <direct.h>
#include <io.h>
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace nvinfer1;

samplesCommon::Args gArgs;
#define Debug(x) std::cout << "Line" << __LINE__ << " " << x << std::endl
#define LClip(x, lmin ,lmax) ((x)<lmin ? lmin : ( (x)>lmax ? lmax : (x) ))


struct TensorRT {
	IExecutionContext* context;
	ICudaEngine* engine;
	IRuntime* runtime;
};


void Mat_2_Buffer(cv::Mat& m, float *buffer)
{
	int width = m.cols;
	int height = m.rows;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			unsigned char* s = (uchar *)&m.at<cv::Vec3b>(i, j)[0];
			float* d = buffer + (i * width + j) * 3;
			d[0] = float(s[0]) / 255.0;
			d[1] = float(s[1]) / 255.0;
			d[2] = float(s[2]) / 255.0;
		}
	}
}

void Mat_2_Buffer2(cv::Mat &m, float *buffer)
{
	int width = m.cols;
	int height = m.rows;
	float* r = buffer + width * height * 0;
	float* g = buffer + width * height * 1;
	float* b = buffer + width * height * 2;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			unsigned char* s = (uchar *)&m.at<cv::Vec3b>(i, j)[0];
			int offset = i * width + j;
			b[offset] = float(s[0]) / 255.0;
			g[offset] = float(s[1]) / 255.0;
			r[offset] = float(s[2]) / 255.0;
		}
	}
}

void Buffer_2_Mat(float *buffer, cv::Mat& m)
{
	int width = m.cols;
	int height = m.rows;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			unsigned char* d = (uchar*)&m.at<cv::Vec3b>(i, j)[0];
			float *rgb = buffer + (i*width + j) * 3;
			d[0] = LClip(rgb[0] * 255, 0, 255);
			d[1] = LClip(rgb[1] * 255, 0, 255);
			d[2] = LClip(rgb[2] * 255, 0, 255);
		}
	}
}

void Buffer_2_Mat2(float *buffer, cv::Mat& m)
{
	int width = m.cols;
	int height = m.rows;
	float* r = buffer + width * height * 0;
	float* g = buffer + width * height * 1;
	float* b = buffer + width * height * 2;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			unsigned char* d = (uchar*)&m.at<cv::Vec3b>(i, j)[0];
			int offset = i * width + j;
			d[2] = LClip(r[offset] * 255, 0, 255);
			d[1] = LClip(g[offset] * 255, 0, 255);
			d[0] = LClip(b[offset] * 255, 0, 255);
		}
	}
}


int ONNX2TRT(char* onnxFileName, char* trtFileName, int batchSize)
{
	//gArgs.runInFp16 = true;
	if (_access(onnxFileName, 02) != 0)
	{
		Debug("Can't Read ONNX File");
		return -1;
	}

	// create the builder
	IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
	if (builder == nullptr)
	{
		Debug("Create Builder Failure");
		return -2;
	}

	// Now We Have BatchSize Here
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(batchSize);

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

	if (!parser->parseFromFile(onnxFileName, static_cast<int>(gLogger.getReportableSeverity())))
	{
		Debug("Parse ONNX Failure");
		return -3;
	}

#if 1
	ITensor *input = network->getInput(0);
	Dims dims = input->getDimensions();
	cout << "dims : " << dims << endl;

	// Create an optimization profile so that we can specify a range of input dimensions.
	auto profile = builder->createOptimizationProfile();
	// This profile will be valid for all images whose size falls in the range of [(1, 1, 1, 1), (1, 1, 56, 56)]
	// but TensorRT will optimize for (1, 1, 28, 28)
	profile->setDimensions("input11", OptProfileSelector::kMIN, Dims4{ 1, 3, 96, 96 });
	profile->setDimensions("input11", OptProfileSelector::kOPT, Dims4{ 1, 3, 1280, 360 });
	profile->setDimensions("input11", OptProfileSelector::kMAX, Dims4{ 1, 3, 1920, 720 });
	config->addOptimizationProfile(profile);
#endif

	builder->setMaxBatchSize(batchSize);
	builder->setMaxWorkspaceSize(16_MiB);
	config->setMaxWorkspaceSize(16_MiB);
	if (gArgs.runInFp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (gArgs.runInInt8)
	{
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
	}

	samplesCommon::enableDLA(builder, config, gArgs.useDLACore);

	//ICudaEngine* engine = builder->buildCudaEngine(*network);
	auto engine = builder->buildEngineWithConfig(*network, *config);
	gLogInfo << "Profile dimensions in preprocessor engine:\n";
	gLogInfo << "    Minimum = " << engine->getProfileDimensions(0, 0, OptProfileSelector::kMIN) << '\n';
	gLogInfo << "    Optimum = " << engine->getProfileDimensions(0, 0, OptProfileSelector::kOPT) << '\n';
	gLogInfo << "    Maximum = " << engine->getProfileDimensions(0, 0, OptProfileSelector::kMAX)
		<< std::endl;
	if (!engine)
	{
		Debug("Engine Build Failure");
		return -4;
	}

	// we can destroy the parser
	parser->destroy();

	// serialize the engine, then close everything down
	IHostMemory* trtModelStream = engine->serialize();

	engine->destroy();
	network->destroy();
	builder->destroy();

	if (!trtModelStream)
	{
		Debug("Serialize Fail");
		return -5;
	}

	ofstream ofs(trtFileName, std::ios::out | std::ios::binary);
	ofs.write((char*)(trtModelStream->data()), trtModelStream->size());
	ofs.close();
	trtModelStream->destroy();

	Debug("Save Success");

	return 0;
}

void* LoadNet(char* trtFileName)
{
	if (_access(trtFileName, 02) != 0)
	{
		Debug("Can't Read TRT File");
		return 0;
	}

	std::ifstream t(trtFileName, std::ios::in | std::ios::binary);
	std::stringstream tempStream;
	tempStream << t.rdbuf();
	t.close();
	Debug("TRT File Loaded");

	tempStream.seekg(0, std::ios::end);
	const int modelSize = tempStream.tellg();
	tempStream.seekg(0, std::ios::beg);
	void* modelMem = malloc(modelSize);
	tempStream.read((char*)modelMem, modelSize);

	IRuntime* runtime = createInferRuntime(gLogger);
	if (runtime == nullptr)
	{
		Debug("Build Runtime Failure");
		return 0;
	}

	if (gArgs.useDLACore >= 0)
	{
		runtime->setDLACore(gArgs.useDLACore);
	}

	ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, nullptr);

	if (engine == nullptr)
	{
		Debug("Build Engine Failure");
		return 0;
	}

	IExecutionContext* context = engine->createExecutionContext();
	if (context == nullptr)
	{
		Debug("Build Context Failure");
		return 0;
	}
	context->setOptimizationProfile(0);
	
	Dims4 dims4 = { 1, 3, 1280, 360 };
	context->setBindingDimensions(0, dims4);
	Dims dims = context->getBindingDimensions(0);
	cout << "dims : " << dims << endl;

	TensorRT* trt = new TensorRT();
	trt->context = context;
	trt->engine = engine;
	trt->runtime = runtime;

	return trt;
}

void ReleaseNet(void* trt)
{
	TensorRT* curr = (TensorRT*)trt;
	curr->context->destroy();
	curr->engine->destroy();
	curr->runtime->destroy();

	delete curr;
	curr = NULL;
	delete curr;
}

void DoInference(void* trt, const char* input_name, const char* output_name, float* input, float* output, int input_size, int output_size)
{
	TensorRT* curr = (TensorRT*)trt;

	const ICudaEngine& engine = curr->context->getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()

	const int inputIndex = engine.getBindingIndex(input_name);
	const int outputIndex = engine.getBindingIndex(output_name);

	// DebugP(inputIndex); DebugP(outputIndex);
	// create GPU buffers and a stream

	CHECK(cudaMalloc(&buffers[inputIndex], input_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// Because we had specified batch size, so we use enqueueV2
	time_t starttime = GetCurrentTime();
	int count = 0;
	while (1) {
		// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
		CHECK(cudaMemcpyAsync(buffers[inputIndex], input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
		cudaStreamSynchronize(stream);
		curr->context->enqueueV2(buffers, stream, nullptr);
		CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
		cudaStreamSynchronize(stream);

		if (++count == 1000)
			break;
	}
	cout << " time : " << GetCurrentTime() - starttime << endl;

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

const char *g_szTRTFile = "d:/nir6_best.trt";
void do_onnx2trt()
{
	const char *szOnnxFile = "d:/nir6_best.onnx";
	ONNX2TRT((char *)szOnnxFile, (char *)g_szTRTFile, 1);
}

void do_infer()
{
	TensorRT *handle = (TensorRT *)LoadNet((char *)g_szTRTFile);
	Mat m;
	m = imread("d:/workroom/testroom/v360.png");

	float *pInput, *pOutput;
	int size = 3 * m.rows * m.cols;
	pInput = new float[size];
	pOutput = new float[size];
	Mat_2_Buffer2(m, pInput);
	
	DoInference(handle, "input11", "output44", pInput, pOutput, size, size);
	Buffer_2_Mat2(pOutput, m);
	imwrite("d:/trt.png", m);
//		break;
//		imshow("1", m);
//		waitKey(0);
	
	cout << "do_infer done!" << endl;
}

int main(int argc, char *argv[])
{
	cout << "Hi, this is a onnx2trt program!" << endl;
	do_onnx2trt();
	do_infer();

	return 0;
}
