#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.pgm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}
//This code was developed using Tutorial 2 as a foundation and was appropriately edited and built upon to carry out this task.
//The first step was to implement a histogram kernel. The hist_simple kernel from tutorial 3 was used as a template. The size of H was predefined as 256 as the pixel values always range from 0 to 255, and the buffer was initialised to 0 using enqueueFillBuffer.
//A histogram is a simple example of the scatter pattern which writes data into output locations indicated by an index array.
//The first purpose of plotting this histogram was to identify the distribution of pixel values in the image. Since the majority of pixels in the image fell between the same small range of values, the image is of very low contrast.
//The histogram implemented uses global memory and atomic operators, which deal with race conditions but serialise the access to global memory and are therefore slow. However, they were appropriate for this task as the amount of data is relatively small, so the execution times remained fast.
//The next step was to create a cumulative histogram of this which plots the total number of pixels in the image against the pixel values. So, by 255, all pixels in the image have been counted. This was done by using the scan_add_atomic kernel from Tutorial 3 and adapting it appropriately.
//This is an example of the Scan pattern which computes all partial reductions of a collection so that every element of the output is a reduction of all the elements of the input up to the position of that output element.
//Then the cumulative histogram needed to be normalised to create a lookup table (LUT) of new values for the pixels to increase the contrast of the image. This was done by creating a kernel to multiply each value in the cumulative histogram by 255/total pixels. This resulted in values ranging from 0-255
//This is an example of the map pattern. Map applies ‘elemental function’ to every element of data – the result is a new collection of the same shape as the input. The result does not depend on the order in which various instances of the function are executed.
//The final kernel was also an example of the map pattern. This time, the kernel simply changes each pixel in the input image to the values defined in the LUT.
//The execution time and memory transfer of each kernel is also logged and printed with each resulting histogram.
//The result of this program is an output image with a higher, more balanced contrast than the input image. The code works on greyscale .pgm images of varying sizes.
//By Gabriella Di Gregorio DIG15624188

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm"; //Change this to change the input image. Images available: test.pgm, test_large.pgm, Einstein.pgm, cat.pgm

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");

		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 3 - memory allocation
		//host - input

		typedef int mytype;
		std::vector<mytype> H(256);
		size_t histsize = H.size() * sizeof(mytype);

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_histogram_output(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer dev_cumulative_histogram_output(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer dev_LUT_output(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

		//Part 4 - device operations

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(dev_histogram_output, 0, 0, histsize);

		//4.2 Setup and execute the kernel (i.e. device code)

		//The first kernel call plots a histogram of the frequency of each pixel value (0-255) in the picture
		cl::Kernel kernel_hist_simple = cl::Kernel(program, "hist_simple");
		kernel_hist_simple.setArg(0, dev_image_input);
		kernel_hist_simple.setArg(1, dev_histogram_output);

		cl::Event prof_event;

		queue.enqueueNDRangeKernel(kernel_hist_simple, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);
		queue.enqueueReadBuffer(dev_histogram_output, CL_TRUE, 0, histsize, &H[0]);

		std::vector<mytype> CH(256);

		queue.enqueueFillBuffer(dev_cumulative_histogram_output, 0, 0, histsize);

		//The second kernel call plots a cumulative histogram of the total pixels in the picture across pixel values 0-255, so by 255, all pixels have been counted
		cl::Kernel kernel_hist_cum = cl::Kernel(program, "hist_cum");
		kernel_hist_cum.setArg(0, dev_histogram_output);
		kernel_hist_cum.setArg(1, dev_cumulative_histogram_output);

		cl::Event prof_event2;

		queue.enqueueNDRangeKernel(kernel_hist_cum, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &prof_event2);
		queue.enqueueReadBuffer(dev_cumulative_histogram_output, CL_TRUE, 0, histsize, &CH[0]);

		std::vector<mytype> LUT(256);

		queue.enqueueFillBuffer(dev_LUT_output, 0, 0, histsize);

		//The third kernel call creates a new histogram that will serve as a look up table of the new pixel vales. It does this by normalising the cumulative histogram, essentially decreasing the value of the pixels to increase the contrast
		cl::Kernel kernel_LUT = cl::Kernel(program, "LUT");
		kernel_LUT.setArg(0, dev_cumulative_histogram_output);
		kernel_LUT.setArg(1, dev_LUT_output);

		cl::Event prof_event3;

		queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &prof_event3);
		queue.enqueueReadBuffer(dev_LUT_output, CL_TRUE, 0, histsize, &LUT[0]);

		//The last kernel assigns the new pixel values from the lookup table to the output image, so that the output is of higher contrast than the input
		cl::Kernel kernel_ReProject = cl::Kernel(program, "ReProject");
		kernel_ReProject.setArg(0, dev_image_input);
		kernel_ReProject.setArg(1, dev_LUT_output);
		kernel_ReProject.setArg(2, dev_image_output);

		cl::Event prof_event4;

		//The values from each histogram are printed, along with the kernel execution times and memory transfer of each kernel.
		vector<unsigned char> output_buffer(image_input.size());
		queue.enqueueNDRangeKernel(kernel_ReProject, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event4);
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		cout << endl;
		std::cout << "Histogram = " << H << std::endl;
		std::cout << "Histogram kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		std::cout << "Cumulative Histogram = " << CH << std::endl;
		std::cout << "Cumulative Histogram kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event2, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		std::cout << "LUT = " << LUT << std::endl;
		std::cout << "LUT kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event3, ProfilingResolution::PROF_US) << endl;
		cout << endl;


		std::cout << "Vector kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event4.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event4, ProfilingResolution::PROF_US) << endl;

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");

 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
