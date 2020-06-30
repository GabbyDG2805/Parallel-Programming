//SCATTER: HISTOGRAM
//a very simple histogram implementation
kernel void hist_simple(global const uchar* A, global int* H) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);
	//Atomic operations deal with race conditions, but serialise the access to global memory, and are slow
}
//SCAN: Cumulative Histogram
//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void hist_cum(global int* H, global int* CH) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)
		atomic_add(&CH[i], H[id]);
}
//MAP: LUT
kernel void LUT(global int* CH, global int* LUT) {
	int id = get_global_id(0);
	LUT[id] = CH[id] * (double)255 / CH[255];
}
//MAP: changing pixel values
//a simple OpenCL kernel which copies all pixels from A to B
kernel void ReProject(global uchar* A, global int* LUT, global uchar* B) {
	int id = get_global_id(0);
	B[id] = LUT[A[id]];
}