#ifndef PR_GLCUDA_H
#define PR_GLCUDA_H

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <stdexcept>

#define cuChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		throw std::logic_error("cuda error");
		//if (abort) exit(code);
	}
}

#define glChk(ans) { (ans); glAssert(__FILE__, __LINE__); }
inline void glAssert(const char *file, int line, bool abort = true){
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
	{
		fprintf(stderr, "gl assert: %d %s %d\n", err, file, line);
		throw std::logic_error("opengl error");
		//if (abort) exit(-1);
	}
}

#endif