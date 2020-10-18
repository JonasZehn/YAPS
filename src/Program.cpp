#include "Program.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <ios>
#include <algorithm>

#include <stdlib.h>
#include <string.h>

int loadFile(const char *filePath, std::string &fileContent, std::string &error)
{
	std::ifstream filestream(filePath);
	if (!filestream.is_open())
	{
		error = "Impossible to open " + std::string(filePath) + ". Are you in the right directory ? \n";
		return -1;
	}

	fileContent.clear();
	filestream.seekg(0, std::ios::end);
	fileContent.reserve(filestream.tellg());
	filestream.seekg(0, std::ios::beg);

	fileContent.assign((std::istreambuf_iterator<char>(filestream)),
		std::istreambuf_iterator<char>());

	return 0;
}

GLuint compileShader(GLenum shaderType, const std::string &filePath, const std::string &shaderSource)
{
	GLuint shaderID = glCreateShader(shaderType);
	printf("Compiling shader : %s\n", filePath.c_str());
	char const * VertexSourcePointer = shaderSource.c_str();
	glShaderSource(shaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(shaderID);

	// Check Vertex Shader
	GLint Result = GL_FALSE;
	int InfoLogLength;
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0)
	{
		std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(shaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}
	return shaderID;
}

GLuint linkProgram(const std::vector<GLuint> &shaderIDs)
{
	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	for (GLuint shaderID : shaderIDs)
	{
		glAttachShader(ProgramID, shaderID);
	}
	glLinkProgram(ProgramID);

	// Check the program
	GLint Result = GL_FALSE;
	int InfoLogLength;
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0)
	{
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}
	return ProgramID;
}
GLuint LoadShaders(const char * vertex_file_path, const char * fragment_file_path){

	std::string error;
	std::string VertexShaderCode;
	int resultCode = loadFile(vertex_file_path, VertexShaderCode, error);
	if (resultCode != 0)
	{
		std::cout << "Could not load vertex shader " << vertex_file_path << " " << error << std::endl;
		throw std::logic_error("Could not load vertex shader");
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	resultCode = loadFile(fragment_file_path, FragmentShaderCode, error);
	if (resultCode != 0)
	{
		std::cout << "Could not load fragment shader " << vertex_file_path << " " << error << std::endl;
		throw std::logic_error("Could not load fragment shader");
	}

	GLuint VertexShaderID = compileShader(GL_VERTEX_SHADER, vertex_file_path, VertexShaderCode);
	GLuint FragmentShaderID = compileShader(GL_FRAGMENT_SHADER, fragment_file_path, FragmentShaderCode);

	GLuint ProgramID = linkProgram({ VertexShaderID, FragmentShaderID });

	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);
	
	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}


GLuint LoadShaders(const char * vertex_file_path, const char * geometry_shader_file_path, const char * fragment_file_path)
{
	std::string error;
	std::string VertexShaderCode;
	int resultCode = loadFile(vertex_file_path, VertexShaderCode, error);
	if (resultCode != 0)
	{
		std::cout << "Could not load vertex shader " << vertex_file_path << " " << error << std::endl;
		throw std::logic_error("Could not load vertex shader");
	}

	// Read the Geometry Shader code from the file
	std::string GeometryShaderCode;
	resultCode = loadFile(geometry_shader_file_path, GeometryShaderCode, error);
	if (resultCode != 0)
	{
		std::cout << "Could not load geometry shader " << vertex_file_path << " " << error << std::endl;
		throw std::logic_error("Could not load geometry shader");
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	resultCode = loadFile(fragment_file_path, FragmentShaderCode, error);
	if (resultCode != 0)
	{
		std::cout << "Could not load fragment shader " << vertex_file_path << " " << error << std::endl;
		throw std::logic_error("Could not load fragment shader");
	}

	GLuint VertexShaderID = compileShader(GL_VERTEX_SHADER, vertex_file_path, VertexShaderCode);
	GLuint GeometryShaderID = compileShader(GL_GEOMETRY_SHADER, geometry_shader_file_path, GeometryShaderCode);
	GLuint FragmentShaderID = compileShader(GL_FRAGMENT_SHADER, fragment_file_path, FragmentShaderCode);

	GLuint ProgramID = linkProgram({ VertexShaderID, GeometryShaderID, FragmentShaderID });

	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, GeometryShaderID);
	glDetachShader(ProgramID, FragmentShaderID);

	glDeleteShader(VertexShaderID);
	glDeleteShader(GeometryShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}
