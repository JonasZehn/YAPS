#ifndef PROGRAM_H
#define PROGRAM_H

#include <Eigen/Core>
#include <glad/glad.h>

#include <string>

int loadFile(const char *filePath, std::string &fileContent, std::string &error);

GLuint LoadShaders(const char * vertex_file_path, const char * fragment_file_path);
GLuint LoadShaders(const char * vertex_file_path, const char * geometry_shader_file_path, const char * fragment_file_path);

class ProgramGL
{
public:

	ProgramGL(GLuint programId)
		:m_program(programId)
	{

	}

	static ProgramGL loadFromFiles(const std::string& vertex_file_path, const std::string& fragment_file_path)
	{
		return ProgramGL(LoadShaders(vertex_file_path.c_str(), fragment_file_path.c_str()));
	}

	static ProgramGL loadFromFiles(const std::string& vertex_file_path, const std::string& geometry_shader_file_path, const std::string& fragment_file_path)
	{
		return ProgramGL(LoadShaders(vertex_file_path.c_str(), geometry_shader_file_path.c_str(), fragment_file_path.c_str()));
	}

	void use()
	{
		assert(glGetError() == GL_NO_ERROR);
		glUseProgram(m_program);
		GLuint error = glGetError();
		assert(error == GL_NO_ERROR);
	}
	void unbind()
	{
		assert(glGetError() == GL_NO_ERROR);
		glUseProgram(0);
		GLuint error = glGetError();
		assert(error == GL_NO_ERROR);
	}

	void setUniform(const char* name, int i)
	{
		glUniform1i(glGetUniformLocation(m_program, name), i);
	}

	void setUniform(const char* name, float f)
	{
		glUniform1f(glGetUniformLocation(m_program, name), f);
	}
	void setUniform(const char* name, const Eigen::Vector2f &vec)
	{
		glUniform2f(glGetUniformLocation(m_program, name), vec[0], vec[1]);
	}
	void setUniform(const char* name, const Eigen::Vector3f &vec)
	{
		glUniform3f(glGetUniformLocation(m_program, name), vec[0], vec[1], vec[2]);
	}
	void setUniform(const char* name, const Eigen::Vector4f &vec)
	{
		glUniform4f(glGetUniformLocation(m_program, name), vec[0], vec[1], vec[2], vec[3]);
	}
	void setUniform(const char* name, const Eigen::Matrix3f &mat)
	{
		glUniformMatrix3fv(glGetUniformLocation(m_program, name), 1, GL_FALSE, mat.data());
	}
	void setUniform(const char* name, const Eigen::Matrix4f &mat)
	{
		glUniformMatrix4fv(glGetUniformLocation(m_program, name), 1, GL_FALSE, mat.data());
	}

private:
	GLuint m_program;
};

#endif
