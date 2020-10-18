#include "Renderer.h"

#include "SimpleTriMesh.h"

#include <cstdio>
#include <array>
#include <fstream>

Renderer::Renderer()
{
	//core.viewport[2] = 1920;
	//core.viewport[3] = 1080;

	////overwrite default values of core
	//core.background_color = Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f);

	//core.depth_test = false; // disable depth test for particle drawing;

							 /*	core.camera_dnear = 1e-1;
							 core.camera_dfar = 1e3; */// z fighting is a larger problem than I thought, but I guess these parameters could be normalized with respect to scene dimensions

	//core.animation_max_fps = 120;
	frame = 0;

	m_lightPosition = Eigen::Vector3f(200.f, 128.f, 256.f);
	background_color = Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f);
	m_baseColor = Eigen::Vector4f(0.0, 0.0, 0.2f, 1.0f);
	m_illuminatedColor = Eigen::Vector4f(0.1f, 0.3f, 0.8f, 1.0f);
	m_illuminationMultiplier = 5.f;
	m_shadowKMultiplier = 2.0f;
	m_radius = 0.2f;
	m_opacity = 0.015f;
	m_play = true;
	m_pauseAtFrame = 1000;
}
void Renderer::init()
{
	setInitialParticles(particles);

	particles.initGL();
	printf("%s\n", glGetString(GL_VERSION));
	std::string shaderFolder = std::string(SHADER_FOLDER);
	m_program.reset(new ProgramGL(ProgramGL::loadFromFiles(shaderFolder + "Particle.vs", shaderFolder + "Particle.gs", shaderFolder + "Particle.fs")));

	m_illuminationProgram.reset(new ProgramGL(ProgramGL::loadFromFiles(shaderFolder + "Illumination.vs", shaderFolder + "Illumination.gs", shaderFolder + "Illumination.fs")));

	Eigen::Vector3i dims = getVelocityGridDims();
	m_grid.reset(new GPUGrid(dims[0], dims[1], dims[2]));
	m_grid2.reset(new GPUGrid(dims[0], dims[1], dims[2]));

	m_illuminationFramebuffer.reset(new Framebuffer(8192, 8192));
	m_texture.reset(new Texture2D(m_illuminationFramebuffer->getWidth(), m_illuminationFramebuffer->getHeight()));

	Eigen::Vector2i outputDims = getOutputDims();
	m_hdrFramebuffer.reset(new FramebufferHDR(outputDims[0], outputDims[1]));

	m_quad.init();

	m_quadProgram.reset(new ProgramGL(ProgramGL::loadFromFiles(shaderFolder + "Quad.vs", shaderFolder + "Quad.fs")));
}
void Renderer::computeIllumination()
{
	m_particleIllTexture.reset(new TextureBuffer(particles.getParticleCount()));
	m_particleIllTexture->setConstant(0);

	m_texture->setZero();

	//static int frame = 0;
	//std::ofstream file(std::string("texture") + std::to_string(frame));
	//std::vector<int>  data = m_texture->read();
	//for (int i = 0; i < m_texture->getHeight(); i++)
	//{
	//	for (int j = 0; j < m_texture->getWidth(); j++)
	//	{
	//		file << " " << data[j + i * m_texture->getWidth()];
	//	}
	//	file << std::endl;
	//}
	//frame++;

	////Eigen::Vector3f center = particles.getGPUParticles().computeCenter();
	//Eigen::Vector3f center(128.f, 64.f, 64.f);
	//m_lightPosition = Eigen::Vector3f(128.f, 64.f, -200.f);
	Eigen::Vector3f center = getVelocityGridDims().cast<float>() * 0.5f;

	Eigen::Vector3f upRes(0.0f, 1.0f, 0.0f);
	Eigen::Vector3f forward = (center - m_lightPosition).normalized();
	Eigen::Vector3f right = upRes.cross(forward).normalized();
	Eigen::Vector3f up = forward.cross(right).normalized();

	Eigen::Vector3f camZ = forward;
	particles.sort(camZ[0], camZ[1], camZ[2]);

	Eigen::Matrix3f view3Inverse;
	view3Inverse << -right, up, -forward;

	Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
	view.topLeftCorner(3, 3) = view3Inverse.transpose();

	view.col(3).head(3) = view.topLeftCorner(3, 3)*-m_lightPosition;

	int width = m_illuminationFramebuffer->getWidth();
	int height = m_illuminationFramebuffer->getHeight();

	Eigen::Matrix4f projection;
	float lightAngle = 60.0f;
	float fH = tan(lightAngle / 360.0 * PR_PI) * m_camera.dnear;
	float fW = fH * (double)width / (double)height;
	CameraMatrix::frustumMatrix(-fW, fW, -fH, fH, m_camera.dnear, m_camera.dfar, projection);

	//PHASE 0: initialize framebuffer
	m_illuminationFramebuffer->bind();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//PHASE 1: initialize special render settings
	glDisable(GL_DEPTH_TEST);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDepthMask(GL_FALSE);

	//PHASE 2: set program and uniforms
	assert(glGetError() == GL_NO_ERROR);
	m_illuminationProgram->use();
	assert(glGetError() == GL_NO_ERROR);
	m_illuminationProgram->setUniform("right", right);
	m_illuminationProgram->setUniform("up", up);
	m_illuminationProgram->setUniform("MVP", Eigen::Matrix4f(projection * view));
	m_illuminationProgram->setUniform("size", getRadius());
	m_illuminationProgram->setUniform("opacity", m_shadowKMultiplier * getOpacity());
	assert(glGetError() == GL_NO_ERROR);

	//PHASE 3: set textures
	//glActiveTexture(GL_TEXTURE0);
	//glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_texture->getBufferId());
	//glBindTexture(GL_TEXTURE_2D, m_texture->getTextureId());
	glBindImageTexture(0, m_texture->getTextureId(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
	m_program->setUniform("img", 0);

	//glActiveTexture(GL_TEXTURE1);
	//glBindBuffer(GL_TEXTURE_BUFFER, m_particleIllTexture->getBufferId());
	//glBindTexture(GL_TEXTURE_BUFFER, m_particleIllTexture->getTextureId());
	glChk(glBindImageTexture(1, m_particleIllTexture->getTextureId(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I));
	m_program->setUniform("pimg", 1);

	glChk(0);

	//PHASE 4: draw
	particles.draw();

	//PHASE 5: unbind textures
	glChk(glBindImageTexture(1, 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I));
	glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);

	//PHASE 6: unbind  program
	m_illuminationProgram->unbind();

	//PHASE 7: reset strange settings:
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDepthMask(GL_TRUE);

	//PHASE 8: unbind framebuffer (wont reset viewport here, we expect that to happen at the beginning somewhere else in phase 1)
	m_illuminationFramebuffer->unbind();

	ComputeColorsParameters parameters;
	parameters.baseColor.x = m_baseColor[0];
	parameters.baseColor.y = m_baseColor[1];
	parameters.baseColor.z = m_baseColor[2];

	parameters.illuminatedColor.x = m_illuminatedColor[0];
	parameters.illuminatedColor.y = m_illuminatedColor[1];
	parameters.illuminatedColor.z = m_illuminatedColor[2];

	parameters.illuminationMultiplier = m_illuminationMultiplier;

	//computeColors(m_particleIllTexture.get(), particles.getGPUParticles().getCUDAIlluminatedResource(), parameters);
	computeColors2(particles.getGPUParticles().getCUDAColorsResource(), m_particleIllTexture.get(), particles.getGPUParticles().getCUDAIlluminatedResource(), parameters);

}
void Renderer::renderToTexture()
{
	computeIllumination();

	Eigen::Matrix4f proj, view;
	m_camera.computeMatrices(m_hdrFramebuffer->getWidth(), m_hdrFramebuffer->getHeight(), m_camera.orthographic, view, proj);

	Eigen::Vector3f camZ = m_camera.trackball_angles.inverse()*Eigen::Vector3f(0.0f, 0.0f, -1.0f);
	particles.sort(-camZ[0], -camZ[1], -camZ[2]);
	//particles.sort(camZ[0], camZ[1], camZ[2]);

	Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f modelView = view * model.cast<float>();
	Eigen::Vector3f right = m_camera.trackball_angles.inverse()*Eigen::Vector3f(1.0f, 0.0f, 0.0f);
	Eigen::Vector3f up = m_camera.trackball_angles.inverse()*Eigen::Vector3f(0.0f, 1.0f, 0.0f);

	//PHASE 0
	m_hdrFramebuffer->bind();
	glClearColor(background_color[0],
		background_color[1],
		background_color[2],
		background_color[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, m_hdrFramebuffer->getWidth(), m_hdrFramebuffer->getHeight());
	//core.init_draw(); //set glviewport after unbinding

	//PHASE 1
	//glEnable(GL_DEPTH_TEST); // Too much aliasing
	glDisable(GL_DEPTH_TEST); // Too much aliasing
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA); // we use gl_one because all light here goes to the eye, not only a the "opaque" part

												 //PHASE 2 - 7 again
												 //renderSolids(proj, view);

	glDepthMask(GL_FALSE);

	//PHASE 2
	assert(glGetError() == GL_NO_ERROR);
	m_program->use();
	assert(glGetError() == GL_NO_ERROR);
	m_program->setUniform("right", right);
	m_program->setUniform("up", up);
	m_program->setUniform("MVP", Eigen::Matrix4f(proj * modelView));
	m_program->setUniform("size", getRadius());
	m_program->setUniform("opacity", getOpacity());
	m_program->setUniform("lightPosition", m_lightPosition);
	assert(glGetError() == GL_NO_ERROR);

	//PHASE 3

	//PHASE 4
	particles.draw();

	//PHASE 5
	//PHASE 6
	m_program->unbind();

	//PHASE 7
	glDepthMask(GL_TRUE);

	//PHASE 8
	m_hdrFramebuffer->unbind();
}

void Renderer::step()
{
	if (frame >= m_pauseAtFrame)
	{
		m_play = false;
	}
	if (m_play)
	{

		//bool useAdvection = true;
		//if (useAdvection)
		//{
		//std::swap(m_grid, m_grid2);
		//if (frame == 0)
		//{
		//	m_grid->load(getVelocityFilename(0).c_str());
		//}
		//m_grid2->load(getVelocityFilename(frame + 1).c_str());
		//m_grid->advectParticles(particles.getGPUParticles().getCUDAResource(), dt);
		////m_grid->advectParticles2(*m_grid2, particles.getGPUParticles().getCUDAResource(), dt);

		m_grid->load(getVelocityFilename(frame).c_str());
		m_grid->advectParticles(particles.getGPUParticles().getCUDAResource(), particles.getGPUParticles().getCUDATypesResource(), getTimestep());

		this->updateParticles(particles);

		bool saveParticles = false;
		if (saveParticles)
		{
			//particles.transferGPUToCPU();

			//std::string outputFilePattern = getOutputFolder() + std::string("/particles_%04d.posgz");

			//char filename[10000];
			//sprintf(filename, outputFilePattern.c_str(), frame);

			//save_particles_posgz(particles.getCPUParticles().getVertices(), filename);
		}

		frame += 1;
	}

}
void Renderer::saveToFile()
{
	std::string outputFilePattern = getOutputFolder() + std::string("/image_%04d.png");

	char filename[10000];
	sprintf(filename, outputFilePattern.c_str(), frame);

	bool succ = m_hdrFramebuffer->render_to_png(filename);
	if (!succ)
	{
		std::cout << " could not save framebuffer to " << filename << std::endl;
		throw std::logic_error("could not save file");
	}
}
void Renderer::renderToScreen(int screenWidth, int screenHeight)
{
	m_quadProgram->use();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_hdrFramebuffer->colorTexture());

	m_quadProgram->setUniform("myTextureSampler", 0);
	Eigen::Matrix4f modelMat = Eigen::Matrix4f::Identity();

	float outputWidth = getOutputDims()[0] * (float(screenHeight) / getOutputDims()[1]);
	modelMat(0, 0) = outputWidth / screenWidth;
	m_quadProgram->setUniform("model", modelMat);

	m_quad.draw();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);

	m_quadProgram->unbind();

	outputGPUMemoryUsage();

}

float Renderer::getRadius() const
{
	return m_radius;
}
float Renderer::getOpacity() const
{
	return m_opacity;
}
Eigen::Vector3f Renderer::getColor() const
{
	Eigen::Vector3f baseColor(1.0, .2, 0.3);
	return baseColor;
}

class Sphere
{
public:
	Sphere(const Eigen::Vector3f &center, float radius)
		:m_center(center),
		m_radius(radius)
	{

	}

	float sdf(const Eigen::Vector3f &x) const
	{
		return (x - m_center).norm() - m_radius;
	}

	const Eigen::Vector3f& getCenter() const
	{
		return m_center;
	}

	float getRadius() const
	{
		return m_radius;
	}

private:
	Eigen::Vector3f m_center;
	float m_radius;
};

void adjustNumberOfParticles(const Sphere &source, std::vector<Eigen::Vector3f> &vtcs, int nd, int np)
{
	double dx = (source.getRadius() * 2) / nd;
	Eigen::Vector3d x0 = source.getCenter().cast<double>() - Eigen::Vector3d::Constant(source.getRadius());
	std::vector<std::vector<std::vector<int> > > cnts(nd, std::vector<std::vector<int> >(nd, std::vector<int>(nd, 0)));
	for (const Eigen::Vector3f &x : vtcs)
	{
		Eigen::Vector3d df = ((x.cast<double>() - x0) / dx).array();
		Eigen::Vector3i idcs;
		for (int j = 0; j < 3; j++) idcs[j] = (int)std::floor(df[j]);

		bool outside = false;
		for (int j = 0; j < 3; j++)
		{
			if (idcs[j] < 0) outside = true;
			if (idcs[j] >= nd) outside = true;
		}
		if (outside) continue;

		cnts[idcs[0]][idcs[1]][idcs[2]] += 1;
	}

	for (int i = 0; i < nd; i++)
	{
		for (int j = 0; j < nd; j++)
		{
			for (int k = 0; k < nd; k++)
			{
				if (cnts[i][j][k] < np)
				{
					Eigen::Vector3f xc = x0.cast<float>() + Eigen::Vector3f(i + 0.5f, j + 0.5f, k + 0.5f) * dx;
					float sdf = source.sdf(xc);
					//if (sdf > 0.0f) continue;

					//estimate based on source.sdf(xc) how much particles should be tried to be added
					// if sdf == 0, we assume half of volume is inside sdf
					// if sdf == -dx, we assume full volume is inside sdf
					// V(0) = 0.5, V(-dx) = 1.0
					// V(sdf) = v0 + dv * sdf
					// => v0 = 0.5, V(-dx) = 1.0 = 0.5 + (-dx) * dv 
					// => dv = 0.5 / (-dx)
					float targetRatio = 0.5 + sdf / (-dx);
					int targetNumberOfParticles = std::max(0, std::min(np, (int)std::ceil(targetRatio * np)));
					int particlesToAdd = targetNumberOfParticles - cnts[i][j][k];

					int numberOfTries = 5 * np;
					for (int l = 0; l < numberOfTries && particlesToAdd > 0; l++)
					{
						Eigen::Vector3f x = xc + Eigen::Vector3f::Random() * (0.5f * float(dx));
						if (source.sdf(x) <= 0.f)
						{
							vtcs.push_back(x);
							particlesToAdd -= 1;
						}
					}
				}
			}
		}
	}
}


PlumeRenderer::PlumeRenderer(
	const std::string &velocityFolder,
	const std::string &outputFolder)
	:
	m_velocityFolder(velocityFolder),
	m_outputFolder(outputFolder)
{
	Eigen::Vector3f center = 0.5f * getVelocityGridDims().cast<float>();
	m_camera.setCameraTrackballCenter(center);
	m_camera.setCameraTrackballRadius(1.5 * 256.f);

	this->background_color = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);

	m_illuminatedColor = Eigen::Vector4f(1.0, 1.0, 1.0f, 1.0f);
	m_baseColor = Eigen::Vector4f(0.8f * 0.238281f, 0.7f * 0.293666f, 0.6f * 0.631375f, 1.0f);


	m_opacity = 0.003f;
	m_illuminationMultiplier = 23;
	m_shadowKMultiplier = 2;

	m_radius = 0.2f;
	m_lightPosition = Eigen::Vector3f(320.f, 128.f, 120.f);

	m_nTarget = 1.5e6;
}
void PlumeRenderer::setInitialParticles(ParticlesManaged &particles)
{
	int res = 128;
	float sigma = 0.5;//instead of considering sigma later, we add it here to the radius
	Sphere source(float(res)*Eigen::Vector3f(0.5, 0.1, 0.5), res*0.05 + 1.5 * sigma);

	std::vector<Eigen::Vector3f> vtcs;

	int nd = (int)std::floor(std::pow(m_nTarget, 1.0 / 3.0));
	int np = 1;

	adjustNumberOfParticles(source, vtcs, nd, np);

	Eigen::MatrixXf vertices(3, vtcs.size());
	for (int i = 0; i < vtcs.size(); i++)
	{
		vertices.col(i) = vtcs[i];
	}
	Eigen::MatrixXf colors = Eigen::MatrixXf::Zero(3, vertices.cols());

	Eigen::VectorXi types = Eigen::VectorXi::Zero(vertices.cols());
	Eigen::MatrixXf additionalData1 = Eigen::MatrixXf::Zero(3, vertices.cols());
	Eigen::MatrixXf additionalData2 = Eigen::MatrixXf::Zero(3, vertices.cols());
	particles.setParticles(vertices, colors, types, additionalData1, additionalData2);
}
void PlumeRenderer::updateParticles(ParticlesManaged &particles)
{
	int res = 128;
	float sigma = 0.5;//instead of considering sigma later, we add it here to the radius
	Sphere source(float(res)*Eigen::Vector3f(0.5, 0.1, 0.5), res*0.05 + 1.5 * sigma);

	int nd = (int)std::floor(std::pow(m_nTarget, 1.0 / 3.0));
	int np = 1;

	particles.transferGPUToCPU();

	Eigen::Vector3i dims = getVelocityGridDims();
	Eigen::Vector3f xmin(2.0f, 1.0f, 2.0f);
	Eigen::Vector3f xmax = dims.cast<float>() - Eigen::Vector3f(2.f, 8.f, 2.f);
	std::vector<Eigen::Vector3f> vtcs;
	for (int i = 0; i < particles.getParticleCount(); i++)
	{
		Eigen::Vector3f x = particles.getCPUParticles().getVertices().col(i);
		if ((x.array() < xmin.array()).any()
			|| (x.array() > xmax.array()).any())
		{
			continue;
		}
		vtcs.push_back(x);
	}

	adjustNumberOfParticles(source, vtcs, nd, np);

	Eigen::MatrixXf vertices(3, vtcs.size());
	for (int i = 0; i < vtcs.size(); i++)
	{
		vertices.col(i) = vtcs[i];
	}
	Eigen::MatrixXf colors = Eigen::MatrixXf::Zero(3, vertices.cols());

	Eigen::VectorXi types = Eigen::VectorXi::Zero(vertices.cols());
	Eigen::MatrixXf additionalData1 = Eigen::MatrixXf::Zero(3, vertices.cols());
	Eigen::MatrixXf additionalData2 = Eigen::MatrixXf::Zero(3, vertices.cols());
	particles.setParticles(vertices, colors, types, additionalData1, additionalData2);
}

std::string PlumeRenderer::getVelocityFilename(int frame) const
{
	std::string filepattern = m_velocityFolder + std::string("/velocityGrid_%04d.uni");

	char filename[10000];
	sprintf(filename, filepattern.c_str(), frame);
	return filename;
}
std::string PlumeRenderer::getOutputFolder() const
{
	return m_outputFolder;
}


PlumeObsRenderer::PlumeObsRenderer(
	const std::string &velocityFolder,
	const std::string &outputFolder)
	:
	m_velocityFolder(velocityFolder),
	m_outputFolder(outputFolder)
{
	Eigen::Vector3f center = 0.5f * getVelocityGridDims().cast<float>();
	m_camera.setCameraTrackballCenter(center);
	m_camera.setCameraTrackballRadius(1.5 * 256.f);

	this->background_color = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);

	m_illuminatedColor = Eigen::Vector4f(1.0, 1.0, 1.0f, 1.0f);
	m_baseColor = Eigen::Vector4f(0.356863, 0.364706, 0.494118, 1.0f);


	m_opacity = 0.005f;
	m_illuminationMultiplier = 12;
	m_shadowKMultiplier = 2;

	m_radius = 0.25f;
	m_lightPosition = Eigen::Vector3f(256.f, 200.f, 256.f);


	Eigen::MatrixXd vertices;
	Eigen::MatrixXi triangles;

	//loadFromFile_obj("C:/Users/Jonas Zehnder/Desktop/sphere.obj", vertices, triangles);

	SimpleTriMesh::icosphere(4).toMatrices(vertices, triangles);
	m_sphere.setMesh(vertices, triangles);

}

struct ObsParticle
{
	Eigen::Vector3f x;
	int type;
	Eigen::Vector3f x1, x2;
};
template<typename T>
void adjustNumberOfParticles(double dx, const Eigen::Vector3d &x0, T &sdfFunction,
	std::vector<ObsParticle> &vtcs, int nd, int np, int pType)
{
	std::vector<std::vector<std::vector<int> > > cnts(nd, std::vector<std::vector<int> >(nd, std::vector<int>(nd, 0)));
	for (const ObsParticle &p : vtcs)
	{
		Eigen::Vector3d df = ((p.x.cast<double>() - x0) / dx).array();
		Eigen::Vector3i idcs;
		for (int j = 0; j < 3; j++) idcs[j] = (int)std::floor(df[j]);

		bool outside = false;
		for (int j = 0; j < 3; j++)
		{
			if (idcs[j] < 0) outside = true;
			if (idcs[j] >= nd) outside = true;
		}
		if (outside) continue;

		cnts[idcs[0]][idcs[1]][idcs[2]] += 1;
	}

	for (int i = 0; i < nd; i++)
	{
		for (int j = 0; j < nd; j++)
		{
			for (int k = 0; k < nd; k++)
			{
				if (cnts[i][j][k] < np)
				{
					Eigen::Vector3f xc = x0.cast<float>() + Eigen::Vector3f(i + 0.5f, j + 0.5f, k + 0.5f) * dx;
					float sdf = sdfFunction(xc);
					//if (sdf > 0.0f) continue;

					//estimate based on source.sdf(xc) how much particles should be tried to be added
					// if sdf == 0, we assume half of volume is inside sdf
					// if sdf == -dx, we assume full volume is inside sdf
					// V(0) = 0.5, V(-dx) = 1.0
					// V(sdf) = v0 + dv * sdf
					// => v0 = 0.5, V(-dx) = 1.0 = 0.5 + (-dx) * dv 
					// => dv = 0.5 / (-dx)
					float targetRatio = 0.5 + sdf / (-dx);
					int targetNumberOfParticles = std::max(0, std::min(np, (int)std::ceil(targetRatio * np)));
					int particlesToAdd = targetNumberOfParticles - cnts[i][j][k];

					int numberOfTries = 5 * np;
					for (int l = 0; l < numberOfTries && particlesToAdd > 0; l++)
					{
						Eigen::Vector3f x = xc + Eigen::Vector3f::Random() * (0.5f * float(dx));
						if (sdfFunction(x) <= 0.f)
						{
							ObsParticle p;
							p.x = x;
							p.type = pType;
							vtcs.push_back(p);
							particlesToAdd -= 1;
						}
					}
				}
			}
		}
	}
}


void PlumeObsRenderer::setInitialParticles(ParticlesManaged &particles)
{
	int res = 128;
	float sigma = 0.5;//instead of considering sigma later, we add it here to the radius
	Sphere source(float(res)*Eigen::Vector3f(0.5, 0.1, 0.5), res*0.05 + 1.5 * sigma);

	std::vector<ObsParticle> vtcs;

	int nd = (int)std::floor(std::pow(1.5e6, 1.0 / 3.0));
	int np = 1;

	double dx = (source.getRadius() * 2) / nd;
	Eigen::Vector3d x0 = source.getCenter().cast<double>() - Eigen::Vector3d::Constant(source.getRadius());
	adjustNumberOfParticles(dx, x0, [&source](const Eigen::Vector3f& x)
	{
		return source.sdf(x);
	}, vtcs, nd, np, 0);

	//////sample obstacle
	////int res = 128;
	Eigen::Vector3f sphereCenter = float(res)*Eigen::Vector3f(0.5, 1.0, 0.5);
	float sphereRadius = res*0.1;
	//float sphereRadius2 = 0.7 * sphereRadius;
	//np = 6;
	//dx = (sphereRadius * 2) / nd;
	//x0 = sphereCenter.cast<double>() - Eigen::Vector3d::Constant(sphereRadius);
	//adjustNumberOfParticles(dx, x0, [sphereRadius, sphereRadius2, &sphereCenter](const Eigen::Vector3f& x)
	//{
	//	float sdf1 = (x - sphereCenter).norm() - sphereRadius;
	//	//float sdf2 =  - ( (x - sphereCenter).norm() - sphereRadius2);
	//	//return std::max(sdf1, sdf2);
	//	return sdf1;
	//}, vtcs, nd, np, 1);

	Eigen::MatrixXd obsVertices;
	Eigen::MatrixXi obsTriangles;

	//loadFromFile_obj("C:/Users/Jonas Zehnder/Desktop/sphere.obj", obsVertices, obsTriangles);
	SimpleTriMesh::icosphere(4).toMatrices(obsVertices, obsTriangles);
	for (int triIdx = 0; triIdx < obsTriangles.cols(); triIdx++)
	{
		auto transform = [sphereRadius, &sphereCenter](const Eigen::Vector3f& v)
		{
			Eigen::Vector3f p = v * sphereRadius + sphereCenter;
			return p;
		};
		Eigen::Vector3f x1 = transform(obsVertices.col(obsTriangles(0, triIdx)).cast<float>());
		Eigen::Vector3f x2 = transform(obsVertices.col(obsTriangles(1, triIdx)).cast<float>());
		Eigen::Vector3f x3 = transform(obsVertices.col(obsTriangles(2, triIdx)).cast<float>());
		ObsParticle p;
		p.x = (x1 + x2 + x3) / 3.0;
		p.type = 1;
		p.x1 = x1;
		p.x2 = x2;
		vtcs.push_back(p);
	}

	Eigen::MatrixXf vertices(3, vtcs.size());
	Eigen::VectorXi types(vtcs.size());
	Eigen::MatrixXf additionalData1 = Eigen::MatrixXf::Zero(3, vertices.cols());
	Eigen::MatrixXf additionalData2 = Eigen::MatrixXf::Zero(3, vertices.cols());
	for (int i = 0; i < vtcs.size(); i++)
	{
		vertices.col(i) = vtcs[i].x;
		types[i] = vtcs[i].type;
		additionalData1.col(i) = vtcs[i].x1;
		additionalData2.col(i) = vtcs[i].x2;
	}
	Eigen::MatrixXf colors = Eigen::MatrixXf::Zero(3, vertices.cols());

	particles.setParticles(vertices, colors, types, additionalData1, additionalData2);
}
void PlumeObsRenderer::updateParticles(ParticlesManaged &particles)
{
	int res = 128;
	float sigma = 0.5;//instead of considering sigma later, we add it here to the radius
	Sphere source(float(res)*Eigen::Vector3f(0.5, 0.1, 0.5), res*0.05 + 1.5 * sigma);

	int nd = (int)std::floor(std::pow(1.5e6, 1.0 / 3.0));
	int np = 1;

	particles.transferGPUToCPU();

	Eigen::Vector3i dims = getVelocityGridDims();
	Eigen::Vector3f xmin(5.0f, 2.0f, 5.0f);
	Eigen::Vector3f xmax = dims.cast<float>() - Eigen::Vector3f(5.f, 8.f, 5.f);

	std::vector<ObsParticle> vtcs;

	for (int i = 0; i < particles.getParticleCount(); i++)
	{
		Eigen::Vector3f x = particles.getCPUParticles().getVertices().col(i);
		if ((x.array() < xmin.array()).any()
			|| (x.array() > xmax.array()).any())
		{
			continue;
		}
		ObsParticle p;
		p.x = x;
		p.type = particles.getCPUParticles().getTypes()[i];
		p.x1 = particles.getCPUParticles().getAdditionalData1().col(i);
		p.x2 = particles.getCPUParticles().getAdditionalData2().col(i);
		vtcs.push_back(p);
	}

	double dx = (source.getRadius() * 2) / nd;
	Eigen::Vector3d x0 = source.getCenter().cast<double>() - Eigen::Vector3d::Constant(source.getRadius());
	adjustNumberOfParticles(dx, x0, [&source](const Eigen::Vector3f& x)
	{
		return source.sdf(x);
	}, vtcs, nd, np, 0);

	Eigen::MatrixXf vertices(3, vtcs.size());
	Eigen::VectorXi types(vtcs.size());
	Eigen::MatrixXf additionalData1 = Eigen::MatrixXf::Zero(3, vertices.cols());
	Eigen::MatrixXf additionalData2 = Eigen::MatrixXf::Zero(3, vertices.cols());
	for (int i = 0; i < vtcs.size(); i++)
	{
		vertices.col(i) = vtcs[i].x;
		types[i] = vtcs[i].type;
		additionalData1.col(i) = vtcs[i].x1;
		additionalData2.col(i) = vtcs[i].x2;
	}
	Eigen::MatrixXf colors = Eigen::MatrixXf::Zero(3, vertices.cols());

	particles.setParticles(vertices, colors, types, additionalData1, additionalData2);
}

std::string PlumeObsRenderer::getVelocityFilename(int frame) const
{
	std::string filepattern = m_velocityFolder + std::string("/velocityGrid_%04d.uni");

	char filename[10000];
	sprintf(filename, filepattern.c_str(), frame);
	return filename;
}
std::string PlumeObsRenderer::getOutputFolder() const
{
	return m_outputFolder;
}
void PlumeObsRenderer::renderSolids(const Eigen::Matrix4f &proj, const Eigen::Matrix4f &view)
{
	int res = 128;
	Eigen::Vector3f sphereCenter = float(res)*Eigen::Vector3f(0.5, 1.0, 0.5);
	float sphereRadius = res*0.1;

	Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
	model(0, 0) = sphereRadius;
	model(1, 1) = sphereRadius;
	model(2, 2) = sphereRadius;
	model(0, 3) = sphereCenter[0];
	model(1, 3) = sphereCenter[1];
	model(2, 3) = sphereCenter[2];

	Eigen::Matrix4f modelView = view * model.cast<float>();
	Eigen::Matrix4f mvp = proj * modelView;

	Eigen::Matrix3f modelInverseTranspose = (model.topLeftCorner<3, 3>().inverse().transpose()).eval().cast<float>();

	Eigen::Vector4f color(1.0f, 0.0f, 0.0f, 1.0);

	Eigen::Vector3f cameraPosition = m_camera.computePosition();

	assert(glGetError() == GL_NO_ERROR);
	m_program->use();
	assert(glGetError() == GL_NO_ERROR);
	m_program->setUniform("color", color);
	m_program->setUniform("MVP", mvp);
	m_program->setUniform("Model", Eigen::Matrix4f(model.cast<float>()));
	m_program->setUniform("ModelInverseTranspose", modelInverseTranspose);
	m_program->setUniform("lighting_factor", 0.6f);
	m_program->setUniform("cameraPositionWorld", cameraPosition);
	assert(glGetError() == GL_NO_ERROR);

	m_sphere.draw();

	m_program->unbind();
}
void PlumeObsRenderer::init()
{
	SuperClass::init();

	m_sphere.initGL();

	m_program.reset(new ProgramGL(ProgramGL::loadFromFiles("Sphere.vs", "Sphere.fs")));
}
void PlumeObsRenderer::shut()
{
	m_sphere.shutGL();

	SuperClass::shut();
}
