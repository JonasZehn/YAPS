#ifndef RENDERER_H
#define RENDERER_H

#include "Camera.h"
#include "GL.h"
#include "Program.h"
#include "Texture.h"
#include "Framebuffer.h"
#include "TexQuad.h"
#include "GPUGrid.h"
#include "Particles.h"
#include <Eigen/Core>

#include <memory>

class Renderer
{
public:
	
	Renderer();
	virtual ~Renderer() {}
	virtual void init();
	virtual void shut() {}
	void computeIllumination();
	void renderToTexture();

	void step();
	void saveToFile();
	void renderToScreen(int screenWidth, int screenHeight);

	float getRadius() const;
	float getOpacity() const;
	Eigen::Vector3f getColor() const;

	virtual void setInitialParticles(ParticlesManaged &particles) = 0;
	virtual void updateParticles(ParticlesManaged &particles) {}

	virtual Eigen::Vector3i getVelocityGridDims() const = 0;
	virtual Eigen::Vector2i getOutputDims() const = 0;
	virtual std::string getVelocityFilename(int frame) const = 0;
	virtual std::string getOutputFolder() const = 0;


	virtual float getTimestep() const = 0;

	int frame;
	bool m_play;
	Camera m_camera;
	Eigen::Vector4f background_color;
	Eigen::Vector4f m_baseColor;
	Eigen::Vector4f m_illuminatedColor;
	float m_radius;
	float m_illuminationMultiplier;
	float m_shadowKMultiplier;
	float m_opacity;
	int m_pauseAtFrame;
	Eigen::Vector3f m_lightPosition;
protected:

	std::unique_ptr<ProgramGL> m_program;
	ParticlesManaged particles;
	std::unique_ptr<TextureBuffer> m_particleIllTexture;
	std::unique_ptr<Texture2D> m_texture;
	std::unique_ptr<ProgramGL> m_illuminationProgram;
	std::unique_ptr<Framebuffer> m_illuminationFramebuffer;
	std::unique_ptr<FramebufferHDR> m_hdrFramebuffer;

	TexQuad m_quad;
	std::unique_ptr<ProgramGL> m_quadProgram;

	std::unique_ptr<GPUGrid> m_grid;
	std::unique_ptr<GPUGrid> m_grid2;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class PlumeRenderer : public Renderer
{
public:
	PlumeRenderer(
		const std::string &velocityFolder,
		const std::string &outputFolder);

	virtual void setInitialParticles(ParticlesManaged &particles);
	virtual void updateParticles(ParticlesManaged &particles);

	virtual Eigen::Vector3i getVelocityGridDims() const override
	{
		return Eigen::Vector3i(128, 256, 128);
	}

	virtual std::string getVelocityFilename(int frame) const;
	virtual std::string getOutputFolder() const;

	virtual float getTimestep() const
	{
		return 1.0f;
	}
	virtual Eigen::Vector2i getOutputDims() const
	{
		return Eigen::Vector2i(1080, 1920);
	}

private:
	std::string m_velocityFolder;
	std::string m_outputFolder;
	double m_nTarget;
};

class PlumeObsRenderer : public Renderer
{
public:
	typedef Renderer SuperClass;

	PlumeObsRenderer(
		const std::string &velocityFolder,
		const std::string &outputFolder);

	virtual void setInitialParticles(ParticlesManaged &particles);
	virtual void updateParticles(ParticlesManaged &particles);

	virtual Eigen::Vector3i getVelocityGridDims() const override
	{
		return Eigen::Vector3i(128, 256, 128);
	}

	virtual std::string getVelocityFilename(int frame) const;
	virtual std::string getOutputFolder() const;

	virtual float getTimestep() const
	{
		return 1.0f;
	}
	virtual Eigen::Vector2i getOutputDims() const
	{
		return Eigen::Vector2i(1080, 1920);
	}

	virtual void renderSolids(const Eigen::Matrix4f &proj, const Eigen::Matrix4f &view);

protected:
	virtual void init() override;
	virtual void shut() override;

private:
	std::string m_velocityFolder;
	std::string m_outputFolder;
	MeshManaged m_sphere;

	std::unique_ptr<ProgramGL> m_program;
};

#endif