cmake_minimum_required(VERSION 3.5.0)

project(YAPS)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

project(YAPS LANGUAGES CXX CUDA)

file(GLOB_RECURSE source_files src/*.h src/*.cpp src/*.cu)

find_package(ZLIB REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PNG REQUIRED)
find_package(Qt5Core QUIET)
if (Qt5Core_FOUND)
	message("Using Qt5")
	find_package(Qt5Widgets REQUIRED)
	find_package(Qt5OpenGL REQUIRED)
endif(Qt5Core_FOUND)

add_executable(${PROJECT_NAME} ${source_files} ${CMAKE_CURRENT_LIST_DIR}/glad/src/glad.c)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
#add_definitions(${PROJECT_NAME} ${PNG_DEFINITIONS})

set(PARTICLE_RENDERER_RES_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/res)

target_include_directories(${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_LIST_DIR}/src
		${ZLIB_INCLUDE_DIRS}
		${EIGEN3_INCLUDE_DIR}
		${CMAKE_CURRENT_LIST_DIR}/glad/include
		${PNG_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
	Qt5::Widgets
	Qt5::OpenGL
	${ZLIB_LIBRARIES}
	${PNG_LIBRARIES}
	)
target_compile_definitions(${PROJECT_NAME}
	PUBLIC
		SHADER_FOLDER="${CMAKE_CURRENT_LIST_DIR}/res/"
		YAPS_MANTA_FOLDER="${CMAKE_CURRENT_LIST_DIR}/manta-scenes/"
	)